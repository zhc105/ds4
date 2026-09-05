#!/usr/bin/env python3
"""Shared pieces for converting Hugging Face safetensors checkpoints to GGUF.

Everything here is numpy-only: no torch, no GGML.  The output follows the
llama.cpp GGUF conventions (tensor names, key names, quant block layouts) so
that ds4 can read either our files or upstream-converted ones with one loader.

NVFP4 is stored as GGML type 40: 64-element super-blocks of four UE4M3 scales
followed by 32 packed E2M1 bytes.  The per-tensor ModelOpt global scale
(`weight_scale_2`) is not folded into the block scales; it is written as a
companion F32 tensor `<name>.scale` (and `<name>.input_scale` when present) so
the runtime can hand exact E4M3 block scales to Blackwell tensor cores.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# GGUF metadata value types.
KV_UINT32, KV_INT32, KV_FLOAT32, KV_BOOL, KV_STRING, KV_ARRAY, KV_UINT64 = 4, 5, 6, 7, 8, 9, 10

# GGML tensor types we emit.
T_F32, T_F16, T_BF16, T_NVFP4 = 0, 1, 30, 40
TYPE_NAMES = {T_F32: "F32", T_F16: "F16", T_BF16: "BF16", T_NVFP4: "NVFP4"}

# llama.cpp token types.
TOK_NORMAL, TOK_CONTROL, TOK_USER_DEFINED, TOK_UNUSED = 1, 3, 4, 5

NVFP4_BLOCK = 16          # elements per E4M3 scale
NVFP4_SUPER = 64          # elements per GGML super-block
NVFP4_SUPER_BYTES = 36    # 4 scales + 32 packed bytes

SAFETENSORS_DTYPES = {
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),   # raw bits; see bf16_to_f32()
    "F8_E4M3": np.dtype("u1"),
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "I32": np.dtype("<i4"),
    "I64": np.dtype("<i8"),
}


def fail(message):
    print(f"hf-gguf: {message}", file=sys.stderr)
    sys.exit(1)


def align(value, alignment=GGUF_ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values):
    """Round-to-nearest-even BF16, matching torch's conversion."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounding = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + rounding) >> 16).astype(np.uint16)


# ---------------------------------------------------------------------------
# safetensors reading
# ---------------------------------------------------------------------------

class SafeTensors:
    """Read-only view of a single-file or sharded safetensors checkpoint."""

    def __init__(self, hf_dir):
        self.hf_dir = hf_dir
        self.entries = {}   # name -> (path, dtype string, shape, start, end)
        self._maps = {}
        index_path = os.path.join(hf_dir, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as fp:
                weight_map = json.load(fp)["weight_map"]
            shards = sorted(set(weight_map.values()))
        elif os.path.isfile(os.path.join(hf_dir, "model.safetensors")):
            shards = ["model.safetensors"]
        else:
            fail(f"no safetensors checkpoint found in {hf_dir}")
        for shard in shards:
            self._index_shard(os.path.join(hf_dir, shard))

    def _index_shard(self, path):
        size = os.path.getsize(path)
        with open(path, "rb") as fp:
            (header_len,) = struct.unpack("<Q", fp.read(8))
            if header_len > size - 8 or header_len > (1 << 30):
                fail(f"{path}: bad safetensors header length")
            header = json.loads(fp.read(header_len))
        base = 8 + header_len
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if entry["dtype"] not in SAFETENSORS_DTYPES:
                fail(f"{path}: unsupported dtype {entry['dtype']} for {name}")
            start, end = entry["data_offsets"]
            nbytes = int(np.prod(entry["shape"], dtype=np.int64)) * SAFETENSORS_DTYPES[entry["dtype"]].itemsize
            if end - start != nbytes or base + end > size:
                fail(f"{path}: inconsistent data span for {name}")
            if name in self.entries:
                fail(f"duplicate tensor {name} across shards")
            self.entries[name] = (path, entry["dtype"], tuple(entry["shape"]), base + start, base + end)

    def __contains__(self, name):
        return name in self.entries

    def names(self):
        return list(self.entries)

    def dtype(self, name):
        return self.entries[name][1]

    def shape(self, name):
        return self.entries[name][2]

    def _map(self, path):
        if path not in self._maps:
            fd = os.open(path, os.O_RDONLY)
            try:
                self._maps[path] = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            finally:
                os.close(fd)
        return self._maps[path]

    def read(self, name):
        """Return the tensor as a numpy array over the mmap (BF16 as uint16 bits)."""
        if name not in self.entries:
            fail(f"missing tensor {name}")
        path, dtype, shape, start, end = self.entries[name]
        view = np.frombuffer(self._map(path), dtype=SAFETENSORS_DTYPES[dtype], offset=start,
                             count=(end - start) // SAFETENSORS_DTYPES[dtype].itemsize)
        return view.reshape(shape)

    def read_f32(self, name):
        data = self.read(name)
        if self.dtype(name) == "BF16":
            return bf16_to_f32(data)
        return data.astype(np.float32)

    def close(self):
        for m in self._maps.values():
            m.close()
        self._maps.clear()


# ---------------------------------------------------------------------------
# NVFP4 repacking (ModelOpt layout -> GGML type 40 + companion scales)
# ---------------------------------------------------------------------------

def nvfp4_repack(weight, scale):
    """Repack ModelOpt NVFP4 (uint8 [out, in/2] nibble pairs, E4M3 [out, in/16])
    into GGML super-blocks.  Returns (raw uint8 [out, nsuper*36], [out, in]).

    ModelOpt packs adjacent elements (2j low nibble, 2j+1 high nibble); GGML
    packs element j in the low nibble and j+8 in the high nibble of byte j
    within each 16-element block.  Scale bits are kept verbatim minus the sign
    bit, so the block scales stay exact E4M3 values."""
    out_features, half = weight.shape
    n_blocks = half * 2 // NVFP4_BLOCK
    if scale.shape != (out_features, n_blocks):
        fail(f"NVFP4 scale shape {scale.shape} does not match weight {weight.shape}")
    if (half * 2) % NVFP4_SUPER != 0:
        fail(f"NVFP4 row length {half * 2} is not a multiple of {NVFP4_SUPER}")
    w = weight.reshape(out_features, n_blocks, 8)
    vals = np.stack([w & 0x0F, w >> 4], axis=-1).reshape(out_features, n_blocks, 16)
    qs = (vals[:, :, :8] | (vals[:, :, 8:] << 4)).astype(np.uint8)
    n_super = n_blocks // 4
    d = (scale.view(np.uint8) & 0x7F).reshape(out_features, n_super, 4)
    raw = np.concatenate([d, qs.reshape(out_features, n_super, 32)], axis=-1)
    return np.ascontiguousarray(raw.reshape(out_features, n_super * NVFP4_SUPER_BYTES)), [out_features, half * 2]


_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
_E2M1_MID = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)   # decision points


def f32_to_ue4m3(x):
    """Nearest E4M3 code (no sign) for non-negative float32, saturating at 448."""
    x = np.minimum(np.asarray(x, dtype=np.float32), np.float32(448.0))
    exp = np.floor(np.log2(np.maximum(x, np.float32(2.0 ** -9)))).astype(np.int32)
    exp = np.clip(exp, -6, 8)
    man = np.rint(x / np.exp2(exp).astype(np.float32) * 8.0 - 8.0).astype(np.int32)   # normal: 1.m * 2^exp
    carry = man >= 8
    exp = np.where(carry, exp + 1, exp)
    man = np.where(carry, 0, man)
    sub = np.rint(x / np.float32(2.0 ** -9)).astype(np.int32)                         # subnormal: m * 2^-9
    code = np.where(x < np.float32(2.0 ** -6), np.clip(sub, 0, 7), np.clip(exp + 7, 1, 15) * 8 + man)
    return np.minimum(code, 0x7E).astype(np.uint8)


def nvfp4_quantize(weight):
    """Quantize a float32 [out, in] matrix the ModelOpt way: one global scale
    (amax / (6 * 448)), an E4M3 scale per 16 values (block amax / 6 over the
    global scale) and nearest E2M1 codes, packed as GGML super-blocks.
    Returns (raw uint8 [out, nsuper*36], global scale)."""
    out_features, n_cols = weight.shape
    if n_cols % NVFP4_SUPER != 0:
        fail(f"NVFP4 row length {n_cols} is not a multiple of {NVFP4_SUPER}")
    w = np.asarray(weight, dtype=np.float32)
    amax = float(np.max(np.abs(w))) if w.size else 0.0
    scale2 = np.float32(amax / (6.0 * 448.0)) if amax > 0 else np.float32(1.0)
    blocks = w.reshape(out_features, n_cols // NVFP4_BLOCK, NVFP4_BLOCK)
    bmax = np.max(np.abs(blocks), axis=-1)
    d = f32_to_ue4m3(bmax / 6.0 / scale2)                                            # [out, n_blocks]
    step = ue4m3_to_f32(d) * scale2
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(step[..., None] > 0, blocks / step[..., None], 0.0).astype(np.float32)
    mag = np.searchsorted(_E2M1_MID, np.abs(q), side="right").astype(np.uint8)       # 0..7
    vals = np.where(q < 0, mag | 8, mag).astype(np.uint8)
    qs = (vals[:, :, :8] | (vals[:, :, 8:] << 4)).astype(np.uint8)                   # element j low, j+8 high
    n_super = n_cols // NVFP4_SUPER
    raw = np.concatenate([d.reshape(out_features, n_super, 4), qs.reshape(out_features, n_super, 32)], axis=-1)
    return np.ascontiguousarray(raw.reshape(out_features, n_super * NVFP4_SUPER_BYTES)), scale2


def ue4m3_to_f32(bits):
    """E4M3 (sign stripped, 0x7F = NaN treated as 0) to float32."""
    bits = np.asarray(bits, dtype=np.uint8) & 0x7F
    exp = (bits >> 3).astype(np.int32)
    man = (bits & 7).astype(np.float32)
    value = np.where(exp == 0, man * 2.0 ** -9, (1.0 + man / 8.0) * np.exp2(exp - 7).astype(np.float32))
    return np.where(bits == 0x7F, 0.0, value).astype(np.float32)


def nvfp4_dequant_modelopt(weight, scale, scale2):
    """Reference dequantization straight from the ModelOpt tensors."""
    out_features, half = weight.shape
    vals = np.stack([weight & 0x0F, weight >> 4], axis=-1).reshape(out_features, half * 2)
    d = np.repeat(ue4m3_to_f32(scale), NVFP4_BLOCK, axis=1)
    return _E2M1[vals] * d * np.float32(scale2)


def nvfp4_dequant_gguf(raw, n_cols, scale2):
    """Reference dequantization of the GGML super-block layout."""
    out_features = raw.shape[0]
    n_super = n_cols // NVFP4_SUPER
    blocks = raw.reshape(out_features, n_super, NVFP4_SUPER_BYTES)
    d = ue4m3_to_f32(blocks[:, :, :4])                     # [out, nsuper, 4]
    qs = blocks[:, :, 4:].reshape(out_features, n_super, 4, 8)
    vals = np.concatenate([qs & 0x0F, qs >> 4], axis=-1)   # [out, nsuper, 4, 16]
    return (_E2M1[vals] * d[..., None] * np.float32(scale2)).reshape(out_features, n_cols)


def nvfp4_stack_experts(read_weight, read_scale, n_expert):
    """Stack per-expert ModelOpt NVFP4 tensors into one [n_expert][out][in]
    GGML NVFP4 tensor.  read_weight(e) / read_scale(e) return expert e's
    packed weight and E4M3 scale arrays; every expert must share one shape."""
    raw0, shape = nvfp4_repack(read_weight(0), read_scale(0))
    out = np.empty((n_expert,) + raw0.shape, dtype=np.uint8)
    out[0] = raw0
    for e in range(1, n_expert):
        raw, s = nvfp4_repack(read_weight(e), read_scale(e))
        if s != shape:
            fail(f"expert {e} has shape {s}, expected {shape}")
        out[e] = raw
    return out, [n_expert] + shape


# ---------------------------------------------------------------------------
# PLE n-gram table sidecar (ds4 .ngram file, not GGUF)
# ---------------------------------------------------------------------------

NGRAM_MAGIC = b"DS4NGRAM"
NGRAM_VERSION = 1
NGRAM_HEADER_BYTES = 4096      # rows start page aligned
NGRAM_DTYPE_E4M3 = 1


def ngram_header(rows, row_bytes, global_scale, multipliers, head_offsets, head_vocab_sizes):
    """Fixed 4 KiB header: magic, version, layout, hash constants.  Row i of
    the table lives at NGRAM_HEADER_BYTES + i * row_bytes; row = 16 heads x
    (row_bytes / 16) E4M3 values scaled by global_scale."""
    if len(head_offsets) != len(head_vocab_sizes):
        fail("PLE head offsets and vocab sizes differ in length")
    body = NGRAM_MAGIC
    body += struct.pack("<IIQIfII", NGRAM_VERSION, row_bytes, rows, NGRAM_DTYPE_E4M3, global_scale,
                        len(multipliers), len(head_offsets))
    body += struct.pack(f"<{len(multipliers)}Q", *multipliers)
    body += struct.pack(f"<{len(head_offsets)}Q", *head_offsets)
    body += struct.pack(f"<{len(head_vocab_sizes)}Q", *head_vocab_sizes)
    if len(body) > NGRAM_HEADER_BYTES:
        fail("PLE header does not fit")
    return body + bytes(NGRAM_HEADER_BYTES - len(body))


def read_ngram_header(path):
    with open(path, "rb") as fp:
        raw = fp.read(NGRAM_HEADER_BYTES)
    if raw[:8] != NGRAM_MAGIC:
        fail(f"{path}: not a ds4 n-gram table")
    version, row_bytes, rows, dtype, scale, n_mult, n_heads = struct.unpack_from("<IIQIfII", raw, 8)
    pos = 8 + struct.calcsize("<IIQIfII")
    multipliers = list(struct.unpack_from(f"<{n_mult}Q", raw, pos))
    pos += 8 * n_mult
    offsets = list(struct.unpack_from(f"<{n_heads}Q", raw, pos))
    pos += 8 * n_heads
    sizes = list(struct.unpack_from(f"<{n_heads}Q", raw, pos))
    return {"version": version, "row_bytes": row_bytes, "rows": rows, "dtype": dtype, "scale": scale,
            "multipliers": multipliers, "head_offsets": offsets, "head_vocab_sizes": sizes}


# ---------------------------------------------------------------------------
# Linear-attention V head reorder (grouped -> tiled), as llama.cpp expects
# ---------------------------------------------------------------------------

def v_head_perm(num_k_heads, num_v_per_k):
    """HF stores V heads grouped by K head [G0_v0..G0_v{r-1}, G1_v0..]; GGML
    wants them tiled [G0_v0, G1_v0, .., G0_v1, G1_v1, ..] so a plain repeat of
    the K heads lines up.  Returns the source head index for each output slot."""
    return np.arange(num_k_heads * num_v_per_k).reshape(num_k_heads, num_v_per_k).T.reshape(-1)


def reorder_heads(array, axis, perm, head_size):
    """Permute `axis` in units of head_size according to perm."""
    if np.array_equal(perm, np.arange(len(perm))):
        return array
    shape = list(array.shape)
    if shape[axis] != len(perm) * head_size:
        fail(f"head reorder: axis {axis} has {shape[axis]} entries, expected {len(perm) * head_size}")
    grouped = array.reshape(shape[:axis] + [len(perm), head_size] + shape[axis + 1:])
    return np.ascontiguousarray(np.take(grouped, perm, axis=axis)).reshape(shape)


# ---------------------------------------------------------------------------
# GGUF writing
# ---------------------------------------------------------------------------

def _pack_string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


class GGUFWriter:
    """Collects metadata and lazily-produced tensors, then streams the file."""

    def __init__(self, path):
        self.path = path
        self.kv = []
        self.tensors = []   # (name, shape, type, nbytes, produce)
        self._names = set()

    def _add(self, key, payload):
        self.kv.append(_pack_string(key) + payload)

    def add_u32(self, key, value):
        self._add(key, struct.pack("<II", KV_UINT32, value))

    def add_u64(self, key, value):
        self._add(key, struct.pack("<IQ", KV_UINT64, value))

    def add_f32(self, key, value):
        self._add(key, struct.pack("<If", KV_FLOAT32, value))

    def add_bool(self, key, value):
        self._add(key, struct.pack("<I?", KV_BOOL, value))

    def add_string(self, key, value):
        self._add(key, struct.pack("<I", KV_STRING) + _pack_string(value))

    def add_i32_array(self, key, values):
        """Integer arrays are INT32: llama.cpp's vocab loader insists on it for token types."""
        self._add(key, struct.pack("<IIQ", KV_ARRAY, KV_INT32, len(values)) + struct.pack(f"<{len(values)}i", *values))

    def add_u64_array(self, key, values):
        """PLE hash constants: 45-bit multipliers and row ranges past int32."""
        self._add(key, struct.pack("<IIQ", KV_ARRAY, KV_UINT64, len(values)) + struct.pack(f"<{len(values)}Q", *values))

    def add_string_array(self, key, values):
        self._add(key, struct.pack("<IIQ", KV_ARRAY, KV_STRING, len(values)) + b"".join(_pack_string(v) for v in values))

    def add_tensor(self, name, shape, qtype, nbytes, produce):
        """shape is in numpy (row-major) order; GGUF stores it reversed.
        produce() must return exactly nbytes of contiguous data."""
        if name in self._names:
            fail(f"duplicate output tensor {name}")
        self._names.add(name)
        self.tensors.append((name, [int(d) for d in shape], qtype, int(nbytes), produce))

    def add_array(self, name, array, qtype):
        array = np.ascontiguousarray(array)
        self.add_tensor(name, array.shape, qtype, array.nbytes, lambda a=array: a)

    def data_offset(self):
        header = 4 + 4 + 8 + 8 + sum(len(r) for r in self.kv)
        for name, shape, _, _, _ in self.tensors:
            header += len(_pack_string(name)) + 4 + 8 * len(shape) + 4 + 8
        return align(header)

    def write(self, log=None):
        data_offset = self.data_offset()
        offsets = []
        cursor = 0
        for _, _, _, nbytes, _ in self.tensors:
            offsets.append(cursor)
            cursor += align(nbytes)
        with open(self.path, "wb") as fp:
            fp.write(GGUF_MAGIC)
            fp.write(struct.pack("<IQQ", GGUF_VERSION, len(self.tensors), len(self.kv)))
            for record in self.kv:
                fp.write(record)
            for (name, shape, qtype, _, _), offset in zip(self.tensors, offsets):
                fp.write(_pack_string(name))
                fp.write(struct.pack("<I", len(shape)))
                fp.write(struct.pack(f"<{len(shape)}Q", *reversed(shape)))
                fp.write(struct.pack("<IQ", qtype, offset))
            fp.write(bytes(data_offset - fp.tell()))
            for index, (name, shape, qtype, nbytes, produce) in enumerate(self.tensors):
                data = np.ascontiguousarray(produce())
                if data.nbytes != nbytes:
                    fail(f"{name}: produced {data.nbytes} bytes, planned {nbytes}")
                fp.write(memoryview(data).cast("B"))
                fp.write(bytes(align(nbytes) - nbytes))
                if log:
                    log(index, name, shape, qtype)
        return data_offset + cursor


# ---------------------------------------------------------------------------
# GGUF reading (enough to validate what we wrote)
# ---------------------------------------------------------------------------

def read_gguf(path):
    """Parse a GGUF file: returns (kv dict, {name: (shape, type, offset, nbytes)}, data_offset)."""
    def read_value(fp, vtype):
        if vtype == KV_STRING:
            (n,) = struct.unpack("<Q", fp.read(8))
            return fp.read(n).decode("utf-8")
        if vtype == KV_ARRAY:
            etype, n = struct.unpack("<IQ", fp.read(12))
            return [read_value(fp, etype) for _ in range(n)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[vtype]
        return struct.unpack(fmt, fp.read(struct.calcsize(fmt)))[0]

    kv = {}
    tensors = {}
    with open(path, "rb") as fp:
        if fp.read(4) != GGUF_MAGIC:
            fail(f"{path}: not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", fp.read(20))
        if version != GGUF_VERSION:
            fail(f"{path}: unsupported GGUF version {version}")
        for _ in range(n_kv):
            key = read_value(fp, KV_STRING)
            (vtype,) = struct.unpack("<I", fp.read(4))
            kv[key] = read_value(fp, vtype)
        infos = []
        for _ in range(n_tensors):
            name = read_value(fp, KV_STRING)
            (n_dims,) = struct.unpack("<I", fp.read(4))
            ne = struct.unpack(f"<{n_dims}Q", fp.read(8 * n_dims))
            qtype, offset = struct.unpack("<IQ", fp.read(12))
            infos.append((name, list(reversed(ne)), qtype, offset))
        data_offset = align(fp.tell(), kv.get("general.alignment", GGUF_ALIGNMENT))
    for name, shape, qtype, offset in infos:
        tensors[name] = (shape, qtype, offset, tensor_nbytes(shape, qtype))
    return kv, tensors, data_offset


def tensor_nbytes(shape, qtype):
    n = int(np.prod(shape, dtype=np.int64))
    if qtype == T_F32:
        return n * 4
    if qtype in (T_F16, T_BF16):
        return n * 2
    if qtype == T_NVFP4:
        return n // NVFP4_SUPER * NVFP4_SUPER_BYTES
    fail(f"unknown tensor type {qtype}")


# ---------------------------------------------------------------------------
# Tokenizer export (GPT-2 style byte-level BPE from tokenizer.json)
# ---------------------------------------------------------------------------

def looks_special(token):
    return (token.startswith("<|") and token.endswith("|>")) or \
           (token.startswith("<｜") and token.endswith("｜>")) or \
           (token.startswith("<unused") and token.endswith(">")) or \
           token in ("<pad>", "<mask>", "<2mass>", "[@BOS@]")


def export_tokenizer(writer, hf_dir, vocab_size, pre_type):
    """Emit tokenizer.ggml.* keys the way llama.cpp's converter does for BPE models."""
    with open(os.path.join(hf_dir, "tokenizer.json"), encoding="utf-8") as fp:
        tok = json.load(fp)
    if tok["model"]["type"] != "BPE":
        fail(f"unsupported tokenizer model {tok['model']['type']}")
    vocab = dict(tok["model"]["vocab"])
    added = {entry["id"]: entry for entry in tok.get("added_tokens", [])}
    config_path = os.path.join(hf_dir, "tokenizer_config.json")
    config = {}
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as fp:
            config = json.load(fp)
    # tokenizer_config.json may declare added tokens that tokenizer.json lacks
    # (transformers merges both); llama.cpp's converter sees the union too.
    for tid, entry in config.get("added_tokens_decoder", {}).items():
        added.setdefault(int(tid), entry)
    for tid, entry in added.items():
        vocab.setdefault(entry["content"], tid)
    if max(vocab.values()) >= vocab_size:
        fail(f"tokenizer vocab id {max(vocab.values())} exceeds vocab_size {vocab_size}")
    reverse = {i: t for t, i in vocab.items()}
    tokens, types = [], []
    for i in range(vocab_size):
        token = reverse.get(i)
        if token is None:
            tokens.append(f"[PAD{i}]")
            types.append(TOK_UNUSED)
        elif i in added:
            tokens.append(token)
            types.append(TOK_CONTROL if added[i].get("special") or looks_special(token) else TOK_USER_DEFINED)
        else:
            tokens.append(token)
            types.append(TOK_NORMAL)
    merges = tok["model"]["merges"]
    if merges and not isinstance(merges[0], str):
        merges = [f"{a} {b}" for a, b in merges]

    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", pre_type)
    writer.add_string_array("tokenizer.ggml.tokens", tokens)
    writer.add_i32_array("tokenizer.ggml.token_type", types)
    writer.add_string_array("tokenizer.ggml.merges", merges)

    def token_id(field):
        value = config.get(field)
        if isinstance(value, dict):
            value = value.get("content")
        return vocab.get(value) if isinstance(value, str) else None

    for typ, key in (("bos", "bos_token_id"), ("eos", "eos_token_id"), ("unk", "unknown_token_id"),
                     ("sep", "seperator_token_id"), ("pad", "padding_token_id")):
        tid = token_id(f"{typ}_token")
        if tid is not None:
            writer.add_u32(f"tokenizer.ggml.{key}", tid)
    if "add_bos_token" in config:
        writer.add_bool("tokenizer.ggml.add_bos_token", bool(config["add_bos_token"]))
    if "add_eos_token" in config:
        writer.add_bool("tokenizer.ggml.add_eos_token", bool(config["add_eos_token"]))

    template = None
    jinja_path = os.path.join(hf_dir, "chat_template.jinja")
    if os.path.isfile(jinja_path):
        with open(jinja_path, encoding="utf-8") as fp:
            template = fp.read()
    elif isinstance(config.get("chat_template"), str):
        template = config["chat_template"]
    if template:
        writer.add_string("tokenizer.chat_template", template)
    return tokens, types

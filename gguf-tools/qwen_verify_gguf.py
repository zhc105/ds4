#!/usr/bin/env python3
"""Check a qwen_convert.py output against its HF source.

Samples tensors of every kind (routed experts, NVFP4 and BF16 projections,
zero-centered norms, GDN constants) and dequantises the GGUF bytes back to
compare with the ModelOpt reference formula on the safetensors side.  For
Flash-Next it also checks the .ngram sidecar: header constants, row count,
and randomly sampled rows against the source shards.

    python3 gguf-tools/qwen_verify_gguf.py hf/Qwen3.8-Flash-Next-NVFP4 gguf/Qwen3.8-Flash-Next-NVFP4.gguf
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_gguf import (  # noqa: E402
    NGRAM_HEADER_BYTES, SafeTensors, T_BF16, T_F32, T_NVFP4,
    nvfp4_dequant_gguf, nvfp4_dequant_modelopt, read_gguf, read_ngram_header, reorder_heads,
    tensor_nbytes, v_head_perm,
)
from qwen_convert import load_config  # noqa: E402


class Checker:
    def __init__(self, hf_dir, gguf_path):
        self.st = SafeTensors(hf_dir)
        self.cfg = load_config(hf_dir)
        self.kv, self.tensors, self.data_offset = read_gguf(gguf_path)
        self.data = np.memmap(gguf_path, dtype=np.uint8, mode="r")
        self.prefix = "model.language_model" if "model.language_model.embed_tokens.weight" in self.st else "model"
        self.arch = self.kv["general.architecture"]
        # GDN tensors indexed by V head are stored in llama.cpp's tiled order
        self.v_perm = v_head_perm(self.cfg["linear_num_key_heads"],
                                  self.cfg["linear_num_value_heads"] // self.cfg["linear_num_key_heads"])
        self.checks = 0
        self.failures = 0

    def raw(self, name):
        shape, qtype, offset, nbytes = self.tensors[name]
        start = self.data_offset + offset
        return np.asarray(self.data[start:start + nbytes]), shape, qtype

    def ok(self, cond, what):
        self.checks += 1
        if not cond:
            self.failures += 1
            print(f"FAIL {what}", file=sys.stderr)

    def check_nvfp4_2d(self, gguf_name, hf_base, rows=None):
        raw, shape, qtype = self.raw(gguf_name)
        self.ok(qtype == T_NVFP4, f"{gguf_name} type")
        got = nvfp4_dequant_gguf(raw.reshape(shape[0], -1), shape[1], 1.0)
        w, s = self.st.read(hf_base + ".weight"), self.st.read(hf_base + ".weight_scale")
        if rows is not None:
            w, s = w[rows], s[rows]
        self.ok(np.array_equal(got, nvfp4_dequant_modelopt(w, s, 1.0)), f"{gguf_name} values")
        scale_name = gguf_name.replace(".weight", ".scale")
        if scale_name in self.tensors:
            scale_raw, _, _ = self.raw(scale_name)
            src = float(self.st.read_f32(hf_base + ".weight_scale_2").reshape(-1)[0])
            self.ok(float(np.frombuffer(scale_raw, dtype=np.float32)[0]) == src, f"{scale_name}")

    def check_expert(self, il, proj, expert):
        gguf_name = f"blk.{il}.ffn_{proj}_exps.weight"
        raw, shape, qtype = self.raw(gguf_name)
        n_expert, out_features, n_cols = shape
        self.ok(qtype == T_NVFP4 and n_expert == self.cfg["num_experts"], f"{gguf_name} shape")
        per = out_features * (n_cols // 64 * 36)
        got = nvfp4_dequant_gguf(raw[expert * per:(expert + 1) * per].reshape(out_features, -1), n_cols, 1.0)
        base = f"{self.prefix}.layers.{il}.mlp.experts.{expert}.{proj}_proj"
        ref = nvfp4_dequant_modelopt(self.st.read(base + ".weight"), self.st.read(base + ".weight_scale"), 1.0)
        self.ok(np.array_equal(got, ref), f"{gguf_name}[{expert}] values")
        scales, _, _ = self.raw(f"blk.{il}.ffn_{proj}_exps.scale")
        src = float(self.st.read_f32(base + ".weight_scale_2").reshape(-1)[0])
        self.ok(float(np.frombuffer(scales, dtype=np.float32)[expert]) == src, f"{gguf_name}[{expert}] scale")

    def check_bf16(self, gguf_name, hf_name, row_perm=None, row_head=1, col_perm=None, col_head=1):
        raw, shape, qtype = self.raw(gguf_name)
        self.ok(qtype == T_BF16, f"{gguf_name} type")
        src = self.st.read(hf_name)
        if row_perm is not None:
            src = reorder_heads(src, 0, row_perm, row_head)
        if col_perm is not None:
            src = reorder_heads(src, 1, col_perm, col_head)
        self.ok(np.array_equal(np.frombuffer(raw, dtype=np.uint16).reshape(shape), src),
                f"{gguf_name} values")

    def check_f32(self, gguf_name, expected):
        raw, shape, qtype = self.raw(gguf_name)
        self.ok(qtype == T_F32, f"{gguf_name} type")
        self.ok(np.array_equal(np.frombuffer(raw, dtype=np.float32).reshape(shape), expected.reshape(shape)),
                f"{gguf_name} values")

    def run(self, rng):
        st, cfg, L = self.st, self.cfg, self.prefix + ".layers"
        n_layer = cfg["num_hidden_layers"]
        moe = self.arch == "qwen4exp"
        layer_types = cfg["layer_types"]
        for il in sorted(rng.choice(n_layer, size=min(6, n_layer), replace=False).tolist()):
            hf = f"{L}.{il}"
            if layer_types[il] == "linear_attention":
                if moe:
                    hv = cfg["linear_value_head_dim"]
                    qk_heads = 2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"] // hv
                    qkv_perm = np.concatenate([np.arange(qk_heads), qk_heads + self.v_perm])
                    self.check_bf16(f"blk.{il}.ssm_beta.weight", f"{hf}.linear_attn.in_proj_b.weight",
                                    row_perm=self.v_perm, row_head=1)
                    self.check_bf16(f"blk.{il}.attn_qkv.weight", f"{hf}.linear_attn.in_proj_qkv.weight",
                                    row_perm=qkv_perm, row_head=hv)
                    self.check_bf16(f"blk.{il}.ssm_out.weight", f"{hf}.linear_attn.out_proj.weight",
                                    col_perm=self.v_perm, col_head=hv)
                else:
                    self.check_nvfp4_2d(f"blk.{il}.attn_gate.weight", f"{hf}.linear_attn.in_proj_z")
                self.check_f32(f"blk.{il}.ssm_a",
                               reorder_heads(-np.exp(st.read_f32(f"{hf}.linear_attn.A_log")), 0, self.v_perm, 1))
                self.check_f32(f"blk.{il}.ssm_norm.weight", st.read_f32(f"{hf}.linear_attn.norm.weight"))
            else:
                if moe:
                    self.check_bf16(f"blk.{il}.attn_k.weight", f"{hf}.self_attn.k_proj.weight")
                    n_q = cfg["indexer_n_heads"] * cfg["indexer_head_dim"]
                    raw, shape, _ = self.raw(f"blk.{il}.indexer.k_proj.weight")
                    src = st.read(f"{hf}.self_attn.indexer.index_qk_proj.weight")[n_q:]
                    self.ok(np.array_equal(np.frombuffer(raw, dtype=np.uint16).reshape(shape), src),
                            f"blk.{il}.indexer.k_proj split")
                    self.check_f32(f"blk.{il}.indexer.q_norm.weight",
                                   st.read_f32(f"{hf}.self_attn.indexer.q_layernorm.weight") + 1)
                else:
                    self.check_nvfp4_2d(f"blk.{il}.attn_q.weight", f"{hf}.self_attn.q_proj")
                self.check_f32(f"blk.{il}.attn_q_norm.weight", st.read_f32(f"{hf}.self_attn.q_norm.weight") + 1)
            if moe:
                self.check_f32(f"blk.{il}.hc_attn_norm.weight",
                               st.read_f32(f"{hf}.attn_hyper_connection.hc_norm.weight") + 1)
                self.check_bf16(f"blk.{il}.hc_ffn_inject.weight",
                                f"{hf}.mlp_hyper_connection.block_inject_weight.weight")
                self.check_f32(f"blk.{il}.ffn_gate_inp.weight", st.read_f32(f"{hf}.mlp.gate.weight"))
                for proj in ("gate", "up", "down"):
                    self.check_expert(il, proj, int(rng.integers(cfg["num_experts"])))
            else:
                self.check_nvfp4_2d(f"blk.{il}.ffn_down.weight", f"{hf}.mlp.down_proj")
                self.check_f32(f"blk.{il}.attn_norm.weight", st.read_f32(f"{hf}.input_layernorm.weight") + 1)
        self.check_bf16("token_embd.weight", f"{self.prefix}.embed_tokens.weight")
        if moe:
            self.check_f32("output_hc_norm.weight",
                           st.read_f32(f"{self.prefix}.hyper_connection_mixer.hc_norm.weight") + 1)
            self.check_ngram(rng)
        for name, (shape, qtype, _, nbytes) in self.tensors.items():
            self.ok(tensor_nbytes(shape, qtype) == nbytes, f"{name} byte size")

    def check_ngram(self, rng):
        a = self.arch
        table = self.kv.get(f"{a}.ple.table_file")
        self.ok(table is not None, "ple.table_file key")
        path = os.path.join(os.path.dirname(self.gguf_path), table)
        h = read_ngram_header(path)
        ple = f"{self.prefix}.layers.{self.kv[f'{a}.ple.layers'][0]}.ple.ple_embedding"
        consts = lambda s: [int(v) for v in self.st.read(f"{ple}.{s}").reshape(-1)]
        self.ok(h["multipliers"] == consts("layer_multipliers") == self.kv[f"{a}.ple.layer_multipliers"], "ngram multipliers")
        self.ok(h["head_offsets"] == consts("ngram_heads_offsets") == self.kv[f"{a}.ple.head_offsets"], "ngram head offsets")
        self.ok(h["head_vocab_sizes"] == consts("ngram_heads_vocab_sizes") == self.kv[f"{a}.ple.head_vocab_sizes"], "ngram head sizes")
        shards = sorted(n for n in self.st.names() if n.startswith(f"{ple}.ngram_embedding.shard_"))
        shards.sort(key=lambda n: int(n.rsplit("shard_", 1)[1].split(".")[0]))
        rows = sum(self.st.shape(n)[0] for n in shards)
        self.ok(h["rows"] == rows, "ngram row count")
        self.ok(os.path.getsize(path) == NGRAM_HEADER_BYTES + rows * h["row_bytes"], "ngram file size")
        self.ok(h["scale"] == float(self.st.read_f32(f"{ple}.ngram_embedding.weight_scale").reshape(-1)[0]), "ngram scale")
        table_mm = np.memmap(path, dtype=np.uint8, mode="r", offset=NGRAM_HEADER_BYTES)
        shard_rows = self.st.shape(shards[0])[0]
        for row in rng.integers(rows, size=32).tolist():
            got = np.asarray(table_mm[row * h["row_bytes"]:(row + 1) * h["row_bytes"]])
            src = self.st.read(shards[row // shard_rows])[row % shard_rows]
            self.ok(np.array_equal(got, src), f"ngram row {row}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hf_dir")
    parser.add_argument("gguf")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    checker = Checker(args.hf_dir, args.gguf)
    checker.gguf_path = args.gguf
    checker.run(np.random.default_rng(args.seed))
    print(f"qwen-verify: {checker.checks} checks, {checker.failures} failures")
    sys.exit(1 if checker.failures else 0)


if __name__ == "__main__":
    main()

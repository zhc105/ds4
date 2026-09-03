#!/usr/bin/env python3
"""Convert a Qwen3.5 (dense, hybrid GDN + gated attention) HF checkpoint to GGUF.

Accepts BF16 checkpoints and ModelOpt NVFP4 checkpoints such as
AxionML/Qwen3.5-2B-NVFP4.  Output follows llama.cpp's `qwen35` layout:

  blk.N.attn_qkv / attn_gate / ssm_alpha / ssm_beta / ssm_out   GDN projections
  blk.N.ssm_conv1d / ssm_dt.bias / ssm_a / ssm_norm             GDN state params
  blk.N.attn_q (query and sigmoid gate fused) / attn_k / attn_v / attn_output
  blk.N.attn_q_norm / attn_k_norm / attn_norm / post_attention_norm
  blk.N.ffn_gate / ffn_up / ffn_down
  blk.L.nextn.*                                                  MTP block, L = layer count

Zero-centered RMSNorm weights are stored as (1 + w) so the runtime applies a
plain RMSNorm.  A_log is stored as -exp(A_log).  V heads of the GDN branch are
reordered from grouped to tiled order when there are more V than K heads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_gguf import (  # noqa: E402
    GGUFWriter, SafeTensors, T_BF16, T_F16, T_F32, T_NVFP4, TYPE_NAMES,
    bf16_to_f32, export_tokenizer, fail, nvfp4_repack, reorder_heads, v_head_perm,
)

FILE_TYPE_MOSTLY_BF16 = 32
FILE_TYPE_MOSTLY_NVFP4 = 39
SOURCE_TYPES = {"BF16": T_BF16, "F16": T_F16, "F32": T_F32}


def load_config(hf_dir):
    with open(os.path.join(hf_dir, "config.json"), encoding="utf-8") as fp:
        config = json.load(fp)
    text = dict(config.get("text_config", config))
    text.setdefault("tie_word_embeddings", config.get("tie_word_embeddings", False))
    quant = config.get("quantization_config", {})
    algo = quant.get("quant_algo") or ""
    if algo and algo != "NVFP4":
        fail(f"unsupported quantization {algo}; only NVFP4 ModelOpt checkpoints or BF16 are supported")
    return text


class Converter:
    def __init__(self, hf_dir, out_path, name):
        self.hf_dir = hf_dir
        self.cfg = load_config(hf_dir)
        self.st = SafeTensors(hf_dir)
        self.writer = GGUFWriter(out_path)
        self.name = name
        self.uses_nvfp4 = False
        cfg = self.cfg
        self.n_layer = cfg["num_hidden_layers"]
        self.n_mtp = cfg.get("mtp_num_hidden_layers", 0)
        if self.n_mtp == 0:
            self.n_mtp = len({n.split(".")[2] for n in self.st.names() if n.startswith("mtp.layers.")})
        self.layer_types = cfg.get("layer_types") or [
            "full_attention" if (i + 1) % cfg.get("full_attention_interval", 4) == 0 else "linear_attention"
            for i in range(self.n_layer)]
        self.num_k_heads = cfg["linear_num_key_heads"]
        self.num_v_heads = cfg["linear_num_value_heads"]
        self.head_k_dim = cfg["linear_key_head_dim"]
        self.head_v_dim = cfg["linear_value_head_dim"]
        if self.num_v_heads % self.num_k_heads:
            fail("linear_num_value_heads must be a multiple of linear_num_key_heads")
        self.v_perm = v_head_perm(self.num_k_heads, self.num_v_heads // self.num_k_heads)

    # -- hyperparameters --------------------------------------------------

    def add_metadata(self):
        cfg, w = self.cfg, self.writer
        head_dim = cfg["head_dim"]
        rope = cfg.get("rope_parameters") or {}
        rope_theta = rope.get("rope_theta", cfg.get("rope_theta"))
        partial = rope.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 0.25))
        mrope = list(rope.get("mrope_section", [11, 11, 10]))
        mrope += [0] * (4 - len(mrope))

        w.add_string("general.architecture", "qwen35")
        w.add_string("general.type", "model")
        w.add_string("general.name", self.name)
        w.add_u32("general.quantization_version", 2)
        w.add_u32("general.file_type", FILE_TYPE_MOSTLY_NVFP4 if self.uses_nvfp4 else FILE_TYPE_MOSTLY_BF16)
        w.add_u32("qwen35.block_count", self.n_layer + self.n_mtp)
        w.add_u32("qwen35.context_length", cfg["max_position_embeddings"])
        w.add_u32("qwen35.embedding_length", cfg["hidden_size"])
        w.add_u32("qwen35.feed_forward_length", cfg["intermediate_size"])
        w.add_u32("qwen35.attention.head_count", cfg["num_attention_heads"])
        w.add_u32("qwen35.attention.head_count_kv", cfg["num_key_value_heads"])
        w.add_u32("qwen35.attention.key_length", head_dim)
        w.add_u32("qwen35.attention.value_length", head_dim)
        w.add_f32("qwen35.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
        w.add_f32("qwen35.rope.freq_base", float(rope_theta))
        w.add_u32("qwen35.rope.dimension_count", int(head_dim * partial))
        w.add_i32_array("qwen35.rope.dimension_sections", mrope[:4])
        w.add_u32("qwen35.ssm.conv_kernel", cfg["linear_conv_kernel_dim"])
        w.add_u32("qwen35.ssm.state_size", self.head_k_dim)
        w.add_u32("qwen35.ssm.group_count", self.num_k_heads)
        w.add_u32("qwen35.ssm.time_step_rank", self.num_v_heads)
        w.add_u32("qwen35.ssm.inner_size", self.head_v_dim * self.num_v_heads)
        w.add_u32("qwen35.full_attention_interval", cfg.get("full_attention_interval", 4))
        w.add_u32("qwen35.vocab_size", cfg["vocab_size"])
        if self.n_mtp:
            w.add_u32("qwen35.nextn_predict_layers", self.n_mtp)
        export_tokenizer(w, self.hf_dir, cfg["vocab_size"], "qwen35")

    # -- tensor helpers ---------------------------------------------------

    def norm(self, gguf_name, hf_name, zero_centered=True):
        """RMSNorm weight as F32; zero-centered norms are stored as 1 + w."""
        st = self.st
        self.writer.add_tensor(gguf_name, st.shape(hf_name), T_F32, 4 * int(np.prod(st.shape(hf_name))),
                               lambda: st.read_f32(hf_name) + (1.0 if zero_centered else 0.0))

    def f32(self, gguf_name, hf_name, transform=lambda x: x, shape=None):
        st = self.st
        shape = shape or st.shape(hf_name)
        self.writer.add_tensor(gguf_name, shape, T_F32, 4 * int(np.prod(shape)),
                               lambda: transform(st.read_f32(hf_name)).reshape(shape).astype(np.float32))

    def linear(self, gguf_base, hf_base, row_perm=None, row_head=1, col_perm=None, col_head=1):
        """Emit a projection: NVFP4 (with companion scales) or its source float type.
        Optional head permutations reorder output rows or input columns."""
        st = self.st
        weight_name = hf_base + ".weight"
        scale_name = hf_base + ".weight_scale"
        if scale_name in st and len(st.shape(scale_name)) == 2:
            self.uses_nvfp4 = True
            out_features, half = st.shape(weight_name)
            n_cols = half * 2

            def produce():
                weight, scale = st.read(weight_name), st.read(scale_name)
                if row_perm is not None:
                    weight = reorder_heads(weight, 0, row_perm, row_head)
                    scale = reorder_heads(scale, 0, row_perm, row_head)
                if col_perm is not None:
                    weight = reorder_heads(weight, 1, col_perm, col_head // 2)
                    scale = reorder_heads(scale, 1, col_perm, col_head // 16)
                return nvfp4_repack(weight, scale)[0]

            self.writer.add_tensor(gguf_base + ".weight", [out_features, n_cols], T_NVFP4,
                                   n_cols // 64 * 36 * out_features, produce)
            for suffix, key in ((".scale", ".weight_scale_2"), (".input_scale", ".input_scale")):
                if hf_base + key in st:
                    value = st.read_f32(hf_base + key).reshape(-1)
                    if value.size != 1 or abs(float(value[0]) - 1.0) >= 1e-6:
                        self.writer.add_array(gguf_base + suffix, value, T_F32)
            return

        dtype = st.dtype(weight_name)
        if dtype not in SOURCE_TYPES:
            fail(f"{weight_name}: unsupported dtype {dtype}")
        shape = st.shape(weight_name)

        def produce_float():
            weight = st.read(weight_name)
            if row_perm is not None:
                weight = reorder_heads(weight, 0, row_perm, row_head)
            if col_perm is not None:
                weight = reorder_heads(weight, 1, col_perm, col_head)
            return weight

        self.writer.add_tensor(gguf_base + ".weight", shape, SOURCE_TYPES[dtype],
                               int(np.prod(shape)) * st.read(weight_name).itemsize, produce_float)

    # -- model layout -----------------------------------------------------

    def add_gdn(self, il, hf):
        """Gated DeltaNet branch.  in_proj_qkv rows are [q | k | v]; only the V
        rows, and everything else indexed by V head, get the tiled reorder."""
        perm, nv, hv = self.v_perm, self.num_v_heads, self.head_v_dim
        qk_rows = 2 * self.num_k_heads * self.head_k_dim
        qkv_rows = qk_rows + nv * hv
        # Rows above the V block are untouched: extend the permutation with an identity prefix.
        qkv_perm = np.concatenate([np.arange(qk_rows // hv), qk_rows // hv + perm]) if qk_rows % hv == 0 else None
        if qkv_perm is None:
            fail("GDN q/k row block is not a multiple of the V head size")
        self.linear(f"blk.{il}.attn_qkv", f"{hf}.in_proj_qkv", row_perm=qkv_perm, row_head=hv)
        self.linear(f"blk.{il}.attn_gate", f"{hf}.in_proj_z", row_perm=perm, row_head=hv)
        self.linear(f"blk.{il}.ssm_alpha", f"{hf}.in_proj_a", row_perm=perm, row_head=1)
        self.linear(f"blk.{il}.ssm_beta", f"{hf}.in_proj_b", row_perm=perm, row_head=1)
        self.linear(f"blk.{il}.ssm_out", f"{hf}.out_proj", col_perm=perm, col_head=hv)
        conv_dim = qkv_rows
        kernel = self.cfg["linear_conv_kernel_dim"]
        self.f32(f"blk.{il}.ssm_conv1d.weight", f"{hf}.conv1d.weight", shape=[conv_dim, kernel],
                 transform=lambda w: reorder_heads(w.reshape(conv_dim, kernel), 0, qkv_perm, hv))
        self.f32(f"blk.{il}.ssm_dt.bias", f"{hf}.dt_bias", transform=lambda b: reorder_heads(b, 0, perm, 1))
        self.f32(f"blk.{il}.ssm_a", f"{hf}.A_log", transform=lambda a: reorder_heads(-np.exp(a), 0, perm, 1))
        self.norm(f"blk.{il}.ssm_norm.weight", f"{hf}.norm.weight", zero_centered=False)

    def add_attention(self, il, hf):
        self.linear(f"blk.{il}.attn_q", f"{hf}.q_proj")
        self.linear(f"blk.{il}.attn_k", f"{hf}.k_proj")
        self.linear(f"blk.{il}.attn_v", f"{hf}.v_proj")
        self.linear(f"blk.{il}.attn_output", f"{hf}.o_proj")
        self.norm(f"blk.{il}.attn_q_norm.weight", f"{hf}.q_norm.weight")
        self.norm(f"blk.{il}.attn_k_norm.weight", f"{hf}.k_norm.weight")

    def add_block(self, il, hf, linear):
        self.norm(f"blk.{il}.attn_norm.weight", f"{hf}.input_layernorm.weight")
        self.norm(f"blk.{il}.post_attention_norm.weight", f"{hf}.post_attention_layernorm.weight")
        if linear:
            self.add_gdn(il, f"{hf}.linear_attn")
        else:
            self.add_attention(il, f"{hf}.self_attn")
        self.linear(f"blk.{il}.ffn_gate", f"{hf}.mlp.gate_proj")
        self.linear(f"blk.{il}.ffn_up", f"{hf}.mlp.up_proj")
        self.linear(f"blk.{il}.ffn_down", f"{hf}.mlp.down_proj")

    def add_tensors(self):
        prefix = "model.language_model" if "model.language_model.embed_tokens.weight" in self.st else "model"
        self.linear("token_embd", f"{prefix}.embed_tokens")
        self.norm("output_norm.weight", f"{prefix}.norm.weight")
        if "lm_head.weight" in self.st and not self.cfg["tie_word_embeddings"]:
            self.linear("output", "lm_head")
        for il in range(self.n_layer):
            self.add_block(il, f"{prefix}.layers.{il}", self.layer_types[il] == "linear_attention")
        for m in range(self.n_mtp):
            il = self.n_layer + m
            self.add_block(il, f"mtp.layers.{m}", linear=False)
            self.linear(f"blk.{il}.nextn.eh_proj", "mtp.fc")
            self.norm(f"blk.{il}.nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight")
            self.norm(f"blk.{il}.nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight")
            self.norm(f"blk.{il}.nextn.shared_head_norm.weight", "mtp.norm.weight")

    def run(self, dry_run, verbose):
        self.add_tensors()
        self.add_metadata()   # after tensors: file_type depends on what was emitted
        totals = {}
        for _, _, qtype, nbytes, _ in self.writer.tensors:
            count, size = totals.get(qtype, (0, 0))
            totals[qtype] = (count + 1, size + nbytes)
        print(f"qwen35-convert: {self.n_layer} layers + {self.n_mtp} MTP, "
              f"{len(self.writer.tensors)} tensors, {len(self.writer.kv)} metadata keys", file=sys.stderr)
        for qtype, (count, size) in sorted(totals.items()):
            print(f"  {TYPE_NAMES[qtype]:6s} {count:5d} tensors {size / 2**30:8.3f} GiB", file=sys.stderr)
        if dry_run:
            for name, shape, qtype, _, _ in self.writer.tensors:
                print(f"  {name:40s} {TYPE_NAMES[qtype]:6s} {shape}", file=sys.stderr)
            return
        total = len(self.writer.tensors)

        def log(index, name, shape, qtype):
            if verbose:
                print(f"[{index + 1}/{total}] {name} {TYPE_NAMES[qtype]} {shape}", file=sys.stderr)

        size = self.writer.write(log)
        print(f"qwen35-convert: wrote {self.writer.path} ({size / 2**30:.3f} GiB)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hf_dir", help="Hugging Face checkpoint directory")
    parser.add_argument("-o", "--out", required=True, help="output GGUF path")
    parser.add_argument("--name", help="general.name (default: directory basename)")
    parser.add_argument("--dry-run", action="store_true", help="print the tensor plan and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.out) and not args.dry_run:
        fail(f"output exists: {args.out}")
    name = args.name or os.path.basename(os.path.normpath(args.hf_dir))
    Converter(args.hf_dir, args.out, name).run(args.dry_run, args.verbose)


if __name__ == "__main__":
    main()

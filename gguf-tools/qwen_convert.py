#!/usr/bin/env python3
"""Convert Qwen3.5 (dense) and Qwen3.8-Flash-Next (qwen4_exp) HF checkpoints to GGUF.

Accepts BF16 checkpoints and ModelOpt NVFP4 checkpoints (AxionML/Qwen3.5-2B-NVFP4,
RadixArk/Qwen3.8-Flash-Next-NVFP4).  Output follows llama.cpp's `qwen35` /
`qwen4exp` layouts:

  blk.N.attn_qkv / attn_gate / ssm_alpha / ssm_beta / ssm_out   GDN projections
  blk.N.ssm_conv1d / ssm_dt.bias / ssm_a / ssm_norm             GDN state params
  blk.N.attn_q (query and sigmoid gate fused) / attn_k / attn_v / attn_output
  blk.N.attn_q_norm / attn_k_norm
  blk.N.attn_norm / post_attention_norm (Qwen3.5)  |  blk.N.hc_attn_* / hc_ffn_* (Flash-Next)
  blk.N.ffn_gate / ffn_up / ffn_down (dense)       |  ffn_*_exps [n_expert] + router + shexp (MoE)
  blk.N.indexer.q_proj / k_proj / q_norm / k_norm  QSA indexer (Flash-Next attention layers)
  blk.N.ple_key / ple_value / ple_norm_* / ple_conv1d      PLE projections (Flash-Next, one layer)
  blk.L.nextn.*                                    MTP block (Qwen3.5 only; Flash-Next drops it)

`--vision` writes the Flash-Next vision tower (the stock Qwen3-VL ViT) as its
own GGUF for ds4 --vision: raw BF16 copies under their HF names plus the
vision config and image preprocessing constants.

The Flash-Next n-gram table (47.7 GiB of E4M3 rows) is not a GGUF tensor: it is
written beside the GGUF as a ds4 .ngram file (see hf_gguf.ngram_header), and the
hash constants go into the GGUF metadata.

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
    GGUFWriter, Q8_0_BLOCK, Q8_0_BLOCK_BYTES, SafeTensors, T_BF16, T_F16, T_F32, T_NVFP4,
    T_Q8_0, TYPE_NAMES, bf16_to_f32, export_tokenizer, fail, ngram_header, nvfp4_quantize,
    nvfp4_repack, nvfp4_stack_experts, q8_0_quantize, reorder_heads, v_head_perm,
)

FILE_TYPE_MOSTLY_BF16 = 32
FILE_TYPE_MOSTLY_NVFP4 = 39
SOURCE_TYPES = {"BF16": T_BF16, "F16": T_F16, "F32": T_F32}
ARCH_BY_MODEL_TYPE = {"qwen3_5": "qwen35", "qwen4_exp": "qwen4exp"}


def load_config(hf_dir):
    with open(os.path.join(hf_dir, "config.json"), encoding="utf-8") as fp:
        config = json.load(fp)
    text = dict(config.get("text_config", config))
    text.setdefault("tie_word_embeddings", config.get("tie_word_embeddings", False))
    text.setdefault("image_token_id", config.get("image_token_id"))
    quant = config.get("quantization_config", {})
    algo = quant.get("quant_algo") or ""
    if algo and algo != "NVFP4":
        fail(f"unsupported quantization {algo}; only NVFP4 ModelOpt checkpoints or BF16 are supported")
    return text


class Converter:
    def __init__(self, hf_dir, out_path, name, quant_head=False, quant_gdn=False):
        self.hf_dir = hf_dir
        self.cfg = load_config(hf_dir)
        self.st = SafeTensors(hf_dir)
        self.writer = GGUFWriter(out_path)
        self.name = name
        self.uses_nvfp4 = False
        # Bypass-layer quantization, split so each half can be built and
        # measured alone: lm_head -> q8_0, the three big GDN projections ->
        # fp8 e4m3 with 128x128 block scales.  Both off by default so the
        # shipped BF16 conversion stays byte-identical.
        self.quant_head = quant_head
        self.quant_gdn = quant_gdn
        cfg = self.cfg
        model_type = cfg.get("model_type", "").replace("_text", "")
        if model_type not in ARCH_BY_MODEL_TYPE:
            fail(f"unsupported model_type {model_type!r}")
        self.arch = ARCH_BY_MODEL_TYPE[model_type]
        self.moe = self.arch == "qwen4exp"
        self.prefix = "model.language_model" if "model.language_model.embed_tokens.weight" in self.st else "model"
        self.n_layer = cfg["num_hidden_layers"]
        # Flash-Next keeps its MTP drafter as a separate head that llama.cpp and vLLM drop.
        self.n_mtp = 0 if self.moe else cfg.get("mtp_num_hidden_layers", 0)
        if not self.moe and self.n_mtp == 0:
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
        # PLE: ple_layer_ids is 1-based in the HF config
        self.ple_layers = [i - 1 for i in cfg.get("ple_layer_ids", [])] if self.moe else []
        self.ngram_path = None

    # -- hyperparameters --------------------------------------------------

    def add_metadata(self):
        cfg, w, a = self.cfg, self.writer, self.arch
        head_dim = cfg["head_dim"]
        rope = cfg.get("rope_parameters") or {}
        rope_theta = rope.get("rope_theta", cfg.get("rope_theta"))
        partial = rope.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 0.25))
        mrope = list(rope.get("mrope_section", [11, 11, 10]))
        mrope += [0] * (4 - len(mrope))

        w.add_string("general.architecture", a)
        w.add_string("general.type", "model")
        w.add_string("general.name", self.name)
        w.add_u32("general.quantization_version", 2)
        w.add_u32("general.file_type", FILE_TYPE_MOSTLY_NVFP4 if self.uses_nvfp4 else FILE_TYPE_MOSTLY_BF16)
        w.add_u32(f"{a}.block_count", self.n_layer + self.n_mtp)
        w.add_u32(f"{a}.context_length", cfg["max_position_embeddings"])
        w.add_u32(f"{a}.embedding_length", cfg["hidden_size"])
        if "intermediate_size" in cfg:
            w.add_u32(f"{a}.feed_forward_length", cfg["intermediate_size"])
        w.add_u32(f"{a}.attention.head_count", cfg["num_attention_heads"])
        w.add_u32(f"{a}.attention.head_count_kv", cfg["num_key_value_heads"])
        w.add_u32(f"{a}.attention.key_length", head_dim)
        w.add_u32(f"{a}.attention.value_length", head_dim)
        w.add_f32(f"{a}.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
        w.add_f32(f"{a}.rope.freq_base", float(rope_theta))
        w.add_u32(f"{a}.rope.dimension_count", int(head_dim * partial))
        w.add_i32_array(f"{a}.rope.dimension_sections", mrope[:4])
        w.add_u32(f"{a}.ssm.conv_kernel", cfg["linear_conv_kernel_dim"])
        w.add_u32(f"{a}.ssm.state_size", self.head_k_dim)
        w.add_u32(f"{a}.ssm.group_count", self.num_k_heads)
        w.add_u32(f"{a}.ssm.time_step_rank", self.num_v_heads)
        w.add_u32(f"{a}.ssm.inner_size", self.head_v_dim * self.num_v_heads)
        w.add_u32(f"{a}.full_attention_interval", cfg.get("full_attention_interval", 4))
        w.add_u32(f"{a}.vocab_size", cfg["vocab_size"])
        if self.n_mtp:
            w.add_u32(f"{a}.nextn_predict_layers", self.n_mtp)
        if self.moe:
            w.add_u32(f"{a}.expert_count", cfg["num_experts"])
            w.add_u32(f"{a}.expert_used_count", cfg["num_experts_per_tok"])
            w.add_u32(f"{a}.expert_feed_forward_length", cfg["moe_intermediate_size"])
            w.add_u32(f"{a}.expert_shared_feed_forward_length", cfg["shared_expert_intermediate_size"])
            w.add_u32(f"{a}.hyper_connection.count", cfg["hc_count"])
            w.add_u32(f"{a}.hyper_connection.low_rank", cfg["hc_lowrank"])
            w.add_u32(f"{a}.attention.indexer.head_count", cfg["indexer_n_heads"])
            w.add_u32(f"{a}.attention.indexer.key_length", cfg["indexer_head_dim"])
            w.add_u32(f"{a}.attention.indexer.top_k", cfg["indexer_budget"])
            ratio = cfg["indexer_compress_ratio"]
            w.add_i32_array(f"{a}.attention.compress_ratios",
                            [ratio if t == "full_attention" else 0 for t in self.layer_types])
            if self.ple_layers:
                self.add_ple_metadata()
        export_tokenizer(w, self.hf_dir, cfg["vocab_size"], "qwen35")

    def ple_prefix(self):
        return f"{self.prefix}.layers.{self.ple_layers[0]}.ple.ple_embedding"

    def ple_constants(self):
        p = self.ple_prefix()
        read = lambda suffix: [int(v) for v in self.st.read(f"{p}.{suffix}").reshape(-1)]
        return read("layer_multipliers"), read("ngram_heads_offsets"), read("ngram_heads_vocab_sizes")

    def add_ple_metadata(self):
        cfg, w, a = self.cfg, self.writer, self.arch
        if len(self.ple_layers) != 1:
            fail("exactly one PLE layer is supported")
        multipliers, offsets, sizes = self.ple_constants()
        w.add_i32_array(f"{a}.ple.layers", self.ple_layers)
        w.add_u32(f"{a}.ple.ngram_size", cfg["ngram_size"])
        w.add_u32(f"{a}.ple.heads_per_ngram", cfg["heads_per_ngram"])
        w.add_u32(f"{a}.ple.conv_kernel", cfg["ple_conv_kernel_size"])
        eos = cfg["eos_token_id"]
        w.add_u32(f"{a}.ple.eos_token_id", int(eos[-1] if isinstance(eos, list) else eos))
        if cfg.get("image_token_id") is not None:
            w.add_u32(f"{a}.ple.image_token_id", int(cfg["image_token_id"]))
        w.add_u32(f"{a}.embedding_length_per_layer_input", self.ple_row_dim())
        w.add_u64_array(f"{a}.ple.layer_multipliers", multipliers)
        w.add_u64_array(f"{a}.ple.head_offsets", offsets)
        w.add_u64_array(f"{a}.ple.head_vocab_sizes", sizes)
        if self.ngram_path:
            w.add_string(f"{a}.ple.table_file", os.path.basename(self.ngram_path))

    def ple_shards(self):
        p = self.ple_prefix() + ".ngram_embedding.shard_"
        shards = sorted((int(n[len(p):].split(".")[0]), n) for n in self.st.names() if n.startswith(p))
        if [i for i, _ in shards] != list(range(len(shards))):
            fail("PLE shards are not contiguous")
        return [n for _, n in shards]

    def ple_row_dim(self):
        return self.st.shape(self.ple_shards()[0])[1]

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

    def linear(self, gguf_base, hf_base, row_perm=None, row_head=1, col_perm=None, col_head=1,
               rows=None):
        """Emit a projection: NVFP4 (with companion scales) or its source float type.
        Optional head permutations reorder output rows or input columns; `rows`
        keeps only that slice of output rows."""
        st = self.st
        weight_name = hf_base + ".weight"
        scale_name = hf_base + ".weight_scale"
        if scale_name in st and len(st.shape(scale_name)) == 2:
            self.uses_nvfp4 = True
            out_features, half = st.shape(weight_name)
            n_cols = half * 2
            if rows is not None:
                out_features = rows.stop - rows.start

            def produce():
                weight, scale = st.read(weight_name), st.read(scale_name)
                if rows is not None:
                    weight, scale = weight[rows], scale[rows]
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
        shape = list(st.shape(weight_name))
        if rows is not None:
            shape[0] = rows.stop - rows.start

        def produce_float():
            weight = st.read(weight_name)
            if rows is not None:
                weight = weight[rows]
            if row_perm is not None:
                weight = reorder_heads(weight, 0, row_perm, row_head)
            if col_perm is not None:
                weight = reorder_heads(weight, 1, col_perm, col_head)
            return weight

        self.writer.add_tensor(gguf_base + ".weight", shape, SOURCE_TYPES[dtype],
                               int(np.prod(shape)) * st.read(weight_name).itemsize, produce_float)

    def experts(self, gguf_base, hf_layer, proj, n_expert):
        """Stack the routed experts' `proj` into one [n_expert][out][in] tensor.
        NVFP4 experts carry per-expert global scales as `.scale` [n_expert]."""
        st = self.st
        name = lambda e, suffix: f"{hf_layer}.mlp.experts.{e}.{proj}.{suffix}"
        fused = f"{hf_layer}.mlp.experts.gate_up_proj"
        if fused in st:
            # The MTP drafter keeps its experts in bf16, stacked and with gate
            # and up fused: quantize them to NVFP4 so the expert kernels apply.
            self.uses_nvfp4 = True
            stacked = fused if proj != "down_proj" else f"{hf_layer}.mlp.experts.down_proj"
            e_count, rows, cols = st.shape(stacked)
            if e_count != n_expert:
                fail(f"{stacked}: {e_count} experts, expected {n_expert}")
            half = rows // 2 if proj != "down_proj" else rows
            row0 = half if proj == "up_proj" else 0
            scales = np.zeros(n_expert, dtype=np.float32)

            def produce():
                out = np.empty((n_expert, half, cols // 64 * 36), dtype=np.uint8)
                for e in range(n_expert):
                    w = bf16_to_f32(st.read(stacked)[e][row0:row0 + half])   # one expert's rows off the mmap
                    out[e], scales[e] = nvfp4_quantize(w)
                return out

            self.writer.add_tensor(gguf_base + ".weight", [n_expert, half, cols], T_NVFP4,
                                   n_expert * half * (cols // 64 * 36), produce)
            self.writer.add_tensor(gguf_base + ".scale", [n_expert], T_F32, 4 * n_expert, lambda: scales)
            return
        if name(0, "weight_scale") in st:
            self.uses_nvfp4 = True
            out_features, half = st.shape(name(0, "weight"))
            n_cols = half * 2
            self.writer.add_tensor(gguf_base + ".weight", [n_expert, out_features, n_cols], T_NVFP4,
                                   n_expert * out_features * (n_cols // 64 * 36),
                                   lambda: nvfp4_stack_experts(lambda e: st.read(name(e, "weight")),
                                                               lambda e: st.read(name(e, "weight_scale")),
                                                               n_expert)[0])
            for suffix, key in ((".scale", "weight_scale_2"), (".input_scale", "input_scale")):
                if name(0, key) in st:
                    self.writer.add_tensor(gguf_base + suffix, [n_expert], T_F32, 4 * n_expert,
                                           lambda key=key: np.array([float(st.read_f32(name(e, key)).reshape(-1)[0])
                                                                     for e in range(n_expert)], dtype=np.float32))
            return
        dtype = st.dtype(name(0, "weight"))
        if dtype not in SOURCE_TYPES:
            fail(f"{name(0, 'weight')}: unsupported dtype {dtype}")
        shape = [n_expert] + list(st.shape(name(0, "weight")))
        self.writer.add_tensor(gguf_base + ".weight", shape, SOURCE_TYPES[dtype],
                               int(np.prod(shape)) * st.read(name(0, "weight")).itemsize,
                               lambda: np.stack([st.read(name(e, "weight")) for e in range(n_expert)]))

    # -- model layout -----------------------------------------------------

    def add_gdn(self, il, hf):
        """Gated DeltaNet branch.  in_proj_qkv rows are [q | k | v]; only the V
        rows, and everything else indexed by V head, get the tiled reorder."""
        perm, nv, hv = self.v_perm, self.num_v_heads, self.head_v_dim
        qk_rows = 2 * self.num_k_heads * self.head_k_dim
        if qk_rows % hv:
            fail("GDN q/k row block is not a multiple of the V head size")
        # Rows above the V block are untouched: extend the permutation with an identity prefix.
        qkv_perm = np.concatenate([np.arange(qk_rows // hv), qk_rows // hv + perm])
        conv_dim = qk_rows + nv * hv
        # The three projections that dominate the decode read shrink to q8_0
        # under --quant-gdn; alpha/beta are tiny and stay as they are.
        proj = self.linear_q8_0 if self.quant_gdn else self.linear
        proj(f"blk.{il}.attn_qkv", f"{hf}.in_proj_qkv", row_perm=qkv_perm, row_head=hv)
        proj(f"blk.{il}.attn_gate", f"{hf}.in_proj_z", row_perm=perm, row_head=hv)
        self.linear(f"blk.{il}.ssm_alpha", f"{hf}.in_proj_a", row_perm=perm, row_head=1)
        self.linear(f"blk.{il}.ssm_beta", f"{hf}.in_proj_b", row_perm=perm, row_head=1)
        proj(f"blk.{il}.ssm_out", f"{hf}.out_proj", col_perm=perm, col_head=hv)
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
        if self.moe:
            # one projection feeds the indexer's q heads and its single k head
            n_q = self.cfg["indexer_n_heads"] * self.cfg["indexer_head_dim"]
            n_k = self.cfg["indexer_kv_heads"] * self.cfg["indexer_head_dim"]
            self.linear(f"blk.{il}.indexer.q_proj", f"{hf}.indexer.index_qk_proj", rows=slice(0, n_q))
            self.linear(f"blk.{il}.indexer.k_proj", f"{hf}.indexer.index_qk_proj", rows=slice(n_q, n_q + n_k))
            self.norm(f"blk.{il}.indexer.q_norm.weight", f"{hf}.indexer.q_layernorm.weight")
            self.norm(f"blk.{il}.indexer.k_norm.weight", f"{hf}.indexer.k_layernorm.weight")

    def add_hc(self, gguf_base, hf):
        """Gated Residual: norm over the widened stream, low-rank mix, block inject."""
        self.norm(f"{gguf_base}_norm.weight", f"{hf}.hc_norm.weight")
        self.linear(f"{gguf_base}_down", f"{hf}.input_mix_weight_down")
        self.linear(f"{gguf_base}_up", f"{hf}.input_mix_weight_up")
        if f"{hf}.block_inject_weight.weight" in self.st:
            self.linear(f"{gguf_base}_inject", f"{hf}.block_inject_weight")

    def add_ple(self, il, hf):
        kernel = self.cfg["ple_conv_kernel_size"]
        channels = self.st.shape(f"{hf}.conv1d.weight")[0]
        self.linear(f"blk.{il}.ple_key", f"{hf}.key_proj")
        self.linear(f"blk.{il}.ple_value", f"{hf}.value_proj")
        self.norm(f"blk.{il}.ple_norm_key.weight", f"{hf}.norm_key.weight")
        self.norm(f"blk.{il}.ple_norm_query.weight", f"{hf}.norm_query.weight")
        self.norm(f"blk.{il}.ple_norm_conv.weight", f"{hf}.norm_conv.weight")
        self.f32(f"blk.{il}.ple_conv1d.weight", f"{hf}.conv1d.weight", shape=[channels, kernel])

    def add_ffn(self, il, hf):
        if not self.moe:
            self.linear(f"blk.{il}.ffn_gate", f"{hf}.mlp.gate_proj")
            self.linear(f"blk.{il}.ffn_up", f"{hf}.mlp.up_proj")
            self.linear(f"blk.{il}.ffn_down", f"{hf}.mlp.down_proj")
            return
        n_expert = self.cfg["num_experts"]
        self.f32(f"blk.{il}.ffn_gate_inp.weight", f"{hf}.mlp.gate.weight")
        self.f32(f"blk.{il}.ffn_gate_inp_shexp.weight", f"{hf}.mlp.shared_expert_gate.weight")
        self.linear(f"blk.{il}.ffn_gate_shexp", f"{hf}.mlp.shared_expert.gate_proj")
        self.linear(f"blk.{il}.ffn_up_shexp", f"{hf}.mlp.shared_expert.up_proj")
        self.linear(f"blk.{il}.ffn_down_shexp", f"{hf}.mlp.shared_expert.down_proj")
        self.experts(f"blk.{il}.ffn_gate_exps", hf, "gate_proj", n_expert)
        self.experts(f"blk.{il}.ffn_up_exps", hf, "up_proj", n_expert)
        self.experts(f"blk.{il}.ffn_down_exps", hf, "down_proj", n_expert)

    def add_block(self, il, hf, linear):
        if self.moe:
            self.add_hc(f"blk.{il}.hc_attn", f"{hf}.attn_hyper_connection")
        else:
            self.norm(f"blk.{il}.attn_norm.weight", f"{hf}.input_layernorm.weight")
        if linear:
            self.add_gdn(il, f"{hf}.linear_attn")
        else:
            self.add_attention(il, f"{hf}.self_attn")
        if il in self.ple_layers:
            self.add_ple(il, f"{hf}.ple")
        if self.moe:
            self.add_hc(f"blk.{il}.hc_ffn", f"{hf}.mlp_hyper_connection")
        else:
            self.norm(f"blk.{il}.post_attention_norm.weight", f"{hf}.post_attention_layernorm.weight")
        self.add_ffn(il, hf)

    def add_tensors(self):
        prefix = self.prefix
        self.linear("token_embd", f"{prefix}.embed_tokens")
        if self.moe:
            self.add_hc("output_hc", f"{prefix}.hyper_connection_mixer")
        else:
            self.norm("output_norm.weight", f"{prefix}.norm.weight")
        if "lm_head.weight" in self.st and not self.cfg["tie_word_embeddings"]:
            head = self.linear_q8_0 if self.quant_head else self.linear
            head("output", "lm_head")
        for il in range(self.n_layer):
            self.add_block(il, f"{prefix}.layers.{il}", self.layer_types[il] == "linear_attention")
        for m in range(self.n_mtp):
            il = self.n_layer + m
            self.add_block(il, f"mtp.layers.{m}", linear=False)
            self.linear(f"blk.{il}.nextn.eh_proj", "mtp.fc")
            self.norm(f"blk.{il}.nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight")
            self.norm(f"blk.{il}.nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight")
            self.norm(f"blk.{il}.nextn.shared_head_norm.weight", "mtp.norm.weight")

    def add_tensors_mtp(self):
        """The Flash-Next MTP drafter as its own GGUF (ds4 --mtp-model): one
        full-attention block as blk.0 (no PLE), the embedding/hidden fusion
        in front of it and the output mixer behind it.  Token embedding and
        lm_head are the main model's."""
        hf = "mtp.layers.0"
        self.add_hc("blk.0.hc_attn", f"{hf}.attn_hyper_connection")
        self.add_attention(0, f"{hf}.self_attn")
        self.add_hc("blk.0.hc_ffn", f"{hf}.mlp_hyper_connection")
        self.add_ffn(0, hf)
        self.linear("nextn.fc_embedding", "mtp.fc_embedding")
        self.linear("nextn.fc_hidden", "mtp.fc_hidden")
        self.norm("nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight")
        self.norm("nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight")
        self.add_hc("output_hc", "mtp.hyper_connection_mixer")
        # The drafter scores through the main lm_head; a NVFP4 copy reads a
        # quarter of the bytes per draft, the drafter only proposing tokens.
        self.linear_nvfp4("output", "lm_head")

    def add_tensors_vision(self):
        """The vision tower as its own GGUF (ds4 --vision): every model.visual.*
        tensor copied raw in BF16, 27 blocks plus patch embedding, learned
        position grid and the 2x2 merger."""
        st = self.st
        names = sorted(n for n in st.names() if n.startswith("model.visual."))
        if len(names) != 333:
            fail(f"--vision expects the 333 Qwen3-VL tower tensors, found {len(names)}")
        for name in names:
            if st.dtype(name) != "BF16":
                fail(f"{name}: expected a BF16 vision tensor, got {st.dtype(name)}")
            shape = st.shape(name)
            self.writer.add_tensor(name, shape, T_BF16, 2 * int(np.prod(shape)), lambda n=name: st.read(n))

    def add_metadata_vision(self):
        with open(os.path.join(self.hf_dir, "config.json"), encoding="utf-8") as fp:
            config = json.load(fp)
        with open(os.path.join(self.hf_dir, "preprocessor_config.json"), encoding="utf-8") as fp:
            processor = json.load(fp)
        vision = config["vision_config"]
        expected = {"depth": 27, "hidden_size": 1152, "intermediate_size": 4304, "num_heads": 16,
                    "out_hidden_size": 2560, "patch_size": 16, "spatial_merge_size": 2,
                    "temporal_patch_size": 2, "num_position_embeddings": 2304,
                    "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": []}
        for key, value in expected.items():
            if vision.get(key) != value:
                fail(f"unexpected vision_config.{key}: {vision.get(key)!r}")
        if processor.get("image_mean") != [0.5, 0.5, 0.5] or processor.get("image_std") != [0.5, 0.5, 0.5]:
            fail("unexpected image normalization constants")
        w, a = self.writer, "qwen4exp-vision"
        w.add_string("general.architecture", a)
        w.add_string("general.type", "model")
        w.add_string("general.name", self.name + " Vision Encoder")
        w.add_u32("general.quantization_version", 2)
        w.add_u32("general.file_type", FILE_TYPE_MOSTLY_BF16)
        w.add_u32(f"{a}.block_count", vision["depth"])
        w.add_u32(f"{a}.embedding_length", vision["hidden_size"])
        w.add_u32(f"{a}.feed_forward_length", vision["intermediate_size"])
        w.add_u32(f"{a}.attention.head_count", vision["num_heads"])
        w.add_u32(f"{a}.projection_length", vision["out_hidden_size"])
        w.add_u32(f"{a}.patch_size", vision["patch_size"])
        w.add_u32(f"{a}.temporal_patch_size", vision["temporal_patch_size"])
        w.add_u32(f"{a}.spatial_merge_size", vision["spatial_merge_size"])
        w.add_u32(f"{a}.position_embedding_count", vision["num_position_embeddings"])
        w.add_u32(f"{a}.image_token_id", config["image_token_id"])
        w.add_u32(f"{a}.vision_start_token_id", config["vision_start_token_id"])
        w.add_u32(f"{a}.vision_end_token_id", config["vision_end_token_id"])
        w.add_u32(f"{a}.image.min_pixels", processor["size"]["shortest_edge"])
        w.add_u32(f"{a}.image.max_pixels", processor["size"]["longest_edge"])

    def linear_nvfp4(self, gguf_base, hf_base):
        """A bf16 projection quantized to NVFP4 in row chunks with one global scale."""
        st = self.st
        name = hf_base + ".weight"
        rows, cols = st.shape(name)
        chunk = 8192
        scale = np.zeros(1, dtype=np.float32)

        def produce():
            amax = 0.0
            for r0 in range(0, rows, chunk):
                amax = max(amax, float(np.max(np.abs(bf16_to_f32(st.read(name)[r0:r0 + chunk])))))
            scale[0] = amax / (6.0 * 448.0) if amax > 0 else 1.0
            out = np.empty((rows, cols // 64 * 36), dtype=np.uint8)
            for r0 in range(0, rows, chunk):
                out[r0:r0 + chunk], _ = nvfp4_quantize(bf16_to_f32(st.read(name)[r0:r0 + chunk]), scale[0])
            return out

        self.uses_nvfp4 = True
        self.writer.add_tensor(gguf_base + ".weight", [rows, cols], T_NVFP4, rows * (cols // 64 * 36), produce)
        self.writer.add_tensor(gguf_base + ".scale", [1], T_F32, 4, lambda: scale)

    def linear_q8_0(self, gguf_base, hf_base, row_perm=None, row_head=1,
                    col_perm=None, col_head=1):
        """A projection quantized to GGML q8_0 (the int8 lm_head, and under
        --quant-gdn the three big GDN projections), with the same head
        permutations the BF16 path applies."""
        st = self.st
        name = hf_base + ".weight"
        if st.dtype(name) not in SOURCE_TYPES:
            fail(f"{name}: the q8_0 path needs float weights, source dtype is {st.dtype(name)}")
        rows, cols = st.shape(name)
        if cols % Q8_0_BLOCK:
            fail(f"{name}: row length {cols} is not a multiple of {Q8_0_BLOCK}")

        def produce():
            w = st.read_f32(name)
            if row_perm is not None:
                w = reorder_heads(w, 0, row_perm, row_head)
            if col_perm is not None:
                w = reorder_heads(w, 1, col_perm, col_head)
            return q8_0_quantize(w)

        self.writer.add_tensor(gguf_base + ".weight", [rows, cols], T_Q8_0,
                               rows * (cols // Q8_0_BLOCK) * Q8_0_BLOCK_BYTES, produce)

    # -- n-gram sidecar ---------------------------------------------------

    def write_ngram_table(self, verbose):
        """Stream the E4M3 shards into the .ngram file in row order."""
        st = self.st
        shards = self.ple_shards()
        row_bytes = self.ple_row_dim()
        rows = sum(st.shape(n)[0] for n in shards)
        multipliers, offsets, sizes = self.ple_constants()
        if max(o + s for o, s in zip(offsets, sizes)) > rows:
            fail("PLE head ranges exceed the table rows")
        scale_name = self.ple_prefix() + ".ngram_embedding.weight_scale"
        scale = float(st.read_f32(scale_name).reshape(-1)[0]) if scale_name in st else 1.0
        with open(self.ngram_path, "wb") as fp:
            fp.write(ngram_header(rows, row_bytes, scale, multipliers, offsets, sizes))
            for i, name in enumerate(shards):
                if st.dtype(name) != "F8_E4M3":
                    fail(f"{name}: PLE shard is {st.dtype(name)}, expected F8_E4M3")
                fp.write(memoryview(st.read(name)).cast("B"))
                if verbose:
                    print(f"[ngram {i + 1}/{len(shards)}] {name}", file=sys.stderr)
        print(f"qwen-convert: wrote {self.ngram_path} ({rows} rows x {row_bytes} B, scale {scale:g})",
              file=sys.stderr)

    # -- driver -----------------------------------------------------------

    def run(self, dry_run, verbose, mtp=False, vision=False):
        if mtp:
            if not self.moe or "mtp.fc_hidden.weight" not in self.st:
                fail("--mtp needs a Flash-Next checkpoint with an mtp.* drafter")
            self.add_tensors_mtp()
        elif vision:
            if not self.moe:
                fail("--vision needs a Flash-Next checkpoint")
            self.add_tensors_vision()
        else:
            if self.ple_layers and not dry_run:
                self.ngram_path = os.path.splitext(self.writer.path)[0] + ".ngram"
            self.add_tensors()
        if vision:
            self.add_metadata_vision()
        else:
            self.add_metadata()   # after tensors: file_type depends on what was emitted
        if mtp:
            self.writer.add_bool(f"{self.arch}.mtp_model", True)   # the main model's shape keys, one drafter block
        totals = {}
        for _, _, qtype, nbytes, _ in self.writer.tensors:
            count, size = totals.get(qtype, (0, 0))
            totals[qtype] = (count + 1, size + nbytes)
        print(f"qwen-convert: {self.arch}, {self.n_layer} layers + {self.n_mtp} MTP, "
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
        print(f"qwen-convert: wrote {self.writer.path} ({size / 2**30:.3f} GiB)", file=sys.stderr)
        if self.ngram_path:
            self.write_ngram_table(verbose)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hf_dir", help="Hugging Face checkpoint directory")
    parser.add_argument("-o", "--out", required=True, help="output GGUF path (the .ngram sidecar goes beside it)")
    parser.add_argument("--name", help="general.name (default: directory basename)")
    parser.add_argument("--dry-run", action="store_true", help="print the tensor plan and exit")
    parser.add_argument("--mtp", action="store_true",
                        help="write the Flash-Next MTP drafter alone (for ds4 --mtp-model), experts quantized to NVFP4")
    parser.add_argument("--vision", action="store_true",
                        help="write the Flash-Next vision tower alone (for ds4 --vision)")
    parser.add_argument("--quant-head", action="store_true",
                        help="quantize the lm_head to q8_0 (int8)")
    parser.add_argument("--quant-gdn", action="store_true",
                        help="quantize the three big GDN projections to q8_0 (int8)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.out) and not args.dry_run:
        fail(f"output exists: {args.out}")
    name = args.name or os.path.basename(os.path.normpath(args.hf_dir))
    Converter(args.hf_dir, args.out, name, quant_head=args.quant_head,
              quant_gdn=args.quant_gdn).run(args.dry_run, args.verbose, mtp=args.mtp, vision=args.vision)


if __name__ == "__main__":
    main()

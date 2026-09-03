#!/usr/bin/env python3
"""Unit tests for gguf-tools/hf_gguf.py: NVFP4 repacking, V head reordering,
GGUF round trip, and tokenizer export.  Runs without a checkpoint."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hf_gguf as hg  # noqa: E402


def random_modelopt(rng, out_features, in_features):
    weight = rng.integers(0, 256, size=(out_features, in_features // 2), dtype=np.uint8)
    # Valid finite E4M3 magnitudes (exclude 0x7F NaN); random sign bit as ModelOpt files may carry it.
    scale = rng.integers(0, 0x7F, size=(out_features, in_features // 16), dtype=np.uint8)
    scale |= (rng.integers(0, 2, size=scale.shape, dtype=np.uint8) << 7)
    return weight, scale


class NVFP4Tests(unittest.TestCase):
    def test_ue4m3_matches_formula(self):
        bits = np.arange(256, dtype=np.uint8)
        got = hg.ue4m3_to_f32(bits)
        for b in range(256):
            x = b & 0x7F
            exp, man = x >> 3, x & 7
            if x == 0x7F:
                expect = 0.0
            elif exp == 0:
                expect = man * 2.0 ** -9
            else:
                expect = (1 + man / 8) * 2.0 ** (exp - 7)
            self.assertEqual(float(got[b]), expect, msg=f"bits {b:#04x}")

    def test_repack_preserves_values(self):
        rng = np.random.default_rng(1)
        weight, scale = random_modelopt(rng, 6, 192)
        raw, shape = hg.nvfp4_repack(weight, scale)
        self.assertEqual(shape, [6, 192])
        self.assertEqual(raw.shape, (6, 192 // 64 * 36))
        expect = hg.nvfp4_dequant_modelopt(weight, scale, 0.75)
        got = hg.nvfp4_dequant_gguf(raw, 192, 0.75)
        np.testing.assert_array_equal(got, expect)

    def test_repack_layout_is_ggml(self):
        # One row, one super-block, element j = code j (0..15) repeated per 16-block.
        codes = np.tile(np.arange(16, dtype=np.uint8), 4)                 # 64 elements
        weight = (codes[0::2] | (codes[1::2] << 4)).reshape(1, 32)        # ModelOpt pairs
        scale = np.array([[0x38, 0x40, 0x48, 0x50]], dtype=np.uint8)      # 1.0, 2.0, 4.0, 8.0
        raw, _ = hg.nvfp4_repack(weight, scale)
        self.assertEqual(list(raw[0, :4]), [0x38, 0x40, 0x48, 0x50])
        for s in range(4):
            block = raw[0, 4 + 8 * s: 4 + 8 * (s + 1)]
            self.assertEqual(list(block & 0x0F), list(range(8)))          # element j low nibble
            self.assertEqual(list(block >> 4), list(range(8, 16)))        # element j+8 high nibble


class ExpertStackTests(unittest.TestCase):
    def test_stack_matches_per_expert_repack(self):
        rng = np.random.default_rng(3)
        experts = [random_modelopt(rng, 4, 128) for _ in range(3)]
        raw, shape = hg.nvfp4_stack_experts(lambda e: experts[e][0], lambda e: experts[e][1], 3)
        self.assertEqual(shape, [3, 4, 128])
        self.assertEqual(raw.shape, (3, 4, 128 // 64 * 36))
        for e in range(3):
            np.testing.assert_array_equal(raw[e], hg.nvfp4_repack(*experts[e])[0])


class NgramHeaderTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.ngram")
            header = hg.ngram_header(320001536, 160, 0.125, [23703573157769, 20109073645365, 8052911324071],
                                     [0, 20000003], [20000003, 20000023])
            self.assertEqual(len(header), hg.NGRAM_HEADER_BYTES)
            with open(path, "wb") as fp:
                fp.write(header)
            h = hg.read_ngram_header(path)
        self.assertEqual(h["rows"], 320001536)
        self.assertEqual(h["row_bytes"], 160)
        self.assertEqual(h["scale"], 0.125)
        self.assertEqual(h["multipliers"], [23703573157769, 20109073645365, 8052911324071])
        self.assertEqual(h["head_offsets"], [0, 20000003])
        self.assertEqual(h["head_vocab_sizes"], [20000003, 20000023])


class ReorderTests(unittest.TestCase):
    def test_perm_is_tiled(self):
        # 2 K heads, 3 V per K: grouped [00 01 02 10 11 12] -> tiled [00 10 01 11 02 12]
        self.assertEqual(list(hg.v_head_perm(2, 3)), [0, 3, 1, 4, 2, 5])
        self.assertEqual(list(hg.v_head_perm(4, 1)), [0, 1, 2, 3])

    def test_identity_is_noop(self):
        x = np.arange(24).reshape(4, 6)
        self.assertIs(hg.reorder_heads(x, 0, np.arange(4), 1), x)

    def test_nibble_reorder_matches_float_reorder(self):
        """Reordering packed NVFP4 rows/cols must equal reordering the dequantized matrix."""
        rng = np.random.default_rng(2)
        num_k, per_k, head = 2, 3, 32
        n_heads = num_k * per_k
        perm = hg.v_head_perm(num_k, per_k)
        # rows = V-head-indexed outputs, cols = 128 arbitrary inputs
        weight, scale = random_modelopt(rng, n_heads * head, 128)
        ref = hg.nvfp4_dequant_modelopt(weight, scale, 1.0)
        got = hg.nvfp4_dequant_gguf(hg.nvfp4_repack(hg.reorder_heads(weight, 0, perm, head),
                                                    hg.reorder_heads(scale, 0, perm, head))[0], 128, 1.0)
        np.testing.assert_array_equal(got, hg.reorder_heads(ref, 0, perm, head))
        # cols = V-head-indexed inputs (out_proj), 64 arbitrary outputs
        weight, scale = random_modelopt(rng, 64, n_heads * head)
        ref = hg.nvfp4_dequant_modelopt(weight, scale, 1.0)
        got = hg.nvfp4_dequant_gguf(hg.nvfp4_repack(hg.reorder_heads(weight, 1, perm, head // 2),
                                                    hg.reorder_heads(scale, 1, perm, head // 16))[0],
                                    n_heads * head, 1.0)
        np.testing.assert_array_equal(got, hg.reorder_heads(ref, 1, perm, head))


class GGUFRoundTrip(unittest.TestCase):
    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.gguf")
            w = hg.GGUFWriter(path)
            w.add_string("general.architecture", "qwen35")
            w.add_u32("qwen35.block_count", 3)
            w.add_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
            w.add_bool("tokenizer.ggml.add_bos_token", False)
            w.add_i32_array("qwen35.rope.dimension_sections", [11, 11, 10, 0])
            w.add_string_array("tokenizer.ggml.tokens", ["a", "b"])
            a = np.arange(6, dtype=np.float32).reshape(2, 3)
            b = np.arange(2 * 64 // 64 * 36, dtype=np.uint8).reshape(2, 36)
            w.add_array("a.weight", a, hg.T_F32)
            w.add_tensor("b.weight", [2, 64], hg.T_NVFP4, b.nbytes, lambda: b)
            w.add_array("c.weight", hg.f32_to_bf16(a), hg.T_BF16)
            w.write()

            kv, tensors, data_offset = hg.read_gguf(path)
            self.assertEqual(kv["general.architecture"], "qwen35")
            self.assertEqual(kv["qwen35.block_count"], 3)
            self.assertAlmostEqual(kv["qwen35.attention.layer_norm_rms_epsilon"], 1e-6)
            self.assertFalse(kv["tokenizer.ggml.add_bos_token"])
            self.assertEqual(kv["qwen35.rope.dimension_sections"], [11, 11, 10, 0])
            self.assertEqual(kv["tokenizer.ggml.tokens"], ["a", "b"])
            self.assertEqual(tensors["a.weight"], ([2, 3], hg.T_F32, 0, 24))
            self.assertEqual(tensors["b.weight"], ([2, 64], hg.T_NVFP4, 32, 72))
            self.assertEqual(tensors["c.weight"][1:], (hg.T_BF16, 32 + 96, 12))
            self.assertEqual(data_offset % hg.GGUF_ALIGNMENT, 0)
            with open(path, "rb") as fp:
                fp.seek(data_offset)
                np.testing.assert_array_equal(np.frombuffer(fp.read(24), dtype=np.float32).reshape(2, 3), a)
                fp.seek(data_offset + 32)
                np.testing.assert_array_equal(np.frombuffer(fp.read(72), dtype=np.uint8).reshape(2, 36), b)

    def test_bf16_rounding(self):
        values = np.array([1.0, 1.00390625, 1.01171875, -3.0e38, 1.5e-38], dtype=np.float32)
        bits = hg.f32_to_bf16(values)
        # 1.00390625 = 1 + 2^-8 ties to even (1.0); 1.01171875 = 1 + 3*2^-8 rounds up to 1 + 2^-6.
        self.assertEqual(list(bits[:3]), [0x3F80, 0x3F80, 0x3F82])
        np.testing.assert_allclose(hg.bf16_to_f32(bits), values, rtol=2 ** -7)


class TokenizerTests(unittest.TestCase):
    def test_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = {
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"]]},
                "added_tokens": [
                    {"id": 3, "content": "<|im_end|>", "special": True},
                    {"id": 4, "content": "<think>", "special": False},
                    {"id": 5, "content": "<|foo|>", "special": False},
                ],
            }
            with open(os.path.join(tmp, "tokenizer.json"), "w") as fp:
                json.dump(tok, fp)
            with open(os.path.join(tmp, "tokenizer_config.json"), "w") as fp:
                json.dump({"eos_token": "<|im_end|>", "pad_token": {"content": "<|foo|>"},
                           "add_bos_token": False, "chat_template": "{{ x }}",
                           "added_tokens_decoder": {"6": {"content": "<|extra|>", "special": True}}}, fp)
            w = hg.GGUFWriter(os.path.join(tmp, "t.gguf"))
            tokens, types = hg.export_tokenizer(w, tmp, 8, "qwen35")
            w.write()
            kv, _, _ = hg.read_gguf(w.path)
        self.assertEqual(tokens, ["a", "b", "ab", "<|im_end|>", "<think>", "<|foo|>", "<|extra|>", "[PAD7]"])
        self.assertEqual(types, [1, 1, 1, 3, 4, 3, 3, 5])
        self.assertEqual(kv["tokenizer.ggml.merges"], ["a b"])
        self.assertEqual(kv["tokenizer.ggml.eos_token_id"], 3)
        self.assertEqual(kv["tokenizer.ggml.padding_token_id"], 5)
        self.assertNotIn("tokenizer.ggml.bos_token_id", kv)
        self.assertFalse(kv["tokenizer.ggml.add_bos_token"])
        self.assertEqual(kv["tokenizer.chat_template"], "{{ x }}")
        self.assertEqual(kv["tokenizer.ggml.pre"], "qwen35")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Reference embeddings for the Qwen3.8-Flash-Next vision tower.

Runs the checkpoint's vision weights through transformers' Qwen3-VL vision
model and image processor on the CPU in f32 and writes the preprocessed
patches and the projected embeddings as raw f32 files, the format
tests/test_qwen_vision_engine writes, for tests/compare_glm53_vision_embeddings.py.

    qwen_vision_ref.py HF_DIR IMAGE EMBED.f32 [PATCHES.f32]
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


def main():
    if len(sys.argv) not in (4, 5):
        print(__doc__, file=sys.stderr)
        return 2
    hf_dir, image_path, embed_path = sys.argv[1:4]
    with open(os.path.join(hf_dir, "config.json"), encoding="utf-8") as fp:
        config = json.load(fp)
    with open(os.path.join(hf_dir, "preprocessor_config.json"), encoding="utf-8") as fp:
        processor_config = json.load(fp)
    with open(os.path.join(hf_dir, "model.safetensors.index.json"), encoding="utf-8") as fp:
        weight_map = json.load(fp)["weight_map"]

    vision = {k: v for k, v in config["vision_config"].items() if k != "model_type"}
    model = Qwen3VLVisionModel(Qwen3VLVisionConfig(**vision)).float().eval()
    state = {}
    shards = sorted({v for k, v in weight_map.items() if k.startswith("model.visual.")})
    for shard in shards:
        with safe_open(os.path.join(hf_dir, shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    state[key[len("model.visual."):]] = f.get_tensor(key).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"state dict mismatch: missing {missing} unexpected {unexpected}", file=sys.stderr)
        return 1

    processor = Qwen2VLImageProcessor(
        patch_size=processor_config["patch_size"],
        temporal_patch_size=processor_config["temporal_patch_size"],
        merge_size=processor_config["merge_size"],
        image_mean=processor_config["image_mean"],
        image_std=processor_config["image_std"],
        size=processor_config["size"],
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")
    pixels = inputs["pixel_values"].float()
    grid = inputs["image_grid_thw"]
    with torch.no_grad():
        embeds = model(pixels, grid).pooler_output
    print(f"{image.size[0]}x{image.size[1]} -> grid {grid.tolist()}, {embeds.shape[0]} image tokens of {embeds.shape[1]}")
    embeds.numpy().astype(np.float32).tofile(embed_path)
    if len(sys.argv) == 5:
        pixels.numpy().astype(np.float32).tofile(sys.argv[4])
    return 0


if __name__ == "__main__":
    sys.exit(main())

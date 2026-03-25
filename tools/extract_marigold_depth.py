#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data import build_frame_key


def iter_scene_frames(scene_root):
    scene_root = Path(scene_root)
    for split in ("train", "val", "test"):
        json_path = scene_root / f"transforms_{split}.json"
        if not json_path.exists():
            continue
        with json_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        for frame in metadata.get("frames", []):
            relative_path = frame["file_path"]
            image_path = scene_root / relative_path.replace("/", os.sep)
            if not image_path.exists():
                continue
            yield build_frame_key(relative_path), image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Marigold monocular depth priors for an official Blender scene.")
    parser.add_argument("scene_root", type=str, help="Path to an official Blender scene root")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the Marigold checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--auxiliary-dir", type=str, default="auxiliaries")
    parser.add_argument("--skip-existing", action="store_true", help="Deprecated. Kept for compatibility; outputs are always overwritten.")
    args = parser.parse_args()

    from marigold import MarigoldPipeline

    pipe = MarigoldPipeline.from_pretrained(args.checkpoint).to(args.device)

    scene_root = Path(args.scene_root)
    depth_root = scene_root / args.auxiliary_dir / "depth"
    depth_root.mkdir(parents=True, exist_ok=True)

    frames = list(iter_scene_frames(scene_root))
    with torch.no_grad():
        for frame_key, image_path in tqdm(frames, desc="Estimating Marigold depth"):
            output_path = depth_root / f"{frame_key}.npy"

            input_image = Image.open(image_path).convert("RGB")
            depth_pred = pipe(input_image).depth_np
            depth_pred = np.clip(depth_pred, 0.0, 1.0).astype(np.float32)
            np.save(output_path, depth_pred)

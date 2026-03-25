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
from core.losses.structure_prior import build_structure_extractor



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
    parser = argparse.ArgumentParser(description="Extract CIConv-based structure priors for an official Blender scene.")
    parser.add_argument("scene_root", type=str, help="Path to an official Blender scene root")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--auxiliary-dir", type=str, default="auxiliaries")
    parser.add_argument("--invariant", type=str, default="W")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--skip-existing", action="store_true", help="Deprecated. Kept for compatibility; outputs are always overwritten.")
    args = parser.parse_args()

    extractor = build_structure_extractor(
        invariant=args.invariant,
        kernel_size=args.kernel_size,
        scale=args.scale,
    ).to(args.device)
    extractor.eval()

    scene_root = Path(args.scene_root)
    structure_root = scene_root / args.auxiliary_dir / "structure"
    structure_root.mkdir(parents=True, exist_ok=True)

    frames = list(iter_scene_frames(scene_root))
    with torch.no_grad():
        for frame_key, image_path in tqdm(frames, desc="Extracting structure priors"):
            output_path = structure_root / f"{frame_key}.png"

            input_image = Image.open(image_path).convert("RGB")
            image_np = np.asarray(input_image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(args.device)

            structure = extractor(image_tensor).squeeze(0).squeeze(0).clamp(0.0, 1.0).cpu().numpy()
            structure_uint8 = np.rint(structure * 255.0).astype(np.uint8)
            Image.fromarray(structure_uint8, mode="L").save(output_path)

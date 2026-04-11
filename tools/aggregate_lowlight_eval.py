#!/usr/bin/env python
"""Aggregate per-scene eval_test.json files into method-wise scene averages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


METHOD_DIRS = {
    "3dgs": "3dgs",
    "alethnerf": "AlethNeRF",
    "i2nerf": "I2NeRF",
    "lita-gs": "LITA-GS",
    "luminance-gs": "Luminance-GS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate lowlight eval_test.json files.")
    parser.add_argument(
        "--lowlight-root",
        type=Path,
        default=Path("lowlight"),
        help="Path to the lowlight root directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="method_scene_average.json",
        help="Filename written into the lowlight root directory.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gather_scene_dirs(lowlight_root: Path) -> list[Path]:
    scene_dirs = sorted(path for path in lowlight_root.iterdir() if path.is_dir())
    valid_scene_dirs = []
    for scene_dir in scene_dirs:
        if all((scene_dir / method_dir / "eval_test.json").exists() for method_dir in METHOD_DIRS.values()):
            valid_scene_dirs.append(scene_dir)
    if not valid_scene_dirs:
        raise FileNotFoundError(f"No valid scene directories found under: {lowlight_root}")
    return valid_scene_dirs


def main() -> None:
    args = parse_args()
    lowlight_root = args.lowlight_root.resolve()
    if not lowlight_root.exists():
        raise FileNotFoundError(f"lowlight root does not exist: {lowlight_root}")

    scene_dirs = gather_scene_dirs(lowlight_root)
    by_method = {method_name: {} for method_name in METHOD_DIRS}

    for scene_dir in scene_dirs:
        for method_name, method_dir in METHOD_DIRS.items():
            eval_path = scene_dir / method_dir / "eval_test.json"
            by_method[method_name][scene_dir.name] = read_json(eval_path)

    method_means = {}
    for method_name, scene_metrics in by_method.items():
        psnr_values = [metrics["psnr"] for metrics in scene_metrics.values()]
        ssim_values = [metrics["ssim"] for metrics in scene_metrics.values()]
        lpips_values = [metrics["lpips"] for metrics in scene_metrics.values()]
        method_means[method_name] = {
            "num_scenes": len(scene_metrics),
            "psnr": mean(psnr_values),
            "ssim": mean(ssim_values),
            "lpips": mean(lpips_values),
        }

    output = {
        "lowlight_root": lowlight_root.as_posix(),
        "num_scenes": len(scene_dirs),
        "scene_names": [scene_dir.name for scene_dir in scene_dirs],
        "method_means": method_means,
        "by_method_by_scene": by_method,
    }

    output_path = lowlight_root / args.output_name
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    for method_name, metrics in method_means.items():
        print(
            f"{method_name:<13} "
            f"PSNR={metrics['psnr']:.6f} "
            f"SSIM={metrics['ssim']:.6f} "
            f"LPIPS={metrics['lpips']:.6f}"
        )
    print(f"Aggregated results written to: {output_path}")


if __name__ == "__main__":
    main()

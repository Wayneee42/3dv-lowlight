#!/usr/bin/env python
"""Evaluate low-light benchmark renders against ground-truth test images.

This script is designed for the directory layout used by this repository:

lowlight/
  <SceneName>/
    transforms_test.json
    test/
    3dgs/test/
    AlethNeRF/test/
    I2NeRF/test/
    LITA-GS/test/
    Luminance-GS/test/

For each scene, the script:
1. reads the official GT test-frame order from transforms_test.json
2. pairs GT images with each method's rendered test images
3. computes PSNR / SSIM / LPIPS
4. writes a per-scene JSON file into the scene root
5. writes an all-scene summary JSON into the lowlight root
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
METHOD_DIRS = {
    "3dgs": "3dgs",
    "alethnerf": "AlethNeRF",
    "i2nerf": "I2NeRF",
    "lita-gs": "LITA-GS",
    "luminance-gs": "Luminance-GS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lowlight scene renders.")
    parser.add_argument(
        "--lowlight-root",
        type=Path,
        default=Path("lowlight"),
        help="Path to the lowlight root directory.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional subset of scene names to evaluate. Default: all valid scenes under lowlight root.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, for example 'cuda', 'cuda:0', or 'cpu'. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--scene-output-name",
        type=str,
        default="benchmark_results.json",
        help="Filename written into each scene root.",
    )
    parser.add_argument(
        "--overall-output-name",
        type=str,
        default="benchmark_results_all_scenes.json",
        help="Filename written into the lowlight root.",
    )
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Skip LPIPS when lpips is unavailable or not needed.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        device = torch.device(device_arg)
        if device.type.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_lpips_model(device: torch.device):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "lpips is required for LPIPS evaluation but is not installed. "
            "Install it with: pip install lpips "
            "or rerun this script with --skip-lpips."
        ) from exc
    model = lpips.LPIPS(net="vgg").to(device)
    model.eval()
    return model


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = to_tensor(image)
    return tensor.permute(1, 2, 0).to(device)


def psnr(rendered: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    mse = ((rendered - target) ** 2).mean().clamp_min(eps)
    return float((-10.0 * torch.log10(mse)).item())


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute SSIM between two HWC float tensors in [0, 1]."""
    c1 = 0.01**2
    c2 = 0.03**2
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    channels = img1.shape[1]
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    gauss_1d = torch.exp(-(coords**2) / (2 * 1.5**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = (gauss_1d[:, None] * gauss_1d[None, :]).expand(channels, 1, window_size, window_size)
    pad = window_size // 2
    mu1 = F.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu12
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def lpips_score(lpips_model, rendered: torch.Tensor, target: torch.Tensor) -> float | None:
    if lpips_model is None:
        return None
    rendered_nchw = rendered.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    target_nchw = target.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    return float(lpips_model(rendered_nchw, target_nchw).mean().item())


def list_image_files(directory: Path) -> list[Path]:
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=lambda path: path.name)


def read_gt_paths(scene_dir: Path) -> list[Path]:
    transform_path = scene_dir / "transforms_test.json"
    metadata = json.loads(transform_path.read_text(encoding="utf-8"))
    gt_paths = []
    for frame in metadata["frames"]:
        frame_path = scene_dir / frame["file_path"]
        if not frame_path.exists():
            raise FileNotFoundError(f"GT image referenced by {transform_path} does not exist: {frame_path}")
        gt_paths.append(frame_path)
    return gt_paths


def validate_scene(scene_dir: Path) -> None:
    required_paths = [scene_dir / "transforms_test.json", scene_dir / "test"]
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Scene is missing required path: {required_path}")
    for method_dir in METHOD_DIRS.values():
        method_test_dir = scene_dir / method_dir / "test"
        if not method_test_dir.exists():
            raise FileNotFoundError(f"Method test directory is missing: {method_test_dir}")


@torch.inference_mode()
def evaluate_method(
    scene_dir: Path,
    method_name: str,
    method_dir_name: str,
    gt_paths: list[Path],
    device: torch.device,
    lpips_model,
) -> dict:
    pred_dir = scene_dir / method_dir_name / "test"
    pred_paths = list_image_files(pred_dir)
    if len(pred_paths) != len(gt_paths):
        raise ValueError(
            f"[{scene_dir.name} | {method_name}] image count mismatch: "
            f"{len(pred_paths)} predictions vs {len(gt_paths)} GT images."
        )

    per_image = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    for gt_path, pred_path in zip(gt_paths, pred_paths):
        gt = load_image(gt_path, device)
        pred = load_image(pred_path, device)
        if pred.shape != gt.shape:
            raise ValueError(
                f"[{scene_dir.name} | {method_name}] shape mismatch: "
                f"{pred_path.name}={tuple(pred.shape)} vs {gt_path.name}={tuple(gt.shape)}"
            )

        psnr_value = psnr(pred, gt)
        ssim_value = float(ssim(pred, gt).item())
        lpips_value = lpips_score(lpips_model, pred, gt)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        if lpips_value is not None:
            lpips_values.append(lpips_value)

        image_result = {
            "gt": gt_path.relative_to(scene_dir).as_posix(),
            "pred": pred_path.relative_to(scene_dir).as_posix(),
            "psnr": psnr_value,
            "ssim": ssim_value,
            "lpips": lpips_value,
        }
        per_image.append(image_result)

    summary = {
        "num_images": len(per_image),
        "psnr": mean(psnr_values),
        "ssim": mean(ssim_values),
        "lpips": mean(lpips_values) if lpips_values else None,
    }

    return {
        "method": method_name,
        "method_dir": method_dir_name,
        "summary": summary,
        "per_image": per_image,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def gather_scene_dirs(lowlight_root: Path, requested_scenes: list[str] | None) -> list[Path]:
    if requested_scenes:
        scene_dirs = [lowlight_root / scene_name for scene_name in requested_scenes]
    else:
        scene_dirs = sorted(path for path in lowlight_root.iterdir() if path.is_dir())
    valid_scene_dirs = [scene_dir for scene_dir in scene_dirs if (scene_dir / "transforms_test.json").exists()]
    if not valid_scene_dirs:
        raise FileNotFoundError(f"No valid scenes found under: {lowlight_root}")
    return valid_scene_dirs


def build_overall_summary(scene_results: list[dict]) -> dict:
    mean_over_scenes = {}
    for method_name in METHOD_DIRS:
        psnr_values = []
        ssim_values = []
        lpips_values = []
        for scene_result in scene_results:
            summary = scene_result["methods"][method_name]["summary"]
            psnr_values.append(summary["psnr"])
            ssim_values.append(summary["ssim"])
            if summary["lpips"] is not None:
                lpips_values.append(summary["lpips"])
        mean_over_scenes[method_name] = {
            "psnr": mean(psnr_values),
            "ssim": mean(ssim_values),
            "lpips": mean(lpips_values) if lpips_values else None,
        }
    return {
        "num_scenes": len(scene_results),
        "mean_over_scenes": mean_over_scenes,
        "scenes": {scene_result["scene"]: scene_result for scene_result in scene_results},
    }


def main() -> None:
    args = parse_args()
    lowlight_root = args.lowlight_root.resolve()
    if not lowlight_root.exists():
        raise FileNotFoundError(f"lowlight root does not exist: {lowlight_root}")

    device = resolve_device(args.device)
    lpips_model = None if args.skip_lpips else build_lpips_model(device)

    scene_dirs = gather_scene_dirs(lowlight_root, args.scenes)
    scene_results = []

    for scene_dir in tqdm(scene_dirs, desc="Scenes"):
        validate_scene(scene_dir)
        gt_paths = read_gt_paths(scene_dir)

        scene_result = {
            "scene": scene_dir.name,
            "scene_dir": scene_dir.as_posix(),
            "gt_test_count": len(gt_paths),
            "methods": {},
        }

        for method_name, method_dir_name in METHOD_DIRS.items():
            method_result = evaluate_method(
                scene_dir=scene_dir,
                method_name=method_name,
                method_dir_name=method_dir_name,
                gt_paths=gt_paths,
                device=device,
                lpips_model=lpips_model,
            )
            scene_result["methods"][method_name] = method_result
            summary = method_result["summary"]
            lpips_text = "N/A" if summary["lpips"] is None else f"{summary['lpips']:.6f}"
            print(
                f"[{scene_dir.name}] {method_name:<13} "
                f"PSNR={summary['psnr']:.6f} SSIM={summary['ssim']:.6f} LPIPS={lpips_text}"
            )

        scene_output_path = scene_dir / args.scene_output_name
        write_json(scene_output_path, scene_result)
        scene_results.append(scene_result)

    overall_result = {
        "lowlight_root": lowlight_root.as_posix(),
        "device": str(device),
        "lpips_enabled": not args.skip_lpips,
        "scene_output_name": args.scene_output_name,
        "overall_summary": build_overall_summary(scene_results),
    }
    overall_output_path = lowlight_root / args.overall_output_name
    write_json(overall_output_path, overall_result)

    print(f"Per-scene results written to: */{args.scene_output_name}")
    print(f"All-scene summary written to: {overall_output_path}")


if __name__ == "__main__":
    main()

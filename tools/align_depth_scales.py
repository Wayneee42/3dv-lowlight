#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data import build_frame_key


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG")
NPY_EXTS = (".npy",)


def load_train_meta(scene_root):
    meta_path = scene_root / "transforms_train.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    required = ("h", "w", "fl_x", "fl_y", "cx", "cy", "frames")
    for key in required:
        if key not in meta:
            raise KeyError(f"transforms_train.json missing key '{key}'")
    return meta


def resolve_depth_path(depth_root, frame_key):
    for ext in NPY_EXTS + IMAGE_EXTS:
        candidate = depth_root / f"{frame_key}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_depth_array(depth_path, width, height):
    if depth_path.suffix.lower() == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    else:
        depth = np.asarray(Image.open(depth_path).convert("L"), dtype=np.float32) / 255.0
    if depth.shape != (height, width):
        depth = np.asarray(
            Image.fromarray(np.asarray(np.clip(depth, 0.0, 1.0) * 255.0, dtype=np.uint8), mode="L").resize(
                (width, height), resample=Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
    return depth


def backproject_points(depth, c2w, intrinsics, stride, depth_eps):
    h, w = depth.shape
    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    sampled_depth = depth[grid_y, grid_x]
    valid = np.isfinite(sampled_depth) & (sampled_depth > depth_eps)
    if not np.any(valid):
        return None

    u = grid_x[valid].astype(np.float32)
    v = grid_y[valid].astype(np.float32)
    d = sampled_depth[valid].astype(np.float32)

    fx, fy, cx, cy = intrinsics
    x = (u - cx) / fx * d
    y = -(v - cy) / fy * d
    z = -d

    points_cam = np.stack([x, y, z, np.ones_like(x)], axis=1)
    points_world = (c2w @ points_cam.T).T[:, :3]
    return points_world


def project_points(points_world, c2w_tgt, intrinsics, width, height, depth_eps):
    w2c = np.linalg.inv(c2w_tgt)
    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
    pts_cam = (w2c @ points_h.T).T

    d = -pts_cam[:, 2]
    valid = np.isfinite(d) & (d > depth_eps)
    if not np.any(valid):
        return None, None, None

    pts_cam = pts_cam[valid]
    d = d[valid]
    fx, fy, cx, cy = intrinsics

    u = fx * (pts_cam[:, 0] / d) + cx
    v = cy - fy * (pts_cam[:, 1] / d)

    inside = (u >= 0.0) & (u <= (width - 1.0)) & (v >= 0.0) & (v <= (height - 1.0))
    if not np.any(inside):
        return None, None, None

    return u[inside], v[inside], d[inside]


def sample_bilinear(image, u, v):
    h, w = image.shape

    x0 = np.floor(u).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(v).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = u - x0
    wy = v - y0

    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]

    top = Ia * (1.0 - wx) + Ib * wx
    bottom = Ic * (1.0 - wx) + Id * wx
    return top * (1.0 - wy) + bottom * wy


def robust_affine_fit(x, y, delta=0.03, n_iter=10):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if x.size < 8:
        raise RuntimeError("Not enough samples to fit affine transform.")

    A = np.stack([x, np.ones_like(x)], axis=1)
    s, b = np.linalg.lstsq(A, y, rcond=None)[0]

    for _ in range(n_iter):
        residual = y - (s * x + b)
        abs_res = np.abs(residual)
        weights = np.ones_like(abs_res)
        mask = abs_res > delta
        weights[mask] = delta / np.maximum(abs_res[mask], 1e-8)

        Aw = A * weights[:, None]
        yw = y * weights
        s, b = np.linalg.lstsq(Aw, yw, rcond=None)[0]

    return float(s), float(b)


def quantiles(values):
    if values.size == 0:
        return None, None
    return float(np.median(values)), float(np.percentile(values, 90))


def compute_global_normalization(aligned_depths, depth_eps, low_q, high_q):
    valid_values = []
    for depth in aligned_depths.values():
        mask = np.isfinite(depth) & (depth > depth_eps)
        if np.any(mask):
            valid_values.append(depth[mask])
    if not valid_values:
        raise RuntimeError("No valid aligned depth values available for normalization.")

    values = np.concatenate(valid_values, axis=0)
    norm_low = float(np.percentile(values, low_q))
    norm_high = float(np.percentile(values, high_q))
    if not np.isfinite(norm_low) or not np.isfinite(norm_high) or norm_high <= norm_low + 1e-8:
        raise RuntimeError(f"Invalid normalization range: low={norm_low}, high={norm_high}")
    return norm_low, norm_high


def main():
    parser = argparse.ArgumentParser(description="Align per-view Marigold depth scales for train split.")
    parser.add_argument("scene_root", type=str)
    parser.add_argument("--auxiliary-dir", type=str, default="auxiliaries")
    parser.add_argument("--input-modality", type=str, default="depth")
    parser.add_argument("--output-modality", type=str, default="depth_aligned")
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--depth-eps", type=float, default=1e-3)
    parser.add_argument("--min-overlap", type=int, default=256)
    parser.add_argument("--huber-delta", type=float, default=0.03)
    parser.add_argument("--anchor-index", type=int, default=0)
    parser.add_argument("--min-positive-scale", type=float, default=1e-4)
    parser.add_argument("--norm-low-quantile", type=float, default=1.0)
    parser.add_argument("--norm-high-quantile", type=float, default=99.0)
    args = parser.parse_args()

    scene_root = Path(args.scene_root)
    meta = load_train_meta(scene_root)
    width = int(meta["w"])
    height = int(meta["h"])
    intrinsics = (float(meta["fl_x"]), float(meta["fl_y"]), float(meta["cx"]), float(meta["cy"]))

    input_root = scene_root / args.auxiliary_dir / args.input_modality
    output_root = scene_root / args.auxiliary_dir / args.output_modality
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"Input depth directory not found: {input_root}")

    frames = []
    frames_by_key = {}
    for frame in meta.get("frames", []):
        frame_key = build_frame_key(frame["file_path"])
        depth_path = resolve_depth_path(input_root, frame_key)
        if depth_path is None:
            raise FileNotFoundError(f"Missing depth prior for train frame '{frame_key}' in {input_root}")
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :] = np.asarray(frame["transform_matrix"], dtype=np.float32)[:3, :]
        raw_depth = load_depth_array(depth_path, width, height)
        record = {
            "frame_key": frame_key,
            "depth_path": str(depth_path),
            "c2w": c2w,
            "raw": raw_depth,
        }
        frames.append(record)
        frames_by_key[frame_key] = record

    if not frames:
        raise RuntimeError("No train frames found for alignment.")

    anchor_idx = int(np.clip(args.anchor_index, 0, len(frames) - 1))
    aligned = {}
    reports = {}

    anchor_key = frames[anchor_idx]["frame_key"]
    aligned[anchor_key] = frames[anchor_idx]["raw"].astype(np.float32).copy()
    reports[anchor_key] = {
        "source_frame": None,
        "s": 1.0,
        "b": 0.0,
        "overlap": 0,
        "median_before": 0.0,
        "p90_before": 0.0,
        "median_after": 0.0,
        "p90_after": 0.0,
    }

    ordered_indices = list(range(len(frames)))
    ordered_indices.remove(anchor_idx)
    ordered_indices.insert(0, anchor_idx)

    for idx in tqdm(ordered_indices[1:], desc="Aligning train depth scales"):
        tgt = frames[idx]
        tgt_key = tgt["frame_key"]
        best = None

        for src_key, src_aligned in aligned.items():
            src = frames_by_key[src_key]
            points_world = backproject_points(src_aligned, src["c2w"], intrinsics, args.sample_stride, args.depth_eps)
            if points_world is None:
                continue

            u, v, d_pred = project_points(points_world, tgt["c2w"], intrinsics, width, height, args.depth_eps)
            if u is None:
                continue

            raw_tgt_samples = sample_bilinear(tgt["raw"], u, v)
            valid = np.isfinite(raw_tgt_samples) & (raw_tgt_samples > args.depth_eps) & np.isfinite(d_pred)
            if int(valid.sum()) < int(args.min_overlap):
                continue

            x = raw_tgt_samples[valid]
            y = d_pred[valid]
            try:
                s, b = robust_affine_fit(x, y, delta=args.huber_delta)
            except Exception:
                continue
            if not np.isfinite(s) or not np.isfinite(b) or s <= float(args.min_positive_scale):
                continue

            before_err = np.abs(x - y)
            after_err = np.abs((s * x + b) - y)
            before_med, before_p90 = quantiles(before_err)
            after_med, after_p90 = quantiles(after_err)

            candidate = {
                "source_frame": src_key,
                "s": s,
                "b": b,
                "overlap": int(valid.sum()),
                "median_before": before_med,
                "p90_before": before_p90,
                "median_after": after_med,
                "p90_after": after_p90,
            }
            if best is None:
                best = candidate
            else:
                if candidate["overlap"] > best["overlap"]:
                    best = candidate
                elif candidate["overlap"] == best["overlap"] and candidate["median_after"] < best["median_after"]:
                    best = candidate

        if best is None:
            raise RuntimeError(
                f"No valid positive-scale overlap constraints for train frame '{tgt_key}'. "
                f"Try reducing --min-overlap or --sample-stride."
            )

        aligned_depth = best["s"] * tgt["raw"] + best["b"]
        if not np.isfinite(aligned_depth).all() or float(np.nanmax(aligned_depth)) <= 0.0:
            raise RuntimeError(f"Aligned depth for frame '{tgt_key}' is invalid.")

        aligned[tgt_key] = aligned_depth.astype(np.float32)
        reports[tgt_key] = best

    norm_low, norm_high = compute_global_normalization(
        aligned_depths=aligned,
        depth_eps=float(args.depth_eps),
        low_q=float(args.norm_low_quantile),
        high_q=float(args.norm_high_quantile),
    )

    summary_stats = {}
    for frame in frames:
        frame_key = frame["frame_key"]
        aligned_depth = aligned[frame_key]
        normalized = np.clip((aligned_depth - norm_low) / (norm_high - norm_low), 0.0, 1.0).astype(np.float32)

        np.save(output_root / f"{frame_key}.npy", normalized)
        png_uint8 = np.rint(normalized * 255.0).astype(np.uint8)
        Image.fromarray(png_uint8, mode="L").save(output_root / f"{frame_key}.png")

        summary_stats[frame_key] = {
            "normalized_min": float(normalized.min()),
            "normalized_max": float(normalized.max()),
            "normalized_mean": float(normalized.mean()),
            "normalized_std": float(normalized.std()),
        }

    report = {
        "scene_root": str(scene_root),
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "num_frames": len(frames),
        "anchor_frame": anchor_key,
        "sample_stride": int(args.sample_stride),
        "depth_eps": float(args.depth_eps),
        "min_overlap": int(args.min_overlap),
        "huber_delta": float(args.huber_delta),
        "min_positive_scale": float(args.min_positive_scale),
        "normalization": {
            "low_quantile": float(args.norm_low_quantile),
            "high_quantile": float(args.norm_high_quantile),
            "low": norm_low,
            "high": norm_high,
        },
        "frames": reports,
        "normalized_stats": summary_stats,
    }

    report_path = output_root / "depth_align_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved aligned depth maps to {output_root}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()

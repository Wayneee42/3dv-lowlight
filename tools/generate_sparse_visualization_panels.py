#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_payload(visual_dir, step=None):
    visual_dir = Path(visual_dir)
    if not visual_dir.exists():
        raise FileNotFoundError(f"visual_data directory not found: {visual_dir}")
    if step is None:
        files = sorted(visual_dir.glob("step_*.pt"))
        if not files:
            raise FileNotFoundError(f"No step_*.pt found in {visual_dir}")
        payload_path = files[-1]
    else:
        payload_path = visual_dir / f"step_{int(step):06d}.pt"
        if not payload_path.exists():
            raise FileNotFoundError(f"Payload not found: {payload_path}")
    return torch_load_cpu(payload_path), payload_path


def list_step_payloads(visual_dir):
    visual_dir = Path(visual_dir)
    files = sorted(visual_dir.glob("step_*.pt"))
    result = []
    for p in files:
        stem = p.stem
        try:
            step = int(stem.split("_")[-1])
        except ValueError:
            continue
        result.append((step, p))
    return result


def load_topology_points_aggregated(visual_dir, target_step, aggregate_window=1):
    step_files = list_step_payloads(visual_dir)
    step_files = [(s, p) for s, p in step_files if int(s) <= int(target_step)]
    if not step_files:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), []

    if int(aggregate_window) > 0:
        step_files = step_files[-int(aggregate_window) :]
    selected_steps = [int(s) for s, _ in step_files]

    hole_list = []
    spawn_list = []
    for _, p in step_files:
        payload = torch_load_cpu(p)
        topo = payload.get("topology", {})
        hole = np.asarray(topo.get("coverage_hole_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        spawn = np.asarray(topo.get("spawn_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        if hole.ndim == 2 and hole.shape[1] == 3 and hole.shape[0] > 0:
            hole_list.append(hole)
        if spawn.ndim == 2 and spawn.shape[1] == 3 and spawn.shape[0] > 0:
            spawn_list.append(spawn)

    holes = np.concatenate(hole_list, axis=0).astype(np.float32) if hole_list else np.zeros((0, 3), dtype=np.float32)
    spawns = np.concatenate(spawn_list, axis=0).astype(np.float32) if spawn_list else np.zeros((0, 3), dtype=np.float32)
    return holes, spawns, selected_steps


def load_sampling_points_aggregated(visual_dir, target_step, aggregate_window=1):
    step_files = list_step_payloads(visual_dir)
    step_files = [(s, p) for s, p in step_files if int(s) <= int(target_step)]
    if not step_files:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), []

    if int(aggregate_window) > 0:
        step_files = step_files[-int(aggregate_window) :]
    selected_steps = [int(s) for s, _ in step_files]

    means_list = []
    score_list = []
    for _, p in step_files:
        payload = torch_load_cpu(p)
        sp = payload.get("sparse_sampling", {})
        means = np.asarray(sp.get("sampled_means", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        if means.ndim != 2 or means.shape[1] != 3 or means.shape[0] == 0:
            continue
        mid = np.asarray(sp.get("sampled_mid_hard_score", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if mid.shape[0] == means.shape[0] and mid.size > 0 and float(np.max(mid)) > 1.0e-8:
            scores = mid
        else:
            diff = np.asarray(sp.get("sampled_difficulty", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            if diff.shape[0] == means.shape[0]:
                scores = diff
            else:
                scores = np.ones((means.shape[0],), dtype=np.float32)
        means_list.append(means)
        score_list.append(scores.astype(np.float32))

    if not means_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), selected_steps
    means_all = np.concatenate(means_list, axis=0).astype(np.float32)
    score_all = np.concatenate(score_list, axis=0).astype(np.float32)
    return means_all, score_all, selected_steps


def resolve_scene_paths(visual_dir):
    visual_dir = Path(visual_dir)
    scene_dir = visual_dir.parent
    config_path = scene_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    dataset_cfg = cfg.get("DATASET", {})
    priors_cfg = cfg.get("PRIORS", {})
    sparse_cfg = priors_cfg.get("SPARSE", {})
    data_path = Path(str(dataset_cfg.get("DATA_PATH", "")))
    if not data_path.is_absolute():
        data_path = (repo_root / data_path).resolve()
    aux_dir = str(dataset_cfg.get("AUXILIARY_DIR", "auxiliaries"))
    colmap_dir = str(sparse_cfg.get("COLMAP_DIR", f"{aux_dir}/colmap_sparse"))
    colmap_root = Path(colmap_dir) if Path(colmap_dir).is_absolute() else (data_path / colmap_dir).resolve()
    return cfg, sparse_cfg, data_path, colmap_root


def load_original_sparse_prior(colmap_root, sparse_cfg):
    points_path = colmap_root / "points.npy"
    if not points_path.exists():
        raise FileNotFoundError(f"points.npy not found: {points_path}")
    points = np.load(points_path).astype(np.float32)
    track_len = np.ones((points.shape[0],), dtype=np.float32)
    reproj_error = np.ones((points.shape[0],), dtype=np.float32)

    meta_path = colmap_root / "points_meta.npz"
    meta_enabled = bool(sparse_cfg.get("META_ENABLED", False))
    if meta_enabled and meta_path.exists():
        meta = np.load(meta_path)
        meta_xyz = np.asarray(meta["xyz"], dtype=np.float32)
        meta_track_len = np.asarray(meta["track_len"], dtype=np.float32).reshape(-1)
        meta_reproj_error = np.asarray(meta["reproj_error"], dtype=np.float32).reshape(-1)
        if (
            meta_xyz.shape == points.shape
            and meta_track_len.shape[0] == points.shape[0]
            and meta_reproj_error.shape[0] == points.shape[0]
            and np.allclose(meta_xyz, points)
        ):
            track_len = meta_track_len
            reproj_error = meta_reproj_error
    return points, track_len, reproj_error


def compute_reliable_mask(points, track_len, reproj_error, sparse_cfg):
    n = int(points.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)
    filter_enabled = bool(sparse_cfg.get("RELIABILITY_FILTER_ENABLED", False))
    if not filter_enabled:
        return np.ones((n,), dtype=bool)

    min_points = max(
        int(sparse_cfg.get("KNN_K", 3)),
        int(sparse_cfg.get("PLANE_K", 8)),
        32,
    )
    all_keep = np.ones((n,), dtype=bool)
    track_keep = all_keep.copy()
    reproj_keep = all_keep.copy()

    track_low = clamp(float(sparse_cfg.get("FILTER_TRACK_P_LOW", 0.20)), 0.0, 1.0)
    reproj_high = clamp(float(sparse_cfg.get("FILTER_REPROJ_P_HIGH", 0.80)), 0.0, 1.0)

    if track_len is not None and track_len.shape[0] == n:
        track_threshold = float(np.quantile(track_len, track_low))
        track_keep = track_len >= track_threshold
    if reproj_error is not None and reproj_error.shape[0] == n:
        reproj_threshold = float(np.quantile(reproj_error, reproj_high))
        reproj_keep = reproj_error <= reproj_threshold

    keep_mask = track_keep & reproj_keep
    if int(keep_mask.sum()) < min_points:
        if int(reproj_keep.sum()) >= min_points:
            keep_mask = reproj_keep
        elif int(track_keep.sum()) >= min_points:
            keep_mask = track_keep
        else:
            keep_mask = all_keep
    return keep_mask


def to_c2w_4x4(camtoworld):
    c2w = np.asarray(camtoworld, dtype=np.float32)
    if c2w.shape == (3, 4):
        full = np.eye(4, dtype=np.float32)
        full[:3, :] = c2w
        return full
    if c2w.shape == (4, 4):
        return c2w
    raise RuntimeError(f"Unexpected camtoworld shape: {c2w.shape}")


def project_points(points_world, camtoworld, intrinsics, width, height):
    if points_world.shape[0] == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    c2w = to_c2w_4x4(camtoworld)
    k = np.asarray(intrinsics, dtype=np.float32)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]

    cam_gl = (points_world - translation[None, :]) @ rotation
    cam_cv = cam_gl * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    z = cam_cv[:, 2]
    valid = np.isfinite(z) & (z > 1.0e-4)
    if not np.any(valid):
        return np.zeros((points_world.shape[0],), dtype=bool), np.zeros((points_world.shape[0],), dtype=np.float32), np.zeros((points_world.shape[0],), dtype=np.float32)

    u = fx * (cam_cv[:, 0] / np.clip(z, 1.0e-4, None)) + cx
    v = fy * (cam_cv[:, 1] / np.clip(z, 1.0e-4, None)) + cy
    valid &= np.isfinite(u) & np.isfinite(v)
    valid &= (u >= 0.0) & (u <= float(width - 1)) & (v >= 0.0) & (v <= float(height - 1))
    return valid, u.astype(np.float32), v.astype(np.float32)


def sample_points(points, max_points, rng):
    n = int(points.shape[0])
    if n <= max_points:
        return points
    keep = rng.choice(n, size=max_points, replace=False)
    return points[keep]


def dedup_points_voxel(points_xyz, voxel_size=0.002):
    if points_xyz.shape[0] == 0 or voxel_size <= 0.0:
        return points_xyz
    grid = np.floor(points_xyz / float(voxel_size)).astype(np.int64)
    _, uniq_idx = np.unique(grid, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return points_xyz[uniq_idx]


def dedup_points_voxel_with_scores(points_xyz, scores, voxel_size=0.002):
    if points_xyz.shape[0] == 0 or voxel_size <= 0.0:
        return points_xyz, scores
    if scores.shape[0] != points_xyz.shape[0]:
        return points_xyz, scores

    grid = np.floor(points_xyz / float(voxel_size)).astype(np.int64)
    # Keep one representative per voxel, biased to larger score for heatmap saliency.
    order = np.lexsort(
        (
            np.arange(points_xyz.shape[0], dtype=np.int64),
            -scores.astype(np.float64),
            grid[:, 2],
            grid[:, 1],
            grid[:, 0],
        )
    )
    sorted_grid = grid[order]
    keep_mask = np.ones(order.shape[0], dtype=bool)
    keep_mask[1:] = np.any(sorted_grid[1:] != sorted_grid[:-1], axis=1)
    keep_idx = order[keep_mask]
    keep_idx = np.sort(keep_idx)
    return points_xyz[keep_idx], scores[keep_idx]


def draw_points_overlay(base_u8, points_uv, color, alpha=0.85, radius=1):
    out = base_u8.astype(np.float32) / 255.0
    h, w = out.shape[:2]
    if points_uv.shape[0] == 0:
        return (out * 255.0).round().astype(np.uint8)
    col = np.asarray(color, dtype=np.float32) / 255.0
    u = np.clip(np.rint(points_uv[:, 0]).astype(np.int32), 0, w - 1)
    v = np.clip(np.rint(points_uv[:, 1]).astype(np.int32), 0, h - 1)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu = np.clip(u + dx, 0, w - 1)
            vv = np.clip(v + dy, 0, h - 1)
            out[vv, uu, :] = out[vv, uu, :] * (1.0 - alpha) + col * alpha
    return (out * 255.0).round().astype(np.uint8)


def add_legend(img_u8, title, items):
    img = Image.fromarray(img_u8)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    pad = 8
    line_h = 16
    box_w = 10
    title_h = 16
    legend_h = title_h + line_h * len(items) + pad * 2
    legend_w = 300
    draw.rectangle([pad, pad, pad + legend_w, pad + legend_h], fill=(0, 0, 0))
    draw.text((pad + 8, pad + 4), title, fill=(255, 255, 255), font=font)
    y = pad + title_h + 2
    for label, color in items:
        draw.rectangle([pad + 8, y + 2, pad + 8 + box_w, y + 2 + box_w], fill=tuple(color))
        draw.text((pad + 24, y), label, fill=(255, 255, 255), font=font)
        y += line_h
    return np.asarray(img, dtype=np.uint8)


def blur5(arr):
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
    h, w = arr.shape
    pad_x = np.pad(arr, ((0, 0), (2, 2)), mode="edge")
    tmp = np.zeros_like(arr)
    for i, k in enumerate(kernel):
        tmp += k * pad_x[:, i : i + w]
    pad_y = np.pad(tmp, ((2, 2), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    for i, k in enumerate(kernel):
        out += k * pad_y[i : i + h, :]
    return out


def gaussian_blur(arr, sigma):
    sigma = float(sigma)
    if sigma <= 1.0e-6:
        return arr
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
    kernel = kernel / np.sum(kernel)

    h, w = arr.shape
    pad_x = np.pad(arr, ((0, 0), (radius, radius)), mode="edge")
    tmp = np.zeros_like(arr, dtype=np.float32)
    for i, k in enumerate(kernel):
        tmp += float(k) * pad_x[:, i : i + w]

    pad_y = np.pad(tmp, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    for i, k in enumerate(kernel):
        out += float(k) * pad_y[i : i + h, :]
    return out


def jet_colormap(x):
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def build_density_map(points_uv, h, w, weights=None):
    density = np.zeros((h, w), dtype=np.float32)
    if points_uv.shape[0] == 0:
        return density
    u = np.clip(np.rint(points_uv[:, 0]).astype(np.int32), 0, w - 1)
    v = np.clip(np.rint(points_uv[:, 1]).astype(np.int32), 0, h - 1)
    if weights is None:
        np.add.at(density, (v, u), 1.0)
    else:
        np.add.at(density, (v, u), weights.astype(np.float32))
    return density


def normalize_density(density, q=99.0):
    valid = density[density > 0]
    if valid.size == 0:
        return np.zeros_like(density, dtype=np.float32)
    scale = float(np.percentile(valid, q))
    if scale <= 1.0e-8:
        scale = float(np.max(valid)) + 1.0e-8
    return np.clip(density / scale, 0.0, 1.0)


def overlay_single_density(base_rgb, density_norm, color_rgb, alpha_max=0.8, gamma=0.7, min_alpha_nonzero=0.0):
    color = (np.asarray(color_rgb, dtype=np.float32) / 255.0).reshape(1, 1, 3)
    alpha = np.clip(np.power(density_norm, gamma) * float(alpha_max), 0.0, float(alpha_max))
    if min_alpha_nonzero > 0.0:
        active = density_norm > 0.0
        alpha = np.where(active, np.maximum(alpha, float(min_alpha_nonzero)), alpha)
    return base_rgb * (1.0 - alpha[..., None]) + color * alpha[..., None]


def render_mid_hard_heatmap(base_u8, points_uv, scores, sigma=4.0, percentile=95.0):
    h, w = base_u8.shape[:2]
    if points_uv.shape[0] == 0:
        return add_legend(
            base_u8.copy(),
            "Mid-hard Scored Heatmap",
            [("low score", (30, 80, 200)), ("high score", (220, 30, 30))],
        )

    weights = np.asarray(scores, dtype=np.float32).reshape(-1)
    if weights.shape[0] != points_uv.shape[0]:
        weights = np.ones((points_uv.shape[0],), dtype=np.float32)
    else:
        weights = np.maximum(weights, 0.0)

    if weights.size > 0 and float(np.max(weights)) > 0.0:
        weights = weights / (float(np.percentile(weights, 95.0)) + 1.0e-6)
        weights = np.clip(weights, 0.0, 1.0)
    else:
        weights = np.ones((points_uv.shape[0],), dtype=np.float32)
    heat = build_density_map(points_uv, h, w, weights=weights)
    heat = gaussian_blur(heat, sigma=float(sigma))
    heat_n = normalize_density(heat, q=float(percentile))
    heat_rgb = jet_colormap(heat_n)
    base = base_u8.astype(np.float32) / 255.0
    alpha = np.clip(np.power(heat_n, 0.65) * 0.85, 0.0, 0.85)
    out = base * (1.0 - alpha[..., None]) + heat_rgb * alpha[..., None]
    out_u8 = (out * 255.0).round().astype(np.uint8)
    return add_legend(out_u8, "Mid-hard Scored Heatmap", [("low score", (30, 80, 200)), ("high score", (220, 30, 30))])


def render_sparse_prior_density(base_u8, uv_reliable, uv_filtered, sigma=3.5, reliable_alpha=0.78, filtered_alpha=0.55):
    h, w = base_u8.shape[:2]
    den_rel = build_density_map(uv_reliable, h, w)
    den_fil = build_density_map(uv_filtered, h, w)
    den_rel = gaussian_blur(den_rel, sigma=float(sigma))
    den_fil = gaussian_blur(den_fil, sigma=float(sigma))
    den_rel_n = normalize_density(den_rel, q=99.0)
    den_fil_n = normalize_density(den_fil, q=99.0)

    out = base_u8.astype(np.float32) / 255.0
    out = overlay_single_density(out, den_fil_n, color_rgb=(255, 170, 50), alpha_max=float(filtered_alpha), gamma=0.70)
    out = overlay_single_density(out, den_rel_n, color_rgb=(80, 240, 255), alpha_max=float(reliable_alpha), gamma=0.65)
    out_u8 = (out * 255.0).round().astype(np.uint8)
    return add_legend(
        out_u8,
        "Projected Sparse Prior",
        [
            (f"reliable ({uv_reliable.shape[0]})", (80, 240, 255)),
            (f"filtered ({uv_filtered.shape[0]})", (255, 170, 50)),
        ],
    )


def render_topology_density(base_u8, uv_hole, uv_spawn, sigma=6.0, hole_alpha=0.72, spawn_alpha=0.90, percentile=90.0, min_alpha=0.06):
    h, w = base_u8.shape[:2]
    hole_den = gaussian_blur(build_density_map(uv_hole, h, w), sigma=float(sigma))
    spawn_den = gaussian_blur(build_density_map(uv_spawn, h, w), sigma=float(sigma))
    hole_n = normalize_density(hole_den, q=float(percentile))
    spawn_n = normalize_density(spawn_den, q=float(percentile))

    before = base_u8.astype(np.float32) / 255.0
    before = overlay_single_density(
        before,
        hole_n,
        color_rgb=(255, 60, 60),
        alpha_max=float(hole_alpha),
        gamma=0.65,
        min_alpha_nonzero=float(min_alpha),
    )
    before_u8 = (before * 255.0).round().astype(np.uint8)
    before_u8 = add_legend(before_u8, "Topology Before Event", [(f"coverage holes ({uv_hole.shape[0]})", (255, 60, 60))])

    after = base_u8.astype(np.float32) / 255.0
    after = overlay_single_density(
        after,
        hole_n,
        color_rgb=(255, 60, 60),
        alpha_max=float(hole_alpha) * 0.45,
        gamma=0.70,
        min_alpha_nonzero=float(min_alpha) * 0.8,
    )
    after = overlay_single_density(
        after,
        spawn_n,
        color_rgb=(80, 255, 80),
        alpha_max=float(spawn_alpha),
        gamma=0.60,
        min_alpha_nonzero=float(min_alpha),
    )
    after_u8 = (after * 255.0).round().astype(np.uint8)
    after_u8 = add_legend(
        after_u8,
        "Topology After Event",
        [
            (f"coverage holes ({uv_hole.shape[0]})", (255, 60, 60)),
            (f"spawned points ({uv_spawn.shape[0]})", (80, 255, 80)),
        ],
    )
    return before_u8, after_u8


def adapt_topology_render_params(base_sigma, base_percentile, base_min_alpha, total_projected_points):
    sigma = float(base_sigma)
    percentile = float(base_percentile)
    min_alpha = float(base_min_alpha)
    n = int(total_projected_points)
    if n <= 0:
        return sigma, percentile, min_alpha
    # When topology points are sparse, broaden support and lower normalization percentile
    # so the event regions remain visible instead of dot-like.
    target_dense = 2000.0
    if n < target_dense:
        sparsity = 1.0 - (float(n) / target_dense)
        sparsity = clamp(sparsity, 0.0, 1.0)
        sigma = sigma * (1.0 + 1.2 * sparsity)
        percentile = max(72.0, percentile - 18.0 * sparsity)
        min_alpha = min(0.18, min_alpha + 0.08 * sparsity)
    return sigma, percentile, min_alpha


def chw_u8_to_hwc(img_chw):
    arr = np.asarray(img_chw)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected CHW image, got shape {arr.shape}")
    if arr.shape[0] == 3:
        return np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    if arr.shape[-1] == 3:
        return arr.astype(np.uint8)
    raise RuntimeError(f"Cannot interpret image shape: {arr.shape}")


def save_png(path, img_u8):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(path)


def main():
    parser = argparse.ArgumentParser(description="Generate sparse-guidance visualization panels from visual_data payloads.")
    parser.add_argument("--visual-dir", required=True, help="Path to outputs/.../visual_data")
    parser.add_argument("--step", type=int, default=None, help="Step to render. Default: latest payload in visual_data.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: <visual-dir>/panels_step_xxxxxx")
    parser.add_argument("--max-reliable-points", type=int, default=30000)
    parser.add_argument("--max-filtered-points", type=int, default=30000)
    parser.add_argument("--max-sampled-points", type=int, default=50000)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--topology-point-radius", type=int, default=2)
    parser.add_argument("--sparse-sigma", type=float, default=3.5)
    parser.add_argument("--midhard-sigma", type=float, default=4.0)
    parser.add_argument("--topology-sigma", type=float, default=6.0)
    parser.add_argument("--midhard-percentile", type=float, default=95.0)
    parser.add_argument("--topology-percentile", type=float, default=90.0)
    parser.add_argument("--topology-min-alpha", type=float, default=0.06)
    parser.add_argument("--sparse-reliable-alpha", type=float, default=0.78)
    parser.add_argument("--sparse-filtered-alpha", type=float, default=0.55)
    parser.add_argument("--topology-hole-alpha", type=float, default=0.72)
    parser.add_argument("--topology-spawn-alpha", type=float, default=0.90)
    parser.add_argument(
        "--topology-aggregate-window",
        type=int,
        default=1,
        help="Number of visual_data snapshots (<= step) to aggregate for topology. Use 1 for current step only, <=0 for all history up to step.",
    )
    parser.add_argument(
        "--topology-auto-fallback-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If topology points are empty under current topology-aggregate-window, auto fallback to all history <= step.",
    )
    parser.add_argument("--topology-voxel-dedup", type=float, default=0.002, help="Voxel size for deduping aggregated topology points in world space.")
    parser.add_argument("--max-topology-points", type=int, default=120000, help="Max projected points kept per topology class after sampling.")
    parser.add_argument(
        "--topology-adaptive-density",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt topology density rendering params (sigma/percentile/min-alpha) when projected points are sparse.",
    )
    parser.add_argument(
        "--sampling-aggregate-window",
        type=int,
        default=1,
        help="Number of snapshots (<= step) to aggregate for sampled Gaussian heatmap. If current step has no sampled points, script auto-fallbacks to all history.",
    )
    parser.add_argument("--sampling-voxel-dedup", type=float, default=0.0, help="Voxel size for deduping aggregated sampled means in world space.")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    payload, payload_path = load_payload(args.visual_dir, step=args.step)
    step = int(payload.get("step", int(payload_path.stem.split("_")[-1])))
    visual_dir = Path(args.visual_dir)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = visual_dir / f"panels_step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, sparse_cfg, _, colmap_root = resolve_scene_paths(visual_dir)
    points_all, track_len_all, reproj_error_all = load_original_sparse_prior(colmap_root, sparse_cfg)
    reliable_mask = compute_reliable_mask(points_all, track_len_all, reproj_error_all, sparse_cfg)
    points_reliable = points_all[reliable_mask]
    points_filtered = points_all[~reliable_mask]

    camtoworld = np.asarray(payload["camera"]["camtoworld"], dtype=np.float32)
    intrinsics = np.asarray(payload["camera"]["intrinsics"], dtype=np.float32)
    base_render = chw_u8_to_hwc(payload["images"]["current_render_u8"])
    h, w = base_render.shape[:2]

    rng = np.random.default_rng(int(args.seed) + step)

    valid_r, u_r, v_r = project_points(points_reliable, camtoworld, intrinsics, w, h)
    uv_reliable = np.stack([u_r[valid_r], v_r[valid_r]], axis=1) if np.any(valid_r) else np.zeros((0, 2), dtype=np.float32)
    valid_f, u_f, v_f = project_points(points_filtered, camtoworld, intrinsics, w, h)
    uv_filtered = np.stack([u_f[valid_f], v_f[valid_f]], axis=1) if np.any(valid_f) else np.zeros((0, 2), dtype=np.float32)
    uv_reliable = sample_points(uv_reliable, int(args.max_reliable_points), rng)
    uv_filtered = sample_points(uv_filtered, int(args.max_filtered_points), rng)

    sparse_overlay = render_sparse_prior_density(
        base_render,
        uv_reliable,
        uv_filtered,
        sigma=float(args.sparse_sigma),
        reliable_alpha=float(args.sparse_reliable_alpha),
        filtered_alpha=float(args.sparse_filtered_alpha),
    )
    if int(args.point_radius) > 0:
        sparse_overlay = draw_points_overlay(
            sparse_overlay,
            uv_filtered,
            color=(255, 170, 50),
            alpha=0.6,
            radius=max(1, int(args.point_radius)),
        )
        sparse_overlay = draw_points_overlay(
            sparse_overlay,
            uv_reliable,
            color=(80, 240, 255),
            alpha=0.7,
            radius=max(1, int(args.point_radius)),
        )

    sampled_means = np.asarray(payload["sparse_sampling"]["sampled_means"], dtype=np.float32)
    if sampled_means.ndim != 2 or sampled_means.shape[1] != 3:
        sampled_means = np.zeros((0, 3), dtype=np.float32)
    sampled_scores = np.asarray(payload["sparse_sampling"]["sampled_mid_hard_score"], dtype=np.float32).reshape(-1)
    use_mid_hard_scores = (
        sampled_scores.shape[0] == sampled_means.shape[0]
        and sampled_scores.size > 0
        and float(np.max(sampled_scores)) > 1.0e-8
    )
    if not use_mid_hard_scores:
        sampled_scores = np.asarray(payload["sparse_sampling"]["sampled_difficulty"], dtype=np.float32).reshape(-1)
    if sampled_scores.shape[0] != sampled_means.shape[0]:
        sampled_scores = np.ones((sampled_means.shape[0],), dtype=np.float32)
    sampled_steps = [step]
    if sampled_means.shape[0] == 0:
        sampled_means, sampled_scores, sampled_steps = load_sampling_points_aggregated(
            visual_dir,
            target_step=step,
            aggregate_window=0,
        )
    elif int(args.sampling_aggregate_window) != 1:
        sampled_means, sampled_scores, sampled_steps = load_sampling_points_aggregated(
            visual_dir,
            target_step=step,
            aggregate_window=int(args.sampling_aggregate_window),
        )
    sampled_means, sampled_scores = dedup_points_voxel_with_scores(
        sampled_means,
        sampled_scores,
        voxel_size=float(args.sampling_voxel_dedup),
    )
    if sampled_scores.shape[0] != sampled_means.shape[0]:
        sampled_scores = np.ones((sampled_means.shape[0],), dtype=np.float32)
    if sampled_means.shape[0] > 0:
        sampled_keep = sample_points(np.arange(sampled_means.shape[0]), int(args.max_sampled_points), rng)
        sampled_means = sampled_means[sampled_keep]
        sampled_scores = sampled_scores[sampled_keep]
    valid_s, u_s, v_s = project_points(sampled_means, camtoworld, intrinsics, w, h)
    uv_sampled = np.stack([u_s[valid_s], v_s[valid_s]], axis=1) if np.any(valid_s) else np.zeros((0, 2), dtype=np.float32)
    score_sampled = sampled_scores[valid_s] if np.any(valid_s) else np.zeros((0,), dtype=np.float32)
    heatmap_img = render_mid_hard_heatmap(
        base_render,
        uv_sampled,
        score_sampled,
        sigma=float(args.midhard_sigma),
        percentile=float(args.midhard_percentile),
    )

    hole_points, spawn_points, topo_steps = load_topology_points_aggregated(
        visual_dir,
        target_step=step,
        aggregate_window=int(args.topology_aggregate_window),
    )
    topology_fallback_used = False
    if (
        bool(args.topology_auto_fallback_history)
        and int(args.topology_aggregate_window) == 1
        and int(hole_points.shape[0]) == 0
        and int(spawn_points.shape[0]) == 0
    ):
        hole_points, spawn_points, topo_steps = load_topology_points_aggregated(
            visual_dir,
            target_step=step,
            aggregate_window=0,
        )
        topology_fallback_used = True

    hole_points = dedup_points_voxel(hole_points, voxel_size=float(args.topology_voxel_dedup))
    spawn_points = dedup_points_voxel(spawn_points, voxel_size=float(args.topology_voxel_dedup))
    valid_h, u_h, v_h = project_points(hole_points, camtoworld, intrinsics, w, h)
    uv_hole = np.stack([u_h[valid_h], v_h[valid_h]], axis=1) if np.any(valid_h) else np.zeros((0, 2), dtype=np.float32)
    valid_p, u_p, v_p = project_points(spawn_points, camtoworld, intrinsics, w, h)
    uv_spawn = np.stack([u_p[valid_p], v_p[valid_p]], axis=1) if np.any(valid_p) else np.zeros((0, 2), dtype=np.float32)
    uv_hole = sample_points(uv_hole, int(args.max_topology_points), rng)
    uv_spawn = sample_points(uv_spawn, int(args.max_topology_points), rng)

    topology_sigma = float(args.topology_sigma)
    topology_percentile = float(args.topology_percentile)
    topology_min_alpha = float(args.topology_min_alpha)
    if bool(args.topology_adaptive_density):
        topology_sigma, topology_percentile, topology_min_alpha = adapt_topology_render_params(
            topology_sigma,
            topology_percentile,
            topology_min_alpha,
            total_projected_points=int(uv_hole.shape[0] + uv_spawn.shape[0]),
        )

    topo_before, topo_after = render_topology_density(
        base_render,
        uv_hole,
        uv_spawn,
        sigma=topology_sigma,
        hole_alpha=float(args.topology_hole_alpha),
        spawn_alpha=float(args.topology_spawn_alpha),
        percentile=topology_percentile,
        min_alpha=topology_min_alpha,
    )
    if int(args.topology_point_radius) > 0:
        topo_before = draw_points_overlay(
            topo_before,
            uv_hole,
            color=(255, 60, 60),
            alpha=0.8,
            radius=max(1, int(args.topology_point_radius)),
        )
        topo_after = draw_points_overlay(
            topo_after,
            uv_hole,
            color=(255, 60, 60),
            alpha=0.8,
            radius=max(1, int(args.topology_point_radius)),
        )
        topo_after = draw_points_overlay(
            topo_after,
            uv_spawn,
            color=(80, 255, 80),
            alpha=0.9,
            radius=max(1, int(args.topology_point_radius)),
        )

    save_png(out_dir / "00_initial_render.png", base_render)
    save_png(out_dir / "01_sparse_prior_projection.png", sparse_overlay)
    save_png(out_dir / "02_mid_hard_scored_heatmap.png", heatmap_img)
    save_png(out_dir / "03_topology_before.png", topo_before)
    save_png(out_dir / "04_topology_after.png", topo_after)

    print(f"[Done] payload: {payload_path}")
    print(f"[Done] output dir: {out_dir}")
    print(f"[Info] reliable projected: {uv_reliable.shape[0]}, filtered projected: {uv_filtered.shape[0]}")
    if int(args.topology_aggregate_window) <= 0:
        agg_desc = "all history <= step"
    else:
        agg_desc = f"last {int(args.topology_aggregate_window)} snapshots"
    if int(args.sampling_aggregate_window) <= 0:
        sampled_desc = "all history <= step"
    else:
        sampled_desc = f"last {int(args.sampling_aggregate_window)} snapshots"
    print(f"[Info] sampled aggregation: {sampled_desc}, steps_used={sampled_steps}")
    print(f"[Info] topology aggregation: {agg_desc}, steps_used={topo_steps}")
    if topology_fallback_used:
        print("[Info] topology aggregation fallback triggered: current step had no topology event, switched to all history <= step")
    print(f"[Info] sampled projected: {uv_sampled.shape[0]}, holes projected: {uv_hole.shape[0]}, spawned projected: {uv_spawn.shape[0]}")
    print(
        "[Info] topology render params: "
        f"sigma={topology_sigma:.2f}, percentile={topology_percentile:.1f}, min_alpha={topology_min_alpha:.3f}"
    )


if __name__ == "__main__":
    main()

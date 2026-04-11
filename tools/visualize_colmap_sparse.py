#!/usr/bin/env python

import argparse
import json
import math
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a transparent 3D COLMAP sparse-point backdrop with optional camera frusta."
    )
    parser.add_argument("--scene-root", type=Path, required=True, help="Scene root containing transforms_train.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument(
        "--style",
        type=str,
        default="paper",
        choices=("paper", "paper_gray", "color"),
        help="Rendering style. 'paper' keeps soft colors with local smoothing, 'paper_gray' uses a monochrome figure style.",
    )
    parser.add_argument("--camera-count", type=int, default=0, help="Number of camera frusta to render.")
    parser.add_argument(
        "--outlier-quantile",
        type=float,
        default=99.0,
        help="Keep sparse points within this radius quantile around the median center.",
    )
    parser.add_argument("--width", type=int, default=2400, help="Canvas width before alpha crop.")
    parser.add_argument("--height", type=int, default=1600, help="Canvas height before alpha crop.")
    parser.add_argument("--view-elev", type=float, default=None, help="Optional manual camera elevation.")
    parser.add_argument("--view-azim", type=float, default=None, help="Optional manual camera azimuth.")
    parser.add_argument("--view-roll", type=float, default=None, help="Optional manual camera roll.")
    return parser.parse_args()


def read_points3d_binary(points3d_path: Path):
    with points3d_path.open("rb") as handle:
        num_points = struct.unpack("<Q", handle.read(8))[0]
        xyz = np.empty((num_points, 3), dtype=np.float32)
        rgb = np.empty((num_points, 3), dtype=np.uint8)
        for idx in range(num_points):
            handle.read(8)  # point3d id
            xyz[idx] = np.asarray(struct.unpack("<ddd", handle.read(24)), dtype=np.float32)
            rgb[idx] = np.asarray(struct.unpack("<BBB", handle.read(3)), dtype=np.uint8)
            handle.read(8)  # reprojection error
            track_len = struct.unpack("<Q", handle.read(8))[0]
            handle.read(8 * track_len)  # image_id + point2d_idx pairs
    return xyz, rgb


def load_sparse_points_with_colors(scene_root: Path):
    sparse_root = scene_root / "auxiliaries" / "colmap_sparse"
    points_path = sparse_root / "points.npy"
    points3d_path = sparse_root / "sparse" / "points3D.bin"
    if not points_path.exists():
        raise FileNotFoundError(f"Missing sparse points: {points_path}")
    if not points3d_path.exists():
        raise FileNotFoundError(f"Missing COLMAP points3D.bin: {points3d_path}")

    sparse_points = np.load(points_path).astype(np.float32)
    if sparse_points.ndim != 2 or sparse_points.shape[1] != 3 or sparse_points.shape[0] == 0:
        raise RuntimeError(f"Expected non-empty [N, 3] sparse points from {points_path}, got {sparse_points.shape}")

    colmap_xyz, colmap_rgb = read_points3d_binary(points3d_path)
    color_lookup = {xyz.tobytes(): rgb for xyz, rgb in zip(colmap_xyz, colmap_rgb)}

    colors = np.empty((sparse_points.shape[0], 3), dtype=np.float32)
    missing = 0
    fallback_color = np.array([0.72, 0.64, 0.56], dtype=np.float32)
    for idx, xyz in enumerate(sparse_points):
        rgb = color_lookup.get(xyz.tobytes())
        if rgb is None:
            missing += 1
            colors[idx] = fallback_color
        else:
            colors[idx] = rgb.astype(np.float32) / 255.0

    if missing > 0:
        raise RuntimeError(
            f"Failed to match {missing} sparse-point colors exactly from {points3d_path}. "
            "The sparse prior and COLMAP model appear misaligned."
        )
    return sparse_points, colors


def load_camera_data(scene_root: Path):
    transforms_path = scene_root / "transforms_train.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms_train.json: {transforms_path}")

    with transforms_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    frames = meta.get("frames", [])
    if not frames:
        raise RuntimeError(f"No train frames found in {transforms_path}")

    fov_x = float(meta.get("camera_angle_x"))
    if "camera_angle_y" in meta:
        fov_y = float(meta["camera_angle_y"])
    elif "h" in meta and "w" in meta:
        aspect = float(meta["h"]) / max(float(meta["w"]), 1.0)
        fov_y = 2.0 * math.atan(math.tan(fov_x * 0.5) * aspect)
    else:
        raise RuntimeError("Unable to resolve vertical FOV from transforms_train.json")

    transforms = []
    centers = []
    for frame in frames:
        matrix = np.asarray(frame["transform_matrix"], dtype=np.float32)
        if matrix.shape == (3, 4):
            full = np.eye(4, dtype=np.float32)
            full[:3, :] = matrix
            matrix = full
        if matrix.shape != (4, 4):
            raise RuntimeError(f"Expected 4x4 transform matrix, got {matrix.shape}")
        transforms.append(matrix)
        centers.append(matrix[:3, 3])

    return np.stack(transforms), np.stack(centers), fov_x, fov_y


def filter_sparse_outliers(points, colors, quantile):
    quantile = float(np.clip(quantile, 50.0, 100.0))
    median_center = np.median(points, axis=0)
    radii = np.linalg.norm(points - median_center, axis=1)
    radius_threshold = np.percentile(radii, quantile)
    keep = radii <= radius_threshold
    return points[keep], colors[keep], keep, median_center, float(radius_threshold)


def build_pca_frame(points, camera_centers):
    center = points.mean(axis=0)
    centered_points = points - center
    _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
    frame = vh.astype(np.float32)

    camera_coords = (camera_centers - center) @ frame.T
    if camera_coords[:, 0].mean() < 0.0:
        frame[0] *= -1.0
    if camera_coords[:, 1].mean() > 0.0:
        frame[1] *= -1.0
    if np.linalg.det(frame) < 0.0:
        frame[2] *= -1.0
    return center, frame


def transform_points(points, center, frame):
    return (points - center) @ frame.T


def sample_camera_indices(num_cameras, camera_count):
    if num_cameras <= 0:
        return np.zeros((0,), dtype=np.int64)
    camera_count = int(max(0, min(camera_count, num_cameras)))
    if camera_count == 0:
        return np.zeros((0,), dtype=np.int64)
    indices = np.linspace(0, num_cameras - 1, num=camera_count)
    indices = np.round(indices).astype(np.int64)
    return np.unique(indices)


def soften_point_colors(colors):
    hsv = rgb_to_hsv(np.clip(colors, 0.0, 1.0))
    hsv[:, 1] = np.clip(hsv[:, 1] * 1.08 + 0.015, 0.0, 1.0)
    bright_mix = np.clip((hsv[:, 2] - 0.76) / 0.24, 0.0, 1.0)
    hsv[:, 2] = np.clip(hsv[:, 2] * (1.0 - 0.16 * bright_mix) + 0.01, 0.0, 1.0)
    rgb = hsv_to_rgb(hsv)
    rgb = np.clip(rgb * 0.97 + 0.03, 0.0, 1.0)
    return rgb


def soften_paper_colors(colors):
    hsv = rgb_to_hsv(np.clip(colors, 0.0, 1.0))
    hsv[:, 1] = np.clip(hsv[:, 1] * 0.72, 0.0, 1.0)
    hsv[:, 2] = np.clip(hsv[:, 2] * 0.92 + 0.06, 0.0, 1.0)
    return np.clip(hsv_to_rgb(hsv), 0.0, 1.0)


def aggregate_points_to_voxels(points, colors, voxel_divisor):
    mins = points.min(axis=0)
    extent = points.max(axis=0) - mins
    voxel_size = float(max(np.max(extent) / max(float(voxel_divisor), 1.0), 1.0e-6))
    voxel_indices = np.floor((points - mins) / voxel_size).astype(np.int32)
    unique_voxels, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)

    aggregated_points = np.zeros((unique_voxels.shape[0], 3), dtype=np.float32)
    aggregated_colors = np.zeros((unique_voxels.shape[0], 3), dtype=np.float32)
    np.add.at(aggregated_points, inverse, points)
    np.add.at(aggregated_colors, inverse, colors)
    aggregated_points /= counts[:, None]
    aggregated_colors /= counts[:, None]
    return aggregated_points, aggregated_colors, counts.astype(np.int32), unique_voxels, voxel_size


def count_voxel_neighbors(unique_voxels):
    if unique_voxels.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)

    voxel_lookup = {tuple(voxel): idx for idx, voxel in enumerate(unique_voxels)}
    neighbor_counts = np.zeros((unique_voxels.shape[0],), dtype=np.int32)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    for idx, voxel in enumerate(unique_voxels):
        x, y, z = voxel
        total = 0
        for dx, dy, dz in offsets:
            if (x + dx, y + dy, z + dz) in voxel_lookup:
                total += 1
        neighbor_counts[idx] = total
    return neighbor_counts


def smooth_colors_with_knn(points, colors, sigma, knn_k=10):
    num_points = int(points.shape[0])
    if num_points == 0:
        return colors
    sigma = float(max(sigma, 1.0e-6))
    knn_k = int(max(1, min(knn_k, num_points)))
    smoothed = np.zeros_like(colors, dtype=np.float32)
    chunk_size = 512
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        distances = np.linalg.norm(points[start:end, None, :] - points[None, :, :], axis=2)
        neighbor_indices = np.argpartition(distances, kth=knn_k - 1, axis=1)[:, :knn_k]
        neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
        weights = np.exp(-(neighbor_distances ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-6)
        smoothed[start:end] = (colors[neighbor_indices] * weights[..., None]).sum(axis=1)
    return smoothed


def build_paper_style_points(points, colors, use_monochrome=False):
    aggregated_points, aggregated_colors, voxel_counts, unique_voxels, voxel_size = aggregate_points_to_voxels(
        points,
        colors,
        voxel_divisor=82.0,
    )
    neighbor_counts = count_voxel_neighbors(unique_voxels)
    keep = (voxel_counts >= 2) | (neighbor_counts >= 4)
    aggregated_points = aggregated_points[keep]
    aggregated_colors = aggregated_colors[keep]
    voxel_counts = voxel_counts[keep]
    neighbor_counts = neighbor_counts[keep]

    if use_monochrome:
        luminance = 0.299 * aggregated_colors[:, 0] + 0.587 * aggregated_colors[:, 1] + 0.114 * aggregated_colors[:, 2]
        mono_value = 0.16 + 0.58 * (1.0 - luminance)
        paper_colors = np.stack(
            [
                np.clip(mono_value * 0.99, 0.0, 1.0),
                np.clip(mono_value * 0.99, 0.0, 1.0),
                np.clip(mono_value * 1.02, 0.0, 1.0),
            ],
            axis=1,
        )
    else:
        paper_colors = soften_paper_colors(aggregated_colors)
        paper_colors = smooth_colors_with_knn(
            aggregated_points,
            paper_colors,
            sigma=voxel_size * 2.2,
            knn_k=10,
        )
        paper_colors = soften_paper_colors(paper_colors)

    count_strength = np.log1p(voxel_counts.astype(np.float32))
    count_strength /= max(float(count_strength.max()), 1.0e-6)
    neighbor_strength = neighbor_counts.astype(np.float32)
    neighbor_strength /= max(float(neighbor_strength.max()), 1.0)
    density_strength = 0.55 * count_strength + 0.45 * neighbor_strength

    sizes = 0.9 + 1.7 * density_strength
    alphas = 0.18 + 0.58 * density_strength
    order = np.argsort(aggregated_points[:, 1])
    return {
        "points": aggregated_points[order],
        "colors": paper_colors[order],
        "sizes": sizes[order],
        "alphas": alphas[order],
        "halo_scale": 2.4,
        "halo_alpha_scale": 0.08,
        "camera_color": (0.53, 0.68, 0.84, 0.26),
        "camera_linewidth": 0.8,
        "view": (15.0, -68.0, 0.0),
        "pad_ratio": 0.03,
    }


def build_color_style_points(points, colors):
    softened_colors = soften_point_colors(colors)
    order = np.argsort(points[:, 1])
    num_points = points.shape[0]
    return {
        "points": points[order],
        "colors": softened_colors[order],
        "sizes": np.full((num_points,), 1.6, dtype=np.float32),
        "alphas": np.full((num_points,), 0.96, dtype=np.float32),
        "halo_scale": 5.6,
        "halo_alpha_scale": 0.045,
        "camera_color": (0.53, 0.68, 0.84, 0.46),
        "camera_linewidth": 0.9,
        "view": (19.0, -62.0, 0.0),
        "pad_ratio": 0.04,
    }


def build_render_style(style, points, colors):
    style = str(style).lower()
    if style == "paper":
        return build_paper_style_points(points, colors, use_monochrome=False)
    if style == "paper_gray":
        return build_paper_style_points(points, colors, use_monochrome=True)
    if style == "color":
        return build_color_style_points(points, colors)
    raise ValueError(f"Unsupported sparse visualization style: {style}")


def build_camera_edges(c2w, fov_x, fov_y, depth):
    half_w = depth * math.tan(fov_x * 0.5)
    half_h = depth * math.tan(fov_y * 0.5)
    corners_local = np.asarray(
        [
            [-half_w, -half_h, -depth],
            [half_w, -half_h, -depth],
            [half_w, half_h, -depth],
            [-half_w, half_h, -depth],
        ],
        dtype=np.float32,
    )
    camera_center = np.zeros((1, 3), dtype=np.float32)
    local_points = np.concatenate([camera_center, corners_local], axis=0)
    world_points = local_points @ c2w[:3, :3].T + c2w[:3, 3]
    edge_indices = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return [(world_points[start], world_points[end]) for start, end in edge_indices]


def configure_axes(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    mid = 0.5 * (mins + maxs)
    max_range = float(np.max(maxs - mins))
    half_extent = max_range * 0.58

    ax.set_xlim(mid[0] - half_extent, mid[0] + half_extent)
    ax.set_ylim(mid[1] - half_extent, mid[1] + half_extent)
    ax.set_zlim(mid[2] - half_extent, mid[2] + half_extent)
    ax.set_box_aspect((1.0, 1.0, 0.72))

    ax.set_axis_off()
    ax.grid(False)
    ax.set_proj_type("ortho")
    ax.patch.set_alpha(0.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_alpha(0.0)
        axis.pane.set_visible(False)
        axis.set_ticks([])


def alpha_crop(image_path: Path, pad_ratio=0.04):
    image = Image.open(image_path).convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise RuntimeError(f"Rendered image has empty alpha channel: {image_path}")

    left, upper, right, lower = bbox
    pad = int(round(max(image.size) * pad_ratio))
    left = max(0, left - pad)
    upper = max(0, upper - pad)
    right = min(image.size[0], right + pad)
    lower = min(image.size[1], lower + pad)
    image.crop((left, upper, right, lower)).save(image_path)


def render_sparse_backdrop(
    render_points,
    render_colors,
    render_sizes,
    render_alphas,
    camera_transforms,
    fov_x,
    fov_y,
    out_path: Path,
    width,
    height,
    halo_scale,
    halo_alpha_scale,
    camera_line_color,
    camera_linewidth,
    view_elev,
    view_azim,
    view_roll,
    pad_ratio,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(width / 200.0, height / 200.0), dpi=200, facecolor=(1.0, 1.0, 1.0, 0.0))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_alpha(0.0)

    configure_axes(ax, render_points)
    ax.view_init(elev=view_elev, azim=view_azim, roll=view_roll)

    halo_color = np.clip(render_colors * 0.92 + 0.08, 0.0, 1.0)
    ax.scatter(
        render_points[:, 0],
        render_points[:, 1],
        render_points[:, 2],
        s=render_sizes * halo_scale,
        c=halo_color,
        alpha=render_alphas * halo_alpha_scale,
        linewidths=0.0,
        depthshade=False,
    )
    ax.scatter(
        render_points[:, 0],
        render_points[:, 1],
        render_points[:, 2],
        s=render_sizes,
        c=render_colors,
        alpha=render_alphas,
        linewidths=0.0,
        depthshade=False,
    )

    scene_extent = float(np.max(render_points.max(axis=0) - render_points.min(axis=0)))
    frustum_depth = scene_extent * 0.065
    for c2w in camera_transforms:
        for start, end in build_camera_edges(c2w, fov_x, fov_y, frustum_depth):
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=camera_line_color,
                linewidth=camera_linewidth,
                solid_capstyle="round",
            )

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(out_path, transparent=True, dpi=200)
    plt.close(fig)
    alpha_crop(out_path, pad_ratio=pad_ratio)


def main():
    args = parse_args()
    scene_root = args.scene_root.resolve()
    points, colors = load_sparse_points_with_colors(scene_root)
    camera_transforms, camera_centers, fov_x, fov_y = load_camera_data(scene_root)

    filtered_points, filtered_colors, keep_mask, _, radius_threshold = filter_sparse_outliers(
        points,
        colors,
        quantile=args.outlier_quantile,
    )
    center, frame = build_pca_frame(filtered_points, camera_centers)

    transformed_points = transform_points(filtered_points, center, frame)
    render_style = build_render_style(args.style, transformed_points, filtered_colors)

    sampled_indices = sample_camera_indices(camera_transforms.shape[0], args.camera_count)
    sampled_transforms = camera_transforms[sampled_indices]
    transformed_camera_transforms = []
    for c2w in sampled_transforms:
        transformed = np.eye(4, dtype=np.float32)
        transformed[:3, :3] = frame @ c2w[:3, :3]
        transformed[:3, 3] = transform_points(c2w[:3, 3][None, :], center, frame)[0]
        transformed_camera_transforms.append(transformed)

    default_elev, default_azim, default_roll = render_style["view"]
    view_elev = default_elev if args.view_elev is None else float(args.view_elev)
    view_azim = default_azim if args.view_azim is None else float(args.view_azim)
    view_roll = default_roll if args.view_roll is None else float(args.view_roll)

    render_sparse_backdrop(
        render_points=render_style["points"],
        render_colors=render_style["colors"],
        render_sizes=render_style["sizes"],
        render_alphas=render_style["alphas"],
        camera_transforms=np.asarray(transformed_camera_transforms),
        fov_x=fov_x,
        fov_y=fov_y,
        out_path=args.out.resolve(),
        width=args.width,
        height=args.height,
        halo_scale=float(render_style["halo_scale"]),
        halo_alpha_scale=float(render_style["halo_alpha_scale"]),
        camera_line_color=render_style["camera_color"],
        camera_linewidth=float(render_style["camera_linewidth"]),
        view_elev=view_elev,
        view_azim=view_azim,
        view_roll=view_roll,
        pad_ratio=float(render_style["pad_ratio"]),
    )

    print(
        "[SparseVis] "
        f"scene={scene_root.name} points_total={points.shape[0]} points_kept={filtered_points.shape[0]} "
        f"outlier_quantile={args.outlier_quantile:.3f} radius_threshold={radius_threshold:.4f} "
        f"style={args.style} render_points={render_style['points'].shape[0]} "
        f"camera_count={len(sampled_indices)} out={args.out.resolve()}"
    )


if __name__ == "__main__":
    main()

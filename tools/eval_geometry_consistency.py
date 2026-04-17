#!/usr/bin/env python

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data import Blender
from core.losses.modules import (
    backproject_to_world,
    build_c2w_4x4,
    normalize_coords,
    project_world_to_target,
    sample_target_map,
)
from core.model import Simple3DGS


NORMAL_CONSISTENCY_ANGLE_DEG = 15.0
NORMAL_CONSISTENCY_KEY = f"normal_consistency_ratio@{int(NORMAL_CONSISTENCY_ANGLE_DEG)}"


class AttrDict:
    def __init__(self, data):
        self._data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                value = AttrDict(value)
            self._data[key] = value

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)

    def to_dict(self):
        output = {}
        for key, value in self._data.items():
            if isinstance(value, AttrDict):
                value = value.to_dict()
            output[key] = value
        return output


def resolve_config_path(ckpt_dir):
    local_config = os.path.join(ckpt_dir, "config.yaml")
    if os.path.exists(local_config):
        return local_config
    parent_config = os.path.join(os.path.dirname(ckpt_dir), "config.yaml")
    if os.path.exists(parent_config):
        return parent_config
    raise FileNotFoundError(f"No config.yaml found in '{ckpt_dir}' or its parent directory.")


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    try:
        return getattr(cfg, key)
    except AttributeError:
        return default


def squeeze_single_channel(tensor, name):
    if tensor is None:
        raise RuntimeError(f"{name} is None")
    if tensor.dim() == 2:
        return tensor
    if tensor.dim() == 3 and tensor.shape[-1] == 1:
        return tensor[..., 0]
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        return tensor[0]
    raise RuntimeError(f"Unsupported tensor shape for {name}: {tuple(tensor.shape)}")


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def infer_scene_name(meta_cfg):
    scene_name = _cfg_get(_cfg_get(meta_cfg, "DATASET", None), "NAME", None)
    if scene_name:
        return str(scene_name)
    data_path = _cfg_get(_cfg_get(meta_cfg, "DATASET", None), "DATA_PATH", "")
    if data_path:
        return Path(str(data_path)).name
    return "unknown"


def build_intrinsics(meta_cfg, dataset, device):
    data_info = dataset._data_info
    return torch.tensor(
        [
            [float(data_info["fl_x"]), 0.0, float(data_info["cx"])],
            [0.0, float(data_info["fl_y"]), float(data_info["cy"])],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )


def build_pixel_grid(height, width, device, dtype):
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return grid_x, grid_y


def backproject_depth_map_to_world(depth_map, camtoworld, intrinsics):
    height, width = depth_map.shape
    device = depth_map.device
    dtype = depth_map.dtype
    grid_x, grid_y = build_pixel_grid(height, width, device, dtype)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = (grid_x - cx) / fx * depth_map
    y = (grid_y - cy) / fy * depth_map
    cam_points_cv = torch.stack([x, y, depth_map], dim=-1)

    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype))
    c2w = build_c2w_4x4(camtoworld, device, dtype)
    return (cam_points_cv @ flip.T) @ c2w[:3, :3].T + c2w[:3, 3]


def compute_normal_map_from_depth(depth_map, alpha_map, camtoworld, intrinsics, min_alpha, eps):
    height, width = depth_map.shape
    device = depth_map.device
    dtype = depth_map.dtype
    normal_map = torch.zeros((height, width, 3), device=device, dtype=dtype)
    normal_valid = torch.zeros((height, width), device=device, dtype=torch.bool)
    if height < 3 or width < 3:
        return normal_map, normal_valid

    depth_valid = torch.isfinite(depth_map) & (depth_map > eps)
    alpha_valid = torch.isfinite(alpha_map) & (alpha_map > min_alpha)
    valid = depth_valid & alpha_valid
    if not valid.any():
        return normal_map, normal_valid

    world_points = backproject_depth_map_to_world(depth_map, camtoworld, intrinsics)
    world_points = torch.where(valid[..., None], world_points, torch.zeros_like(world_points))

    dx = world_points[1:-1, 2:] - world_points[1:-1, :-2]
    dy = world_points[2:, 1:-1] - world_points[:-2, 1:-1]
    normals = torch.cross(dx, dy, dim=-1)
    normal_norm = torch.linalg.norm(normals, dim=-1)

    inner_valid = valid[1:-1, 1:-1]
    inner_valid &= valid[1:-1, :-2]
    inner_valid &= valid[1:-1, 2:]
    inner_valid &= valid[:-2, 1:-1]
    inner_valid &= valid[2:, 1:-1]
    inner_valid &= torch.isfinite(normal_norm)
    inner_valid &= normal_norm > eps
    if not inner_valid.any():
        return normal_map, normal_valid

    normals = normals / normal_norm.unsqueeze(-1).clamp_min(eps)
    c2w = build_c2w_4x4(camtoworld, device, dtype)
    cam_center = c2w[:3, 3]
    view_dirs = cam_center.view(1, 1, 3) - world_points[1:-1, 1:-1]
    flip_mask = (normals * view_dirs).sum(dim=-1) < 0.0
    normals = torch.where(flip_mask[..., None], -normals, normals)
    normals = F.normalize(normals, dim=-1, eps=eps)
    normals = torch.where(inner_valid[..., None], normals, torch.zeros_like(normals))

    normal_map[1:-1, 1:-1] = normals
    normal_valid[1:-1, 1:-1] = inner_valid
    return normal_map, normal_valid


def sample_target_vector_map(target_map, coords_norm, mode="nearest"):
    sampled = F.grid_sample(
        target_map.permute(2, 0, 1).unsqueeze(0),
        coords_norm.view(1, 1, -1, 2),
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(target_map.shape[-1], -1).T


def sample_target_scalar_map(target_map, coords_norm, mode="nearest"):
    sampled = F.grid_sample(
        target_map[None, None],
        coords_norm.view(1, 1, -1, 2),
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(-1)


def load_checkpoint_state(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def build_pose_neighbors(dataset):
    frame_keys = list(dataset._records_keys)
    centers = torch.stack([dataset._records[key]["camera_center"] for key in frame_keys], dim=0)
    forwards = torch.stack([dataset._records[key]["camera_forward"] for key in frame_keys], dim=0)
    neighbors = {}
    for index, frame_key in enumerate(frame_keys):
        center = centers[index:index + 1]
        forward = forwards[index:index + 1]
        distances = torch.norm(centers - center, dim=1)
        alignment = torch.sum(forwards * forward, dim=1).clamp(-1.0, 1.0)
        scores = distances + 0.25 * (1.0 - alignment)
        scores[index] = float("inf")
        neighbor_index = int(torch.argmin(scores).item())
        neighbors[frame_key] = frame_keys[neighbor_index]
    return neighbors


@torch.no_grad()
def render_geometry_views(model, dataset, frame_keys, intrinsics, device, min_alpha, eps):
    img_h = int(dataset._data_info["img_h"])
    img_w = int(dataset._data_info["img_w"])
    rendered = {}
    ordered_keys = list(frame_keys)
    for frame_key in tqdm(ordered_keys, desc="Rendering geometry views"):
        record = dataset._records[frame_key]
        camtoworld = record["transform_matrix"].to(device)
        outputs = model(
            camtoworld,
            img_h,
            img_w,
            render_heads=(),
            render_geom_depth=True,
            render_rgb=False,
        )
        geom_depth = squeeze_single_channel(outputs["geom_depth"], f"geom_depth[{frame_key}]")
        alphas = squeeze_single_channel(outputs["alphas"], f"alphas[{frame_key}]")
        geom_normal_world, geom_normal_valid = compute_normal_map_from_depth(
            depth_map=geom_depth,
            alpha_map=alphas,
            camtoworld=camtoworld,
            intrinsics=intrinsics,
            min_alpha=min_alpha,
            eps=eps,
        )
        rendered[frame_key] = {
            "geom_depth": geom_depth,
            "alphas": alphas,
            "geom_normal_world": geom_normal_world,
            "geom_normal_valid": geom_normal_valid,
        }
    return rendered


def compute_directional_stats(
    source_depth,
    source_alpha,
    source_normal_world,
    source_normal_valid,
    source_camtoworld,
    target_depth,
    target_alpha,
    target_normal_world,
    target_normal_valid,
    target_camtoworld,
    intrinsics,
    sample_stride,
    min_alpha,
    relative_depth_thresh,
    absolute_depth_thresh,
    eps,
):
    total_samples = int(math.ceil(source_depth.shape[0] / sample_stride) * math.ceil(source_depth.shape[1] / sample_stride))
    world_points, source_u, source_v = backproject_to_world(
        depth_map=source_depth,
        camtoworld=source_camtoworld,
        intrinsics=intrinsics,
        stride=sample_stride,
        eps=eps,
        min_alpha=min_alpha,
        alpha_map=source_alpha,
    )
    source_valid_count = int(world_points.shape[0])
    stats = {
        "sample_count": total_samples,
        "source_valid_count": source_valid_count,
        "projected_count": 0,
        "overlap_count": 0,
        "consistent_count": 0,
        "source_valid_ratio": float(source_valid_count / max(total_samples, 1)),
        "projected_ratio": 0.0,
        "overlap_ratio": 0.0,
        "overlap_ratio_total": 0.0,
        "consistency_ratio": 0.0,
        "reproj_depth_error": None,
        "reproj_depth_error_median": None,
        "normal_overlap_count": 0,
        "reproj_normal_error": None,
        "reproj_normal_error_median": None,
        NORMAL_CONSISTENCY_KEY: 0.0,
    }
    aux_stats = {"normal_angle_errors": torch.empty((0,), dtype=source_depth.dtype)}
    if source_valid_count == 0:
        return stats, aux_stats

    target_c2w = build_c2w_4x4(target_camtoworld, source_depth.device, source_depth.dtype)
    u, v, z_proj = project_world_to_target(world_points, target_c2w, intrinsics, eps)
    width = int(target_depth.shape[1])
    height = int(target_depth.shape[0])

    projected_mask = torch.isfinite(z_proj) & (z_proj > eps)
    projected_mask &= (u >= 0.0) & (u <= float(width - 1))
    projected_mask &= (v >= 0.0) & (v <= float(height - 1))
    projected_count = int(projected_mask.sum().item())
    stats["projected_count"] = projected_count
    stats["projected_ratio"] = float(projected_count / max(source_valid_count, 1))
    if projected_count == 0:
        return stats, aux_stats

    u = u[projected_mask]
    v = v[projected_mask]
    z_proj = z_proj[projected_mask]
    source_u = source_u[projected_mask]
    source_v = source_v[projected_mask]

    coords_norm = normalize_coords(u, v, width, height)
    sampled_target_depth = sample_target_map(target_depth, coords_norm)
    sampled_target_alpha = sample_target_map(target_alpha, coords_norm)

    overlap_mask = torch.isfinite(sampled_target_depth) & (sampled_target_depth > eps)
    overlap_mask &= sampled_target_alpha > min_alpha
    overlap_count = int(overlap_mask.sum().item())
    stats["overlap_count"] = overlap_count
    stats["overlap_ratio"] = float(overlap_count / max(projected_count, 1))
    stats["overlap_ratio_total"] = float(overlap_count / max(total_samples, 1))
    if overlap_count == 0:
        return stats, aux_stats

    z_proj = z_proj[overlap_mask]
    sampled_target_depth = sampled_target_depth[overlap_mask]
    coords_norm = coords_norm[overlap_mask]
    source_u = source_u[overlap_mask]
    source_v = source_v[overlap_mask]
    depth_delta = torch.abs(z_proj - sampled_target_depth)
    relative_depth_error = depth_delta / sampled_target_depth.abs().clamp_min(eps)
    consistent = depth_delta <= (
        absolute_depth_thresh + relative_depth_thresh * sampled_target_depth.abs().clamp_min(eps)
    )
    consistent_count = int(consistent.sum().item())

    stats["consistent_count"] = consistent_count
    stats["consistency_ratio"] = float(consistent_count / max(overlap_count, 1))
    stats["reproj_depth_error"] = float(relative_depth_error.mean().item())
    stats["reproj_depth_error_median"] = float(relative_depth_error.median().item())
    
    source_px = source_u.long()
    source_py = source_v.long()
    source_normals = source_normal_world[source_py, source_px]
    source_normal_mask = source_normal_valid[source_py, source_px]
    sampled_target_normals = sample_target_vector_map(target_normal_world, coords_norm, mode="nearest")
    sampled_target_normal_mask = sample_target_scalar_map(target_normal_valid.float(), coords_norm, mode="nearest") > 0.5

    source_normal_norm = torch.linalg.norm(source_normals, dim=-1)
    target_normal_norm = torch.linalg.norm(sampled_target_normals, dim=-1)
    normal_overlap_mask = source_normal_mask & sampled_target_normal_mask
    normal_overlap_mask &= torch.isfinite(source_normal_norm) & (source_normal_norm > eps)
    normal_overlap_mask &= torch.isfinite(target_normal_norm) & (target_normal_norm > eps)

    normal_overlap_count = int(normal_overlap_mask.sum().item())
    stats["normal_overlap_count"] = normal_overlap_count
    if normal_overlap_count > 0:
        source_normals = F.normalize(source_normals[normal_overlap_mask], dim=-1, eps=eps)
        sampled_target_normals = F.normalize(sampled_target_normals[normal_overlap_mask], dim=-1, eps=eps)
        cosine = (source_normals * sampled_target_normals).sum(dim=-1).clamp(-1.0, 1.0)
        normal_angle_errors = torch.rad2deg(torch.acos(cosine))
        stats["reproj_normal_error"] = float(normal_angle_errors.mean().item())
        stats["reproj_normal_error_median"] = float(normal_angle_errors.median().item())
        stats[NORMAL_CONSISTENCY_KEY] = float((normal_angle_errors <= NORMAL_CONSISTENCY_ANGLE_DEG).float().mean().item())
        aux_stats["normal_angle_errors"] = normal_angle_errors.detach().cpu()
    return stats, aux_stats


def aggregate_pair_stats(pair_stats, pair_aux_stats=None):
    total_sample_count = sum(stat["sample_count"] for stat in pair_stats.values())
    total_source_valid_count = sum(stat["source_valid_count"] for stat in pair_stats.values())
    total_projected_count = sum(stat["projected_count"] for stat in pair_stats.values())
    total_overlap_count = sum(stat["overlap_count"] for stat in pair_stats.values())
    total_consistent_count = sum(stat["consistent_count"] for stat in pair_stats.values())

    weighted_error_num = 0.0
    weighted_error_den = 0
    unweighted_errors = []
    for stat in pair_stats.values():
        if stat["reproj_depth_error"] is None or stat["overlap_count"] <= 0:
            continue
        weighted_error_num += float(stat["reproj_depth_error"]) * int(stat["overlap_count"])
        weighted_error_den += int(stat["overlap_count"])
        unweighted_errors.append(float(stat["reproj_depth_error"]))

    normal_error_tensors = []
    if pair_aux_stats is not None:
        for aux in pair_aux_stats.values():
            normal_errors = aux.get("normal_angle_errors", None)
            if normal_errors is None or int(normal_errors.numel()) == 0:
                continue
            normal_error_tensors.append(normal_errors.reshape(-1).to(dtype=torch.float32))

    if normal_error_tensors:
        all_normal_errors = torch.cat(normal_error_tensors, dim=0)
        reproj_normal_error = float(all_normal_errors.mean().item())
        reproj_normal_error_median = float(all_normal_errors.median().item())
        normal_consistency_ratio = float((all_normal_errors <= NORMAL_CONSISTENCY_ANGLE_DEG).float().mean().item())
        normal_overlap_count = int(all_normal_errors.numel())
    else:
        reproj_normal_error = None
        reproj_normal_error_median = None
        normal_consistency_ratio = 0.0
        normal_overlap_count = 0

    return {
        "num_pairs": len(pair_stats),
        "valid_pairs": int(sum(1 for stat in pair_stats.values() if stat["overlap_count"] > 0)),
        "sample_count": int(total_sample_count),
        "source_valid_count": int(total_source_valid_count),
        "projected_count": int(total_projected_count),
        "overlap_count": int(total_overlap_count),
        "consistent_count": int(total_consistent_count),
        "source_valid_ratio": float(total_source_valid_count / max(total_sample_count, 1)),
        "projected_ratio": float(total_projected_count / max(total_source_valid_count, 1)),
        "overlap_ratio": float(total_overlap_count / max(total_projected_count, 1)),
        "overlap_ratio_total": float(total_overlap_count / max(total_sample_count, 1)),
        "consistency_ratio": float(total_consistent_count / max(total_overlap_count, 1)),
        "reproj_depth_error": None if weighted_error_den == 0 else float(weighted_error_num / weighted_error_den),
        "reproj_depth_error_unweighted": None if not unweighted_errors else float(sum(unweighted_errors) / len(unweighted_errors)),
        "normal_overlap_count": int(normal_overlap_count),
        "reproj_normal_error": reproj_normal_error,
        "reproj_normal_error_median": reproj_normal_error_median,
        NORMAL_CONSISTENCY_KEY: normal_consistency_ratio,
    }


def build_summary_lines(summary, summary_path, per_view_path, log_path):
    return [
        f"Scene: {summary['scene']}",
        f"Split: {summary['split']}",
        f"Pairs: {summary['valid_pairs']}/{summary['num_pairs']}",
        f"Reproj. Depth Error: {summary['reproj_depth_error']}",
        f"Consistency Ratio: {summary['consistency_ratio']}",
        f"Overlap Ratio: {summary['overlap_ratio']}",
        f"Reproj. Normal Error: {summary['reproj_normal_error']}",
        f"Reproj. Normal Error Median: {summary['reproj_normal_error_median']}",
        f"Normal Consistency Ratio@15: {summary[NORMAL_CONSISTENCY_KEY]}",
        f"Summary JSON: {summary_path}",
        f"Per-view JSON: {per_view_path}",
        f"Text Log: {log_path}",
    ]


def save_outputs(summary_path, per_view_path, log_path, summary, per_view):
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(per_view_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(per_view_path, "w", encoding="utf-8") as handle:
        json.dump(per_view, handle, indent=2)
    summary_lines = build_summary_lines(summary, summary_path, per_view_path, log_path)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n\n")
        handle.write("Summary Payload:\n")
        handle.write(json.dumps(summary, indent=2))
        handle.write("\n")
    print(f"Summary written to {summary_path}")
    print(f"Per-view metrics written to {per_view_path}")
    print(f"Text log written to {log_path}")


@torch.no_grad()
def evaluate_geometry_consistency(args):
    checkpoint_path = os.path.abspath(args.checkpoint)
    ckpt_dir = os.path.dirname(checkpoint_path)
    config_path = resolve_config_path(ckpt_dir)

    with open(config_path, "r", encoding="utf-8") as handle:
        config_dict = yaml.load(handle, Loader=yaml.Loader)
    config_dict["EXP_STR"] = ""
    config_dict["TIME_STR"] = ""
    meta_cfg = AttrDict(config_dict)

    dataset = Blender(meta_cfg.DATASET, split=args.split, load_images=False)
    device = torch.device(args.device)
    model = Simple3DGS(meta_cfg.MODEL, dataset._data_info).to(device)
    ckpt = load_checkpoint_state(checkpoint_path, device)
    for key, value in ckpt.items():
        model.splats[key] = torch.nn.Parameter(value)
    model.sh_degree = model.sh_degree_max
    model.eval()

    all_frame_keys = list(dataset._records_keys)
    source_keys = all_frame_keys[: args.max_views] if args.max_views is not None else all_frame_keys
    neighbors = build_pose_neighbors(dataset)
    intrinsics = build_intrinsics(meta_cfg, dataset, device)
    render_keys = set(source_keys)
    for frame_key in source_keys:
        render_keys.add(neighbors[frame_key])
    render_cache = render_geometry_views(
        model,
        dataset,
        sorted(render_keys),
        intrinsics,
        device,
        args.min_alpha,
        args.eps,
    )

    pair_stats = {}
    pair_aux_stats = {}
    for frame_key in tqdm(source_keys, desc="Computing geometry consistency"):
        neighbor_key = neighbors[frame_key]
        source_record = dataset._records[frame_key]
        target_record = dataset._records[neighbor_key]
        source_render = render_cache[frame_key]
        target_render = render_cache[neighbor_key]
        stats, aux_stats = compute_directional_stats(
            source_depth=source_render["geom_depth"],
            source_alpha=source_render["alphas"],
            source_normal_world=source_render["geom_normal_world"],
            source_normal_valid=source_render["geom_normal_valid"],
            source_camtoworld=source_record["transform_matrix"].to(device),
            target_depth=target_render["geom_depth"],
            target_alpha=target_render["alphas"],
            target_normal_world=target_render["geom_normal_world"],
            target_normal_valid=target_render["geom_normal_valid"],
            target_camtoworld=target_record["transform_matrix"].to(device),
            intrinsics=intrinsics,
            sample_stride=args.sample_stride,
            min_alpha=args.min_alpha,
            relative_depth_thresh=args.relative_depth_thresh,
            absolute_depth_thresh=args.absolute_depth_thresh,
            eps=args.eps,
        )
        stats["neighbor_frame_key"] = neighbor_key
        pair_stats[frame_key] = stats
        pair_aux_stats[frame_key] = aux_stats

    summary = aggregate_pair_stats(pair_stats, pair_aux_stats)
    summary.update(
        {
            "tag": args.tag,
            "checkpoint": checkpoint_path,
            "scene": infer_scene_name(meta_cfg),
            "split": args.split,
            "sample_stride": int(args.sample_stride),
            "min_alpha": float(args.min_alpha),
            "relative_depth_thresh": float(args.relative_depth_thresh),
            "absolute_depth_thresh": float(args.absolute_depth_thresh),
            "eps": float(args.eps),
        }
    )

    summary_path = (
        os.path.abspath(args.summary_json)
        if args.summary_json
        else os.path.join(ckpt_dir, f"geometry_consistency_{args.split}.json")
    )
    per_view_path = (
        os.path.abspath(args.per_view_json)
        if args.per_view_json
        else os.path.join(ckpt_dir, f"geometry_consistency_{args.split}_per_view.json")
    )
    log_path = (
        os.path.abspath(args.log_txt)
        if args.log_txt
        else os.path.join(ckpt_dir, f"geometry_consistency_{args.split}.txt")
    )
    save_outputs(summary_path, per_view_path, log_path, summary, pair_stats)

    for line in build_summary_lines(summary, summary_path, per_view_path, log_path)[:9]:
        print(line)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multi-view geometry consistency from rendered geom_depth maps.")
    parser.add_argument("--checkpoint", "-w", required=True, type=str, help="Path to a .pt checkpoint.")
    parser.add_argument("--device", type=str, default=default_device(), help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--split", type=str, default="test", choices=("train", "test", "val"), help="Dataset split used for view pairs.")
    parser.add_argument("--sample-stride", type=int, default=4, help="Pixel sampling stride for source geom_depth.")
    parser.add_argument("--min-alpha", type=float, default=0.2, help="Minimum alpha threshold for valid geometry pixels.")
    parser.add_argument("--relative-depth-thresh", type=float, default=0.05, help="Relative depth consistency threshold.")
    parser.add_argument("--absolute-depth-thresh", type=float, default=0.02, help="Absolute depth consistency threshold.")
    parser.add_argument("--eps", type=float, default=1.0e-4, help="Numerical epsilon.")
    parser.add_argument("--max-views", type=int, default=None, help="Optional limit on the number of source views, for smoke tests.")
    parser.add_argument("--tag", type=str, default="", help="Optional label stored in the summary JSON.")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional summary JSON output path.")
    parser.add_argument("--per-view-json", type=str, default=None, help="Optional per-view JSON output path.")
    parser.add_argument("--log-txt", type=str, default=None, help="Optional plain-text summary log output path.")
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    evaluate_geometry_consistency(parse_args())

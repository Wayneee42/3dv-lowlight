#!/usr/bin/env python

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import torch
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
def render_geometry_views(model, dataset, frame_keys, device):
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
        rendered[frame_key] = {
            "geom_depth": squeeze_single_channel(outputs["geom_depth"], f"geom_depth[{frame_key}]"),
            "alphas": squeeze_single_channel(outputs["alphas"], f"alphas[{frame_key}]"),
        }
    return rendered


def compute_directional_stats(
    source_depth,
    source_alpha,
    source_camtoworld,
    target_depth,
    target_alpha,
    target_camtoworld,
    intrinsics,
    sample_stride,
    min_alpha,
    relative_depth_thresh,
    absolute_depth_thresh,
    eps,
):
    total_samples = int(math.ceil(source_depth.shape[0] / sample_stride) * math.ceil(source_depth.shape[1] / sample_stride))
    world_points, _, _ = backproject_to_world(
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
    }
    if source_valid_count == 0:
        return stats

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
        return stats

    u = u[projected_mask]
    v = v[projected_mask]
    z_proj = z_proj[projected_mask]

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
        return stats

    z_proj = z_proj[overlap_mask]
    sampled_target_depth = sampled_target_depth[overlap_mask]
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
    return stats


def aggregate_pair_stats(pair_stats):
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
    }


def save_outputs(summary_path, per_view_path, summary, per_view):
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(per_view_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(per_view_path, "w", encoding="utf-8") as handle:
        json.dump(per_view, handle, indent=2)
    print(f"Summary written to {summary_path}")
    print(f"Per-view metrics written to {per_view_path}")


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
    render_keys = set(source_keys)
    for frame_key in source_keys:
        render_keys.add(neighbors[frame_key])
    render_cache = render_geometry_views(model, dataset, sorted(render_keys), device)
    intrinsics = build_intrinsics(meta_cfg, dataset, device)

    pair_stats = {}
    for frame_key in tqdm(source_keys, desc="Computing geometry consistency"):
        neighbor_key = neighbors[frame_key]
        source_record = dataset._records[frame_key]
        target_record = dataset._records[neighbor_key]
        source_render = render_cache[frame_key]
        target_render = render_cache[neighbor_key]
        stats = compute_directional_stats(
            source_depth=source_render["geom_depth"],
            source_alpha=source_render["alphas"],
            source_camtoworld=source_record["transform_matrix"].to(device),
            target_depth=target_render["geom_depth"],
            target_alpha=target_render["alphas"],
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

    summary = aggregate_pair_stats(pair_stats)
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
    save_outputs(summary_path, per_view_path, summary, pair_stats)

    print(f"Scene: {summary['scene']}")
    print(f"Split: {summary['split']}")
    print(f"Pairs: {summary['valid_pairs']}/{summary['num_pairs']}")
    print(f"Reproj. Depth Error: {summary['reproj_depth_error']}")
    print(f"Consistency Ratio: {summary['consistency_ratio']}")
    print(f"Overlap Ratio: {summary['overlap_ratio']}")
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
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    evaluate_geometry_consistency(parse_args())

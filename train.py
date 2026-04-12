#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import argparse
import json
import math
import os
import random
import warnings
from pathlib import Path

import gsplat
import numpy as np
import torch
import yaml
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from core.data import Blender
from core.libs import ConfigDict
from core.libs.augment import prepare_low_light_batch
from core.losses import (
    build_loss_modules,
    compute_loss_modules,
    required_aux_heads,
)
from core.model import Simple3DGS



def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    try:
        return getattr(cfg, key)
    except AttributeError:
        return default



def infer_stage_name(config_path, meta_cfg):
    experiment_cfg = _cfg_get(meta_cfg, "EXPERIMENT", None)
    explicit_stage = _cfg_get(experiment_cfg, "STAGE", None)
    if explicit_stage is not None:
        return str(explicit_stage)

    config_parts = Path(config_path).parts
    if "config" in config_parts:
        config_index = config_parts.index("config")
        if config_index + 1 < len(config_parts) - 1:
            return config_parts[config_index + 1]
    return "manual"



def build_output_dir(config_path, meta_cfg):
    stage_name = infer_stage_name(config_path, meta_cfg)
    scene_name = str(meta_cfg.DATASET.NAME)
    return os.path.join("outputs", stage_name, scene_name)


def should_write_sparse_diagnostics(config_path, sparse_cfg):
    config_stage_dir = Path(config_path).resolve().parent.name.lower()
    return config_stage_dir in {"stage5b_ft_on", "stage5b_ft_v1", "stage5b_ft_v2", "stage5c_sparse_topology", "stage5c_vsurface"} and bool(_cfg_get(sparse_cfg, "ENABLED", False))


def append_json_txt_record(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def grad_norm_value(grad_tensor):
    if grad_tensor is None:
        return 0.0
    return float(grad_tensor.detach().norm().item())


def summarize_sparse_records(records, sparse_load_info):
    if not records:
        return None
    mean_grad_ratio = sum(record["sparse_grad_ratio_means"] for record in records) / len(records)
    mean_sparse_loss = sum(record["sparse_loss"] for record in records) / len(records)
    mean_distance = sum(record["sparse_distance_mean"] for record in records) / len(records)
    mean_active_ratio = sum(record["sparse_active_ratio"] for record in records) / len(records)
    mean_effective_weight = sum(record["effective_sparse_weight_after_schedule"] for record in records) / len(records)
    return {
        "record_type": "summary",
        "num_snapshots": len(records),
        "first_step": int(records[0]["step"]),
        "last_step": int(records[-1]["step"]),
        "meta_exists": int(bool(sparse_load_info.get("meta_exists", False))),
        "meta_used": int(bool(sparse_load_info.get("meta_used", False))),
        "mean_sparse_loss": float(mean_sparse_loss),
        "mean_sparse_grad_ratio_means": float(mean_grad_ratio),
        "max_sparse_grad_ratio_means": float(max(record["sparse_grad_ratio_means"] for record in records)),
        "mean_sparse_distance_mean": float(mean_distance),
        "final_sparse_distance_mean": float(records[-1]["sparse_distance_mean"]),
        "mean_sparse_active_ratio": float(mean_active_ratio),
        "final_sparse_active_ratio": float(records[-1]["sparse_active_ratio"]),
        "mean_effective_sparse_weight_after_schedule": float(mean_effective_weight),
        "filter_kept_ratio": float(sparse_load_info.get("filter_kept_ratio", 1.0)),
        "mean_point_to_plane_fallback_ratio": float(
            sum(record.get("point_to_plane_fallback_ratio", 0.0) for record in records) / len(records)
        ),
    }


def summarize_topology_records(records, final_num_gaussians):
    if not records:
        return None
    event_count = len(records)
    total_spawn = sum(int(record.get("spawn_count", 0)) for record in records)
    total_prune = sum(int(record.get("prune_count", 0)) for record in records)
    total_prune_decayed = sum(int(record.get("prune_decayed_count", 0)) for record in records)
    mean_hole_ratio = sum(float(record.get("coverage_hole_ratio", 0.0)) for record in records) / event_count
    return {
        "record_type": "topology_summary",
        "num_events": int(event_count),
        "total_spawn_count": int(total_spawn),
        "total_prune_count": int(total_prune),
        "total_prune_decayed_count": int(total_prune_decayed),
        "mean_spawn_count": float(total_spawn / max(1, event_count)),
        "mean_prune_count": float(total_prune / max(1, event_count)),
        "mean_coverage_hole_ratio": float(mean_hole_ratio),
        "mean_prune_sparse_core_ratio": float(sum(float(record.get("prune_sparse_core_ratio", 0.0)) for record in records) / event_count),
        "final_num_gaussians": int(final_num_gaussians),
    }


def rgb_chw_to_gray(image_tensor):
    if image_tensor is None:
        return None
    return 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]


def is_multiview_active(loss_modules, step):
    context = {"step": int(step)}
    return any(module.name == "multiview_reproj" and module.is_active(context) for module in loss_modules)



def build_step_dir(output_dir, step):
    return os.path.join(output_dir, f"step_{int(step)}")



def save_config(path, meta_cfg):
    with open(path, "w") as handle:
        yaml.dump(dict(meta_cfg), handle, default_flow_style=False)



def resolve_checkpoint_steps(cfg):
    checkpoint_steps = _cfg_get(cfg, "CHECKPOINT_STEPS", None)
    if checkpoint_steps is None:
        checkpoint_steps = [7000, cfg.TRAIN_TOTAL_STEP]
    resolved = sorted({int(step) for step in checkpoint_steps if 0 < int(step) <= int(cfg.TRAIN_TOTAL_STEP)})
    if int(cfg.TRAIN_TOTAL_STEP) not in resolved:
        resolved.append(int(cfg.TRAIN_TOTAL_STEP))
    return resolved


def _project_sparse_points_to_pixels(points_world, transform_matrix, data_info):
    if points_world.shape[0] == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    width = int(data_info["img_w"])
    height = int(data_info["img_h"])
    fx = float(data_info["fl_x"])
    fy = float(data_info["fl_y"])
    cx = float(data_info["cx"])
    cy = float(data_info["cy"])

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :] = transform_matrix.detach().cpu().numpy().astype(np.float32)
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


def _robust_normalize_numpy(values, high_q=95.0):
    values = np.asarray(values, dtype=np.float32)
    normalized = np.zeros_like(values, dtype=np.float32)
    valid = np.isfinite(values)
    if not np.any(valid):
        return normalized
    positive = values[valid]
    scale = float(np.percentile(positive, high_q))
    if not np.isfinite(scale) or scale <= 1.0e-6:
        normalized[valid] = np.clip(positive, 0.0, 1.0)
        return normalized
    normalized[valid] = np.clip(positive / scale, 0.0, 1.0)
    return normalized


def build_sparse_lowlight_scores(train_dataset, sparse_points, sparse_cfg):
    num_points = int(sparse_points.shape[0])
    brightness_scores = np.ones((num_points,), dtype=np.float32)
    gradient_scores = np.ones((num_points,), dtype=np.float32)
    visibility_counts = np.zeros((num_points,), dtype=np.int32)

    brightness_enabled = bool(_cfg_get(sparse_cfg, "LOWLIGHT_BRIGHTNESS_ENABLED", False))
    gradient_enabled = bool(_cfg_get(sparse_cfg, "LOWLIGHT_GRADIENT_ENABLED", False))
    if num_points == 0 or (not brightness_enabled and not gradient_enabled):
        return brightness_scores, gradient_scores, visibility_counts

    brightness_sigma = float(max(_cfg_get(sparse_cfg, "BRIGHTNESS_SIGMA", 0.20), 1.0e-6))
    gradient_sigma = float(max(_cfg_get(sparse_cfg, "GRADIENT_SIGMA", 0.10), 1.0e-6))
    brightness_sum = np.zeros((num_points,), dtype=np.float32)
    gradient_sum = np.zeros((num_points,), dtype=np.float32)

    for frame_key in train_dataset._records_keys:
        record = train_dataset._records[frame_key]
        source_tensor = record["low_light_tensor"] if record.get("low_light_tensor", None) is not None else record.get("img_tensor", None)
        if source_tensor is None:
            continue

        image = source_tensor[:3].detach().cpu().numpy().astype(np.float32)
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        gray = _robust_normalize_numpy(gray)
        grad_y, grad_x = np.gradient(gray)
        grad_mag = _robust_normalize_numpy(np.abs(grad_x) + np.abs(grad_y))

        valid, u, v = _project_sparse_points_to_pixels(sparse_points, record["transform_matrix"], train_dataset._data_info)
        visible_indices = np.nonzero(valid)[0]
        if visible_indices.shape[0] == 0:
            continue

        u_idx = np.clip(np.rint(u[visible_indices]).astype(np.int64), 0, gray.shape[1] - 1)
        v_idx = np.clip(np.rint(v[visible_indices]).astype(np.int64), 0, gray.shape[0] - 1)
        visibility_counts[visible_indices] += 1
        if brightness_enabled:
            brightness_sum[visible_indices] += gray[v_idx, u_idx]
        if gradient_enabled:
            gradient_sum[visible_indices] += grad_mag[v_idx, u_idx]

    visible_mask = visibility_counts > 0
    if brightness_enabled:
        brightness_mean = np.ones((num_points,), dtype=np.float32)
        brightness_mean[visible_mask] = brightness_sum[visible_mask] / visibility_counts[visible_mask].astype(np.float32)
        brightness_scores = brightness_mean / (brightness_mean + brightness_sigma)
    if gradient_enabled:
        gradient_mean = np.ones((num_points,), dtype=np.float32)
        gradient_mean[visible_mask] = gradient_sum[visible_mask] / visibility_counts[visible_mask].astype(np.float32)
        gradient_scores = gradient_mean / (gradient_mean + gradient_sigma)

    return brightness_scores.astype(np.float32), gradient_scores.astype(np.float32), visibility_counts


def filter_sparse_prior(points, track_len, reproj_error, brightness_scores, gradient_scores, sparse_cfg):
    num_points = int(points.shape[0])
    info = {
        "filter_enabled": int(bool(_cfg_get(sparse_cfg, "RELIABILITY_FILTER_ENABLED", False))),
        "filter_mode": "disabled",
        "filter_kept_ratio": 1.0 if num_points > 0 else 0.0,
        "track_threshold": None,
        "reproj_threshold": None,
    }
    if num_points == 0 or not bool(_cfg_get(sparse_cfg, "RELIABILITY_FILTER_ENABLED", False)):
        return points, track_len, reproj_error, brightness_scores, gradient_scores, info

    min_points = max(
        int(_cfg_get(sparse_cfg, "KNN_K", 3)),
        int(_cfg_get(sparse_cfg, "PLANE_K", 8)),
        32,
    )
    all_keep = np.ones((num_points,), dtype=bool)
    track_keep = all_keep.copy()
    reproj_keep = all_keep.copy()

    track_low = float(min(max(_cfg_get(sparse_cfg, "FILTER_TRACK_P_LOW", 0.20), 0.0), 1.0))
    reproj_high = float(min(max(_cfg_get(sparse_cfg, "FILTER_REPROJ_P_HIGH", 0.80), 0.0), 1.0))

    if track_len is not None and track_len.shape[0] == num_points:
        track_threshold = float(np.quantile(track_len, track_low))
        track_keep = track_len >= track_threshold
        info["track_threshold"] = track_threshold
    if reproj_error is not None and reproj_error.shape[0] == num_points:
        reproj_threshold = float(np.quantile(reproj_error, reproj_high))
        reproj_keep = reproj_error <= reproj_threshold
        info["reproj_threshold"] = reproj_threshold

    keep_mask = track_keep & reproj_keep
    info["filter_mode"] = "track_reproj"
    if int(keep_mask.sum()) < min_points:
        if int(reproj_keep.sum()) >= min_points:
            keep_mask = reproj_keep
            info["filter_mode"] = "reproj_only"
        elif int(track_keep.sum()) >= min_points:
            keep_mask = track_keep
            info["filter_mode"] = "track_only"
        else:
            keep_mask = all_keep
            info["filter_mode"] = "all"

    kept_ratio = float(keep_mask.mean()) if keep_mask.shape[0] > 0 else 0.0
    info["filter_kept_ratio"] = kept_ratio
    return (
        points[keep_mask],
        track_len[keep_mask] if track_len is not None else track_len,
        reproj_error[keep_mask] if reproj_error is not None else reproj_error,
        brightness_scores[keep_mask],
        gradient_scores[keep_mask],
        info,
    )


def load_colmap_sparse_points(meta_cfg, sparse_cfg, device, train_dataset=None):
    if not bool(_cfg_get(sparse_cfg, "ENABLED", False)):
        return None, None, None, None, None, {
            "points_path": None,
            "meta_path": None,
            "meta_exists": False,
            "meta_used": False,
            "num_points": 0,
        }
    scene_root = Path(str(meta_cfg.DATASET.DATA_PATH))
    colmap_dir = str(_cfg_get(sparse_cfg, "COLMAP_DIR", "auxiliaries/colmap_sparse"))
    colmap_root = Path(colmap_dir) if os.path.isabs(colmap_dir) else scene_root / colmap_dir
    points_path = colmap_root / "points.npy"
    if not points_path.exists():
        raise RuntimeError(f"Sparse prior is enabled, but COLMAP points were not found: {points_path}")
    points = np.load(points_path).astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise RuntimeError(f"Sparse prior requires non-empty [N,3] points.npy, got {points.shape}: {points_path}")
    track_len = np.ones((points.shape[0],), dtype=np.float32)
    reproj_error = np.ones((points.shape[0],), dtype=np.float32)
    meta_enabled = bool(_cfg_get(sparse_cfg, "META_ENABLED", False))
    meta_path = colmap_root / "points_meta.npz"
    used_metadata = False
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
            used_metadata = True

    brightness_scores, gradient_scores, visibility_counts = build_sparse_lowlight_scores(train_dataset, points, sparse_cfg) if train_dataset is not None else (
        np.ones((points.shape[0],), dtype=np.float32),
        np.ones((points.shape[0],), dtype=np.float32),
        np.zeros((points.shape[0],), dtype=np.int32),
    )
    points, track_len, reproj_error, brightness_scores, gradient_scores, filter_info = filter_sparse_prior(
        points,
        track_len,
        reproj_error,
        brightness_scores,
        gradient_scores,
        sparse_cfg,
    )

    points_tensor = torch.from_numpy(points).to(device)
    track_len_tensor = torch.from_numpy(track_len).to(device)
    reproj_error_tensor = torch.from_numpy(reproj_error).to(device)
    brightness_tensor = torch.from_numpy(brightness_scores).to(device)
    gradient_tensor = torch.from_numpy(gradient_scores).to(device)
    print(
        f"[Sparse] loaded {points_tensor.shape[0]} COLMAP sparse points from {points_path}, "
        f"meta_used={int(used_metadata)}, filter_mode={filter_info['filter_mode']}, "
        f"kept_ratio={filter_info['filter_kept_ratio']:.3f}"
    )
    return points_tensor, track_len_tensor, reproj_error_tensor, brightness_tensor, gradient_tensor, {
        "points_path": str(points_path),
        "meta_path": str(meta_path),
        "meta_exists": bool(meta_path.exists()),
        "meta_used": bool(used_metadata),
        "num_points": int(points_tensor.shape[0]),
        "original_num_points": int(visibility_counts.shape[0]),
        "brightness_score_mean": float(brightness_scores.mean()) if brightness_scores.size > 0 else 1.0,
        "gradient_score_mean": float(gradient_scores.mean()) if gradient_scores.size > 0 else 1.0,
        "visibility_count_mean": float(visibility_counts.mean()) if visibility_counts.size > 0 else 0.0,
        "visibility_count_max": int(visibility_counts.max()) if visibility_counts.size > 0 else 0,
        "filter_enabled": int(filter_info["filter_enabled"]),
        "filter_mode": str(filter_info["filter_mode"]),
        "filter_kept_ratio": float(filter_info["filter_kept_ratio"]),
        "track_threshold": filter_info["track_threshold"],
        "reproj_threshold": filter_info["reproj_threshold"],
    }


def save_checkpoint(model, output_dir, step, meta_cfg):
    step_dir = build_step_dir(output_dir, step)
    os.makedirs(step_dir, exist_ok=True)
    save_config(os.path.join(step_dir, "config.yaml"), meta_cfg)
    checkpoint_path = os.path.join(step_dir, f"step_{int(step)}.pt")
    torch.save(model.splats.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_warmstart_checkpoint(model, checkpoint_path, device):
    checkpoint_path = os.path.expanduser(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "means" not in checkpoint:
        raise RuntimeError(f"Warm-start checkpoint must be a splat state_dict with 'means': {checkpoint_path}")

    expected_keys = list(model.splats.keys())
    means = checkpoint["means"]
    if not torch.is_tensor(means) or means.ndim != 2 or means.shape[1] != 3:
        raise RuntimeError(f"Warm-start checkpoint has invalid means tensor: {checkpoint_path}")

    num_points = int(means.shape[0])
    normalized = {}
    missing_keys = []
    converted_illum = False
    for key in expected_keys:
        tensor = checkpoint.get(key)
        if tensor is None:
            if key in {"depth_feat", "prior_feat", "illum_feat"}:
                tensor = torch.zeros(num_points, 1, dtype=means.dtype)
                missing_keys.append(key)
            elif key == "chroma_feat":
                tensor = torch.zeros(num_points, 2, dtype=means.dtype)
                missing_keys.append(key)
            else:
                raise RuntimeError(f"Warm-start checkpoint is missing required key '{key}': {checkpoint_path}")
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"Warm-start checkpoint key '{key}' is not a tensor: {checkpoint_path}")

        tensor = tensor.detach().to(device)
        if tensor.shape[0] != num_points:
            raise RuntimeError(
                f"Warm-start checkpoint key '{key}' has inconsistent gaussian count {tensor.shape[0]} != {num_points}: {checkpoint_path}"
            )

        if key in {"depth_feat", "prior_feat", "illum_feat"}:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            if key == "illum_feat" and tensor.ndim == 2 and tensor.shape[1] != 1:
                tensor = tensor.mean(dim=1, keepdim=True)
                converted_illum = True
            elif tensor.ndim != 2 or tensor.shape[1] != 1:
                tensor = tensor[:, :1]
        elif key == "chroma_feat":
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            if tensor.ndim != 2:
                tensor = tensor.reshape(num_points, -1)
            if tensor.shape[1] < 2:
                pad = torch.zeros(num_points, 2 - tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=1)
            elif tensor.shape[1] > 2:
                tensor = tensor[:, :2]

        normalized[key] = torch.nn.Parameter(tensor.contiguous())

    model.splats = torch.nn.ParameterDict(normalized)
    model.sh_degree = model.sh_degree_max
    missing_str = ",".join(missing_keys) if missing_keys else "none"
    print(
        f"[WarmStart] checkpoint={checkpoint_path}, gaussians={num_points}, missing={missing_str}, "
        f"illum_converted={int(converted_illum)}, sh_degree={model.sh_degree}"
    )


def _rotation_matrix_to_quaternion(rotation):
    if rotation.numel() == 0:
        return torch.zeros((0, 4), device=rotation.device, dtype=rotation.dtype)
    m00 = rotation[:, 0, 0]
    m11 = rotation[:, 1, 1]
    m22 = rotation[:, 2, 2]
    trace = m00 + m11 + m22
    quats = torch.zeros((rotation.shape[0], 4), device=rotation.device, dtype=rotation.dtype)

    positive_trace = trace > 0.0
    if positive_trace.any():
        s = torch.sqrt((trace[positive_trace] + 1.0).clamp_min(1.0e-12)) * 2.0
        quats[positive_trace, 0] = 0.25 * s
        quats[positive_trace, 1] = (rotation[positive_trace, 2, 1] - rotation[positive_trace, 1, 2]) / s
        quats[positive_trace, 2] = (rotation[positive_trace, 0, 2] - rotation[positive_trace, 2, 0]) / s
        quats[positive_trace, 3] = (rotation[positive_trace, 1, 0] - rotation[positive_trace, 0, 1]) / s

    cond_x = (~positive_trace) & (m00 >= m11) & (m00 >= m22)
    if cond_x.any():
        s = torch.sqrt((1.0 + m00[cond_x] - m11[cond_x] - m22[cond_x]).clamp_min(1.0e-12)) * 2.0
        quats[cond_x, 0] = (rotation[cond_x, 2, 1] - rotation[cond_x, 1, 2]) / s
        quats[cond_x, 1] = 0.25 * s
        quats[cond_x, 2] = (rotation[cond_x, 0, 1] + rotation[cond_x, 1, 0]) / s
        quats[cond_x, 3] = (rotation[cond_x, 0, 2] + rotation[cond_x, 2, 0]) / s

    cond_y = (~positive_trace) & (~cond_x) & (m11 >= m22)
    if cond_y.any():
        s = torch.sqrt((1.0 + m11[cond_y] - m00[cond_y] - m22[cond_y]).clamp_min(1.0e-12)) * 2.0
        quats[cond_y, 0] = (rotation[cond_y, 0, 2] - rotation[cond_y, 2, 0]) / s
        quats[cond_y, 1] = (rotation[cond_y, 0, 1] + rotation[cond_y, 1, 0]) / s
        quats[cond_y, 2] = 0.25 * s
        quats[cond_y, 3] = (rotation[cond_y, 1, 2] + rotation[cond_y, 2, 1]) / s

    cond_z = (~positive_trace) & (~cond_x) & (~cond_y)
    if cond_z.any():
        s = torch.sqrt((1.0 + m22[cond_z] - m00[cond_z] - m11[cond_z]).clamp_min(1.0e-12)) * 2.0
        quats[cond_z, 0] = (rotation[cond_z, 1, 0] - rotation[cond_z, 0, 1]) / s
        quats[cond_z, 1] = (rotation[cond_z, 0, 2] + rotation[cond_z, 2, 0]) / s
        quats[cond_z, 2] = (rotation[cond_z, 1, 2] + rotation[cond_z, 2, 1]) / s
        quats[cond_z, 3] = 0.25 * s

    return torch.nn.functional.normalize(quats, dim=-1, eps=1.0e-12)


class TwoPhaseExponentialLRScheduler:
    def __init__(self, optimizer, initial_lr, final_lr, total_steps, current_step=0, tail_start_step=None, tail_target_lr=None):
        self.optimizer = optimizer
        self.initial_lr = float(initial_lr)
        self.final_lr = float(final_lr)
        self.total_steps = int(max(1, total_steps))
        self.current_step = int(max(0, current_step))
        self.tail_start_step = None if tail_start_step is None else int(max(0, min(int(tail_start_step), self.total_steps)))
        self.tail_target_lr = None if tail_target_lr is None else float(max(0.0, tail_target_lr))
        if self.tail_target_lr is not None:
            lr_low = min(self.initial_lr, self.final_lr)
            lr_high = max(self.initial_lr, self.final_lr)
            self.tail_target_lr = float(min(max(self.tail_target_lr, lr_low), lr_high))
        self._apply_lr(self.current_step)

    def _single_phase_lr(self, step):
        progress = min(max(float(step) / float(self.total_steps), 0.0), 1.0)
        return self.initial_lr * ((self.final_lr / self.initial_lr) ** progress)

    def _lr_at_step(self, step):
        step = int(max(0, min(step, self.total_steps)))
        if (
            self.tail_start_step is None
            or self.tail_target_lr is None
            or self.tail_start_step <= 0
            or self.tail_start_step >= self.total_steps
        ):
            return self._single_phase_lr(step)
        if step <= self.tail_start_step:
            return self._single_phase_lr(step)
        tail_span = max(1, self.total_steps - self.tail_start_step)
        tail_progress = min(max(float(step - self.tail_start_step) / float(tail_span), 0.0), 1.0)
        return self.tail_target_lr * ((self.final_lr / self.tail_target_lr) ** tail_progress)

    def _apply_lr(self, step):
        lr_value = float(self._lr_at_step(step))
        for group in self.optimizer.param_groups:
            group["lr"] = lr_value

    def step(self):
        self.current_step += 1
        self._apply_lr(self.current_step)

    def get_last_lr(self):
        return [float(group["lr"]) for group in self.optimizer.param_groups]


def build_optimizers_and_schedulers(model, cfg, lr_map, frozen_geometry_keys, total_steps, existing_optimizers=None, current_step=0):
    optimizers = {}
    for name, param in model.splats.items():
        if name in frozen_geometry_keys:
            param.requires_grad_(False)
            continue
        lr_value = float(lr_map[name])
        if existing_optimizers is not None and name in existing_optimizers and existing_optimizers[name].param_groups:
            lr_value = float(existing_optimizers[name].param_groups[0]["lr"])
        optimizers[name] = torch.optim.Adam([param], lr=lr_value, eps=1e-15)

    schedulers = {}
    scheduler_specs = {
        "means": ("LR_MEANS", "LR_MEANS_FINAL", "LR_MEANS_TAIL_TARGET"),
        "scales": ("LR_SCALES", "LR_SCALES_FINAL", "LR_SCALES_TAIL_TARGET"),
        "quats": ("LR_QUATS", "LR_QUATS_FINAL", "LR_QUATS_TAIL_TARGET"),
    }
    lr_tail_start_step = _cfg_get(cfg, "LR_TAIL_START_STEP", None)
    for name, (initial_key, final_key, tail_target_key) in scheduler_specs.items():
        if name not in optimizers:
            continue
        initial_lr = _cfg_get(cfg, initial_key, None)
        final_lr = _cfg_get(cfg, final_key, None)
        if initial_lr is None or final_lr is None:
            continue
        initial_lr = float(initial_lr)
        final_lr = float(final_lr)
        if initial_lr <= 0.0 or final_lr <= 0.0:
            continue
        tail_target_lr = _cfg_get(cfg, tail_target_key, None)
        schedulers[name] = TwoPhaseExponentialLRScheduler(
            optimizers[name],
            initial_lr=initial_lr,
            final_lr=final_lr,
            total_steps=total_steps,
            current_step=current_step,
            tail_start_step=lr_tail_start_step,
            tail_target_lr=tail_target_lr,
        )
    return optimizers, schedulers


def chunked_nearest_neighbors(query_points, reference_points, chunk_size):
    query_count = int(query_points.shape[0])
    if query_count == 0:
        return (
            torch.zeros((0,), device=query_points.device, dtype=query_points.dtype),
            torch.zeros((0,), device=query_points.device, dtype=torch.long),
        )
    if int(reference_points.shape[0]) == 0:
        return (
            torch.full((query_count,), float("inf"), device=query_points.device, dtype=query_points.dtype),
            torch.full((query_count,), -1, device=query_points.device, dtype=torch.long),
        )

    distances = []
    indices = []
    for start in range(0, query_count, chunk_size):
        end = min(start + chunk_size, query_count)
        chunk = query_points[start:end]
        chunk_dist = torch.cdist(chunk, reference_points)
        chunk_min_dist, chunk_min_idx = chunk_dist.min(dim=1)
        distances.append(chunk_min_dist)
        indices.append(chunk_min_idx)
    return torch.cat(distances, dim=0), torch.cat(indices, dim=0)


def chunked_topk_neighbors(query_points, reference_points, chunk_size, k):
    query_count = int(query_points.shape[0])
    ref_count = int(reference_points.shape[0])
    if query_count == 0 or ref_count == 0:
        empty_dist = query_points.new_empty((query_count, 0))
        empty_idx = torch.empty((query_count, 0), device=query_points.device, dtype=torch.long)
        return empty_dist, empty_idx

    k = int(max(1, min(k, ref_count)))
    distances = []
    indices = []
    for start in range(0, query_count, chunk_size):
        end = min(start + chunk_size, query_count)
        chunk = query_points[start:end]
        chunk_dist = torch.cdist(chunk, reference_points)
        chunk_knn_dist, chunk_knn_idx = torch.topk(chunk_dist, k=k, dim=1, largest=False)
        distances.append(chunk_knn_dist)
        indices.append(chunk_knn_idx)
    return torch.cat(distances, dim=0), torch.cat(indices, dim=0)


def estimate_sparse_planes(query_points, sparse_points, plane_k, chunk_size, plane_eps, plane_min_eigen_gap):
    query_count = int(query_points.shape[0])
    if query_count == 0 or int(sparse_points.shape[0]) < 3:
        zero_quat = torch.zeros((query_count, 4), device=query_points.device, dtype=query_points.dtype)
        zero_quat[:, 0] = 1.0
        return {
            "stable_mask": torch.zeros((query_count,), device=query_points.device, dtype=torch.bool),
            "quats": zero_quat,
            "local_radius": query_points.new_zeros((query_count,)),
        }

    neighbor_k = min(max(3, int(plane_k)), int(sparse_points.shape[0]))
    quats_parts = []
    stable_parts = []
    local_radius_parts = []
    for start in range(0, query_count, chunk_size):
        end = min(start + chunk_size, query_count)
        chunk = query_points[start:end]
        knn_dist, knn_idx = chunked_topk_neighbors(chunk, sparse_points, chunk_size, neighbor_k)
        knn_points = sparse_points[knn_idx]
        center = knn_points.mean(dim=1, keepdim=True)
        centered = knn_points - center
        cov = torch.matmul(centered.transpose(1, 2), centered) / float(max(1, neighbor_k))
        cov = cov + plane_eps * torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        normal = torch.nn.functional.normalize(eigvecs[:, :, 0], dim=-1, eps=1.0e-12)
        eigen_gap = eigvals[:, 1] - eigvals[:, 0]
        stable_mask = torch.isfinite(eigen_gap) & torch.isfinite(normal).all(dim=1) & (eigen_gap >= plane_min_eigen_gap)

        helper = torch.tensor([0.0, 0.0, 1.0], device=chunk.device, dtype=chunk.dtype).expand(end - start, -1).clone()
        near_parallel = torch.abs((helper * normal).sum(dim=1)) > 0.9
        if near_parallel.any():
            helper[near_parallel] = torch.tensor([0.0, 1.0, 0.0], device=chunk.device, dtype=chunk.dtype)
        tangent1 = torch.nn.functional.normalize(torch.cross(helper, normal, dim=1), dim=1, eps=1.0e-12)
        tangent2 = torch.nn.functional.normalize(torch.cross(normal, tangent1, dim=1), dim=1, eps=1.0e-12)
        rotation = torch.stack([tangent1, tangent2, normal], dim=-1)
        quats = _rotation_matrix_to_quaternion(rotation)
        fallback_quats = torch.zeros_like(quats)
        fallback_quats[:, 0] = 1.0
        quats = torch.where(stable_mask.unsqueeze(-1), quats, fallback_quats)

        quats_parts.append(quats)
        stable_parts.append(stable_mask)
        if neighbor_k > 1:
            local_radius = knn_dist[:, 1:].mean(dim=1)
        else:
            local_radius = knn_dist[:, 0]
        local_radius_parts.append(local_radius)

    return {
        "stable_mask": torch.cat(stable_parts, dim=0),
        "quats": torch.cat(quats_parts, dim=0),
        "local_radius": torch.cat(local_radius_parts, dim=0),
    }


def prepare_spawn_tensors(model, spawn_points, nearest_indices, active_means, active_indices, sparse_points, topology_cfg):
    spawn_count = int(spawn_points.shape[0])
    device = model.splats["means"].device
    dtype = model.splats["means"].dtype
    if spawn_count == 0:
        return {key: tensor.new_empty((0, *tensor.shape[1:])) for key, tensor in model.splats.items()}

    plane_info = estimate_sparse_planes(
        query_points=spawn_points,
        sparse_points=sparse_points,
        plane_k=int(_cfg_get(topology_cfg, "SPAWN_SUPPORT_K", 8)),
        chunk_size=int(max(32, _cfg_get(topology_cfg, "DISTANCE_CHUNK_SIZE", 128))),
        plane_eps=float(max(1.0e-12, _cfg_get(topology_cfg, "PLANE_EPS", 1.0e-6))),
        plane_min_eigen_gap=float(max(0.0, _cfg_get(topology_cfg, "PLANE_MIN_EIGEN_GAP", 1.0e-4))),
    )
    nearest_scales = torch.exp(model.splats["scales"].detach()[nearest_indices])
    coverage_radius = float(_cfg_get(topology_cfg, "COVERAGE_RADIUS", 0.06))
    feature_init_mode = str(_cfg_get(topology_cfg, "SEED_FEATURE_INIT", "nearest_gaussian_copy")).lower()
    tangent_ratio = float(max(0.05, _cfg_get(topology_cfg, "SPARSE_TANGENT_SCALE_RATIO", 0.70)))
    normal_ratio = float(max(0.02, _cfg_get(topology_cfg, "SPARSE_NORMAL_SCALE_RATIO", 0.18)))
    local_radius = plane_info["local_radius"].clamp_min(1.0e-6)
    base_scale = nearest_scales.mean(dim=1)
    tangent_scale_sparse = torch.clamp(
        local_radius * tangent_ratio,
        min=max(0.004, coverage_radius * 0.12),
        max=max(0.012, coverage_radius * 0.90),
    )
    normal_scale_sparse = torch.clamp(
        local_radius * normal_ratio,
        min=max(0.0015, coverage_radius * 0.03),
        max=max(0.006, coverage_radius * 0.35),
    )
    tangent_scale_fallback = torch.clamp(
        base_scale * 1.5,
        min=max(0.005, coverage_radius * 0.15),
        max=max(0.01, coverage_radius * 0.75),
    )
    normal_scale_fallback = torch.clamp(
        base_scale * 0.5,
        min=max(0.002, coverage_radius * 0.05),
        max=max(0.003, coverage_radius * 0.35),
    )
    tangent_scale = torch.where(plane_info["stable_mask"], tangent_scale_sparse, tangent_scale_fallback)
    normal_scale = torch.where(plane_info["stable_mask"], normal_scale_sparse, normal_scale_fallback)
    scales = torch.stack([tangent_scale, tangent_scale, torch.minimum(normal_scale, tangent_scale)], dim=1).log()
    fallback_scale = torch.full_like(scales, fill_value=max(0.005, coverage_radius * 0.2)).log()
    scales = torch.where(plane_info["stable_mask"].unsqueeze(-1), scales, fallback_scale)

    feature_k = int(max(1, _cfg_get(topology_cfg, "SPAWN_FEATURE_K", 4)))
    knn_dist, knn_local_idx = chunked_topk_neighbors(
        spawn_points,
        active_means,
        int(max(32, _cfg_get(topology_cfg, "DISTANCE_CHUNK_SIZE", 128))),
        feature_k,
    )
    if int(knn_local_idx.shape[1]) > 0:
        knn_indices = active_indices[knn_local_idx]
        knn_weights = 1.0 / knn_dist.clamp_min(1.0e-6)
        knn_opacity = torch.sigmoid(model.splats["opacities"].detach()[knn_indices])
        knn_weights = knn_weights * knn_opacity.clamp_min(1.0e-3)
        knn_weights = knn_weights / knn_weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    else:
        knn_indices = nearest_indices[:, None]
        knn_weights = torch.ones((spawn_count, 1), device=device, dtype=dtype)

    spawn_state = {}
    for key, param in model.splats.items():
        base = param.detach()
        if key == "means":
            spawn_state[key] = spawn_points.to(device=device, dtype=dtype)
        elif key == "quats":
            spawn_state[key] = plane_info["quats"].to(device=device, dtype=base.dtype)
        elif key == "scales":
            spawn_state[key] = scales.to(device=device, dtype=base.dtype)
        elif key == "opacities":
            spawn_opacity = float(min(max(_cfg_get(topology_cfg, "SPAWN_OPACITY", 0.05), 1.0e-4), 1.0 - 1.0e-4))
            logit_value = float(torch.logit(torch.tensor(spawn_opacity, device=device, dtype=base.dtype)).item())
            spawn_state[key] = torch.full((spawn_count,), logit_value, device=device, dtype=base.dtype)
        else:
            if feature_init_mode == "knn_weighted_copy":
                neighbor_values = base[knn_indices]
                view_shape = [spawn_count, int(knn_weights.shape[1])] + [1] * (neighbor_values.ndim - 2)
                weighted = neighbor_values * knn_weights.reshape(view_shape).to(device=neighbor_values.device, dtype=neighbor_values.dtype)
                spawn_state[key] = weighted.sum(dim=1)
            elif feature_init_mode == "nearest_gaussian_copy":
                spawn_state[key] = base[nearest_indices].clone()
            else:
                spawn_state[key] = torch.zeros((spawn_count, *base.shape[1:]), device=device, dtype=base.dtype)
    return spawn_state


def mutate_model_splats(model, keep_mask, append_state, frozen_geometry_keys):
    updated = {}
    for key, param in model.splats.items():
        base_tensor = param.detach()
        if keep_mask is not None:
            base_tensor = base_tensor[keep_mask]
        append_tensor = None if append_state is None else append_state.get(key, None)
        if append_tensor is not None and int(append_tensor.shape[0]) > 0:
            append_tensor = append_tensor.to(device=base_tensor.device, dtype=base_tensor.dtype)
            base_tensor = torch.cat([base_tensor, append_tensor], dim=0)
        updated[key] = torch.nn.Parameter(base_tensor.contiguous())
        if key in frozen_geometry_keys:
            updated[key].requires_grad_(False)
    model.splats = torch.nn.ParameterDict(updated)


def should_run_sparse_topology(current_step, topology_cfg):
    if not bool(_cfg_get(topology_cfg, "ENABLED", False)):
        return False
    start_step = int(_cfg_get(topology_cfg, "START_STEP", 0))
    end_step = int(_cfg_get(topology_cfg, "END_STEP", current_step))
    interval = int(max(1, _cfg_get(topology_cfg, "TOPOLOGY_REFRESH_INTERVAL", _cfg_get(topology_cfg, "INTERVAL", 250))))
    return start_step <= int(current_step) <= end_step and int(current_step) % interval == 0


def find_sparse_guided_module(loss_modules):
    for module in loss_modules:
        if getattr(module, "name", "") == "sparse_guided":
            return module
    return None


def compute_vsurface_topology_metrics(
    model,
    sparse_module,
    query_points,
    gaussian_indices,
    sparse_points,
    sparse_track_len,
    sparse_reproj_error,
    sparse_brightness_score,
    sparse_gradient_score,
    normal_score_weight,
    scale_score_weight,
):
    if (
        sparse_module is None
        or query_points is None
        or sparse_points is None
        or int(query_points.shape[0]) == 0
        or int(sparse_points.shape[0]) < 3
    ):
        return None

    with torch.no_grad():
        _, _, _, _, support_score = sparse_module._get_sparse_support_scores(
            sparse_points,
            sparse_track_len,
            sparse_reproj_error,
            sparse_brightness_score,
            sparse_gradient_score,
        )
        neighborhood = sparse_module._build_support_neighborhood(query_points, sparse_points, support_score)
        if int(neighborhood["neighbor_k"]) < 3:
            return None

        query_quats = None
        query_scales = None
        if gaussian_indices is not None and "quats" in model.splats and "scales" in model.splats and int(model.num_gaussians) > 0:
            safe_indices = gaussian_indices.reshape(-1).clamp(min=0, max=int(model.num_gaussians) - 1)
            query_quats = model.splats["quats"].detach()[safe_indices]
            query_scales = model.splats["scales"].detach()[safe_indices]

        residual_bundle = sparse_module._compute_residual_bundle(query_points, neighborhood, query_quats, query_scales)
        stable_mask = residual_bundle["stable_mask"]
        plane_confidence = residual_bundle.get("plane_confidence", stable_mask.float())
        local_radius = residual_bundle["local_radius"].clamp_min(sparse_module.knn_eps)
        normalized_normal_residual = residual_bundle["normal_residual"] / local_radius
        orientation_metric = residual_bundle.get("orientation_alignment_raw", residual_bundle["orientation_alignment"])
        scale_metric = residual_bundle.get("anisotropic_scale_target_raw", residual_bundle["anisotropic_scale_target"])
        surface_score = (
            orientation_metric
            + float(scale_score_weight) * scale_metric
            + float(normal_score_weight) * normalized_normal_residual
        )
        surface_score = plane_confidence * surface_score
        return {
            "surface_score": surface_score,
            "stable_mask": stable_mask,
            "plane_confidence": plane_confidence,
            "orientation_alignment": orientation_metric,
            "anisotropic_scale_target": scale_metric,
            "normalized_normal_residual": normalized_normal_residual,
        }


def build_prune_reliable_core(
    sparse_points,
    sparse_track_len,
    sparse_reproj_error,
    sparse_brightness_score,
    sparse_gradient_score,
    topology_cfg,
    sparse_cfg,
    sparse_module,
):
    num_sparse = 0 if sparse_points is None else int(sparse_points.shape[0])
    support_quantile = float(min(max(_cfg_get(topology_cfg, "PRUNE_CORE_SUPPORT_QUANTILE", 0.7), 0.0), 1.0))
    info = {
        "enabled": int(bool(_cfg_get(topology_cfg, "PRUNE_RELIABLE_CORE_ENABLED", False))),
        "rule": str(_cfg_get(topology_cfg, "PRUNE_RELIABLE_CORE_RULE", "support_quantile")).lower(),
        "core_count": int(num_sparse),
        "core_ratio": float(1.0 if num_sparse > 0 else 0.0),
        "core_support_mean": 0.0,
        "support_quantile": support_quantile,
    }
    if num_sparse == 0:
        return sparse_points, sparse_track_len, sparse_reproj_error, sparse_brightness_score, sparse_gradient_score, info

    if sparse_module is None or not bool(info["enabled"]) or info["rule"] != "support_quantile":
        return sparse_points, sparse_track_len, sparse_reproj_error, sparse_brightness_score, sparse_gradient_score, info

    with torch.no_grad():
        _, _, _, _, support_score = sparse_module._get_sparse_support_scores(
            sparse_points,
            sparse_track_len,
            sparse_reproj_error,
            sparse_brightness_score,
            sparse_gradient_score,
        )
        min_core_points = int(
            max(
                _cfg_get(topology_cfg, "PRUNE_CORE_MIN_POINTS", 256),
                _cfg_get(sparse_cfg, "KNN_K", 3),
                _cfg_get(sparse_cfg, "PLANE_K", 8),
            )
        )
        min_core_points = max(1, min(int(num_sparse), min_core_points))
        if int(num_sparse) <= min_core_points:
            core_mask = torch.ones((num_sparse,), device=sparse_points.device, dtype=torch.bool)
        else:
            support_threshold = torch.quantile(support_score, support_quantile)
            core_mask = support_score >= support_threshold
            if int(core_mask.sum().item()) < min_core_points:
                topk = torch.topk(support_score, k=min_core_points, largest=True).indices
                core_mask = torch.zeros((num_sparse,), device=sparse_points.device, dtype=torch.bool)
                core_mask[topk] = True
        core_count = int(core_mask.sum().item())
        core_support = support_score[core_mask]
        info["core_count"] = core_count
        info["core_ratio"] = float(core_count / max(1, num_sparse))
        info["core_support_mean"] = float(core_support.mean().item()) if int(core_support.numel()) > 0 else 0.0
        return (
            sparse_points[core_mask],
            sparse_track_len[core_mask] if sparse_track_len is not None else sparse_track_len,
            sparse_reproj_error[core_mask] if sparse_reproj_error is not None else sparse_reproj_error,
            sparse_brightness_score[core_mask] if sparse_brightness_score is not None else sparse_brightness_score,
            sparse_gradient_score[core_mask] if sparse_gradient_score is not None else sparse_gradient_score,
            info,
        )


def run_sparse_topology_event(
    model,
    sparse_points,
    sparse_track_len,
    sparse_reproj_error,
    sparse_brightness_score,
    sparse_gradient_score,
    topology_cfg,
    sparse_cfg,
    current_step,
    lr_map,
    optimizers,
    cfg,
    frozen_geometry_keys,
    total_steps,
    loss_modules,
):
    event_info = {
        "record_type": "topology_event",
        "step": int(current_step),
        "topology_event": 1,
        "spawn_count": 0,
        "prune_count": 0,
        "coverage_hole_count": 0,
        "coverage_hole_ratio": 0.0,
        "spawn_distance_mean": 0.0,
        "prune_distance_mean": 0.0,
        "coverage_threshold_mean": 0.0,
        "coverage_excess_mean": 0.0,
        "num_gaussians_before": int(model.num_gaussians),
        "num_gaussians_after": int(model.num_gaussians),
        "prune_mode": str(_cfg_get(topology_cfg, "PRUNE_MODE", "hard_remove")),
        "prune_active": 0,
        "prune_decayed_count": 0,
        "prune_sparse_core_count": 0,
        "prune_sparse_core_ratio": 0.0,
        "prune_sparse_core_support_mean": 0.0,
        "spawn_feature_init_mode": str(_cfg_get(topology_cfg, "SEED_FEATURE_INIT", "nearest_gaussian_copy")),
        "surface_coupling_active": 0,
        "spawn_surface_score_mean": 0.0,
        "spawn_surface_stable_ratio": 0.0,
        "spawn_surface_confidence_mean": 0.0,
        "spawn_surface_extra_count": 0,
        "prune_surface_score_mean": 0.0,
        "prune_surface_stable_ratio": 0.0,
    }
    if sparse_points is None or int(sparse_points.shape[0]) == 0:
        return optimizers, {}, event_info

    device = model.splats["means"].device
    means = model.splats["means"].detach()
    opacity_values = torch.sigmoid(model.splats["opacities"].detach())
    active_thresh = float(_cfg_get(sparse_cfg, "MIN_OPACITY", 0.2))
    active_indices = torch.nonzero(opacity_values > active_thresh, as_tuple=False).squeeze(-1)
    if int(active_indices.numel()) == 0:
        active_indices = torch.arange(int(model.num_gaussians), device=device, dtype=torch.long)
    active_means = means[active_indices]
    distance_chunk = int(max(32, _cfg_get(topology_cfg, "DISTANCE_CHUNK_SIZE", 128)))
    coverage_radius = float(max(1.0e-6, _cfg_get(topology_cfg, "COVERAGE_RADIUS", 0.06)))
    surface_coupling_enabled = bool(_cfg_get(topology_cfg, "SURFACE_COUPLING_ENABLED", False))
    spawn_surface_score_weight = float(max(0.0, _cfg_get(topology_cfg, "SPAWN_SURFACE_SCORE_WEIGHT", 1.5)))
    spawn_surface_additive_weight = float(max(0.0, _cfg_get(topology_cfg, "SPAWN_SURFACE_ADDITIVE_WEIGHT", 1.0)))
    spawn_surface_confidence_weight = float(max(0.0, _cfg_get(topology_cfg, "SPAWN_SURFACE_CONFIDENCE_WEIGHT", 0.75)))
    spawn_surface_slack_weight = float(max(0.0, _cfg_get(topology_cfg, "SPAWN_SURFACE_SLACK_WEIGHT", 0.5)))
    spawn_surface_score_clamp = float(max(0.1, _cfg_get(topology_cfg, "SPAWN_SURFACE_SCORE_CLAMP", 4.0)))
    spawn_surface_min_coverage_ratio = float(max(0.0, _cfg_get(topology_cfg, "SPAWN_SURFACE_MIN_COVERAGE_RATIO", 0.75)))
    spawn_surface_min_confidence = float(max(0.0, _cfg_get(topology_cfg, "SPAWN_SURFACE_MIN_CONFIDENCE", 0.15)))
    max_surface_spawn = int(max(0, _cfg_get(topology_cfg, "MAX_SURFACE_SPAWN_PER_EVENT", 64)))
    prune_surface_score_weight = float(max(0.0, _cfg_get(topology_cfg, "PRUNE_SURFACE_SCORE_WEIGHT", 1.0)))
    surface_score_normal_weight = float(max(0.0, _cfg_get(topology_cfg, "SURFACE_SCORE_NORMAL_WEIGHT", 0.5)))
    surface_score_scale_weight = float(max(0.0, _cfg_get(topology_cfg, "SURFACE_SCORE_SCALE_WEIGHT", 2.0)))
    sparse_guided_module = find_sparse_guided_module(loss_modules)
    sparse_surface_module = sparse_guided_module if surface_coupling_enabled else None
    event_info["surface_coupling_active"] = int(
        sparse_surface_module is not None and (spawn_surface_score_weight > 0.0 or prune_surface_score_weight > 0.0)
    )

    sparse_to_active_dist, nearest_active_local = chunked_nearest_neighbors(sparse_points, active_means, distance_chunk)
    adaptive_coverage_enabled = bool(_cfg_get(topology_cfg, "ADAPTIVE_COVERAGE_ENABLED", True))
    if adaptive_coverage_enabled and int(sparse_points.shape[0]) > 1:
        coverage_density_k = int(max(2, _cfg_get(topology_cfg, "COVERAGE_DENSITY_K", _cfg_get(topology_cfg, "SPAWN_SUPPORT_K", 8))))
        sparse_knn_dist, _ = chunked_topk_neighbors(
            sparse_points,
            sparse_points,
            distance_chunk,
            min(coverage_density_k + 1, int(sparse_points.shape[0])),
        )
        if int(sparse_knn_dist.shape[1]) > 1:
            local_sparse_radius = sparse_knn_dist[:, 1:].mean(dim=1)
        else:
            local_sparse_radius = sparse_knn_dist[:, 0]
        adaptive_threshold = (
            local_sparse_radius * float(max(0.1, _cfg_get(topology_cfg, "COVERAGE_RADIUS_DENSITY_SCALE", 2.0)))
        ).clamp(
            min=coverage_radius * float(max(0.1, _cfg_get(topology_cfg, "COVERAGE_RADIUS_MIN_SCALE", 0.5))),
            max=coverage_radius * float(max(0.2, _cfg_get(topology_cfg, "COVERAGE_RADIUS_MAX_SCALE", 1.25))),
        )
    else:
        adaptive_threshold = torch.full_like(sparse_to_active_dist, fill_value=coverage_radius)

    coverage_excess = sparse_to_active_dist - adaptive_threshold
    coverage_hole_mask = coverage_excess > 0.0
    coverage_hole_indices = torch.nonzero(coverage_hole_mask, as_tuple=False).squeeze(-1)
    coverage_hole_count = int(coverage_hole_indices.numel())
    event_info["coverage_hole_count"] = coverage_hole_count
    event_info["coverage_hole_ratio"] = float(coverage_hole_count / max(1, int(sparse_points.shape[0])))
    event_info["coverage_threshold_mean"] = float(adaptive_threshold.mean().item())
    if coverage_hole_count > 0:
        event_info["coverage_excess_mean"] = float(coverage_excess[coverage_hole_indices].mean().item())

    spawn_enabled = bool(_cfg_get(topology_cfg, "SPAWN_ENABLED", True))
    max_spawn = int(max(0, _cfg_get(topology_cfg, "MAX_SPAWN_PER_EVENT", 2048))) if spawn_enabled else 0
    spawn_sparse_indices = coverage_hole_indices
    spawn_surface_metrics = None
    normalized_surface_all = None
    normalized_confidence_all = None
    nearest_existing_all = None
    if event_info["surface_coupling_active"]:
        nearest_existing_all = active_indices[nearest_active_local.clamp_min(0)]
        spawn_surface_metrics = compute_vsurface_topology_metrics(
            model=model,
            sparse_module=sparse_surface_module,
            query_points=sparse_points,
            gaussian_indices=nearest_existing_all,
            sparse_points=sparse_points,
            sparse_track_len=sparse_track_len,
            sparse_reproj_error=sparse_reproj_error,
            sparse_brightness_score=sparse_brightness_score,
            sparse_gradient_score=sparse_gradient_score,
            normal_score_weight=surface_score_normal_weight,
            scale_score_weight=surface_score_scale_weight,
        )
        if spawn_surface_metrics is not None:
            normalized_surface_all = spawn_surface_metrics["surface_score"] / spawn_surface_metrics["surface_score"].mean().clamp_min(1.0e-6)
            normalized_surface_all = normalized_surface_all.clamp(0.0, spawn_surface_score_clamp)
            normalized_confidence_all = spawn_surface_metrics["plane_confidence"] / spawn_surface_metrics["plane_confidence"].mean().clamp_min(1.0e-6)
            normalized_confidence_all = normalized_confidence_all.clamp(0.0, 2.5)
    if coverage_hole_count > 0 and max_spawn > 0:
        spawn_priority = coverage_excess[coverage_hole_indices] / coverage_excess[coverage_hole_indices].mean().clamp_min(1.0e-6)
        if normalized_surface_all is not None and normalized_confidence_all is not None:
            hole_surface = normalized_surface_all[coverage_hole_indices]
            hole_confidence = normalized_confidence_all[coverage_hole_indices]
            spawn_priority = spawn_priority + spawn_surface_additive_weight * hole_surface + spawn_surface_confidence_weight * hole_confidence
            spawn_priority = spawn_priority * (1.0 + spawn_surface_score_weight * hole_surface)
        hole_order = torch.argsort(spawn_priority, descending=True)
        spawn_sparse_indices = coverage_hole_indices[hole_order[: min(max_spawn, coverage_hole_count)]]
    else:
        spawn_sparse_indices = coverage_hole_indices[:0]

    if normalized_surface_all is not None and normalized_confidence_all is not None and max_spawn > int(spawn_sparse_indices.numel()) and max_surface_spawn > 0:
        coverage_ratio = sparse_to_active_dist / adaptive_threshold.clamp_min(1.0e-6)
        normalized_slack = ((coverage_ratio - spawn_surface_min_coverage_ratio) / max(1.0e-6, 1.0 - spawn_surface_min_coverage_ratio)).clamp(0.0, 1.0)
        extra_mask = ~coverage_hole_mask
        extra_mask &= coverage_ratio >= spawn_surface_min_coverage_ratio
        extra_mask &= spawn_surface_metrics["plane_confidence"] >= spawn_surface_min_confidence
        extra_candidates = torch.nonzero(extra_mask, as_tuple=False).squeeze(-1)
        if int(extra_candidates.numel()) > 0:
            extra_budget = min(max_spawn - int(spawn_sparse_indices.numel()), max_surface_spawn, int(extra_candidates.numel()))
            if extra_budget > 0:
                extra_priority = (
                    spawn_surface_additive_weight * normalized_surface_all[extra_candidates]
                    + spawn_surface_confidence_weight * normalized_confidence_all[extra_candidates]
                    + spawn_surface_slack_weight * normalized_slack[extra_candidates]
                )
                extra_keep = torch.topk(extra_priority, k=extra_budget, largest=True).indices
                extra_indices = extra_candidates[extra_keep]
                if int(extra_indices.numel()) > 0:
                    spawn_sparse_indices = torch.cat([spawn_sparse_indices, extra_indices], dim=0)
                    event_info["spawn_surface_extra_count"] = int(extra_indices.numel())

    spawn_points = sparse_points[spawn_sparse_indices]
    spawn_count = int(spawn_points.shape[0])
    event_info["spawn_count"] = spawn_count
    if spawn_count > 0:
        nearest_existing = nearest_active_local[spawn_sparse_indices]
        nearest_existing = active_indices[nearest_existing.clamp_min(0)]
        event_info["spawn_distance_mean"] = float(sparse_to_active_dist[spawn_sparse_indices].mean().item())
        if spawn_surface_metrics is not None:
            event_info["spawn_surface_score_mean"] = float(spawn_surface_metrics["surface_score"][spawn_sparse_indices].mean().item())
            event_info["spawn_surface_stable_ratio"] = float(spawn_surface_metrics["stable_mask"][spawn_sparse_indices].float().mean().item())
            event_info["spawn_surface_confidence_mean"] = float(spawn_surface_metrics["plane_confidence"][spawn_sparse_indices].mean().item())
        spawn_state = prepare_spawn_tensors(model, spawn_points, nearest_existing, active_means, active_indices, sparse_points, topology_cfg)
    else:
        nearest_existing = torch.zeros((0,), device=device, dtype=torch.long)
        spawn_state = None

    prune_start_step = int(_cfg_get(topology_cfg, "PRUNE_START_STEP", _cfg_get(topology_cfg, "START_STEP", 0)))
    prune_enabled = bool(_cfg_get(topology_cfg, "PRUNE_ENABLED", True)) and int(current_step) >= prune_start_step
    event_info["prune_active"] = int(prune_enabled)
    prune_mode = str(_cfg_get(topology_cfg, "PRUNE_MODE", "hard_remove")).lower()
    prune_indices = torch.zeros((0,), device=device, dtype=torch.long)
    if prune_enabled:
        (
            prune_sparse_points,
            prune_sparse_track_len,
            prune_sparse_reproj_error,
            prune_sparse_brightness_score,
            prune_sparse_gradient_score,
            prune_core_info,
        ) = build_prune_reliable_core(
            sparse_points=sparse_points,
            sparse_track_len=sparse_track_len,
            sparse_reproj_error=sparse_reproj_error,
            sparse_brightness_score=sparse_brightness_score,
            sparse_gradient_score=sparse_gradient_score,
            topology_cfg=topology_cfg,
            sparse_cfg=sparse_cfg,
            sparse_module=sparse_guided_module,
        )
        event_info["prune_sparse_core_count"] = int(prune_core_info.get("core_count", 0))
        event_info["prune_sparse_core_ratio"] = float(prune_core_info.get("core_ratio", 0.0))
        event_info["prune_sparse_core_support_mean"] = float(prune_core_info.get("core_support_mean", 0.0))
        prune_opacity_thresh = float(min(max(_cfg_get(topology_cfg, "PRUNE_OPACITY_THRESH", 0.08), 0.0), 1.0))
        prune_distance_thresh = float(max(_cfg_get(topology_cfg, "PRUNE_DISTANCE_THRESH", 0.12), 0.0))
        candidate_indices = torch.nonzero(opacity_values < prune_opacity_thresh, as_tuple=False).squeeze(-1)
        if int(candidate_indices.numel()) > 0:
            candidate_means = means[candidate_indices]
            candidate_sparse_dist, _ = chunked_nearest_neighbors(candidate_means, prune_sparse_points, distance_chunk)
            far_mask = candidate_sparse_dist > prune_distance_thresh
            if spawn_count > 0:
                candidate_spawn_dist, _ = chunked_nearest_neighbors(candidate_means, spawn_points, distance_chunk)
                far_mask &= candidate_spawn_dist > coverage_radius
            prune_candidates = candidate_indices[far_mask]
            prune_candidate_dist = candidate_sparse_dist[far_mask]
            if int(prune_candidates.numel()) > 0:
                max_prune = int(max(0, math.floor(float(model.num_gaussians) * float(_cfg_get(topology_cfg, "MAX_PRUNE_RATIO_PER_EVENT", 0.01)))))
                max_prune = min(max_prune, int(prune_candidates.numel()))
                if max_prune > 0:
                    candidate_opacity = opacity_values[prune_candidates]
                    prune_score = prune_candidate_dist * (1.0 - candidate_opacity).clamp_min(0.0)
                    prune_surface_metrics = None
                    if event_info["surface_coupling_active"] and prune_surface_score_weight > 0.0:
                        prune_surface_metrics = compute_vsurface_topology_metrics(
                            model=model,
                            sparse_module=sparse_surface_module,
                            query_points=means[prune_candidates],
                            gaussian_indices=prune_candidates,
                            sparse_points=prune_sparse_points,
                            sparse_track_len=prune_sparse_track_len,
                            sparse_reproj_error=prune_sparse_reproj_error,
                            sparse_brightness_score=prune_sparse_brightness_score,
                            sparse_gradient_score=prune_sparse_gradient_score,
                            normal_score_weight=surface_score_normal_weight,
                            scale_score_weight=surface_score_scale_weight,
                        )
                        if prune_surface_metrics is not None:
                            prune_score = prune_score * (1.0 + prune_surface_score_weight * prune_surface_metrics["surface_score"])
                    keep_pos = torch.topk(prune_score, k=max_prune, largest=True).indices
                    prune_indices = prune_candidates[keep_pos]
                    event_info["prune_distance_mean"] = float(prune_candidate_dist[keep_pos].mean().item())
                    if prune_surface_metrics is not None:
                        event_info["prune_surface_score_mean"] = float(prune_surface_metrics["surface_score"][keep_pos].mean().item())
                        event_info["prune_surface_stable_ratio"] = float(prune_surface_metrics["stable_mask"][keep_pos].float().mean().item())

    event_info["prune_count"] = int(prune_indices.numel())

    if prune_mode == "opacity_decay" and int(prune_indices.numel()) > 0:
        decayed_count = int(prune_indices.numel())
        decay_value = float(max(_cfg_get(topology_cfg, "PRUNE_OPACITY_DECAY", 0.5), 0.0))
        with torch.no_grad():
            current_opacity = torch.sigmoid(model.splats["opacities"].data[prune_indices])
            current_opacity = (current_opacity * decay_value).clamp(1.0e-4, 1.0 - 1.0e-4)
            model.splats["opacities"].data[prune_indices] = torch.logit(current_opacity)
        for module in loss_modules:
            if hasattr(module, "reset_runtime_cache"):
                module.reset_runtime_cache()
        prune_indices = torch.zeros((0,), device=device, dtype=torch.long)
        event_info["prune_mode"] = "opacity_decay"
        event_info["prune_decayed_count"] = decayed_count

    keep_mask = torch.ones((int(model.num_gaussians),), device=device, dtype=torch.bool)
    if int(prune_indices.numel()) > 0:
        keep_mask[prune_indices] = False

    if int(prune_indices.numel()) > 0 or spawn_count > 0:
        mutate_model_splats(model, keep_mask, spawn_state, frozen_geometry_keys)
        optimizers, schedulers = build_optimizers_and_schedulers(
            model=model,
            cfg=cfg,
            lr_map=lr_map,
            frozen_geometry_keys=frozen_geometry_keys,
            total_steps=total_steps,
            existing_optimizers=optimizers,
            current_step=current_step,
        )
        for module in loss_modules:
            if hasattr(module, "reset_runtime_cache"):
                module.reset_runtime_cache()
    else:
        schedulers = {}

    event_info["num_gaussians_after"] = int(model.num_gaussians)
    return optimizers, schedulers, event_info

def train(config_path, device="cuda"):
    meta_cfg = ConfigDict(config_path=config_path)
    seed = int(_cfg_get(meta_cfg, "SEED", 3407))
    set_seed(seed)
    print(f"[Seed] {seed}")
    print(meta_cfg)
    cfg = meta_cfg.MODEL
    augmentation_cfg = _cfg_get(meta_cfg, "AUGMENTATION", None)
    proxy_cfg = _cfg_get(meta_cfg, "PROXY_TARGET", None)
    checkpoint_steps = set(resolve_checkpoint_steps(cfg))
    loss_modules = build_loss_modules(meta_cfg, cfg)
    aux_heads = required_aux_heads(loss_modules, cfg)
    has_reconstruction = "illum" in aux_heads

    output_dir = build_output_dir(config_path, meta_cfg)
    os.makedirs(output_dir, exist_ok=True)
    save_config(os.path.join(output_dir, "config.yaml"), meta_cfg)

    multiview_cfg = _cfg_get(_cfg_get(meta_cfg, "PRIORS", None), "MULTIVIEW", None)
    sparse_cfg = _cfg_get(_cfg_get(meta_cfg, "PRIORS", None), "SPARSE", None)
    topology_cfg = _cfg_get(_cfg_get(meta_cfg, "PRIORS", None), "SPARSE_TOPOLOGY", None)
    train_dataset = Blender(meta_cfg.DATASET, split="train")
    num_train = len(train_dataset._records_keys)

    init_records = []
    for key in train_dataset._records_keys:
        rec = train_dataset._records[key]
        init_records.append(
            {
                "frame_key": rec["frame_key"],
                "transform_matrix": rec["transform_matrix"],
                "file_path": rec["file_path"],
            }
        )
    init_context = {
        "scene_root": str(meta_cfg.DATASET.DATA_PATH),
        "records": init_records,
    }
    (
        colmap_sparse_points,
        colmap_sparse_track_len,
        colmap_sparse_reproj_error,
        colmap_sparse_brightness_score,
        colmap_sparse_gradient_score,
        sparse_load_info,
    ) = load_colmap_sparse_points(meta_cfg, sparse_cfg, device, train_dataset=train_dataset)

    model = Simple3DGS(cfg, train_dataset._data_info, init_context=init_context).to(device)
    warmstart_checkpoint = _cfg_get(cfg, "WARMSTART_CHECKPOINT", None)
    if warmstart_checkpoint:
        load_warmstart_checkpoint(model, warmstart_checkpoint, device)
    print(f"Initialized {model.num_gaussians} Gaussians")
    freeze_geometry = bool(_cfg_get(cfg, "FREEZE_GEOMETRY", False))
    frozen_geometry_keys = {"means", "quats", "scales"} if freeze_geometry else set()
    if freeze_geometry:
        for key in frozen_geometry_keys:
            if key in model.splats:
                model.splats[key].requires_grad_(False)
        print("[Freeze] geometry parameters frozen: means,quats,scales")

    lr_map = {
        "means": cfg.LR_MEANS,
        "quats": cfg.LR_QUATS,
        "scales": cfg.LR_SCALES,
        "opacities": cfg.LR_OPACITIES,
        "sh0": cfg.LR_SH0,
        "shN": cfg.LR_SHN,
        "depth_feat": cfg.LR_SHN,
        "prior_feat": cfg.LR_SHN,
        "illum_feat": float(_cfg_get(cfg, "LR_ILLUM", cfg.LR_SHN)),
        "chroma_feat": float(_cfg_get(cfg, "LR_CHROMA", _cfg_get(cfg, "LR_ILLUM", cfg.LR_SHN))),
    }
    total_steps = int(cfg.TRAIN_TOTAL_STEP)
    optimizers, schedulers = build_optimizers_and_schedulers(
        model=model,
        cfg=cfg,
        lr_map=lr_map,
        frozen_geometry_keys=frozen_geometry_keys,
        total_steps=total_steps,
        current_step=0,
    )

    strategy = None
    strategy_state = None
    topology_enabled = bool(_cfg_get(topology_cfg, "ENABLED", False)) and not freeze_geometry
    if not freeze_geometry and not topology_enabled:
        strategy = gsplat.DefaultStrategy(
            verbose=False,
            refine_start_iter=cfg.DENSIFY_START_STEP,
            refine_stop_iter=cfg.DENSIFY_STOP_STEP,
            refine_every=cfg.DENSIFY_INTERVAL,
            grow_grad2d=cfg.DENSIFY_GRAD_THRESH,
            reset_every=cfg.OPACITY_RESET_INTERVAL,
        )
        strategy_state = strategy.initialize_state(scene_scale=cfg.SCENE_SCALE)
    intrinsics = torch.tensor(
        [
            [float(train_dataset._data_info["fl_x"]), 0.0, float(train_dataset._data_info["cx"])],
            [0.0, float(train_dataset._data_info["fl_y"]), float(train_dataset._data_info["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    train_aug_images = []
    train_proxy_images = []
    train_proxy_global_images = []
    train_proxy_shadow_images = []
    train_proxy_weight_images = []
    sparse_diag_enabled = should_write_sparse_diagnostics(config_path, sparse_cfg)
    sparse_diag_interval = max(1, int(cfg.LOG_INTERVAL_STEP))
    sparse_diag_path = os.path.join(output_dir, "sparse_signal_diagnostics.txt")
    sparse_diag_records = []
    topology_diag_records = []
    if sparse_diag_enabled:
        stage_dir = Path(config_path).resolve().parent.name
        with open(sparse_diag_path, "w", encoding="utf-8") as handle:
            handle.write(f"# Sparse signal diagnostics for config/{stage_dir}\n")
        append_json_txt_record(
            sparse_diag_path,
            {
                "record_type": "header",
                "config_path": str(Path(config_path).resolve()),
                "scene": str(meta_cfg.DATASET.NAME),
                "output_dir": str(Path(output_dir).resolve()),
                "interval": int(sparse_diag_interval),
                "points_path": sparse_load_info.get("points_path"),
                "meta_path": sparse_load_info.get("meta_path"),
                "meta_exists": int(bool(sparse_load_info.get("meta_exists", False))),
                "meta_used": int(bool(sparse_load_info.get("meta_used", False))),
                "num_sparse_points": int(sparse_load_info.get("num_points", 0)),
                "original_num_sparse_points": int(sparse_load_info.get("original_num_points", 0)),
                "filter_enabled": int(sparse_load_info.get("filter_enabled", 0)),
                "filter_mode": str(sparse_load_info.get("filter_mode", "disabled")),
                "filter_kept_ratio": float(sparse_load_info.get("filter_kept_ratio", 1.0)),
                "track_threshold": sparse_load_info.get("track_threshold"),
                "reproj_threshold": sparse_load_info.get("reproj_threshold"),
                "brightness_score_mean": float(sparse_load_info.get("brightness_score_mean", 1.0)),
                "gradient_score_mean": float(sparse_load_info.get("gradient_score_mean", 1.0)),
                "sampling_mode": str(_cfg_get(sparse_cfg, "SAMPLING_MODE", "random")),
                "sparse_mode": str(_cfg_get(sparse_cfg, "MODE", "point_to_barycenter")),
                "hard_ratio": float(_cfg_get(sparse_cfg, "HARD_RATIO", 0.5)),
                "difficulty_score_mode": str(_cfg_get(sparse_cfg, "DIFFICULTY_SCORE", "min_sparse_dist")),
                "random_sample_fallback": int(bool(_cfg_get(sparse_cfg, "RANDOM_SAMPLE_FALLBACK", True))),
                "global_mining_chunk_size": int(_cfg_get(sparse_cfg, "GLOBAL_MINING_CHUNK_SIZE", 4096)),
                "global_mining_refresh_interval": int(_cfg_get(sparse_cfg, "GLOBAL_MINING_REFRESH_INTERVAL", 25)),
                "plane_k": int(_cfg_get(sparse_cfg, "PLANE_K", 8)),
                "tangent_weight": float(_cfg_get(sparse_cfg, "TANGENT_WEIGHT", 0.15)),
                "normal_scale_weight": float(_cfg_get(sparse_cfg, "NORMAL_SCALE_WEIGHT", 0.0)),
                "orientation_enabled": int(bool(_cfg_get(sparse_cfg, "ORIENTATION_ENABLED", False))),
                "orientation_weight": float(_cfg_get(sparse_cfg, "ORIENTATION_WEIGHT", 0.0)),
                "anisotropic_scale_target_enabled": int(bool(_cfg_get(sparse_cfg, "ANISOTROPIC_SCALE_TARGET_ENABLED", False))),
                "anisotropic_scale_target_weight": float(_cfg_get(sparse_cfg, "ANISOTROPIC_SCALE_TARGET_WEIGHT", 0.0)),
                "tangent_scale_ratio": float(_cfg_get(sparse_cfg, "TANGENT_SCALE_RATIO", 0.9)),
                "normal_scale_ratio": float(_cfg_get(sparse_cfg, "NORMAL_SCALE_RATIO", 0.24)),
                "target_tangent_scale_min": float(_cfg_get(sparse_cfg, "TARGET_TANGENT_SCALE_MIN", 0.005)),
                "target_tangent_scale_max": float(_cfg_get(sparse_cfg, "TARGET_TANGENT_SCALE_MAX", 0.05)),
                "target_normal_scale_min": float(_cfg_get(sparse_cfg, "TARGET_NORMAL_SCALE_MIN", 0.002)),
                "target_normal_scale_max": float(_cfg_get(sparse_cfg, "TARGET_NORMAL_SCALE_MAX", 0.02)),
                "target_tangent_quantile": float(_cfg_get(sparse_cfg, "TARGET_TANGENT_QUANTILE", 0.75)),
                "target_normal_quantile": float(_cfg_get(sparse_cfg, "TARGET_NORMAL_QUANTILE", 0.75)),
                "target_tangent_std_blend": float(_cfg_get(sparse_cfg, "TARGET_TANGENT_STD_BLEND", 0.75)),
                "target_tangent_std_cap_ratio": float(_cfg_get(sparse_cfg, "TARGET_TANGENT_STD_CAP_RATIO", 1.6)),
                "target_tangent_local_radius_cap_ratio": float(_cfg_get(sparse_cfg, "TARGET_TANGENT_LOCAL_RADIUS_CAP_RATIO", 0.85)),
                "target_tangent_std_floor_ratio": float(_cfg_get(sparse_cfg, "TARGET_TANGENT_STD_FLOOR_RATIO", 1.0)),
                "tail_start_step": int(_cfg_get(sparse_cfg, "TAIL_START_STEP", _cfg_get(sparse_cfg, "END_STEP", cfg.TRAIN_TOTAL_STEP))),
                "tail_weight_hold_end_step": int(_cfg_get(sparse_cfg, "TAIL_WEIGHT_HOLD_END_STEP", _cfg_get(sparse_cfg, "TAIL_START_STEP", _cfg_get(sparse_cfg, "END_STEP", cfg.TRAIN_TOTAL_STEP)))),
                "tail_light_mode_enabled": int(bool(_cfg_get(sparse_cfg, "TAIL_LIGHT_MODE_ENABLED", False))),
                "tail_sampling_mode": str(_cfg_get(sparse_cfg, "TAIL_SAMPLING_MODE", _cfg_get(sparse_cfg, "SAMPLING_MODE", "random"))),
                "tail_hard_ratio": float(_cfg_get(sparse_cfg, "TAIL_HARD_RATIO", _cfg_get(sparse_cfg, "HARD_RATIO", 0.5))),
                "tail_random_sample_fallback": int(bool(_cfg_get(sparse_cfg, "TAIL_RANDOM_SAMPLE_FALLBACK", _cfg_get(sparse_cfg, "RANDOM_SAMPLE_FALLBACK", True)))),
                "tail_min_plane_confidence": float(_cfg_get(sparse_cfg, "TAIL_MIN_PLANE_CONFIDENCE", 0.35)),
                "tail_global_mining_refresh_interval": int(_cfg_get(sparse_cfg, "TAIL_GLOBAL_MINING_REFRESH_INTERVAL", _cfg_get(sparse_cfg, "GLOBAL_MINING_REFRESH_INTERVAL", 25))),
                "tail_candidate_subset_ratio": float(_cfg_get(sparse_cfg, "TAIL_CANDIDATE_SUBSET_RATIO", 1.0)),
                "tail_candidate_subset_min": int(_cfg_get(sparse_cfg, "TAIL_CANDIDATE_SUBSET_MIN", 0)),
                "tail_candidate_subset_max": int(_cfg_get(sparse_cfg, "TAIL_CANDIDATE_SUBSET_MAX", 0)),
                "tail_stable_sample_ratio_floor": float(_cfg_get(sparse_cfg, "TAIL_STABLE_SAMPLE_RATIO_FLOOR", 0.5)),
                "tail_keep_point_to_plane": int(bool(_cfg_get(sparse_cfg, "TAIL_KEEP_POINT_TO_PLANE", True))),
                "tail_point_to_plane_no_fallback": int(bool(_cfg_get(sparse_cfg, "TAIL_POINT_TO_PLANE_NO_FALLBACK", True))),
                "tail_point_to_plane_min_confidence": float(_cfg_get(sparse_cfg, "TAIL_POINT_TO_PLANE_MIN_CONFIDENCE", 0.35)),
                "tail_point_to_plane_confidence_power": float(_cfg_get(sparse_cfg, "TAIL_POINT_TO_PLANE_CONFIDENCE_POWER", 1.0)),
                "tail_point_to_plane_weight_scale": float(_cfg_get(sparse_cfg, "TAIL_POINT_TO_PLANE_WEIGHT_SCALE", 0.2)),
                "tail_keep_orientation": int(bool(_cfg_get(sparse_cfg, "TAIL_KEEP_ORIENTATION", True))),
                "tail_keep_anisotropic_scale": int(bool(_cfg_get(sparse_cfg, "TAIL_KEEP_ANISOTROPIC_SCALE", True))),
                "tail_keep_normal_scale": int(bool(_cfg_get(sparse_cfg, "TAIL_KEEP_NORMAL_SCALE", True))),
                "tail_difficulty_score": str(_cfg_get(sparse_cfg, "TAIL_DIFFICULTY_SCORE", "stable_surface_mixed")),
                "tail_difficulty_distance_weight": float(_cfg_get(sparse_cfg, "TAIL_DIFFICULTY_DISTANCE_WEIGHT", 0.75)),
                "tail_difficulty_orientation_weight": float(_cfg_get(sparse_cfg, "TAIL_DIFFICULTY_ORIENTATION_WEIGHT", 0.75)),
                "tail_difficulty_scale_weight": float(_cfg_get(sparse_cfg, "TAIL_DIFFICULTY_SCALE_WEIGHT", 1.25)),
                "tail_difficulty_normal_weight": float(_cfg_get(sparse_cfg, "TAIL_DIFFICULTY_NORMAL_WEIGHT", 0.5)),
                "tail_difficulty_confidence_weight": float(_cfg_get(sparse_cfg, "TAIL_DIFFICULTY_CONFIDENCE_WEIGHT", 1.0)),
                "difficulty_distance_weight": float(_cfg_get(sparse_cfg, "DIFFICULTY_DISTANCE_WEIGHT", 1.0)),
                "difficulty_orientation_weight": float(_cfg_get(sparse_cfg, "DIFFICULTY_ORIENTATION_WEIGHT", 0.5)),
                "difficulty_scale_weight": float(_cfg_get(sparse_cfg, "DIFFICULTY_SCALE_WEIGHT", 1.0)),
                "difficulty_normal_weight": float(_cfg_get(sparse_cfg, "DIFFICULTY_NORMAL_WEIGHT", 0.5)),
                "weight_schedule": str(_cfg_get(sparse_cfg, "WEIGHT_SCHEDULE", "constant")),
                "weight_start_scale": float(_cfg_get(sparse_cfg, "WEIGHT_START_SCALE", 1.0)),
                "weight_end_scale": float(_cfg_get(sparse_cfg, "WEIGHT_END_SCALE", 1.0)),
                "weight_decay_end_step": int(_cfg_get(sparse_cfg, "WEIGHT_DECAY_END_STEP", cfg.TRAIN_TOTAL_STEP)),
                "sparse_end_step": int(_cfg_get(sparse_cfg, "END_STEP", cfg.TRAIN_TOTAL_STEP)),
                "lr_means": float(_cfg_get(cfg, "LR_MEANS", 0.0)),
                "lr_means_final": float(_cfg_get(cfg, "LR_MEANS_FINAL", _cfg_get(cfg, "LR_MEANS", 0.0))),
                "lr_means_tail_target": float(_cfg_get(cfg, "LR_MEANS_TAIL_TARGET", _cfg_get(cfg, "LR_MEANS_FINAL", _cfg_get(cfg, "LR_MEANS", 0.0)))),
                "lr_scales": float(_cfg_get(cfg, "LR_SCALES", 0.0)),
                "lr_scales_final": float(_cfg_get(cfg, "LR_SCALES_FINAL", _cfg_get(cfg, "LR_SCALES", 0.0))),
                "lr_scales_tail_target": float(_cfg_get(cfg, "LR_SCALES_TAIL_TARGET", _cfg_get(cfg, "LR_SCALES_FINAL", _cfg_get(cfg, "LR_SCALES", 0.0)))),
                "lr_quats": float(_cfg_get(cfg, "LR_QUATS", 0.0)),
                "lr_quats_final": float(_cfg_get(cfg, "LR_QUATS_FINAL", _cfg_get(cfg, "LR_QUATS", 0.0))),
                "lr_quats_tail_target": float(_cfg_get(cfg, "LR_QUATS_TAIL_TARGET", _cfg_get(cfg, "LR_QUATS_FINAL", _cfg_get(cfg, "LR_QUATS", 0.0)))),
                "lr_tail_start_step": int(_cfg_get(cfg, "LR_TAIL_START_STEP", cfg.TRAIN_TOTAL_STEP)),
                "topology_enabled": int(topology_enabled),
                "topology_interval": int(_cfg_get(topology_cfg, "TOPOLOGY_REFRESH_INTERVAL", _cfg_get(topology_cfg, "INTERVAL", 0))) if topology_enabled else 0,
                "topology_start_step": int(_cfg_get(topology_cfg, "START_STEP", 0)) if topology_enabled else 0,
                "topology_end_step": int(_cfg_get(topology_cfg, "END_STEP", 0)) if topology_enabled else 0,
                "topology_prune_start_step": int(_cfg_get(topology_cfg, "PRUNE_START_STEP", _cfg_get(topology_cfg, "START_STEP", 0))) if topology_enabled else 0,
                "topology_max_spawn_per_event": int(_cfg_get(topology_cfg, "MAX_SPAWN_PER_EVENT", 0)) if topology_enabled else 0,
                "topology_max_prune_ratio_per_event": float(_cfg_get(topology_cfg, "MAX_PRUNE_RATIO_PER_EVENT", 0.0)) if topology_enabled else 0.0,
                "topology_prune_mode": str(_cfg_get(topology_cfg, "PRUNE_MODE", "hard_remove")) if topology_enabled else "disabled",
                "topology_prune_opacity_decay": float(_cfg_get(topology_cfg, "PRUNE_OPACITY_DECAY", 0.5)) if topology_enabled else 0.0,
                "topology_prune_reliable_core_enabled": int(bool(_cfg_get(topology_cfg, "PRUNE_RELIABLE_CORE_ENABLED", False))) if topology_enabled else 0,
                "topology_prune_reliable_core_rule": str(_cfg_get(topology_cfg, "PRUNE_RELIABLE_CORE_RULE", "support_quantile")) if topology_enabled else "disabled",
                "topology_prune_core_support_quantile": float(_cfg_get(topology_cfg, "PRUNE_CORE_SUPPORT_QUANTILE", 0.7)) if topology_enabled else 0.0,
                "topology_prune_core_min_points": int(_cfg_get(topology_cfg, "PRUNE_CORE_MIN_POINTS", 0)) if topology_enabled else 0,
                "adaptive_coverage_enabled": int(bool(_cfg_get(topology_cfg, "ADAPTIVE_COVERAGE_ENABLED", True))) if topology_enabled else 0,
                "coverage_radius_density_scale": float(_cfg_get(topology_cfg, "COVERAGE_RADIUS_DENSITY_SCALE", 2.0)) if topology_enabled else 0.0,
                "coverage_radius_min_scale": float(_cfg_get(topology_cfg, "COVERAGE_RADIUS_MIN_SCALE", 0.5)) if topology_enabled else 0.0,
                "coverage_radius_max_scale": float(_cfg_get(topology_cfg, "COVERAGE_RADIUS_MAX_SCALE", 1.25)) if topology_enabled else 0.0,
                "spawn_feature_init_mode": str(_cfg_get(topology_cfg, "SEED_FEATURE_INIT", "nearest_gaussian_copy")) if topology_enabled else "disabled",
                "spawn_feature_k": int(_cfg_get(topology_cfg, "SPAWN_FEATURE_K", 4)) if topology_enabled else 0,
                "surface_coupling_enabled": int(bool(_cfg_get(topology_cfg, "SURFACE_COUPLING_ENABLED", False))) if topology_enabled else 0,
                "spawn_surface_score_weight": float(_cfg_get(topology_cfg, "SPAWN_SURFACE_SCORE_WEIGHT", 0.0)) if topology_enabled else 0.0,
                "spawn_surface_additive_weight": float(_cfg_get(topology_cfg, "SPAWN_SURFACE_ADDITIVE_WEIGHT", 0.0)) if topology_enabled else 0.0,
                "spawn_surface_confidence_weight": float(_cfg_get(topology_cfg, "SPAWN_SURFACE_CONFIDENCE_WEIGHT", 0.0)) if topology_enabled else 0.0,
                "spawn_surface_slack_weight": float(_cfg_get(topology_cfg, "SPAWN_SURFACE_SLACK_WEIGHT", 0.0)) if topology_enabled else 0.0,
                "spawn_surface_score_clamp": float(_cfg_get(topology_cfg, "SPAWN_SURFACE_SCORE_CLAMP", 0.0)) if topology_enabled else 0.0,
                "spawn_surface_min_coverage_ratio": float(_cfg_get(topology_cfg, "SPAWN_SURFACE_MIN_COVERAGE_RATIO", 0.0)) if topology_enabled else 0.0,
                "spawn_surface_min_confidence": float(_cfg_get(topology_cfg, "SPAWN_SURFACE_MIN_CONFIDENCE", 0.0)) if topology_enabled else 0.0,
                "max_surface_spawn_per_event": int(_cfg_get(topology_cfg, "MAX_SURFACE_SPAWN_PER_EVENT", 0)) if topology_enabled else 0,
                "prune_surface_score_weight": float(_cfg_get(topology_cfg, "PRUNE_SURFACE_SCORE_WEIGHT", 0.0)) if topology_enabled else 0.0,
                "surface_score_normal_weight": float(_cfg_get(topology_cfg, "SURFACE_SCORE_NORMAL_WEIGHT", 0.0)) if topology_enabled else 0.0,
                "surface_score_scale_weight": float(_cfg_get(topology_cfg, "SURFACE_SCORE_SCALE_WEIGHT", 0.0)) if topology_enabled else 0.0,
            },
        )
    pbar = tqdm(range(total_steps))
    for step in pbar:
        current_step = step + 1

        if step > 0 and step % cfg.SH_UPGRADE_INTERVAL == 0:
            model.sh_degree = min(model.sh_degree + 1, model.sh_degree_max)

        data = train_dataset[random.randint(0, num_train - 1)]
        input_image = data["images"].to(device)
        train_batch = prepare_low_light_batch(input_image, augmentation_cfg, training=True, proxy_cfg=proxy_cfg)
        supervision_image = train_batch["supervision"]
        reference_image = train_batch["reference"]
        proxy_target_image = train_batch["proxy_target"]

        camtoworld = data["transforms"].to(device)
        H, W = supervision_image.shape[1], supervision_image.shape[2]
        multiview_active = is_multiview_active(loss_modules, current_step)
        render_outputs = model(camtoworld, H, W, render_heads=aux_heads, render_geom_depth=multiview_active)
        rendered = render_outputs["recon_rgb"]

        neighbor_record = train_dataset.get_pose_neighbor(data["infos"]["frame_key"]) if multiview_active else None
        neighbor_outputs = None
        neighbor_camtoworld = None
        neighbor_distance = 0.0
        if neighbor_record is not None:
            neighbor_camtoworld = neighbor_record["transform_matrix"].to(device)
            neighbor_outputs = model(neighbor_camtoworld, H, W, render_heads=(), render_geom_depth=True, render_rgb=False)
            neighbor_distance = float(
                torch.norm(
                    data["transforms"][:3, 3].to(dtype=torch.float32) - neighbor_record["transform_matrix"][:3, 3].to(dtype=torch.float32)
                ).item()
            )

        context = {
            "step": current_step,
            "rendered": rendered,
            "rgb_base_hwc": render_outputs["rgb"],
            "rgb_model_hwc": render_outputs["rgb"],
            "recon_hwc": render_outputs["recon_rgb"],
            "gaussian_means": model.splats["means"],
            "gaussian_quats": model.splats["quats"],
            "gaussian_scales": model.splats["scales"],
            "gaussian_opacities": model.splats["opacities"],
            "depth_aux": render_outputs["depth_aux"],
            "geom_depth": render_outputs["geom_depth"],
            "alphas": render_outputs["alphas"],
            "prior_aux": render_outputs["prior_aux"],
            "illum_aux": render_outputs["illum_aux"],
            "chroma_aux": render_outputs.get("chroma_aux"),
            "chroma_factor": render_outputs.get("chroma_factor"),
            "chroma_delta": render_outputs.get("chroma_delta"),
            "chroma_mode": render_outputs.get("chroma_mode", "multiplicative"),
            "chroma_scale": render_outputs.get("chroma_scale", 0.10),
            "colmap_sparse_points": colmap_sparse_points,
            "colmap_sparse_track_len": colmap_sparse_track_len,
            "colmap_sparse_reproj_error": colmap_sparse_reproj_error,
            "colmap_sparse_brightness_score": colmap_sparse_brightness_score,
            "colmap_sparse_gradient_score": colmap_sparse_gradient_score,
            "supervision_hwc": supervision_image.permute(1, 2, 0),
            "proxy_shadow_weight_hwc": train_batch["proxy_shadow_weight"],
            "reference_hwc": reference_image.permute(1, 2, 0),
            "proxy_target_hwc": proxy_target_image.permute(1, 2, 0),
            "base_lit_rgb": render_outputs.get("base_lit_rgb"),
            "target_mean": train_batch["target_mean"],
            "data": data,
            "batch": train_batch,
            "camtoworld": camtoworld,
            "neighbor_camtoworld": neighbor_camtoworld,
            "neighbor_geom_depth": None if neighbor_outputs is None else neighbor_outputs["geom_depth"],
            "neighbor_alphas": None if neighbor_outputs is None else neighbor_outputs["alphas"],
            "intrinsics": intrinsics,
            "depth": data["depth"].to(device) if data["depth"] is not None else None,
            "structure": data["structure"].to(device) if data["structure"] is not None else None,
        }
        capture_sparse_diag = sparse_diag_enabled and (current_step % sparse_diag_interval == 0 or current_step == total_steps)
        loss_details = None
        if capture_sparse_diag:
            loss, loss_logs, loss_details = compute_loss_modules(loss_modules, context, return_details=True)
        else:
            loss, loss_logs = compute_loss_modules(loss_modules, context)
        loss_logs["illumination_available"] = float(render_outputs["illum_aux"] is not None)
        loss_logs["proxy_mean"] = float(train_batch["proxy_mean"])
        loss_logs["proxy_stat_mean"] = float(train_batch["proxy_stat_mean"])
        loss_logs["proxy_gain"] = float(train_batch["proxy_scale"])
        loss_logs["proxy_form"] = str(train_batch["proxy_form"])
        loss_logs["proxy_global_mean"] = float(train_batch["proxy_global_mean"])
        loss_logs["proxy_shadow_mean"] = float(train_batch["proxy_shadow_mean"])
        loss_logs["proxy_blend_mean"] = float(train_batch["proxy_blend_mean"])
        loss_logs["proxy_shadow_weight_mean"] = float(train_batch["proxy_shadow_weight_mean"])
        loss_logs["low_mean"] = float(train_batch["low_mean"])
        loss_logs["neighbor_distance"] = float(neighbor_distance)
        loss_logs["chroma_available"] = float(render_outputs.get("chroma_aux") is not None)
        sparse_diag_snapshot = None
        topology_event_info = {
            "topology_event": 0,
            "spawn_count": 0,
            "prune_count": 0,
            "coverage_hole_count": 0,
            "coverage_hole_ratio": 0.0,
            "spawn_distance_mean": 0.0,
            "prune_distance_mean": 0.0,
            "coverage_threshold_mean": 0.0,
            "coverage_excess_mean": 0.0,
            "num_gaussians_before": int(model.num_gaussians),
            "num_gaussians_after": int(model.num_gaussians),
            "prune_mode": str(_cfg_get(topology_cfg, "PRUNE_MODE", "hard_remove")) if topology_enabled else "disabled",
            "prune_active": 0,
            "prune_decayed_count": 0,
            "prune_sparse_core_count": 0,
            "prune_sparse_core_ratio": 0.0,
            "prune_sparse_core_support_mean": 0.0,
            "spawn_feature_init_mode": str(_cfg_get(topology_cfg, "SEED_FEATURE_INIT", "nearest_gaussian_copy")) if topology_enabled else "disabled",
            "surface_coupling_active": 0,
            "spawn_surface_score_mean": 0.0,
            "spawn_surface_stable_ratio": 0.0,
            "spawn_surface_confidence_mean": 0.0,
            "spawn_surface_extra_count": 0,
            "prune_surface_score_mean": 0.0,
            "prune_surface_stable_ratio": 0.0,
        }

        if strategy is not None:
            strategy.step_pre_backward(model.splats, optimizers, strategy_state, step, render_outputs["info"])
        if capture_sparse_diag and loss_details is not None and "sparse_guided" in loss_details:
            means_param = model.splats["means"]
            total_grad = None
            if means_param.requires_grad and loss.requires_grad:
                total_grad = torch.autograd.grad(loss, means_param, retain_graph=True, allow_unused=True)[0]

            sparse_weighted_loss = loss_details["sparse_guided"]["weighted_loss"]
            sparse_effective_weight = float(loss_details["sparse_guided"].get("effective_weight", 0.0))
            sparse_grad = None
            if (
                means_param.requires_grad
                and sparse_effective_weight > 0.0
                and isinstance(sparse_weighted_loss, torch.Tensor)
                and sparse_weighted_loss.requires_grad
            ):
                sparse_grad = torch.autograd.grad(sparse_weighted_loss, means_param, retain_graph=True, allow_unused=True)[0]
            total_grad_norm = grad_norm_value(total_grad)
            sparse_grad_norm = grad_norm_value(sparse_grad)
            sparse_grad_ratio = sparse_grad_norm / max(total_grad_norm, 1.0e-12) if total_grad_norm > 0.0 else 0.0
            sparse_diag_snapshot = {
                "record_type": "snapshot",
                "step": int(current_step),
                "final_step": int(current_step == total_steps),
                "meta_exists": int(bool(sparse_load_info.get("meta_exists", False))),
                "meta_used": int(bool(sparse_load_info.get("meta_used", False))),
                "filter_enabled": int(sparse_load_info.get("filter_enabled", 0)),
                "filter_mode": str(sparse_load_info.get("filter_mode", "disabled")),
                "filter_kept_ratio": float(sparse_load_info.get("filter_kept_ratio", 1.0)),
                "num_gaussians": int(model.num_gaussians),
                "sparse_loss": float(loss_logs.get("sparse_guided", 0.0)),
                "sparse_weight": float(loss_logs.get("sparse_guided_weight", 0.0)),
                "effective_sparse_weight_after_schedule": float(loss_logs.get("sparse_guided_weight", 0.0)),
                "sparse_distance_mean": float(loss_logs.get("sparse_guided_distance_mean", 0.0)),
                "sparse_distance_p50": float(loss_logs.get("sparse_guided_distance_p50", 0.0)),
                "sparse_distance_p90": float(loss_logs.get("sparse_guided_distance_p90", 0.0)),
                "sparse_distance_p99": float(loss_logs.get("sparse_guided_distance_p99", 0.0)),
                "sparse_robust_mean": float(loss_logs.get("sparse_guided_robust_mean", 0.0)),
                "sparse_active_count": float(loss_logs.get("sparse_guided_active_count", 0.0)),
                "sparse_active_ratio": float(loss_logs.get("sparse_guided_active_ratio", 0.0)),
                "sparse_sampled_count": float(loss_logs.get("sparse_guided_sampled", 0.0)),
                "sparse_sampled_ratio": float(loss_logs.get("sparse_guided_sampled_ratio", 0.0)),
                "sparse_quality_p10": float(loss_logs.get("sparse_guided_quality_score_p10", 0.0)),
                "sparse_quality_p50": float(loss_logs.get("sparse_guided_quality_score_p50", 0.0)),
                "sparse_quality_p90": float(loss_logs.get("sparse_guided_quality_score_p90", 0.0)),
                "sparse_density_p10": float(loss_logs.get("sparse_guided_density_score_p10", 0.0)),
                "sparse_density_p50": float(loss_logs.get("sparse_guided_density_score_p50", 0.0)),
                "sparse_density_p90": float(loss_logs.get("sparse_guided_density_score_p90", 0.0)),
                "sparse_support_p10": float(loss_logs.get("sparse_guided_support_score_p10", 0.0)),
                "sparse_support_p50": float(loss_logs.get("sparse_guided_support_score_p50", 0.0)),
                "sparse_support_p90": float(loss_logs.get("sparse_guided_support_score_p90", 0.0)),
                "sampling_mode": loss_logs.get("sparse_guided_sampling_mode", "unknown"),
                "sparse_mode": loss_logs.get("sparse_guided_mode", "unknown"),
                "hard_ratio": float(loss_logs.get("sparse_guided_hard_ratio", 0.0)),
                "difficulty_score_mode": loss_logs.get("sparse_guided_difficulty_score_mode", "unknown"),
                "hard_sample_count": float(loss_logs.get("sparse_guided_hard_sample_count", 0.0)),
                "random_sample_count": float(loss_logs.get("sparse_guided_random_sample_count", 0.0)),
                "candidate_count": float(loss_logs.get("sparse_guided_candidate_count", 0.0)),
                "difficulty_mean": float(loss_logs.get("sparse_guided_difficulty_mean", 0.0)),
                "difficulty_p50": float(loss_logs.get("sparse_guided_difficulty_p50", 0.0)),
                "difficulty_p90": float(loss_logs.get("sparse_guided_difficulty_p90", 0.0)),
                "brightness_p10": float(loss_logs.get("sparse_guided_brightness_score_p10", 0.0)),
                "brightness_p50": float(loss_logs.get("sparse_guided_brightness_score_p50", 0.0)),
                "brightness_p90": float(loss_logs.get("sparse_guided_brightness_score_p90", 0.0)),
                "gradient_p10": float(loss_logs.get("sparse_guided_gradient_score_p10", 0.0)),
                "gradient_p50": float(loss_logs.get("sparse_guided_gradient_score_p50", 0.0)),
                "gradient_p90": float(loss_logs.get("sparse_guided_gradient_score_p90", 0.0)),
                "point_to_plane_fallback_ratio": float(loss_logs.get("sparse_guided_point_to_plane_fallback_ratio", 0.0)),
                "plane_residual_p50": float(loss_logs.get("sparse_guided_plane_residual_p50", 0.0)),
                "plane_residual_p90": float(loss_logs.get("sparse_guided_plane_residual_p90", 0.0)),
                "normal_residual_mean": float(loss_logs.get("sparse_guided_normal_residual_mean", 0.0)),
                "tangent_residual_mean": float(loss_logs.get("sparse_guided_tangent_residual_mean", 0.0)),
                "orientation_loss": float(loss_logs.get("sparse_guided_orientation_loss", 0.0)),
                "orientation_alignment_mean": float(loss_logs.get("sparse_guided_orientation_alignment_mean", 0.0)),
                "orientation_alignment_p50": float(loss_logs.get("sparse_guided_orientation_alignment_p50", 0.0)),
                "orientation_alignment_p90": float(loss_logs.get("sparse_guided_orientation_alignment_p90", 0.0)),
                "anisotropic_scale_target_loss": float(loss_logs.get("sparse_guided_anisotropic_scale_target_loss", 0.0)),
                "normal_scale_loss": float(loss_logs.get("sparse_guided_normal_scale_loss", 0.0)),
                "target_tangent_scale_mean": float(loss_logs.get("sparse_guided_target_tangent_scale_mean", 0.0)),
                "target_tangent_scale_cap_mean": float(loss_logs.get("sparse_guided_target_tangent_scale_cap_mean", 0.0)),
                "target_normal_scale_mean": float(loss_logs.get("sparse_guided_target_normal_scale_mean", 0.0)),
                "gaussian_tangent_scale_mean": float(loss_logs.get("sparse_guided_gaussian_tangent_scale_mean", 0.0)),
                "gaussian_normal_scale_mean": float(loss_logs.get("sparse_guided_gaussian_normal_scale_mean", 0.0)),
                "stable_plane_ratio": float(loss_logs.get("sparse_guided_stable_plane_ratio", 0.0)),
                "loss_residual_mean": float(loss_logs.get("sparse_guided_loss_residual_mean", 0.0)),
                "tail_phase_active": int(float(loss_logs.get("sparse_guided_tail_phase_active", 0.0)) > 0.5),
                "point_to_plane_loss_active": int(float(loss_logs.get("sparse_guided_point_to_plane_loss_active", 0.0)) > 0.5),
                "tail_stable_candidate_ratio": float(loss_logs.get("sparse_guided_tail_stable_candidate_ratio", 0.0)),
                "tail_high_conf_candidate_ratio": float(loss_logs.get("sparse_guided_tail_high_conf_candidate_ratio", 0.0)),
                "tail_sample_high_conf_ratio": float(loss_logs.get("sparse_guided_tail_sample_high_conf_ratio", 0.0)),
                "tail_scan_candidate_ratio": float(loss_logs.get("sparse_guided_tail_scan_candidate_ratio", 0.0)),
                "tail_point_to_plane_effective_ratio": float(loss_logs.get("sparse_guided_tail_point_to_plane_effective_ratio", 0.0)),
                "tail_confidence_mask_ratio": float(loss_logs.get("sparse_guided_tail_confidence_mask_ratio", 0.0)),
                "tail_plane_confidence_mean": float(loss_logs.get("sparse_guided_tail_plane_confidence_mean", 0.0)),
                "lr_means_current": float(optimizers["means"].param_groups[0]["lr"]) if "means" in optimizers and optimizers["means"].param_groups else 0.0,
                "lr_scales_current": float(optimizers["scales"].param_groups[0]["lr"]) if "scales" in optimizers and optimizers["scales"].param_groups else 0.0,
                "lr_quats_current": float(optimizers["quats"].param_groups[0]["lr"]) if "quats" in optimizers and optimizers["quats"].param_groups else 0.0,
                "sparse_grad_norm_means": float(sparse_grad_norm),
                "total_grad_norm_means": float(total_grad_norm),
                "sparse_grad_ratio_means": float(sparse_grad_ratio),
            }
        loss.backward()
        if strategy is not None:
            strategy.step_post_backward(model.splats, optimizers, strategy_state, step, render_outputs["info"], packed=False)

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers.values():
            scheduler.step()
        if topology_enabled and should_run_sparse_topology(current_step, topology_cfg):
            topology_event_info["num_gaussians_before"] = int(model.num_gaussians)
            optimizers, topology_schedulers, topology_record = run_sparse_topology_event(
                model=model,
                sparse_points=colmap_sparse_points,
                sparse_track_len=colmap_sparse_track_len,
                sparse_reproj_error=colmap_sparse_reproj_error,
                sparse_brightness_score=colmap_sparse_brightness_score,
                sparse_gradient_score=colmap_sparse_gradient_score,
                topology_cfg=topology_cfg,
                sparse_cfg=sparse_cfg,
                current_step=current_step,
                lr_map=lr_map,
                optimizers=optimizers,
                cfg=cfg,
                frozen_geometry_keys=frozen_geometry_keys,
                total_steps=total_steps,
                loss_modules=loss_modules,
            )
            if topology_schedulers:
                schedulers = topology_schedulers
            topology_event_info.update({
                "topology_event": int(topology_record.get("topology_event", 0)),
                "spawn_count": int(topology_record.get("spawn_count", 0)),
                "prune_count": int(topology_record.get("prune_count", 0)),
                "coverage_hole_count": int(topology_record.get("coverage_hole_count", 0)),
                "coverage_hole_ratio": float(topology_record.get("coverage_hole_ratio", 0.0)),
                "spawn_distance_mean": float(topology_record.get("spawn_distance_mean", 0.0)),
                "prune_distance_mean": float(topology_record.get("prune_distance_mean", 0.0)),
                "coverage_threshold_mean": float(topology_record.get("coverage_threshold_mean", 0.0)),
                "coverage_excess_mean": float(topology_record.get("coverage_excess_mean", 0.0)),
                "num_gaussians_before": int(topology_record.get("num_gaussians_before", topology_event_info["num_gaussians_before"])),
                "num_gaussians_after": int(topology_record.get("num_gaussians_after", model.num_gaussians)),
                "prune_mode": str(topology_record.get("prune_mode", topology_event_info["prune_mode"])),
                "prune_active": int(topology_record.get("prune_active", topology_event_info["prune_active"])),
                "prune_decayed_count": int(topology_record.get("prune_decayed_count", topology_event_info["prune_decayed_count"])),
                "prune_sparse_core_count": int(topology_record.get("prune_sparse_core_count", topology_event_info["prune_sparse_core_count"])),
                "prune_sparse_core_ratio": float(topology_record.get("prune_sparse_core_ratio", topology_event_info["prune_sparse_core_ratio"])),
                "prune_sparse_core_support_mean": float(topology_record.get("prune_sparse_core_support_mean", topology_event_info["prune_sparse_core_support_mean"])),
                "spawn_feature_init_mode": str(topology_record.get("spawn_feature_init_mode", topology_event_info["spawn_feature_init_mode"])),
                "surface_coupling_active": int(topology_record.get("surface_coupling_active", topology_event_info["surface_coupling_active"])),
                "spawn_surface_score_mean": float(topology_record.get("spawn_surface_score_mean", topology_event_info["spawn_surface_score_mean"])),
                "spawn_surface_stable_ratio": float(topology_record.get("spawn_surface_stable_ratio", topology_event_info["spawn_surface_stable_ratio"])),
                "spawn_surface_confidence_mean": float(topology_record.get("spawn_surface_confidence_mean", topology_event_info["spawn_surface_confidence_mean"])),
                "spawn_surface_extra_count": int(topology_record.get("spawn_surface_extra_count", topology_event_info["spawn_surface_extra_count"])),
                "prune_surface_score_mean": float(topology_record.get("prune_surface_score_mean", topology_event_info["prune_surface_score_mean"])),
                "prune_surface_stable_ratio": float(topology_record.get("prune_surface_stable_ratio", topology_event_info["prune_surface_stable_ratio"])),
            })
            topology_diag_records.append(topology_record)
            if sparse_diag_enabled:
                append_json_txt_record(sparse_diag_path, topology_record)
        topology_event_info["num_gaussians_after"] = int(model.num_gaussians)
        if sparse_diag_snapshot is not None:
            sparse_diag_snapshot.update(topology_event_info)
            sparse_diag_snapshot["num_gaussians"] = int(model.num_gaussians)
        if sparse_diag_snapshot is not None:
            sparse_diag_records.append(sparse_diag_snapshot)
            append_json_txt_record(sparse_diag_path, sparse_diag_snapshot)
        if step % cfg.LOG_INTERVAL_STEP == 0:
            with torch.no_grad():
                base_target = context["supervision_hwc"]
                base_render = context["rgb_base_hwc"] if has_reconstruction else context["rendered"]
                mse_base = ((base_render - base_target) ** 2).mean()
                psnr_base = -10.0 * math.log10(mse_base.clamp_min(1e-10).item())
                if has_reconstruction:
                    recon_target = context["proxy_target_hwc"]
                    mse_recon = ((context["recon_hwc"] - recon_target) ** 2).mean()
                    psnr_recon = -10.0 * math.log10(mse_recon.clamp_min(1e-10).item())
                else:
                    psnr_recon = psnr_base
            postfix = {
                "n_gs": model.num_gaussians,
            }
            if has_reconstruction:
                postfix["rgb_b"] = f"{loss_logs.get('rgb_base', 0.0):.4f}"
                if "luminance_reconstruction" in loss_logs:
                    postfix["rec_y"] = f"{loss_logs.get('luminance_reconstruction', 0.0):.4f}"
                if "chroma_reconstruction" in loss_logs:
                    postfix["rec_c"] = f"{loss_logs.get('chroma_reconstruction', 0.0):.4f}"
                if "reconstruction" in loss_logs:
                    postfix["rec"] = f"{loss_logs.get('reconstruction', 0.0):.4f}"
                postfix["psnr_b"] = f"{psnr_base:.2f}"
                postfix["psnr_r"] = f"{psnr_recon:.2f}"
            else:
                postfix["rgb"] = f"{loss_logs.get('rgb', 0.0):.4f}"

            if loss_logs.get("depth_prior_weight", 0.0) > 0.0:
                postfix["dep"] = f"{loss_logs.get('depth_prior', 0.0):.4f}"
                postfix["dep_w"] = f"{loss_logs.get('depth_prior_weight', 0.0):.3f}"
            if loss_logs.get("multiview_reproj_weight", 0.0) > 0.0:
                postfix["mv"] = f"{loss_logs.get('multiview_reproj', 0.0):.4f}"
                postfix["mv_w"] = f"{loss_logs.get('multiview_reproj_weight', 0.0):.3f}"
                postfix["mv_v"] = f"{loss_logs.get('multiview_reproj_valid_ratio', 0.0):.2f}"
            if loss_logs.get("sparse_guided_weight", 0.0) > 0.0:
                postfix["spr"] = f"{loss_logs.get('sparse_guided', 0.0):.4f}"
            if loss_logs.get("chroma_reg_weight", 0.0) > 0.0:
                postfix["chr"] = f"{loss_logs.get('chroma_reg', 0.0):.4f}"
            if loss_logs.get("structure_prior", 0.0) > 0.0:
                postfix["st"] = f"{loss_logs.get('structure_prior', 0.0):.4f}"
            pbar.set_postfix(**postfix)

        if train_aug_images is not None:
            train_aug_images.append(supervision_image.clamp(0, 1))
            train_proxy_images.append(proxy_target_image.clamp(0, 1))
            train_proxy_global_images.append(train_batch["proxy_global"].clamp(0, 1))
            train_proxy_shadow_images.append(train_batch["proxy_shadow"].clamp(0, 1))
            proxy_weight_image = train_batch["proxy_shadow_weight"].unsqueeze(0).repeat(3, 1, 1).clamp(0, 1)
            train_proxy_weight_images.append(proxy_weight_image)
            if len(train_aug_images) >= 4:
                step_dir = build_step_dir(output_dir, current_step)
                examples_dir = os.path.join(step_dir, "examples")
                os.makedirs(examples_dir, exist_ok=True)
                aug_grid = make_grid(train_aug_images[:4], nrow=2)
                proxy_grid = make_grid(train_proxy_images[:4], nrow=2)
                proxy_global_grid = make_grid(train_proxy_global_images[:4], nrow=2)
                proxy_shadow_grid = make_grid(train_proxy_shadow_images[:4], nrow=2)
                proxy_weight_grid = make_grid(train_proxy_weight_images[:4], nrow=2)
                save_image(aug_grid, os.path.join(examples_dir, "train_aug.jpg"))
                save_image(proxy_grid, os.path.join(examples_dir, "proxy_target.jpg"))
                save_image(proxy_global_grid, os.path.join(examples_dir, "proxy_global.jpg"))
                save_image(proxy_shadow_grid, os.path.join(examples_dir, "proxy_shadow.jpg"))
                save_image(proxy_weight_grid, os.path.join(examples_dir, "proxy_shadow_weight.jpg"))
                train_aug_images = None
                train_proxy_images = None
                train_proxy_global_images = None
                train_proxy_shadow_images = None
                train_proxy_weight_images = None

        if current_step in checkpoint_steps:
            save_checkpoint(model, output_dir, current_step, meta_cfg)

    if sparse_diag_enabled:
        sparse_summary = summarize_sparse_records(sparse_diag_records, sparse_load_info)
        if sparse_summary is not None:
            append_json_txt_record(sparse_diag_path, sparse_summary)
        topology_summary = summarize_topology_records(topology_diag_records, final_num_gaussians=int(model.num_gaussians))
        if topology_summary is not None:
            append_json_txt_record(sparse_diag_path, topology_summary)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", required=True, type=str)
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))

    train(args.config_path)

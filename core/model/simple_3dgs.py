# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from gsplat import rasterization

from core.libs.losses import apply_ycbcr_chroma_residual, apply_ycbcr_luminance_gain


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG")
NPY_EXTS = (".npy",)


class Simple3DGS(nn.Module):
    def __init__(self, model_cfg, data_info, init_context=None):
        super().__init__()
        self.fl_x = data_info["fl_x"]
        self.fl_y = data_info["fl_y"]
        self.cx = data_info["cx"]
        self.cy = data_info["cy"]
        self.bg_color = data_info["bg_color"]
        self.sh_degree_max = model_cfg.SH_DEGREE
        self.sh_degree = 0
        self.chroma_residual_enabled = bool(getattr(model_cfg, "CHROMA_RESIDUAL_ENABLED", False))
        self.chroma_colorspace = str(getattr(model_cfg, "CHROMA_COLORSPACE", "ycbcr")).lower()
        self.chroma_residual_mode = str(getattr(model_cfg, "CHROMA_RESIDUAL_MODE", "multiplicative")).lower()
        self.chroma_scale = float(getattr(model_cfg, "CHROMA_SCALE", 0.10))

        configured_num_points = int(model_cfg.NUM_INIT_POINTS)
        num_sh_bases = (self.sh_degree_max + 1) ** 2

        init_mode = str(getattr(model_cfg, "INIT_MODE", "random")).lower()
        if init_mode == "depth_backproject" and init_context is not None:
            means = self._init_means_from_depth_backproject(model_cfg, data_info, init_context)
        elif init_mode == "hybrid_anchor" and init_context is not None:
            means = self._init_means_from_hybrid_anchor(model_cfg, data_info, init_context)
        elif init_mode == "hybrid_anchor_colmap_sparse" and init_context is not None:
            means = self._init_means_from_hybrid_anchor_colmap_sparse(model_cfg, data_info, init_context)
        elif init_mode in {"depth_backproject", "hybrid_anchor", "hybrid_anchor_colmap_sparse"} and init_context is None:
            print(f"[Init] {init_mode} requested but no init_context provided; fallback to random means.")
            means = self._random_means(configured_num_points)
        else:
            means = self._random_means(configured_num_points)

        num_points = int(means.shape[0])
        if num_points != configured_num_points:
            print(
                f"[Init] actual gaussian count differs from MODEL.NUM_INIT_POINTS: "
                f"configured={configured_num_points}, actual={num_points}"
            )

        quats = torch.zeros(num_points, 4)
        quats[:, 0] = 1.0
        scales = torch.log(torch.full((num_points, 3), 0.005))
        opacities = torch.logit(torch.full((num_points,), 0.1))
        sh0 = torch.zeros(num_points, 1, 3)
        shN = torch.zeros(num_points, num_sh_bases - 1, 3)
        depth_feat = torch.zeros(num_points, 1)
        prior_feat = torch.zeros(num_points, 1)
        illum_feat = torch.zeros(num_points, 1)
        chroma_feat = torch.zeros(num_points, 2)

        self.splats = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "quats": nn.Parameter(quats),
                "scales": nn.Parameter(scales),
                "opacities": nn.Parameter(opacities),
                "sh0": nn.Parameter(sh0),
                "shN": nn.Parameter(shN),
                "depth_feat": nn.Parameter(depth_feat),
                "prior_feat": nn.Parameter(prior_feat),
                "illum_feat": nn.Parameter(illum_feat),
                "chroma_feat": nn.Parameter(chroma_feat),
            }
        )

    @property
    def num_gaussians(self):
        return self.splats["means"].shape[0]

    def _random_means(self, num_points):
        return (torch.rand(num_points, 3) - 0.5) * 10.0

    def _random_means_numpy(self, num_points):
        return self._random_means(num_points).cpu().numpy().astype(np.float32)

    def _resolve_npy_depth_path(self, depth_root, frame_key):
        candidate = depth_root / f"{frame_key}.npy"
        if candidate.exists():
            return candidate
        return None

    def _resolve_depth_path(self, depth_root, frame_key):
        for ext in NPY_EXTS + IMAGE_EXTS:
            candidate = depth_root / f"{frame_key}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _load_depth(self, depth_path, width, height):
        if depth_path.suffix.lower() == ".npy":
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth_raw = np.asarray(Image.open(depth_path))
            if depth_raw.ndim == 3:
                depth_raw = depth_raw[..., 0]
            depth = depth_raw.astype(np.float32)
            if np.issubdtype(depth_raw.dtype, np.integer):
                max_value = float(np.iinfo(depth_raw.dtype).max)
                if max_value > 0.0:
                    depth = depth / max_value
        if depth.shape != (height, width):
            depth = np.asarray(
                Image.fromarray(depth.astype(np.float32), mode="F").resize(
                    (width, height), resample=Image.BILINEAR
                ),
                dtype=np.float32,
            )
        return depth.astype(np.float32)

    def _load_gray_image(self, image_path, width, height):
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        if image.shape[:2] != (height, width):
            image = np.asarray(
                Image.fromarray(np.asarray(np.clip(image, 0.0, 1.0) * 255.0, dtype=np.uint8), mode="RGB").resize(
                    (width, height), resample=Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        valid = np.isfinite(gray)
        if np.any(valid):
            high = float(np.percentile(gray[valid], 95.0))
            if np.isfinite(high) and high > 1e-6:
                gray = np.clip(gray / high, 0.0, 1.0)
        return gray.astype(np.float32)

    def _normalize_depth_for_init(self, depth, model_cfg, depth_eps):
        mode = str(getattr(model_cfg, "INIT_DEPTH_NORMALIZATION", "none")).lower()
        if mode == "none":
            return depth
        if mode != "per_frame_robust":
            raise RuntimeError(f"Unsupported INIT_DEPTH_NORMALIZATION mode: {mode}")

        valid = np.isfinite(depth) & (depth > depth_eps)
        if not np.any(valid):
            return depth

        low_q = float(getattr(model_cfg, "INIT_DEPTH_NORM_LOW_Q", 1.0))
        high_q = float(getattr(model_cfg, "INIT_DEPTH_NORM_HIGH_Q", 99.0))
        low = float(np.percentile(depth[valid], low_q))
        high = float(np.percentile(depth[valid], high_q))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-8:
            return depth

        normalized = np.zeros_like(depth, dtype=np.float32)
        normalized[valid] = np.clip((depth[valid] - low) / (high - low), 0.0, 1.0)
        return normalized

    def _voxel_downsample_points(self, points, voxel_size):
        if voxel_size <= 0.0 or points.shape[0] == 0:
            return points
        voxel_indices = np.floor(points / voxel_size).astype(np.int64)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        return points[np.sort(unique_indices)]

    def _select_voxel_indices(self, points, voxel_size, quality=None):
        if points.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        if voxel_size <= 0.0:
            return np.arange(points.shape[0], dtype=np.int64)

        voxel_indices = np.floor(points / voxel_size).astype(np.int64)
        if quality is None:
            _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
            return np.sort(unique_indices).astype(np.int64)

        quality = np.asarray(quality, dtype=np.float32)
        order = np.lexsort(
            (
                np.arange(points.shape[0], dtype=np.int64),
                -quality.astype(np.float64),
                voxel_indices[:, 2],
                voxel_indices[:, 1],
                voxel_indices[:, 0],
            )
        )
        ordered_voxels = voxel_indices[order]
        keep_mask = np.ones(order.shape[0], dtype=bool)
        keep_mask[1:] = np.any(ordered_voxels[1:] != ordered_voxels[:-1], axis=1)
        return order[keep_mask].astype(np.int64)

    def _select_quality_uniform_anchor_indices(self, frame_ids, quality, anchor_target, frame_count):
        if anchor_target <= 0 or frame_ids.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)

        valid_frame_mask = np.bincount(frame_ids, minlength=frame_count) > 0
        valid_frames = np.nonzero(valid_frame_mask)[0]
        if valid_frames.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)

        selected_parts = []
        taken = np.zeros(frame_ids.shape[0], dtype=bool)
        base_quota = anchor_target // valid_frames.shape[0]

        if base_quota > 0:
            for frame_id in valid_frames:
                frame_indices = np.nonzero(frame_ids == frame_id)[0]
                frame_order = frame_indices[np.argsort(-quality[frame_indices], kind="mergesort")]
                take = frame_order[: min(base_quota, frame_order.shape[0])]
                if take.shape[0] > 0:
                    selected_parts.append(take)
                    taken[take] = True

        selected_count = int(sum(part.shape[0] for part in selected_parts))
        remaining = max(0, anchor_target - selected_count)
        if remaining > 0:
            remaining_indices = np.nonzero(~taken)[0]
            global_order = remaining_indices[np.argsort(-quality[remaining_indices], kind="mergesort")]
            take = global_order[:remaining]
            if take.shape[0] > 0:
                selected_parts.append(take)

        if not selected_parts:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(selected_parts).astype(np.int64)
    def _sample_bilinear(self, image, u, v):
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

    def _backproject_sampled_depth(self, sampled_depth, grid_x, grid_y, valid_mask, c2w, intrinsics, near, far, invert_depth):
        fx, fy, cx, cy = intrinsics
        d = sampled_depth[valid_mask]
        if invert_depth:
            d = 1.0 - d
        z = near + d * (far - near)
        u = grid_x[valid_mask]
        v = grid_y[valid_mask]
        x_cam = (u - cx) / fx * z
        y_cam = -(v - cy) / fy * z
        z_cam = -z
        points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=1)
        points_world = (c2w @ points_cam.T).T[:, :3]
        return points_world.astype(np.float32)

    def _project_points(self, points_world, c2w_tgt, intrinsics, width, height):
        w2c = np.linalg.inv(c2w_tgt)
        points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
        pts_cam = (w2c @ points_h.T).T
        d = -pts_cam[:, 2]
        valid = np.isfinite(d) & (d > 1e-6)
        if not np.any(valid):
            return None, None, None, None
        pts_cam = pts_cam[valid]
        d = d[valid]
        valid_idx = np.nonzero(valid)[0]
        fx, fy, cx, cy = intrinsics
        u = fx * (pts_cam[:, 0] / d) + cx
        v = cy - fy * (pts_cam[:, 1] / d)
        inside = (u >= 0.0) & (u <= (width - 1.0)) & (v >= 0.0) & (v <= (height - 1.0))
        if not np.any(inside):
            return None, None, None, None
        return u[inside], v[inside], d[inside], valid_idx[inside]

    def _collect_backproject_points(self, model_cfg, data_info, init_context):
        records = list(init_context.get("records", []))
        scene_root = init_context.get("scene_root", None)
        if scene_root is None:
            raise RuntimeError("Initialization requires init_context['scene_root']")
        if not records:
            raise RuntimeError("Initialization requires non-empty train records")

        depth_rel_dir = str(getattr(model_cfg, "INIT_DEPTH_DIR", "auxiliaries/depth"))
        depth_root = Path(depth_rel_dir) if os.path.isabs(depth_rel_dir) else Path(scene_root) / depth_rel_dir
        if not depth_root.exists():
            raise RuntimeError(f"Depth init directory does not exist: {depth_root}")

        near = float(getattr(model_cfg, "INIT_BACKPROJECT_NEAR", 0.2))
        far = float(getattr(model_cfg, "INIT_BACKPROJECT_FAR", 6.0))
        stride = int(max(1, int(getattr(model_cfg, "INIT_BACKPROJECT_SAMPLE_STRIDE", 4))))
        depth_eps = float(getattr(model_cfg, "INIT_BACKPROJECT_DEPTH_EPS", 1e-3))
        invert_depth = bool(getattr(model_cfg, "INIT_BACKPROJECT_DEPTH_INVERT", False))

        width = int(data_info["img_w"])
        height = int(data_info["img_h"])
        intrinsics = (float(self.fl_x), float(self.fl_y), float(self.cx), float(self.cy))

        xs = np.arange(0, width, stride, dtype=np.float32)
        ys = np.arange(0, height, stride, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)

        points_world_all = []
        used_frames = 0
        dropped_frames = 0

        for rec in records:
            frame_key = rec.get("frame_key")
            transform_matrix = rec.get("transform_matrix")
            if frame_key is None or transform_matrix is None:
                raise RuntimeError("Each init record must include frame_key and transform_matrix")
            depth_path = self._resolve_depth_path(depth_root, frame_key)
            if depth_path is None:
                raise RuntimeError(f"Missing init depth for train frame '{frame_key}' in {depth_root}")
            depth = self._load_depth(depth_path, width, height)
            depth = self._normalize_depth_for_init(depth, model_cfg, depth_eps)
            sampled = depth[::stride, ::stride]
            valid = np.isfinite(sampled) & (sampled > depth_eps)
            if not np.any(valid):
                dropped_frames += 1
                continue
            c2w = np.eye(4, dtype=np.float32)
            if torch.is_tensor(transform_matrix):
                c2w[:3, :] = transform_matrix.detach().cpu().numpy().astype(np.float32)
            else:
                c2w[:3, :] = np.asarray(transform_matrix, dtype=np.float32)
            points_world = self._backproject_sampled_depth(
                sampled_depth=sampled,
                grid_x=grid_x,
                grid_y=grid_y,
                valid_mask=valid,
                c2w=c2w,
                intrinsics=intrinsics,
                near=near,
                far=far,
                invert_depth=invert_depth,
            )
            points_world_all.append(points_world)
            used_frames += 1

        if not points_world_all:
            raise RuntimeError("Depth back-projection produced no valid 3D points.")

        points_world_all = np.concatenate(points_world_all, axis=0)
        return {
            "points": points_world_all,
            "raw_total_points": int(points_world_all.shape[0]),
            "used_frames": used_frames,
            "dropped_frames": dropped_frames,
            "depth_root": depth_root,
        }


    def _load_colmap_sparse_points(self, model_cfg, init_context):
        scene_root = init_context.get("scene_root", None)
        if scene_root is None:
            raise RuntimeError("COLMAP sparse init requires init_context['scene_root']")

        colmap_rel_dir = str(getattr(model_cfg, "INIT_COLMAP_DIR", "auxiliaries/colmap_sparse"))
        colmap_root = Path(colmap_rel_dir) if os.path.isabs(colmap_rel_dir) else Path(scene_root) / colmap_rel_dir
        points_path = colmap_root / "points.npy"
        if not points_path.exists():
            raise RuntimeError(f"Missing COLMAP sparse points file: {points_path}")

        points = np.load(points_path)
        if points.ndim != 2 or points.shape[1] != 3:
            raise RuntimeError(f"COLMAP sparse points must have shape [N, 3], got {tuple(points.shape)} from {points_path}")
        points = points.astype(np.float32)
        raw_points = int(points.shape[0])

        voxel_size = float(getattr(model_cfg, "INIT_COLMAP_VOXEL_SIZE", 0.02))
        points = self._voxel_downsample_points(points, voxel_size=voxel_size)
        voxel_points = int(points.shape[0])
        min_points = int(getattr(model_cfg, "INIT_COLMAP_MIN_POINTS", 1000))
        if voxel_points < min_points:
            raise RuntimeError(
                f"COLMAP sparse points after voxel downsample ({voxel_points}) below INIT_COLMAP_MIN_POINTS ({min_points})"
            )

        return {
            "points": points.astype(np.float32),
            "raw_points": raw_points,
            "voxel_points": voxel_points,
            "colmap_root": colmap_root,
            "points_path": points_path,
            "voxel_size": voxel_size,
        }

    def _sample_colmap_sparse_points(self, model_cfg, init_context, colmap_target):
        if colmap_target <= 0:
            return {
                "points": np.zeros((0, 3), dtype=np.float32),
                "raw_points": 0,
                "voxel_points": 0,
                "used_points": 0,
                "colmap_root": None,
                "points_path": None,
                "voxel_size": float(getattr(model_cfg, "INIT_COLMAP_VOXEL_SIZE", 0.02)),
            }

        colmap_data = self._load_colmap_sparse_points(model_cfg, init_context)
        sparse_points = colmap_data["points"]
        colmap_used = min(int(colmap_target), int(sparse_points.shape[0]))
        if colmap_used > 0:
            choice = np.random.choice(sparse_points.shape[0], size=colmap_used, replace=False)
            sampled_points = sparse_points[choice].astype(np.float32)
        else:
            sampled_points = np.zeros((0, 3), dtype=np.float32)

        return {
            "points": sampled_points,
            "raw_points": int(colmap_data["raw_points"]),
            "voxel_points": int(colmap_data["voxel_points"]),
            "used_points": int(colmap_used),
            "colmap_root": colmap_data["colmap_root"],
            "points_path": colmap_data["points_path"],
            "voxel_size": float(colmap_data["voxel_size"]),
        }

    def _collect_anchor_points(self, model_cfg, data_info, init_context):
        records = list(init_context.get("records", []))
        scene_root = init_context.get("scene_root", None)
        if scene_root is None or not records:
            raise RuntimeError("Hybrid anchor init requires scene_root and train records")

        depth_rel_dir = str(getattr(model_cfg, "INIT_DEPTH_DIR", "auxiliaries/depth"))
        depth_root = Path(depth_rel_dir) if os.path.isabs(depth_rel_dir) else Path(scene_root) / depth_rel_dir
        if not depth_root.exists():
            raise RuntimeError(f"Depth init directory does not exist: {depth_root}")

        near = float(getattr(model_cfg, "INIT_BACKPROJECT_NEAR", 0.2))
        far = float(getattr(model_cfg, "INIT_BACKPROJECT_FAR", 6.0))
        stride = int(max(1, int(getattr(model_cfg, "INIT_BACKPROJECT_SAMPLE_STRIDE", 4))))
        depth_eps = float(getattr(model_cfg, "INIT_BACKPROJECT_DEPTH_EPS", 1e-3))
        invert_depth = bool(getattr(model_cfg, "INIT_BACKPROJECT_DEPTH_INVERT", False))

        brightness_quantile = float(getattr(model_cfg, "INIT_ANCHOR_BRIGHTNESS_QUANTILE", 0.0))
        max_depth_grad = float(getattr(model_cfg, "INIT_ANCHOR_MAX_DEPTH_GRAD", 0.08))

        width = int(data_info["img_w"])
        height = int(data_info["img_h"])
        intrinsics = (float(self.fl_x), float(self.fl_y), float(self.cx), float(self.cy))

        xs = np.arange(0, width, stride, dtype=np.float32)
        ys = np.arange(0, height, stride, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)

        normalized_depths = {}
        c2ws = {}
        for rec in records:
            frame_key = rec["frame_key"]
            depth_path = self._resolve_depth_path(depth_root, frame_key)
            if depth_path is None:
                raise RuntimeError(f"Missing init depth for train frame '{frame_key}' in {depth_root}")
            depth = self._load_depth(depth_path, width, height)
            normalized_depths[frame_key] = self._normalize_depth_for_init(depth, model_cfg, depth_eps)
            c2w = np.eye(4, dtype=np.float32)
            transform_matrix = rec["transform_matrix"]
            if torch.is_tensor(transform_matrix):
                c2w[:3, :] = transform_matrix.detach().cpu().numpy().astype(np.float32)
            else:
                c2w[:3, :] = np.asarray(transform_matrix, dtype=np.float32)
            c2ws[frame_key] = c2w

        anchor_points = []
        frame_ids = []
        gray_values = []
        depth_grads = []
        qualities = []
        anchor_frames = 0

        for idx, rec in enumerate(records):
            frame_key = rec["frame_key"]
            image_path = rec.get("file_path", None)
            if image_path is None:
                continue

            depth = normalized_depths[frame_key]
            gray = self._load_gray_image(image_path, width, height)
            sampled_depth = depth[::stride, ::stride]
            sampled_gray = gray[::stride, ::stride]
            depth_grad_y, depth_grad_x = np.gradient(sampled_depth)
            depth_grad = np.abs(depth_grad_x) + np.abs(depth_grad_y)

            candidate_mask = np.isfinite(sampled_depth) & (sampled_depth > depth_eps)
            candidate_mask &= depth_grad <= max_depth_grad

            brightness_threshold = 0.0
            if brightness_quantile > 0.0:
                valid_gray = sampled_gray[np.isfinite(sampled_gray)]
                if valid_gray.size > 0:
                    brightness_threshold = float(np.percentile(valid_gray, brightness_quantile * 100.0))
                    candidate_mask &= np.isfinite(sampled_gray)
                    candidate_mask &= sampled_gray >= brightness_threshold

            if not np.any(candidate_mask):
                continue

            candidate_gray = np.clip(sampled_gray[candidate_mask], 0.0, 1.0).astype(np.float32)
            candidate_depth_grad = depth_grad[candidate_mask].astype(np.float32)
            depth_score = 1.0 - np.clip(candidate_depth_grad / max(max_depth_grad, 1e-6), 0.0, 1.0)
            gray_score = np.clip(
                (candidate_gray - brightness_threshold) / max(1.0 - brightness_threshold, 1e-6),
                0.0,
                1.0,
            )
            quality = (0.7 * depth_score + 0.3 * gray_score).astype(np.float32)

            points_world = self._backproject_sampled_depth(
                sampled_depth=sampled_depth,
                grid_x=grid_x,
                grid_y=grid_y,
                valid_mask=candidate_mask,
                c2w=c2ws[frame_key],
                intrinsics=intrinsics,
                near=near,
                far=far,
                invert_depth=invert_depth,
            )
            if points_world.shape[0] == 0:
                continue

            anchor_points.append(points_world)
            frame_ids.append(np.full(points_world.shape[0], idx, dtype=np.int32))
            gray_values.append(candidate_gray)
            depth_grads.append(candidate_depth_grad)
            qualities.append(quality)
            anchor_frames += 1

        if not anchor_points:
            return {
                "points": np.zeros((0, 3), dtype=np.float32),
                "frame_ids": np.zeros((0,), dtype=np.int32),
                "gray": np.zeros((0,), dtype=np.float32),
                "depth_grad": np.zeros((0,), dtype=np.float32),
                "quality": np.zeros((0,), dtype=np.float32),
                "anchor_frames": 0,
                "frame_count": len(records),
                "depth_root": depth_root,
            }

        return {
            "points": np.concatenate(anchor_points, axis=0).astype(np.float32),
            "frame_ids": np.concatenate(frame_ids, axis=0).astype(np.int32),
            "gray": np.concatenate(gray_values, axis=0).astype(np.float32),
            "depth_grad": np.concatenate(depth_grads, axis=0).astype(np.float32),
            "quality": np.concatenate(qualities, axis=0).astype(np.float32),
            "anchor_frames": anchor_frames,
            "frame_count": len(records),
            "depth_root": depth_root,
        }

    def _init_means_from_depth_backproject(self, model_cfg, data_info, init_context):
        num_points = int(model_cfg.NUM_INIT_POINTS)
        min_valid_points = int(getattr(model_cfg, "INIT_BACKPROJECT_MIN_VALID_POINTS", num_points))
        voxel_enabled = bool(getattr(model_cfg, "INIT_VOXEL_DOWNSAMPLE", True))
        voxel_size = float(getattr(model_cfg, "INIT_VOXEL_SIZE", 0.01))

        collected = self._collect_backproject_points(model_cfg, data_info, init_context)
        points_world_all = collected["points"]
        raw_total_points = collected["raw_total_points"]
        if voxel_enabled:
            points_world_all = self._voxel_downsample_points(points_world_all, voxel_size=voxel_size)

        total_points = int(points_world_all.shape[0])
        if total_points < min_valid_points:
            raise RuntimeError(
                f"Depth back-projection points after voxel downsample ({total_points}) below "
                f"INIT_BACKPROJECT_MIN_VALID_POINTS ({min_valid_points})."
            )
        if total_points < num_points:
            raise RuntimeError(
                f"Depth back-projection points after voxel downsample ({total_points}) below "
                f"NUM_INIT_POINTS ({num_points}). Reduce INIT_VOXEL_SIZE or NUM_INIT_POINTS."
            )

        choice = np.random.choice(total_points, size=num_points, replace=False)
        sampled_points = points_world_all[choice]
        print(
            "[Init] depth_backproject: "
            f"depth_dir={collected['depth_root']}, used_frames={collected['used_frames']}, dropped_frames={collected['dropped_frames']}, "
            f"raw_points={raw_total_points}, voxel_points={total_points}, sampled_points={num_points}, "
            f"normalization={getattr(model_cfg, 'INIT_DEPTH_NORMALIZATION', 'none')}, "
            f"voxel_enabled={voxel_enabled}, voxel_size={voxel_size}"
        )
        return torch.from_numpy(sampled_points).float()

    def _init_means_from_hybrid_anchor(self, model_cfg, data_info, init_context):
        num_points = int(model_cfg.NUM_INIT_POINTS)
        anchor_ratio = float(np.clip(getattr(model_cfg, "INIT_ANCHOR_RATIO", 0.1), 0.0, 1.0))
        anchor_target = int(round(num_points * anchor_ratio))
        voxel_enabled = bool(getattr(model_cfg, "INIT_VOXEL_DOWNSAMPLE", True))
        voxel_size = float(getattr(model_cfg, "INIT_VOXEL_SIZE", 0.03))
        selection_mode = str(getattr(model_cfg, "INIT_ANCHOR_SELECTION", "quality_uniform")).lower()

        anchor_data = self._collect_anchor_points(model_cfg, data_info, init_context)
        anchor_points = anchor_data["points"]
        frame_ids = anchor_data["frame_ids"]
        quality = anchor_data["quality"]
        raw_anchor_points = int(anchor_points.shape[0])

        if selection_mode == "random":
            dedup_indices = self._select_voxel_indices(anchor_points, voxel_size, quality=None) if voxel_enabled else np.arange(raw_anchor_points, dtype=np.int64)
        elif selection_mode == "quality_uniform":
            dedup_indices = self._select_voxel_indices(anchor_points, voxel_size, quality=quality) if voxel_enabled else np.arange(raw_anchor_points, dtype=np.int64)
        else:
            raise RuntimeError(f"Unsupported INIT_ANCHOR_SELECTION mode: {selection_mode}")

        anchor_points = anchor_points[dedup_indices]
        frame_ids = frame_ids[dedup_indices]
        quality = quality[dedup_indices]
        voxel_anchor_points = int(anchor_points.shape[0])

        if selection_mode == "random":
            anchor_count = min(anchor_target, voxel_anchor_points)
            if anchor_count > 0:
                selected_indices = np.random.choice(voxel_anchor_points, size=anchor_count, replace=False).astype(np.int64)
            else:
                selected_indices = np.zeros((0,), dtype=np.int64)
        else:
            selected_indices = self._select_quality_uniform_anchor_indices(
                frame_ids=frame_ids,
                quality=quality,
                anchor_target=anchor_target,
                frame_count=int(anchor_data["frame_count"]),
            )
            anchor_count = int(selected_indices.shape[0])

        if anchor_count > 0:
            sampled_anchor_points = anchor_points[selected_indices]
            selected_quality = quality[selected_indices]
            selected_frame_counts = np.bincount(frame_ids[selected_indices], minlength=int(anchor_data["frame_count"]))
        else:
            sampled_anchor_points = np.zeros((0, 3), dtype=np.float32)
            selected_quality = np.zeros((0,), dtype=np.float32)
            selected_frame_counts = np.zeros((int(anchor_data["frame_count"]),), dtype=np.int64)

        random_count = num_points - anchor_count
        random_points = self._random_means(random_count).cpu().numpy().astype(np.float32)
        means = np.concatenate([random_points, sampled_anchor_points], axis=0)
        np.random.shuffle(means)

        valid_frame_counts = np.bincount(frame_ids, minlength=int(anchor_data["frame_count"])) if frame_ids.shape[0] > 0 else np.zeros((int(anchor_data["frame_count"]),), dtype=np.int64)
        valid_anchor_frames = int(np.count_nonzero(valid_frame_counts))
        selected_valid_counts = selected_frame_counts[valid_frame_counts > 0]
        selected_frame_min = int(selected_valid_counts.min()) if selected_valid_counts.size > 0 else 0
        selected_frame_median = float(np.median(selected_valid_counts)) if selected_valid_counts.size > 0 else 0.0
        selected_frame_max = int(selected_valid_counts.max()) if selected_valid_counts.size > 0 else 0
        quality_mean_all = float(quality.mean()) if quality.shape[0] > 0 else 0.0
        quality_mean_selected = float(selected_quality.mean()) if selected_quality.shape[0] > 0 else 0.0

        print(
            "[Init] hybrid_anchor: "
            f"depth_dir={anchor_data['depth_root']}, anchor_frames={anchor_data['anchor_frames']}, "
            f"raw_anchor_points={raw_anchor_points}, voxel_anchor_points={voxel_anchor_points}, "
            f"anchor_target={anchor_target}, anchor_used={anchor_count}, random_used={random_count}, "
            f"selection_mode={selection_mode}, valid_anchor_frames={valid_anchor_frames}, "
            f"selected_frame_min={selected_frame_min}, selected_frame_median={selected_frame_median:.1f}, "
            f"selected_frame_max={selected_frame_max}, quality_mean_selected={quality_mean_selected:.4f}, "
            f"quality_mean_all={quality_mean_all:.4f}, voxel_enabled={voxel_enabled}, voxel_size={voxel_size}, "
            f"max_depth_grad={getattr(model_cfg, 'INIT_ANCHOR_MAX_DEPTH_GRAD', 0.08)}, "
            f"brightness_quantile={getattr(model_cfg, 'INIT_ANCHOR_BRIGHTNESS_QUANTILE', 0.0)}"
        )
        return torch.from_numpy(means).float()


    def _init_means_from_hybrid_anchor_colmap_sparse(self, model_cfg, data_info, init_context):
        num_points = int(model_cfg.NUM_INIT_POINTS)
        anchor_ratio = float(np.clip(getattr(model_cfg, "INIT_ANCHOR_RATIO", 0.1), 0.0, 1.0))
        colmap_ratio = float(np.clip(getattr(model_cfg, "INIT_COLMAP_RATIO", 0.20), 0.0, 1.0))
        anchor_target = int(round(num_points * anchor_ratio))
        colmap_use_all = bool(getattr(model_cfg, "INIT_COLMAP_USE_ALL", False))
        fixed_random_points = int(max(0, int(getattr(model_cfg, "INIT_RANDOM_POINTS", -1))))
        voxel_enabled = bool(getattr(model_cfg, "INIT_VOXEL_DOWNSAMPLE", True))
        voxel_size = float(getattr(model_cfg, "INIT_VOXEL_SIZE", 0.03))
        selection_mode = str(getattr(model_cfg, "INIT_ANCHOR_SELECTION", "quality_uniform")).lower()

        if anchor_target > 0:
            anchor_data = self._collect_anchor_points(model_cfg, data_info, init_context)
            anchor_points = anchor_data["points"]
            frame_ids = anchor_data["frame_ids"]
            quality = anchor_data["quality"]
            raw_anchor_points = int(anchor_points.shape[0])

            if selection_mode == "random":
                dedup_indices = self._select_voxel_indices(anchor_points, voxel_size, quality=None) if voxel_enabled else np.arange(raw_anchor_points, dtype=np.int64)
            elif selection_mode == "quality_uniform":
                dedup_indices = self._select_voxel_indices(anchor_points, voxel_size, quality=quality) if voxel_enabled else np.arange(raw_anchor_points, dtype=np.int64)
            else:
                raise RuntimeError(f"Unsupported INIT_ANCHOR_SELECTION mode: {selection_mode}")

            anchor_points = anchor_points[dedup_indices]
            frame_ids = frame_ids[dedup_indices]
            quality = quality[dedup_indices]
            voxel_anchor_points = int(anchor_points.shape[0])

            if selection_mode == "random":
                anchor_count = min(anchor_target, voxel_anchor_points)
                if anchor_count > 0:
                    selected_indices = np.random.choice(voxel_anchor_points, size=anchor_count, replace=False).astype(np.int64)
                else:
                    selected_indices = np.zeros((0,), dtype=np.int64)
            else:
                selected_indices = self._select_quality_uniform_anchor_indices(
                    frame_ids=frame_ids,
                    quality=quality,
                    anchor_target=anchor_target,
                    frame_count=int(anchor_data["frame_count"]),
                )
                anchor_count = int(selected_indices.shape[0])

            if anchor_count > 0:
                sampled_anchor_points = anchor_points[selected_indices]
                selected_quality = quality[selected_indices]
                selected_frame_counts = np.bincount(frame_ids[selected_indices], minlength=int(anchor_data["frame_count"]))
            else:
                sampled_anchor_points = np.zeros((0, 3), dtype=np.float32)
                selected_quality = np.zeros((0,), dtype=np.float32)
                selected_frame_counts = np.zeros((int(anchor_data["frame_count"]),), dtype=np.int64)
        else:
            anchor_data = None
            sampled_anchor_points = np.zeros((0, 3), dtype=np.float32)
            selected_quality = np.zeros((0,), dtype=np.float32)
            selected_frame_counts = np.zeros((0,), dtype=np.int64)
            frame_ids = np.zeros((0,), dtype=np.int32)
            quality = np.zeros((0,), dtype=np.float32)
            raw_anchor_points = 0
            voxel_anchor_points = 0
            anchor_count = 0

        if colmap_use_all:
            colmap_data = self._load_colmap_sparse_points(model_cfg, init_context)
            colmap_points = colmap_data["points"].astype(np.float32)
            colmap_used = int(colmap_points.shape[0])
            colmap_target = colmap_used
            colmap_result = {
                "points": colmap_points,
                "raw_points": int(colmap_data["raw_points"]),
                "voxel_points": int(colmap_data["voxel_points"]),
                "used_points": colmap_used,
                "colmap_root": colmap_data["colmap_root"],
                "points_path": colmap_data["points_path"],
                "voxel_size": float(colmap_data["voxel_size"]),
            }
        else:
            remaining_capacity = max(0, num_points - anchor_count)
            colmap_target = min(int(round(num_points * colmap_ratio)), remaining_capacity)
            colmap_result = self._sample_colmap_sparse_points(model_cfg, init_context, colmap_target)
            colmap_points = colmap_result["points"]
            colmap_used = int(colmap_result["used_points"])

        if fixed_random_points >= 0:
            random_count = fixed_random_points
        else:
            random_count = num_points - anchor_count - colmap_used
        random_points = self._random_means(random_count).cpu().numpy().astype(np.float32)
        means = np.concatenate([random_points, colmap_points, sampled_anchor_points], axis=0)
        np.random.shuffle(means)
        if fixed_random_points < 0 and not colmap_use_all and means.shape[0] != num_points:
            raise RuntimeError(
                f"hybrid_anchor_colmap_sparse initialized {means.shape[0]} points, expected {num_points}"
            )

        frame_count = int(anchor_data["frame_count"]) if anchor_data is not None else 0
        valid_frame_counts = np.bincount(frame_ids, minlength=frame_count) if frame_ids.shape[0] > 0 else np.zeros((frame_count,), dtype=np.int64)
        valid_anchor_frames = int(np.count_nonzero(valid_frame_counts))
        selected_valid_counts = selected_frame_counts[valid_frame_counts > 0]
        selected_frame_min = int(selected_valid_counts.min()) if selected_valid_counts.size > 0 else 0
        selected_frame_median = float(np.median(selected_valid_counts)) if selected_valid_counts.size > 0 else 0.0
        selected_frame_max = int(selected_valid_counts.max()) if selected_valid_counts.size > 0 else 0
        quality_mean_all = float(quality.mean()) if quality.shape[0] > 0 else 0.0
        quality_mean_selected = float(selected_quality.mean()) if selected_quality.shape[0] > 0 else 0.0

        print(
            "[Init] hybrid_anchor_colmap_sparse: "
            f"depth_dir={None if anchor_data is None else anchor_data['depth_root']}, "
            f"anchor_frames={0 if anchor_data is None else anchor_data['anchor_frames']}, "
            f"raw_anchor_points={raw_anchor_points}, voxel_anchor_points={voxel_anchor_points}, "
            f"anchor_target={anchor_target}, anchor_used={anchor_count}, selection_mode={selection_mode}, "
            f"valid_anchor_frames={valid_anchor_frames}, selected_frame_min={selected_frame_min}, "
            f"selected_frame_median={selected_frame_median:.1f}, selected_frame_max={selected_frame_max}, "
            f"quality_mean_selected={quality_mean_selected:.4f}, quality_mean_all={quality_mean_all:.4f}, "
            f"colmap_dir={colmap_result['colmap_root']}, colmap_raw_points={colmap_result['raw_points']}, "
            f"colmap_voxel_points={colmap_result['voxel_points']}, colmap_target={colmap_target}, "
            f"colmap_used={colmap_used}, colmap_voxel_size={colmap_result['voxel_size']}, random_used={random_count}, "
            f"voxel_enabled={voxel_enabled}, voxel_size={voxel_size}, "
            f"colmap_use_all={colmap_use_all}, total_used={means.shape[0]}"
        )
        return torch.from_numpy(means).float()

    def _build_camera(self, camtoworld):
        device = self.splats["means"].device
        c2w = torch.eye(4, device=device, dtype=torch.float32)
        c2w[:3, :] = camtoworld.to(device)
        viewmat = torch.linalg.inv(c2w)
        viewmat[1, :] *= -1
        viewmat[2, :] *= -1
        viewmat = viewmat[None]

        intrinsics = torch.tensor(
            [
                [self.fl_x, 0.0, self.cx],
                [0.0, self.fl_y, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )[None]
        return viewmat, intrinsics

    def _rasterize(self, colors, viewmats, intrinsics, img_h, img_w, backgrounds, sh_degree, render_mode="RGB"):
        return rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=viewmats,
            Ks=intrinsics,
            width=img_w,
            height=img_h,
            sh_degree=sh_degree,
            backgrounds=backgrounds,
            render_mode=render_mode,
            packed=False,
        )

    def render_rgb(self, camtoworld, img_h, img_w):
        device = self.splats["means"].device
        viewmats, intrinsics = self._build_camera(camtoworld)
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        backgrounds = torch.full((1, 3), self.bg_color, dtype=torch.float32, device=device)
        renders, alphas, info = self._rasterize(
            colors=colors,
            viewmats=viewmats,
            intrinsics=intrinsics,
            img_h=img_h,
            img_w=img_w,
            backgrounds=backgrounds,
            sh_degree=self.sh_degree,
            render_mode="RGB",
        )
        return renders[0], alphas[0], info

    def render_geom_depth(self, camtoworld, img_h, img_w):
        device = self.splats["means"].device
        viewmats, intrinsics = self._build_camera(camtoworld)
        dummy_colors = torch.zeros((self.num_gaussians, 3), dtype=torch.float32, device=device)
        backgrounds = torch.zeros((1, 1), dtype=torch.float32, device=device)
        renders, alphas, _ = self._rasterize(
            colors=dummy_colors,
            viewmats=viewmats,
            intrinsics=intrinsics,
            img_h=img_h,
            img_w=img_w,
            backgrounds=backgrounds,
            sh_degree=None,
            render_mode="ED",
        )
        return renders[0], alphas[0]

    def render_aux_heads(self, camtoworld, img_h, img_w, heads):
        device = self.splats["means"].device
        viewmats, intrinsics = self._build_camera(camtoworld)
        backgrounds = torch.zeros((1, 3), dtype=torch.float32, device=device)
        outputs = {}
        head_specs = {
            "depth": ("depth_feat", "scalar"),
            "prior": ("prior_feat", "scalar"),
            "illum": ("illum_feat", "scalar"),
            "chroma": ("chroma_feat", "rgb"),
        }
        for head in heads:
            feature_name, head_type = head_specs.get(head, (f"{head}_feat", "scalar"))
            if feature_name not in self.splats:
                outputs[head] = None
                continue
            feature = self.splats[feature_name]
            if head_type == "rgb":
                if feature.shape[1] >= 3:
                    colors = feature[:, :3]
                else:
                    pad_width = 3 - feature.shape[1]
                    colors = torch.cat(
                        [
                            feature,
                            torch.zeros(feature.shape[0], pad_width, device=feature.device, dtype=feature.dtype),
                        ],
                        dim=1,
                    )
            else:
                colors = feature.repeat(1, 3)
            renders, _, _ = self._rasterize(
                colors=colors,
                viewmats=viewmats,
                intrinsics=intrinsics,
                img_h=img_h,
                img_w=img_w,
                backgrounds=backgrounds,
                sh_degree=None,
                render_mode="RGB",
            )
            if head_type == "rgb":
                outputs[head] = renders[0][..., : feature.shape[1]]
            else:
                outputs[head] = renders[0].mean(dim=-1, keepdim=True)
        return outputs

    def forward(self, camtoworld, img_h, img_w, render_heads=(), render_geom_depth=False, render_rgb=True):
        rgb = None
        alphas = None
        info = {}
        if render_rgb:
            rgb, alphas, info = self.render_rgb(camtoworld, img_h, img_w)
        head_outputs = self.render_aux_heads(camtoworld, img_h, img_w, render_heads) if render_heads else {}
        geom_depth = None
        if render_geom_depth:
            geom_depth, geom_alphas = self.render_geom_depth(camtoworld, img_h, img_w)
            if alphas is None:
                alphas = geom_alphas
        illum_aux = head_outputs.get("illum")
        chroma_aux = head_outputs.get("chroma")
        chroma_factor = None
        chroma_delta = None
        base_lit_ycbcr = None
        if illum_aux is not None and rgb is not None:
            illum_factor = 2.0 * torch.sigmoid(illum_aux)
            base_lit_rgb, base_lit_ycbcr = apply_ycbcr_luminance_gain(rgb, illum_factor)
        else:
            base_lit_rgb = rgb
        if chroma_aux is not None and base_lit_rgb is not None and self.chroma_residual_enabled:
            if self.chroma_colorspace != "ycbcr":
                raise RuntimeError(f"Unsupported chroma colorspace: {self.chroma_colorspace}")
            recon_rgb, chroma_factor, chroma_delta = apply_ycbcr_chroma_residual(
                base_lit_rgb,
                chroma_aux,
                self.chroma_scale,
                mode=self.chroma_residual_mode,
            )
        else:
            recon_rgb = base_lit_rgb
        return {
            "rgb": rgb,
            "depth_aux": head_outputs.get("depth"),
            "geom_depth": geom_depth,
            "prior_aux": head_outputs.get("prior"),
            "illum_aux": illum_aux,
            "chroma_aux": chroma_aux,
            "chroma_factor": chroma_factor,
            "chroma_delta": chroma_delta,
            "chroma_mode": self.chroma_residual_mode,
            "chroma_scale": self.chroma_scale,
            "base_lit_rgb": base_lit_rgb,
            "base_lit_ycbcr": base_lit_ycbcr,
            "recon_rgb": recon_rgb,
            "alphas": alphas,
            "info": info,
        }





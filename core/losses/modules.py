import math

from core.libs.losses import (
    chroma_delta_from_aux,
    chroma_factor_from_aux,
    exposure_control_loss,
    illum_delta_from_aux,
    low_light_consistency_loss,
    rgb_to_ycbcr_hwc,
    rgb_reconstruction_loss,
)
import torch
import torch.nn.functional as F


def _quantile_logs(name, values, quantiles):
    flattened = values.detach().reshape(-1)
    if flattened.numel() == 0:
        return {f"{name}_{label}": 0.0 for _, label in quantiles}
    q_values = torch.tensor([float(q) for q, _ in quantiles], device=flattened.device, dtype=flattened.dtype)
    q_result = torch.quantile(flattened, q_values)
    return {
        f"{name}_{label}": float(q_result[idx].item())
        for idx, (_, label) in enumerate(quantiles)
    }


def _masked_mean(values, mask):
    if values.numel() == 0 or mask.numel() == 0:
        return 0.0
    valid_mask = mask.detach().reshape(-1).bool()
    if not bool(valid_mask.any().item()):
        return 0.0
    return float(values.detach().reshape(-1)[valid_mask].mean().item())


def _masked_quantile_logs(name, values, mask, quantiles):
    if values.numel() == 0 or mask.numel() == 0:
        return {f"{name}_{label}": 0.0 for _, label in quantiles}
    valid_mask = mask.detach().reshape(-1).bool()
    if not bool(valid_mask.any().item()):
        return {f"{name}_{label}": 0.0 for _, label in quantiles}
    return _quantile_logs(name, values.detach().reshape(-1)[valid_mask], quantiles)


def _quaternion_to_rotation_matrix(quats):
    quats = F.normalize(quats, dim=-1, eps=1.0e-12)
    w, x, y, z = quats.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack(
        [
            ww + xx - yy - zz,
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            ww - xx + yy - zz,
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    ).reshape(quats.shape[0], 3, 3)


class BaseLossModule:
    def __init__(
        self,
        name,
        weight=1.0,
        enabled=True,
        start_step=0,
        end_step=None,
        ramp_up_steps=0,
        ramp_down_steps=0,
        start_scale=1.0,
        end_scale=0.0,
    ):
        self.name = name
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.start_step = int(start_step)
        self.end_step = None if end_step is None else int(end_step)
        self.ramp_up_steps = int(max(0, ramp_up_steps))
        self.ramp_down_steps = int(max(0, ramp_down_steps))
        self.start_scale = float(start_scale)
        self.end_scale = float(end_scale)

    def compute(self, context):
        raise NotImplementedError

    def is_active(self, context):
        return self.schedule_scale(context) > 0.0

    def schedule_scale(self, context):
        step = int(context.get("step", 0))
        if step < self.start_step:
            return 0.0
        if self.end_step is not None and step > self.end_step:
            return 0.0

        scale = 1.0
        if self.ramp_up_steps > 0:
            ramp_up_progress = (step - self.start_step) / float(self.ramp_up_steps)
            ramp_up_progress = min(max(ramp_up_progress, 0.0), 1.0)
            scale *= self.start_scale + (1.0 - self.start_scale) * ramp_up_progress

        if self.end_step is not None and self.ramp_down_steps > 0:
            ramp_down_start = self.end_step - self.ramp_down_steps
            if step > ramp_down_start:
                ramp_down_progress = (self.end_step - step) / float(self.ramp_down_steps)
                ramp_down_progress = min(max(ramp_down_progress, 0.0), 1.0)
                scale *= self.end_scale + (1.0 - self.end_scale) * ramp_down_progress

        return scale

    def current_weight(self, context):
        return self.weight * self.schedule_scale(context)


class RGBReconstructionLoss(BaseLossModule):
    def __init__(self, lambda_ssim, name="rgb", input_key="rendered", target_key="supervision_hwc"):
        super().__init__(name=name, weight=1.0, enabled=True, start_step=0)
        self.lambda_ssim = float(lambda_ssim)
        self.input_key = str(input_key)
        self.target_key = str(target_key)

    def compute(self, context):
        result = rgb_reconstruction_loss(
            context[self.input_key],
            context[self.target_key],
            lambda_ssim=self.lambda_ssim,
        )
        return result["total"], {
            "l1": float(result["l1"].detach().item()),
            "ssim": float(result["ssim"].detach().item()),
        }


def normalize_weight_map(weight_map, weight_min, weight_max):
    weight_map = weight_map / weight_map.mean().clamp_min(1.0e-6)
    return weight_map.clamp(weight_min, weight_max)


def build_bright_factor(target, bright_threshold, bright_suppression):
    if bright_suppression <= 0.0:
        return torch.ones(target.shape[:2], device=target.device, dtype=target.dtype)
    target_luma = 0.299 * target[..., 0] + 0.587 * target[..., 1] + 0.114 * target[..., 2]
    bright_mask = (target_luma > bright_threshold).to(dtype=target.dtype)
    return 1.0 - bright_suppression * bright_mask


def build_alpha_confidence(context, target, confidence_floor):
    alphas = context.get("alphas")
    if alphas is None:
        return torch.ones(target.shape[:2], device=target.device, dtype=target.dtype)
    alpha_map = squeeze_single_channel(alphas.to(device=target.device, dtype=target.dtype), "alphas").clamp(0.0, 1.0)
    return confidence_floor + (1.0 - confidence_floor) * alpha_map


def build_structure_confidence(context, target, confidence_floor, structure_power):
    structure = context.get("structure")
    if structure is None:
        return torch.ones(target.shape[:2], device=target.device, dtype=target.dtype)
    structure = squeeze_single_channel(structure.to(device=target.device, dtype=target.dtype), "structure")
    structure = minmax_normalize_map(structure).clamp(0.0, 1.0)
    structure_conf = structure.pow(structure_power)
    return confidence_floor + (1.0 - confidence_floor) * structure_conf


def weighted_l1_loss(prediction, target, weight_map=None):
    if weight_map is None:
        return torch.abs(prediction - target).mean()
    if weight_map.dim() == prediction.dim() - 1:
        weight_map = weight_map.unsqueeze(-1)
    weight_map = weight_map.to(device=prediction.device, dtype=prediction.dtype)
    if weight_map.shape[-1] == 1 and prediction.shape[-1] != 1:
        weight_map = weight_map.expand(*prediction.shape[:-1], prediction.shape[-1])
    if weight_map.shape != prediction.shape:
        raise RuntimeError(
            f"weighted_l1_loss weight_map shape {tuple(weight_map.shape)} does not match prediction {tuple(prediction.shape)}"
        )
    return (torch.abs(prediction - target) * weight_map).sum() / weight_map.sum().clamp_min(1.0e-6)


class LuminanceReconstructionLoss(BaseLossModule):
    def __init__(
        self,
        weight,
        start_step=0,
        input_key="recon_hwc",
        target_key="proxy_target_hwc",
        use_weight_map=False,
        dark_boost=0.0,
        bright_threshold=0.70,
        bright_suppression=0.0,
        confidence_floor=1.0,
        structure_power=1.0,
        weight_min=0.25,
        weight_max=2.0,
        anchor_weight=0.0,
        anchor_shadow_alpha=0.0,
    ):
        super().__init__(name="luminance_reconstruction", weight=weight, enabled=weight > 0.0, start_step=start_step)
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.use_weight_map = bool(use_weight_map)
        self.dark_boost = float(dark_boost)
        self.bright_threshold = float(bright_threshold)
        self.bright_suppression = float(bright_suppression)
        self.confidence_floor = float(confidence_floor)
        self.structure_power = float(structure_power)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.anchor_weight = float(max(anchor_weight, 0.0))
        self.anchor_shadow_alpha = float(max(anchor_shadow_alpha, 0.0))

    def _build_weight_map(self, context, target):
        weight_map = torch.ones(target.shape[:2], device=target.device, dtype=target.dtype)
        shadow_weight = context.get("proxy_shadow_weight_hwc")
        if shadow_weight is not None and self.dark_boost > 0.0:
            shadow_weight = squeeze_single_channel(shadow_weight.to(device=target.device, dtype=target.dtype), "proxy_shadow_weight")
            weight_map = weight_map + self.dark_boost * shadow_weight
        weight_map = weight_map * build_bright_factor(target, self.bright_threshold, self.bright_suppression)
        weight_map = weight_map * build_alpha_confidence(context, target, self.confidence_floor)
        weight_map = weight_map * build_structure_confidence(context, target, self.confidence_floor, self.structure_power)
        return normalize_weight_map(weight_map, self.weight_min, self.weight_max)

    def _weighted_mean(self, values, weight_map):
        values = values.squeeze(-1)
        if weight_map is None:
            return values.mean()
        weights = weight_map.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0e-6)

    def compute(self, context):
        target = context.get(self.target_key)
        if target is None:
            raise RuntimeError("LuminanceReconstructionLoss requires proxy_target_hwc, but it is missing.")
        if not self.is_active(context):
            zero = zero_scalar_like(context)
            return zero, {"active": 0.0, "weight_mean": 1.0}

        pred_y = rgb_to_ycbcr_hwc(context[self.input_key])[..., :1]
        target_y = rgb_to_ycbcr_hwc(target)[..., :1]
        weight_map = self._build_weight_map(context, target) if self.use_weight_map else None
        pixel_loss = weighted_l1_loss(pred_y, target_y, weight_map)
        anchor_loss = torch.zeros((), device=pred_y.device, dtype=pred_y.dtype)
        pred_anchor_mean = self._weighted_mean(pred_y, None)
        target_anchor_scalar = context.get("proxy_effective_y_target_mean", context.get("proxy_target_mean"))
        if target_anchor_scalar is None:
            target_anchor_mean = self._weighted_mean(target_y, None)
        else:
            target_anchor_mean = pred_y.new_tensor(float(target_anchor_scalar))
        anchor_weight_mean = 1.0
        if self.anchor_weight > 0.0:
            pred_anchor_mean = self._weighted_mean(pred_y, None)
            anchor_loss = torch.abs(pred_anchor_mean - target_anchor_mean)
        loss = pixel_loss + self.anchor_weight * anchor_loss
        weight_mean = 1.0 if weight_map is None else float(weight_map.detach().mean().item())
        return loss, {
            "l1": float(pixel_loss.detach().item()),
            "anchor": float(anchor_loss.detach().item()),
            "anchor_weight": float(self.anchor_weight),
            "anchor_shadow_alpha": float(self.anchor_shadow_alpha),
            "anchor_pred_mean": float(pred_anchor_mean.detach().item()),
            "anchor_target_mean": float(target_anchor_mean.detach().item()),
            "anchor_weight_mean": anchor_weight_mean,
            "active": 1.0,
            "weight_mean": weight_mean,
        }


class ChromaReconstructionLoss(BaseLossModule):
    def __init__(
        self,
        weight,
        start_step=0,
        input_key="recon_hwc",
        target_key="proxy_target_hwc",
        reference_key="supervision_hwc",
        use_weight_map=False,
        bright_threshold=0.70,
        bright_suppression=0.0,
        confidence_floor=0.45,
        structure_power=1.0,
        weight_min=0.20,
        weight_max=1.50,
        shadow_power=0.5,
        global_mean_weight=0.0,
        proxy_blend=1.0,
        cb_weight=1.0,
        cr_weight=1.0,
        cb_target_bias=0.0,
        cr_target_bias=0.0,
    ):
        super().__init__(name="chroma_reconstruction", weight=weight, enabled=weight > 0.0, start_step=start_step)
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.reference_key = str(reference_key)
        self.use_weight_map = bool(use_weight_map)
        self.bright_threshold = float(bright_threshold)
        self.bright_suppression = float(bright_suppression)
        self.confidence_floor = float(confidence_floor)
        self.structure_power = float(structure_power)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.shadow_power = float(max(1.0e-3, shadow_power))
        self.global_mean_weight = float(max(0.0, global_mean_weight))
        self.proxy_blend = float(min(max(proxy_blend, 0.0), 1.0))
        self.cb_weight = float(max(0.0, cb_weight))
        self.cr_weight = float(max(0.0, cr_weight))
        self.cb_target_bias = float(cb_target_bias)
        self.cr_target_bias = float(cr_target_bias)

    def _build_weight_map(self, context, target):
        weight_map = build_bright_factor(target, self.bright_threshold, self.bright_suppression)
        weight_map = weight_map * build_alpha_confidence(context, target, self.confidence_floor)
        weight_map = weight_map * build_structure_confidence(context, target, self.confidence_floor, self.structure_power)
        shadow_weight = context.get("proxy_shadow_weight_hwc")
        if shadow_weight is not None:
            shadow_weight = squeeze_single_channel(shadow_weight.to(device=target.device, dtype=target.dtype), "proxy_shadow_weight").clamp(0.0, 1.0)
            color_conf = (1.0 - shadow_weight).pow(self.shadow_power)
            color_conf = self.confidence_floor + (1.0 - self.confidence_floor) * color_conf
            weight_map = weight_map * color_conf
        return normalize_weight_map(weight_map, self.weight_min, self.weight_max)

    def compute(self, context):
        target = context.get(self.target_key)
        if target is None:
            raise RuntimeError("ChromaReconstructionLoss requires proxy_target_hwc, but it is missing.")
        if not self.is_active(context):
            zero = zero_scalar_like(context)
            return zero, {"active": 0.0, "weight_mean": 1.0}

        pred_cbcr = rgb_to_ycbcr_hwc(context[self.input_key])[..., 1:]
        proxy_cbcr = rgb_to_ycbcr_hwc(target)[..., 1:]
        target_cbcr = proxy_cbcr
        reference = context.get(self.reference_key)
        if reference is not None:
            reference_cbcr = rgb_to_ycbcr_hwc(reference)[..., 1:]
            if self.proxy_blend <= 0.0:
                target_cbcr = reference_cbcr
            elif self.proxy_blend < 1.0:
                target_cbcr = self.proxy_blend * proxy_cbcr + (1.0 - self.proxy_blend) * reference_cbcr
        if self.cb_target_bias != 0.0 or self.cr_target_bias != 0.0:
            target_cbcr = target_cbcr.clone()
            target_cbcr[..., 0] = target_cbcr[..., 0] + self.cb_target_bias
            target_cbcr[..., 1] = target_cbcr[..., 1] + self.cr_target_bias
            target_cbcr = target_cbcr.clamp(-0.5, 0.5)
        weight_map = self._build_weight_map(context, target) if self.use_weight_map else None
        diff = torch.abs(pred_cbcr - target_cbcr)
        channel_weights = pred_cbcr.new_tensor([self.cb_weight, self.cr_weight]).view(1, 1, 2)
        weighted_diff = diff * channel_weights
        if weight_map is not None:
            expanded_weight_map = weight_map.unsqueeze(-1).expand_as(weighted_diff)
            pixel_loss = (weighted_diff * expanded_weight_map).sum() / expanded_weight_map.sum().clamp_min(1.0e-6)
        else:
            pixel_loss = weighted_diff.mean()
        global_mean_loss = torch.zeros((), device=pred_cbcr.device, dtype=pred_cbcr.dtype)
        if self.global_mean_weight > 0.0:
            pred_global_mean = pred_cbcr.mean(dim=(0, 1))
            target_global_mean = target_cbcr.mean(dim=(0, 1))
            global_mean_loss = (torch.abs(pred_global_mean - target_global_mean) * channel_weights.view(-1)).mean()
        loss = pixel_loss + self.global_mean_weight * global_mean_loss
        if weight_map is not None:
            cb_l1 = (diff[..., 0] * weight_map).sum() / weight_map.sum().clamp_min(1.0e-6)
            cr_l1 = (diff[..., 1] * weight_map).sum() / weight_map.sum().clamp_min(1.0e-6)
        else:
            cb_l1 = diff[..., 0].mean()
            cr_l1 = diff[..., 1].mean()
        weight_mean = 1.0 if weight_map is None else float(weight_map.detach().mean().item())
        pred_global_mean = pred_cbcr.detach().mean(dim=(0, 1))
        target_global_mean = target_cbcr.detach().mean(dim=(0, 1))
        return loss, {
            "l1": float(pixel_loss.detach().item()),
            "global_mean": float(global_mean_loss.detach().item()),
            "cb_l1": float(cb_l1.detach().item()),
            "cr_l1": float(cr_l1.detach().item()),
            "cb_global_mean_pred": float(pred_global_mean[0].item()),
            "cr_global_mean_pred": float(pred_global_mean[1].item()),
            "cb_global_mean_tgt": float(target_global_mean[0].item()),
            "cr_global_mean_tgt": float(target_global_mean[1].item()),
            "cb_weight": float(self.cb_weight),
            "cr_weight": float(self.cr_weight),
            "cb_target_bias": float(self.cb_target_bias),
            "cr_target_bias": float(self.cr_target_bias),
            "proxy_blend": float(self.proxy_blend),
            "active": 1.0,
            "weight_mean": weight_mean,
        }


class IlluminationRegularizationLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="illum_reg", weight=weight, enabled=weight > 0.0, start_step=0)

    def compute(self, context):
        illum_aux = context.get("illum_aux")
        if illum_aux is None:
            raise RuntimeError("IlluminationRegularizationLoss requires illum_aux, but the model did not render the illumination head.")
        illum_delta = illum_delta_from_aux(illum_aux)
        loss = torch.abs(illum_delta).mean()
        return loss, {"delta_mean": float(illum_delta.detach().mean().item())}


class ChromaResidualRegularizationLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="chroma_reg", weight=weight, enabled=weight > 0.0, start_step=0)

    def compute(self, context):
        chroma_delta = context.get("chroma_delta")
        if chroma_delta is not None:
            loss = torch.abs(chroma_delta).mean()
            return loss, {
                "delta_mean": float(chroma_delta.detach().mean().item()),
                "delta_std": float(chroma_delta.detach().std(unbiased=False).item()),
                "cb_delta_mean": float(chroma_delta[..., 0].detach().mean().item()),
                "cr_delta_mean": float(chroma_delta[..., 1].detach().mean().item()),
            }
        chroma_factor = context.get("chroma_factor")
        if chroma_factor is None:
            chroma_aux = context.get("chroma_aux")
            if chroma_aux is None:
                raise RuntimeError("ChromaResidualRegularizationLoss requires chroma_aux/chroma_factor, but the model did not render the chroma head.")
            chroma_mode = str(context.get("chroma_mode", "multiplicative")).lower()
            if chroma_mode == "additive":
                chroma_delta = chroma_delta_from_aux(
                    chroma_aux,
                    float(context.get("chroma_scale", 0.10)),
                )
                loss = torch.abs(chroma_delta).mean()
                return loss, {
                    "delta_mean": float(chroma_delta.detach().mean().item()),
                    "delta_std": float(chroma_delta.detach().std(unbiased=False).item()),
                    "cb_delta_mean": float(chroma_delta[..., 0].detach().mean().item()),
                    "cr_delta_mean": float(chroma_delta[..., 1].detach().mean().item()),
                }
            chroma_factor = chroma_factor_from_aux(
                chroma_aux,
                float(context.get("chroma_scale", 0.10)),
            )
        loss = torch.abs(chroma_factor - 1.0).mean()
        return loss, {
            "factor_mean": float(chroma_factor.detach().mean().item()),
            "factor_std": float(chroma_factor.detach().std(unbiased=False).item()),
            "cb_factor_mean": float(chroma_factor[..., 0].detach().mean().item()),
            "cr_factor_mean": float(chroma_factor[..., 1].detach().mean().item()),
        }


class SparsePointRegularizationLoss(BaseLossModule):
    def __init__(
        self,
        weight,
        start_step=0,
        end_step=None,
        sample_points=1024,
        min_opacity=0.2,
        robust_scale=0.05,
        knn_k=3,
        knn_eps=1.0e-6,
        meta_enabled=True,
        density_k=8,
        density_clamp_min=0.5,
        density_clamp_max=2.0,
        quality_error_scale_mode="median",
        quality_track_mode="log_median_norm",
        mode="point_to_barycenter",
        plane_k=8,
        plane_eps=1.0e-6,
        plane_min_eigen_gap=1.0e-4,
        tangent_weight=0.15,
        normal_scale_weight=0.0,
        sampling_mode="random",
        hard_ratio=0.5,
        difficulty_score="min_sparse_dist",
        random_sample_fallback=True,
        global_mining_chunk_size=4096,
        global_mining_refresh_interval=25,
        reliability_filter_enabled=False,
        lowlight_brightness_enabled=True,
        lowlight_gradient_enabled=True,
        loss_support_use_brightness=True,
        loss_support_use_gradient=True,
        prune_support_use_brightness=True,
        prune_support_use_gradient=True,
        weight_schedule="constant",
        weight_start_scale=1.0,
        weight_end_scale=1.0,
        weight_decay_end_step=None,
        orientation_enabled=False,
        orientation_weight=0.0,
        anisotropic_scale_target_enabled=False,
        anisotropic_scale_target_weight=0.0,
        tangent_scale_ratio=0.9,
        normal_scale_ratio=0.24,
        target_tangent_scale_min=0.005,
        target_tangent_scale_max=0.05,
        target_normal_scale_min=0.002,
        target_normal_scale_max=0.02,
        target_tangent_quantile=0.75,
        target_normal_quantile=0.75,
        target_tangent_std_blend=0.75,
        target_tangent_std_cap_ratio=1.6,
        target_tangent_local_radius_cap_ratio=0.85,
        target_tangent_std_floor_ratio=1.0,
        tail_start_step=None,
        tail_weight_hold_end_step=None,
        tail_light_mode_enabled=False,
        tail_sampling_mode=None,
        tail_hard_ratio=None,
        tail_random_sample_fallback=None,
        tail_min_plane_confidence=0.35,
        tail_global_mining_refresh_interval=None,
        tail_candidate_subset_ratio=1.0,
        tail_candidate_subset_min=None,
        tail_candidate_subset_max=None,
        tail_stable_sample_ratio_floor=0.5,
        tail_keep_point_to_plane=True,
        tail_point_to_plane_no_fallback=True,
        tail_point_to_plane_min_confidence=0.35,
        tail_point_to_plane_confidence_power=1.0,
        tail_point_to_plane_weight_scale=0.2,
        tail_keep_orientation=True,
        tail_keep_anisotropic_scale=True,
        tail_keep_normal_scale=True,
        tail_difficulty_score="stable_surface_mixed",
        tail_difficulty_distance_weight=0.75,
        tail_difficulty_orientation_weight=0.75,
        tail_difficulty_scale_weight=1.25,
        tail_difficulty_normal_weight=0.5,
        tail_difficulty_confidence_weight=1.0,
        difficulty_distance_weight=1.0,
        difficulty_orientation_weight=0.5,
        difficulty_scale_weight=1.0,
        difficulty_normal_weight=0.5,
        mid_hard_distance_q_low=0.40,
        mid_hard_distance_q_high=0.90,
        mid_hard_normal_weight=1.0,
        mid_hard_tangent_weight=0.25,
        mid_hard_orientation_weight=0.5,
        mid_hard_scale_weight=0.5,
        mid_hard_opacity_weight_power=0.5,
        mid_hard_candidate_subset_ratio=1.0,
        mid_hard_candidate_subset_min=None,
        mid_hard_candidate_subset_max=None,
        local_geometry_enabled=False,
        local_geometry_newborn_steps=0,
        local_geometry_opacity_thresh=0.0,
        local_geometry_render_conf_thresh=0.0,
        local_geometry_mismatch_quantile=0.75,
        local_geometry_low_render_conf_requires_mismatch=True,
        local_geometry_newborn_quota=128,
        local_geometry_low_opacity_quota=128,
    ):
        super().__init__(name="sparse_guided", weight=weight, enabled=weight > 0.0, start_step=start_step, end_step=end_step)
        self.sample_points = int(max(1, sample_points))
        self.min_opacity = float(min_opacity)
        self.robust_scale = float(max(0.0, robust_scale))
        self.knn_k = int(max(1, knn_k))
        self.knn_eps = float(max(1.0e-12, knn_eps))
        self.meta_enabled = bool(meta_enabled)
        self.density_k = int(max(1, density_k))
        self.density_clamp_min = float(density_clamp_min)
        self.density_clamp_max = float(density_clamp_max)
        self.quality_error_scale_mode = str(quality_error_scale_mode).lower()
        self.quality_track_mode = str(quality_track_mode).lower()
        self.mode = str(mode).lower()
        self.plane_k = int(max(3, plane_k))
        self.plane_eps = float(max(1.0e-12, plane_eps))
        self.plane_min_eigen_gap = float(max(0.0, plane_min_eigen_gap))
        self.tangent_weight = float(max(0.0, tangent_weight))
        self.normal_scale_weight = float(max(0.0, normal_scale_weight))
        self.sampling_mode = str(sampling_mode).lower()
        self.hard_ratio = float(min(max(hard_ratio, 0.0), 1.0))
        self.difficulty_score = str(difficulty_score).lower()
        self.random_sample_fallback = bool(random_sample_fallback)
        self.global_mining_chunk_size = int(max(64, global_mining_chunk_size))
        self.global_mining_refresh_interval = int(max(1, global_mining_refresh_interval))
        self.reliability_filter_enabled = bool(reliability_filter_enabled)
        self.lowlight_brightness_enabled = bool(lowlight_brightness_enabled)
        self.lowlight_gradient_enabled = bool(lowlight_gradient_enabled)
        self.loss_support_use_brightness = bool(loss_support_use_brightness)
        self.loss_support_use_gradient = bool(loss_support_use_gradient)
        self.prune_support_use_brightness = bool(prune_support_use_brightness)
        self.prune_support_use_gradient = bool(prune_support_use_gradient)
        self.weight_schedule = str(weight_schedule).lower()
        self.weight_start_scale = float(max(0.0, weight_start_scale))
        self.weight_end_scale = float(max(0.0, weight_end_scale))
        self.weight_decay_end_step = None if weight_decay_end_step is None else int(weight_decay_end_step)
        self.orientation_enabled = bool(orientation_enabled)
        self.orientation_weight = float(max(0.0, orientation_weight))
        self.anisotropic_scale_target_enabled = bool(anisotropic_scale_target_enabled)
        self.anisotropic_scale_target_weight = float(max(0.0, anisotropic_scale_target_weight))
        self.tangent_scale_ratio = float(max(0.0, tangent_scale_ratio))
        self.normal_scale_ratio = float(max(0.0, normal_scale_ratio))
        self.target_tangent_scale_min = float(max(0.0, target_tangent_scale_min))
        self.target_tangent_scale_max = float(max(self.target_tangent_scale_min, target_tangent_scale_max))
        self.target_normal_scale_min = float(max(0.0, target_normal_scale_min))
        self.target_normal_scale_max = float(max(self.target_normal_scale_min, target_normal_scale_max))
        self.target_tangent_quantile = float(min(max(target_tangent_quantile, 0.0), 1.0))
        self.target_normal_quantile = float(min(max(target_normal_quantile, 0.0), 1.0))
        self.target_tangent_std_blend = float(min(max(target_tangent_std_blend, 0.0), 1.0))
        self.target_tangent_std_cap_ratio = float(max(1.0, target_tangent_std_cap_ratio))
        self.target_tangent_local_radius_cap_ratio = float(max(0.0, target_tangent_local_radius_cap_ratio))
        self.target_tangent_std_floor_ratio = float(max(0.0, target_tangent_std_floor_ratio))
        self.tail_start_step = int(self.start_step if tail_start_step is None else max(self.start_step, int(tail_start_step)))
        self.tail_weight_hold_end_step = (
            None if tail_weight_hold_end_step is None else int(max(self.tail_start_step, int(tail_weight_hold_end_step)))
        )
        self.tail_light_mode_enabled = bool(tail_light_mode_enabled)
        self.tail_sampling_mode = self.sampling_mode if tail_sampling_mode is None else str(tail_sampling_mode).lower()
        self.tail_hard_ratio = self.hard_ratio if tail_hard_ratio is None else float(min(max(tail_hard_ratio, 0.0), 1.0))
        self.tail_random_sample_fallback = self.random_sample_fallback if tail_random_sample_fallback is None else bool(tail_random_sample_fallback)
        self.tail_min_plane_confidence = float(min(max(tail_min_plane_confidence, 0.0), 1.0))
        self.tail_global_mining_refresh_interval = (
            self.global_mining_refresh_interval
            if tail_global_mining_refresh_interval is None
            else int(max(1, tail_global_mining_refresh_interval))
        )
        self.tail_candidate_subset_ratio = float(min(max(tail_candidate_subset_ratio, 0.0), 1.0))
        self.tail_candidate_subset_min = 0 if tail_candidate_subset_min is None else int(max(0, tail_candidate_subset_min))
        self.tail_candidate_subset_max = 0 if tail_candidate_subset_max is None else int(max(0, tail_candidate_subset_max))
        self.tail_stable_sample_ratio_floor = float(min(max(tail_stable_sample_ratio_floor, 0.0), 1.0))
        self.tail_keep_point_to_plane = bool(tail_keep_point_to_plane)
        self.tail_point_to_plane_no_fallback = bool(tail_point_to_plane_no_fallback)
        self.tail_point_to_plane_min_confidence = float(min(max(tail_point_to_plane_min_confidence, 0.0), 1.0))
        self.tail_point_to_plane_confidence_power = float(max(0.0, tail_point_to_plane_confidence_power))
        self.tail_point_to_plane_weight_scale = float(max(0.0, tail_point_to_plane_weight_scale))
        self.tail_keep_orientation = bool(tail_keep_orientation)
        self.tail_keep_anisotropic_scale = bool(tail_keep_anisotropic_scale)
        self.tail_keep_normal_scale = bool(tail_keep_normal_scale)
        self.tail_difficulty_score = str(tail_difficulty_score).lower()
        self.tail_difficulty_distance_weight = float(max(0.0, tail_difficulty_distance_weight))
        self.tail_difficulty_orientation_weight = float(max(0.0, tail_difficulty_orientation_weight))
        self.tail_difficulty_scale_weight = float(max(0.0, tail_difficulty_scale_weight))
        self.tail_difficulty_normal_weight = float(max(0.0, tail_difficulty_normal_weight))
        self.tail_difficulty_confidence_weight = float(min(max(tail_difficulty_confidence_weight, 0.0), 1.0))
        self.difficulty_distance_weight = float(max(0.0, difficulty_distance_weight))
        self.difficulty_orientation_weight = float(max(0.0, difficulty_orientation_weight))
        self.difficulty_scale_weight = float(max(0.0, difficulty_scale_weight))
        self.difficulty_normal_weight = float(max(0.0, difficulty_normal_weight))
        self.mid_hard_distance_q_low = float(min(max(mid_hard_distance_q_low, 0.0), 1.0))
        self.mid_hard_distance_q_high = float(min(max(mid_hard_distance_q_high, self.mid_hard_distance_q_low), 1.0))
        self.mid_hard_normal_weight = float(max(0.0, mid_hard_normal_weight))
        self.mid_hard_tangent_weight = float(max(0.0, mid_hard_tangent_weight))
        self.mid_hard_orientation_weight = float(max(0.0, mid_hard_orientation_weight))
        self.mid_hard_scale_weight = float(max(0.0, mid_hard_scale_weight))
        self.mid_hard_opacity_weight_power = float(max(0.0, mid_hard_opacity_weight_power))
        self.mid_hard_candidate_subset_ratio = float(min(max(mid_hard_candidate_subset_ratio, 0.0), 1.0))
        self.mid_hard_candidate_subset_min = 0 if mid_hard_candidate_subset_min is None else int(max(0, mid_hard_candidate_subset_min))
        self.mid_hard_candidate_subset_max = 0 if mid_hard_candidate_subset_max is None else int(max(0, mid_hard_candidate_subset_max))
        self.local_geometry_enabled = bool(local_geometry_enabled)
        self.local_geometry_newborn_steps = int(max(0, local_geometry_newborn_steps))
        self.local_geometry_opacity_thresh = float(min(max(local_geometry_opacity_thresh, 0.0), 1.0))
        self.local_geometry_render_conf_thresh = float(min(max(local_geometry_render_conf_thresh, 0.0), 1.0))
        self.local_geometry_mismatch_quantile = float(min(max(local_geometry_mismatch_quantile, 0.0), 1.0))
        self.local_geometry_low_render_conf_requires_mismatch = bool(local_geometry_low_render_conf_requires_mismatch)
        self.local_geometry_newborn_quota = int(max(0, local_geometry_newborn_quota))
        self.local_geometry_low_opacity_quota = int(max(0, local_geometry_low_opacity_quota))
        self.hard_candidate_multiplier = 4
        self.distance_chunk_size = 1024
        self._cached_support_key = None
        self._cached_quality_score = None
        self._cached_density_score = None
        self._cached_brightness_score = None
        self._cached_gradient_score = None
        self._cached_loss_support_score = None
        self._cached_prune_support_score = None
        self._cached_hard_positions = None
        self._cached_hard_positions_step = -1
        self._cached_hard_active_count = -1
        self._cached_hard_count = -1
        self._cached_tail_hard_payload = None
        self._cached_tail_hard_step = -1
        self._cached_tail_hard_active_count = -1
        self._cached_tail_hard_count = -1

    def reset_runtime_cache(self):
        self._cached_hard_positions = None
        self._cached_hard_positions_step = -1
        self._cached_hard_active_count = -1
        self._cached_hard_count = -1
        self._cached_tail_hard_payload = None
        self._cached_tail_hard_step = -1
        self._cached_tail_hard_active_count = -1
        self._cached_tail_hard_count = -1

    def _support_cache_key(self, sparse_points, track_len, reproj_error, brightness_score, gradient_score):
        return (
            int(sparse_points.data_ptr()),
            int(track_len.data_ptr()) if track_len is not None else -1,
            int(reproj_error.data_ptr()) if reproj_error is not None else -1,
            int(brightness_score.data_ptr()) if brightness_score is not None else -1,
            int(gradient_score.data_ptr()) if gradient_score is not None else -1,
            tuple(sparse_points.shape),
        )

    def _compute_density_radius(self, sparse_points):
        num_points = int(sparse_points.shape[0])
        if num_points <= 1:
            return torch.ones((num_points,), device=sparse_points.device, dtype=sparse_points.dtype)
        knn_k = min(self.density_k + 1, num_points)
        radii = []
        chunk_size = 1024
        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            chunk = sparse_points[start:end]
            distances = torch.cdist(chunk, sparse_points)
            knn_dist = torch.topk(distances, k=knn_k, dim=1, largest=False).values
            if knn_k > 1:
                local_radius = knn_dist[:, 1:].mean(dim=1)
            else:
                local_radius = knn_dist[:, 0]
            radii.append(local_radius)
        return torch.cat(radii, dim=0)

    def _get_sparse_support_scores(self, sparse_points, track_len, reproj_error, brightness_score, gradient_score):
        cache_key = self._support_cache_key(sparse_points, track_len, reproj_error, brightness_score, gradient_score)
        if self._cached_support_key == cache_key:
            return (
                self._cached_quality_score,
                self._cached_density_score,
                self._cached_brightness_score,
                self._cached_gradient_score,
                self._cached_loss_support_score,
                self._cached_prune_support_score,
            )

        ones = torch.ones((sparse_points.shape[0],), device=sparse_points.device, dtype=sparse_points.dtype)
        quality_score = ones
        density_score = ones

        if self.meta_enabled and track_len is not None and reproj_error is not None:
            track_len = track_len.reshape(-1).to(device=sparse_points.device, dtype=sparse_points.dtype).clamp_min(0.0)
            reproj_error = reproj_error.reshape(-1).to(device=sparse_points.device, dtype=sparse_points.dtype).clamp_min(0.0)

            track_score = torch.log1p(track_len)
            if self.quality_track_mode == "log_median_norm":
                positive_track = track_score[track_score > 0.0]
                track_median = positive_track.median() if positive_track.numel() > 0 else torch.tensor(1.0, device=track_score.device, dtype=track_score.dtype)
                track_score = track_score / track_median.clamp_min(self.knn_eps)

            if self.quality_error_scale_mode == "median":
                positive_error = reproj_error[reproj_error > 0.0]
                if positive_error.numel() > 0:
                    error_scale = positive_error.median()
                else:
                    error_scale = torch.tensor(1.0, device=reproj_error.device, dtype=reproj_error.dtype)
            else:
                error_scale = reproj_error.mean() if reproj_error.numel() > 0 else torch.tensor(1.0, device=reproj_error.device, dtype=reproj_error.dtype)
            error_score = torch.exp(-reproj_error / error_scale.clamp_min(self.knn_eps))
            quality_score = (track_score * error_score).clamp_min(self.knn_eps)

            density_radius = self._compute_density_radius(sparse_points)
            positive_radius = density_radius[density_radius > 0.0]
            radius_median = positive_radius.median() if positive_radius.numel() > 0 else torch.tensor(1.0, device=density_radius.device, dtype=density_radius.dtype)
            density_score = (radius_median / density_radius.clamp_min(self.knn_eps)).clamp(self.density_clamp_min, self.density_clamp_max)

        if self.lowlight_brightness_enabled and brightness_score is not None:
            brightness_score = brightness_score.reshape(-1).to(device=sparse_points.device, dtype=sparse_points.dtype).clamp_min(self.knn_eps)
        else:
            brightness_score = ones
        if self.lowlight_gradient_enabled and gradient_score is not None:
            gradient_score = gradient_score.reshape(-1).to(device=sparse_points.device, dtype=sparse_points.dtype).clamp_min(self.knn_eps)
        else:
            gradient_score = ones

        base_support_score = quality_score * density_score
        loss_support_score = base_support_score
        if self.loss_support_use_brightness:
            loss_support_score = loss_support_score * brightness_score
        if self.loss_support_use_gradient:
            loss_support_score = loss_support_score * gradient_score

        prune_support_score = base_support_score
        if self.prune_support_use_brightness:
            prune_support_score = prune_support_score * brightness_score
        if self.prune_support_use_gradient:
            prune_support_score = prune_support_score * gradient_score
        self._cached_support_key = cache_key
        self._cached_quality_score = quality_score
        self._cached_density_score = density_score
        self._cached_brightness_score = brightness_score
        self._cached_gradient_score = gradient_score
        self._cached_loss_support_score = loss_support_score
        self._cached_prune_support_score = prune_support_score
        return quality_score, density_score, brightness_score, gradient_score, loss_support_score, prune_support_score

    def _sparse_weight_scale(self, context):
        if self.weight_schedule == "constant":
            return 1.0
        if self.weight_schedule == "plateau_decay":
            step = int(context.get("step", 0))
            decay_end_step = self.weight_decay_end_step
            hold_end_step = self.tail_start_step if self.tail_weight_hold_end_step is None else self.tail_weight_hold_end_step
            if decay_end_step is None or decay_end_step <= hold_end_step:
                return self.weight_end_scale
            if step <= hold_end_step:
                return self.weight_start_scale
            progress = (step - hold_end_step) / float(decay_end_step - hold_end_step)
            progress = min(max(progress, 0.0), 1.0)
            return self.weight_start_scale + (self.weight_end_scale - self.weight_start_scale) * progress
        if self.weight_schedule != "decay":
            raise RuntimeError(f"Unsupported sparse weight schedule: {self.weight_schedule}")

        step = int(context.get("step", 0))
        decay_end_step = self.weight_decay_end_step
        if decay_end_step is None or decay_end_step <= self.start_step:
            return self.weight_end_scale
        if step <= self.start_step:
            return self.weight_start_scale

        progress = (step - self.start_step) / float(decay_end_step - self.start_step)
        progress = min(max(progress, 0.0), 1.0)
        return self.weight_start_scale + (self.weight_end_scale - self.weight_start_scale) * progress

    def current_weight(self, context):
        return super().current_weight(context) * self._sparse_weight_scale(context)

    def _is_tail_phase(self, context):
        if not self.tail_light_mode_enabled:
            return False
        return int(context.get("step", 0)) > self.tail_start_step

    def _effective_sampling_mode(self, context):
        if self._is_tail_phase(context):
            return self.tail_sampling_mode
        return self.sampling_mode

    def _effective_difficulty_mode(self, context, sampling_mode=None):
        sampling_mode = self._effective_sampling_mode(context) if sampling_mode is None else str(sampling_mode).lower()
        if self._is_tail_phase(context) and sampling_mode == "stable_surface_mixed":
            return self.tail_difficulty_score
        return self.difficulty_score

    def _tail_confidence_term(self, plane_confidence):
        blended = 0.5 + 0.5 * plane_confidence
        return (1.0 - self.tail_difficulty_confidence_weight) + self.tail_difficulty_confidence_weight * blended

    def _resolve_sampled_render_confidence(self, context, sampled_indices, sampled_opacity):
        render_confidence = context.get("gaussian_render_confidence")
        if render_confidence is None:
            return sampled_opacity.detach().clamp(0.0, 1.0)
        render_confidence = render_confidence.reshape(-1)
        if int(render_confidence.numel()) == 0:
            return sampled_opacity.detach().clamp(0.0, 1.0)
        sampled_render_confidence = render_confidence[sampled_indices]
        return sampled_render_confidence.to(device=sampled_opacity.device, dtype=sampled_opacity.dtype).clamp(0.0, 1.0)

    def _build_local_geometry_supplement_indices(self, context, means, opacity_values, sparse_points, active_mask):
        empty_indices = torch.zeros((0,), device=opacity_values.device, dtype=torch.long)
        logs = {
            "supplement_count": 0.0,
            "newborn_quota_count": 0.0,
            "low_opacity_quota_count": 0.0,
        }
        if not self.local_geometry_enabled:
            return empty_indices, logs

        supplement_groups = []
        birth_steps = context.get("gaussian_birth_step")
        if self.local_geometry_newborn_quota > 0 and birth_steps is not None and self.local_geometry_newborn_steps > 0:
            birth_steps = birth_steps.reshape(-1).to(device=opacity_values.device)
            if int(birth_steps.numel()) == int(opacity_values.numel()):
                current_step = int(context.get("step", 0))
                newborn_mask = (birth_steps >= 0) & ((current_step - birth_steps) <= self.local_geometry_newborn_steps) & (~active_mask)
                newborn_candidates = torch.nonzero(newborn_mask, as_tuple=False).squeeze(-1)
                if int(newborn_candidates.numel()) > 0:
                    if int(newborn_candidates.numel()) > self.local_geometry_newborn_quota and int(sparse_points.shape[0]) > 0:
                        newborn_dist = self._compute_min_sparse_distance(means[newborn_candidates].detach(), sparse_points)
                        keep_pos = torch.topk(newborn_dist, k=self.local_geometry_newborn_quota, largest=True).indices
                        newborn_candidates = newborn_candidates[keep_pos]
                    supplement_groups.append(newborn_candidates)
                    logs["newborn_quota_count"] = float(int(newborn_candidates.numel()))

        if self.local_geometry_low_opacity_quota > 0 and self.local_geometry_opacity_thresh > 0.0 and int(sparse_points.shape[0]) > 0:
            low_opacity_mask = (opacity_values > 1.0e-4) & (opacity_values <= self.local_geometry_opacity_thresh) & (~active_mask)
            low_opacity_candidates = torch.nonzero(low_opacity_mask, as_tuple=False).squeeze(-1)
            if int(low_opacity_candidates.numel()) > 0:
                low_opacity_dist = self._compute_min_sparse_distance(means[low_opacity_candidates].detach(), sparse_points)
                mismatch_threshold = torch.quantile(low_opacity_dist, self.local_geometry_mismatch_quantile)
                mismatch_keep = low_opacity_dist >= mismatch_threshold
                low_opacity_candidates = low_opacity_candidates[mismatch_keep]
                low_opacity_dist = low_opacity_dist[mismatch_keep]
                if int(low_opacity_candidates.numel()) > self.local_geometry_low_opacity_quota:
                    keep_pos = torch.topk(low_opacity_dist, k=self.local_geometry_low_opacity_quota, largest=True).indices
                    low_opacity_candidates = low_opacity_candidates[keep_pos]
                if int(low_opacity_candidates.numel()) > 0:
                    supplement_groups.append(low_opacity_candidates)
                    logs["low_opacity_quota_count"] = float(int(low_opacity_candidates.numel()))

        if not supplement_groups:
            return empty_indices, logs
        supplement_indices = torch.unique(torch.cat(supplement_groups, dim=0), sorted=False)
        logs["supplement_count"] = float(int(supplement_indices.numel()))
        return supplement_indices, logs

    def _build_local_geometry_masks(self, context, sampled_indices, sampled_opacity, sampled_render_confidence, mismatch_values=None):
        zero_mask = torch.zeros_like(sampled_opacity, dtype=torch.bool)
        if not self.local_geometry_enabled:
            return torch.ones_like(sampled_opacity, dtype=torch.bool), zero_mask, zero_mask, zero_mask, zero_mask

        newborn_mask = zero_mask
        birth_steps = context.get("gaussian_birth_step")
        if birth_steps is not None and self.local_geometry_newborn_steps > 0:
            birth_steps = birth_steps.reshape(-1)
            if int(birth_steps.numel()) > 0:
                sampled_birth_steps = birth_steps[sampled_indices].to(device=sampled_opacity.device)
                current_step = int(context.get("step", 0))
                newborn_mask = (sampled_birth_steps >= 0) & ((current_step - sampled_birth_steps) <= self.local_geometry_newborn_steps)

        low_opacity_mask = zero_mask
        if self.local_geometry_opacity_thresh > 0.0:
            low_opacity_mask = sampled_opacity <= self.local_geometry_opacity_thresh

        low_render_conf_mask = zero_mask
        if self.local_geometry_render_conf_thresh > 0.0:
            low_render_conf_mask = sampled_render_confidence <= self.local_geometry_render_conf_thresh

        high_mismatch_mask = zero_mask
        if mismatch_values is not None and int(mismatch_values.numel()) > 0:
            mismatch_values = mismatch_values.detach().reshape(-1).to(device=sampled_opacity.device, dtype=sampled_opacity.dtype)
            mismatch_threshold = torch.quantile(mismatch_values, self.local_geometry_mismatch_quantile)
            high_mismatch_mask = mismatch_values >= mismatch_threshold

        low_opacity_trigger = low_opacity_mask & high_mismatch_mask
        if self.local_geometry_low_render_conf_requires_mismatch:
            low_render_conf_trigger = low_render_conf_mask & high_mismatch_mask
        else:
            low_render_conf_trigger = low_render_conf_mask

        local_geometry_mask = newborn_mask | low_opacity_trigger | low_render_conf_trigger
        return local_geometry_mask, newborn_mask, low_opacity_trigger, low_render_conf_trigger, high_mismatch_mask

    def _update_topk(self, top_scores, top_positions, new_scores, new_positions, k):
        if k <= 0 or new_scores.numel() == 0:
            return top_scores, top_positions
        if top_scores is None or top_positions is None:
            keep = min(k, int(new_scores.numel()))
            keep_pos = torch.topk(new_scores, k=keep, largest=True).indices
            return new_scores[keep_pos], new_positions[keep_pos]
        combined_scores = torch.cat([top_scores, new_scores], dim=0)
        combined_positions = torch.cat([top_positions, new_positions], dim=0)
        keep = min(k, int(combined_scores.numel()))
        keep_pos = torch.topk(combined_scores, k=keep, largest=True).indices
        return combined_scores[keep_pos], combined_positions[keep_pos]

    def _tail_candidate_target_count(self, active_count, hard_count):
        if active_count <= 0:
            return 0
        required_count = min(active_count, max(1, int(hard_count)))
        ratio_target = int(math.ceil(float(active_count) * self.tail_candidate_subset_ratio))
        candidate_target = max(required_count, hard_count * self.hard_candidate_multiplier, ratio_target)
        if self.tail_candidate_subset_min > 0:
            candidate_target = max(candidate_target, self.tail_candidate_subset_min)
        if self.tail_candidate_subset_max > 0:
            candidate_target = min(candidate_target, self.tail_candidate_subset_max)
        return min(active_count, max(required_count, candidate_target))

    def _sample_tail_candidate_positions(self, active_count, candidate_count, device):
        if candidate_count >= active_count:
            return torch.arange(active_count, device=device, dtype=torch.long)
        return torch.randperm(active_count, device=device)[:candidate_count]

    def _mid_hard_candidate_target_count(self, active_count, hard_count):
        if active_count <= 0:
            return 0
        required_count = min(active_count, max(1, int(hard_count)))
        ratio_target = int(math.ceil(float(active_count) * self.mid_hard_candidate_subset_ratio))
        candidate_target = max(required_count, hard_count * self.hard_candidate_multiplier, ratio_target)
        if self.mid_hard_candidate_subset_min > 0:
            candidate_target = max(candidate_target, self.mid_hard_candidate_subset_min)
        if self.mid_hard_candidate_subset_max > 0:
            candidate_target = min(candidate_target, self.mid_hard_candidate_subset_max)
        return min(active_count, max(required_count, candidate_target))

    def _sample_mid_hard_candidate_positions(self, active_count, candidate_count, device):
        if candidate_count >= active_count:
            return torch.arange(active_count, device=device, dtype=torch.long)
        return torch.randperm(active_count, device=device)[:candidate_count]

    def _compute_min_sparse_distance(self, query_points, sparse_points):
        if query_points.numel() == 0:
            return torch.zeros((0,), device=sparse_points.device, dtype=sparse_points.dtype)
        min_distances = []
        for start in range(0, int(query_points.shape[0]), self.distance_chunk_size):
            end = min(start + self.distance_chunk_size, int(query_points.shape[0]))
            chunk = query_points[start:end]
            chunk_min = torch.cdist(chunk, sparse_points).min(dim=1).values
            min_distances.append(chunk_min)
        return torch.cat(min_distances, dim=0)

    def _build_support_neighborhood(self, query_points, sparse_points, support_score):
        if query_points.numel() == 0:
            zero = torch.zeros((0, 1), device=sparse_points.device, dtype=sparse_points.dtype)
            return {
                "knn_dist": zero,
                "knn_idx": torch.zeros((0, 1), device=sparse_points.device, dtype=torch.long),
                "knn_points": torch.zeros((0, 1, 3), device=sparse_points.device, dtype=sparse_points.dtype),
                "knn_weights": zero,
                "barycenter": torch.zeros((0, 3), device=sparse_points.device, dtype=sparse_points.dtype),
                "neighbor_k": 1,
            }

        neighbor_k = min(max(self.knn_k, self.plane_k), int(sparse_points.shape[0]))
        distances = torch.cdist(query_points, sparse_points)
        knn_dist, knn_idx = torch.topk(distances, k=neighbor_k, dim=1, largest=False)
        knn_points = sparse_points[knn_idx]
        knn_support = support_score[knn_idx]
        knn_weights = knn_support / knn_dist.clamp_min(self.knn_eps)
        knn_weights = knn_weights / knn_weights.sum(dim=1, keepdim=True).clamp_min(self.knn_eps)
        barycenter = (knn_points * knn_weights.unsqueeze(-1)).sum(dim=1)
        return {
            "knn_dist": knn_dist,
            "knn_idx": knn_idx,
            "knn_points": knn_points,
            "knn_weights": knn_weights,
            "barycenter": barycenter,
            "neighbor_k": neighbor_k,
        }

    def _charbonnier_penalty(self, values):
        eps = values.new_tensor(self.knn_eps)
        return torch.sqrt(values.square() + eps) - torch.sqrt(eps)

    def _extract_surface_aligned_scales(self, gaussian_scales, rotation, normal):
        scale_values = torch.exp(gaussian_scales)
        if rotation is None:
            gaussian_normal_scale = scale_values[:, 2]
            gaussian_tangent_scale = scale_values[:, :2].mean(dim=1)
            return scale_values, gaussian_tangent_scale, gaussian_normal_scale, None

        axis_alignment = torch.abs(torch.einsum("bij,bj->bi", rotation.transpose(1, 2), normal))
        normal_axis = axis_alignment.argmax(dim=1, keepdim=True)
        gaussian_normal_scale = torch.gather(scale_values, 1, normal_axis).squeeze(1)
        gaussian_tangent_scale = (scale_values.sum(dim=1) - gaussian_normal_scale) * 0.5
        return scale_values, gaussian_tangent_scale, gaussian_normal_scale, axis_alignment

    def _row_quantile(self, values, q):
        if values.ndim != 2:
            raise RuntimeError(f"Expected [B,N] tensor for row quantile, got {tuple(values.shape)}")
        if values.shape[1] == 0:
            return torch.zeros((int(values.shape[0]),), device=values.device, dtype=values.dtype)
        if values.shape[1] == 1:
            return values[:, 0]
        q = float(min(max(q, 0.0), 1.0))
        sorted_values, _ = torch.sort(values, dim=1)
        quantile_index = int(round(q * float(values.shape[1] - 1)))
        return sorted_values[:, quantile_index]

    def _build_scale_targets(self, neighborhood, normal, eigvals, eigvecs):
        knn_points = neighborhood["knn_points"]
        neighbor_k = int(knn_points.shape[1])
        if neighbor_k > 1:
            anchor_points = knn_points[:, :1, :]
            anchor_offsets = knn_points[:, 1:] - anchor_points
            normal_view = normal.unsqueeze(1)
            signed_normal_offsets = torch.sum(anchor_offsets * normal_view, dim=2)
            tangent_basis = eigvecs[:, :, 1:3]
            tangent_axis_offsets = torch.abs(torch.einsum("bnd,bdk->bnk", anchor_offsets, tangent_basis))
            anchor_radius = torch.norm(anchor_offsets, dim=2)
            local_radius = anchor_radius.mean(dim=1)
            tangent_extent = 0.5 * (
                self._row_quantile(tangent_axis_offsets[:, :, 0], self.target_tangent_quantile)
                + self._row_quantile(tangent_axis_offsets[:, :, 1], self.target_tangent_quantile)
            )
            normal_extent = self._row_quantile(signed_normal_offsets.abs(), self.target_normal_quantile)
            tangent_std = 0.5 * (
                torch.sqrt(eigvals[:, 1].clamp_min(self.knn_eps))
                + torch.sqrt(eigvals[:, 2].clamp_min(self.knn_eps))
            )
            normal_std = torch.sqrt(eigvals[:, 0].clamp_min(self.knn_eps))
            tangent_extent = torch.minimum(tangent_extent, tangent_std * self.target_tangent_std_cap_ratio)
            tangent_extent = self.target_tangent_std_blend * tangent_std + (1.0 - self.target_tangent_std_blend) * tangent_extent
            tangent_extent = torch.maximum(tangent_extent, tangent_std * self.target_tangent_std_floor_ratio)
            normal_extent = torch.maximum(normal_extent, normal_std)
        else:
            local_radius = neighborhood["knn_dist"][:, 0]
            tangent_extent = local_radius
            normal_extent = local_radius
        tangent_scale_cap = torch.minimum(
            torch.full_like(local_radius, self.target_tangent_scale_max),
            local_radius * self.target_tangent_local_radius_cap_ratio,
        )
        tangent_scale_cap = torch.maximum(tangent_scale_cap, torch.full_like(local_radius, self.target_tangent_scale_min))
        target_tangent_scale = torch.clamp(
            tangent_extent * self.tangent_scale_ratio,
            min=self.target_tangent_scale_min,
        )
        target_tangent_scale = torch.minimum(target_tangent_scale, tangent_scale_cap)
        target_normal_scale = torch.clamp(
            normal_extent * self.normal_scale_ratio,
            min=self.target_normal_scale_min,
            max=self.target_normal_scale_max,
        )
        return local_radius, target_tangent_scale, target_normal_scale, tangent_scale_cap

    def _compute_residual_bundle(self, query_points, neighborhood, gaussian_quats=None, gaussian_scales=None, tail_phase_active=False):
        barycenter = neighborhood["barycenter"]
        barycenter_residual = torch.norm(query_points - barycenter, dim=1)
        zero = torch.zeros_like(barycenter_residual)

        if self.mode != "point_to_plane" or neighborhood["neighbor_k"] < 3:
            return {
                "residual": barycenter_residual,
                "barycenter_residual": barycenter_residual,
                "point_to_plane_residual": zero,
                "plane_residual": barycenter_residual,
                "normal_residual": zero,
                "tangent_residual": zero,
                "normal_scale_penalty": zero,
                "orientation_alignment": zero,
                "orientation_loss": zero,
                "anisotropic_scale_target": zero,
                "anisotropic_scale_target_loss": zero,
                "normal_scale_loss": zero,
                "local_radius": zero,
                "target_tangent_scale": zero,
                "target_normal_scale": zero,
                "target_tangent_scale_cap": zero,
                "gaussian_tangent_scale": zero,
                "gaussian_normal_scale": zero,
                "monitor_residual": barycenter_residual,
                "plane_confidence": zero,
                "orientation_alignment_raw": zero,
                "anisotropic_scale_target_raw": zero,
                "tail_point_to_plane_active": False,
                "tail_confidence_mask": torch.zeros_like(barycenter_residual, dtype=torch.bool),
                "tail_confidence_weight": zero,
                "stable_mask": torch.zeros_like(barycenter_residual, dtype=torch.bool),
                "fallback_mask": torch.ones_like(barycenter_residual, dtype=torch.bool),
            }

        knn_points = neighborhood["knn_points"]
        knn_weights = neighborhood["knn_weights"]
        centered = knn_points - barycenter.unsqueeze(1)
        cov = torch.matmul((knn_weights.unsqueeze(-1) * centered).transpose(1, 2), centered)
        cov = cov + self.plane_eps * torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        normal = F.normalize(eigvecs[:, :, 0], dim=-1, eps=self.knn_eps)
        eigen_gap = eigvals[:, 1] - eigvals[:, 0]
        stable_mask = torch.isfinite(eigen_gap) & torch.isfinite(normal).all(dim=1) & (eigen_gap >= self.plane_min_eigen_gap)
        plane_confidence = eigen_gap / (eigen_gap + self.plane_min_eigen_gap + self.knn_eps)
        plane_confidence = torch.where(torch.isfinite(plane_confidence), plane_confidence.clamp(0.0, 1.0), zero)

        delta = query_points - barycenter
        signed_normal = torch.sum(delta * normal, dim=1)
        normal_residual = signed_normal.abs()
        tangent_vec = delta - signed_normal.unsqueeze(-1) * normal
        tangent_residual = torch.norm(tangent_vec, dim=1)
        point_to_plane_residual = normal_residual + self.tangent_weight * tangent_residual
        plane_residual = point_to_plane_residual
        tail_point_to_plane_residual = torch.where(stable_mask, point_to_plane_residual, zero)
        local_radius, target_tangent_scale, target_normal_scale, target_tangent_scale_cap = self._build_scale_targets(neighborhood, normal, eigvals, eigvecs)
        tail_confidence_mask = stable_mask
        if self.tail_point_to_plane_no_fallback:
            tail_confidence_mask = tail_confidence_mask & (plane_confidence >= self.tail_point_to_plane_min_confidence)
        tail_confidence_weight = torch.where(
            tail_confidence_mask,
            plane_confidence.clamp_min(self.knn_eps).pow(self.tail_point_to_plane_confidence_power),
            zero,
        )

        rotation = None
        if gaussian_quats is not None:
            rotation = _quaternion_to_rotation_matrix(gaussian_quats)

        orientation_alignment = zero
        orientation_alignment_raw = zero
        orientation_loss = zero
        if rotation is not None and self.orientation_enabled and self.orientation_weight > 0.0:
            gaussian_normal_axis = F.normalize(rotation[:, :, 2], dim=-1, eps=self.knn_eps)
            orientation_alignment_raw = 1.0 - torch.abs(torch.sum(gaussian_normal_axis * normal, dim=1)).clamp(0.0, 1.0)
            orientation_alignment = orientation_alignment_raw
            orientation_alignment = torch.where(stable_mask, orientation_alignment, zero)
            orientation_loss = self.orientation_weight * orientation_alignment
            plane_residual = plane_residual + orientation_loss

        gaussian_tangent_scale = zero
        gaussian_normal_scale = zero
        anisotropic_scale_target = zero
        anisotropic_scale_target_raw = zero
        anisotropic_scale_target_loss = zero
        normal_scale_loss = zero
        if gaussian_scales is not None:
            _, gaussian_tangent_scale, gaussian_normal_scale, _ = self._extract_surface_aligned_scales(
                gaussian_scales,
                rotation,
                normal,
            )
            if self.anisotropic_scale_target_enabled and self.anisotropic_scale_target_weight > 0.0:
                tangent_delta = gaussian_tangent_scale - target_tangent_scale
                normal_delta = gaussian_normal_scale - target_normal_scale
                anisotropic_scale_target_raw = 0.5 * (
                    self._charbonnier_penalty(tangent_delta) + self._charbonnier_penalty(normal_delta)
                )
                anisotropic_scale_target = anisotropic_scale_target_raw
                anisotropic_scale_target = torch.where(stable_mask, anisotropic_scale_target, zero)
                anisotropic_scale_target_loss = self.anisotropic_scale_target_weight * anisotropic_scale_target
                plane_residual = plane_residual + anisotropic_scale_target_loss

        normal_scale_penalty = zero
        if gaussian_quats is not None and gaussian_scales is not None and self.normal_scale_weight > 0.0:
            if rotation is None:
                rotation = _quaternion_to_rotation_matrix(gaussian_quats)
            _, tangent_scale, normal_scale, _ = self._extract_surface_aligned_scales(
                gaussian_scales,
                rotation,
                normal,
            )
            normal_scale_penalty = F.relu(normal_scale - tangent_scale)
            normal_scale_penalty = torch.where(stable_mask, normal_scale_penalty, zero)
            normal_scale_loss = self.normal_scale_weight * normal_scale_penalty
            plane_residual = plane_residual + normal_scale_loss

        monitor_residual = torch.where(stable_mask, point_to_plane_residual, barycenter_residual)
        tail_point_to_plane_active = False
        if tail_phase_active:
            tail_residual = zero
            if self.tail_keep_point_to_plane and self.tail_point_to_plane_weight_scale > 0.0:
                tail_residual = tail_residual + self.tail_point_to_plane_weight_scale * tail_point_to_plane_residual * tail_confidence_weight
                tail_point_to_plane_active = bool(tail_confidence_mask.any().item())
            if self.tail_keep_orientation:
                tail_residual = tail_residual + orientation_loss * tail_confidence_weight
            if self.tail_keep_anisotropic_scale:
                tail_residual = tail_residual + anisotropic_scale_target_loss * tail_confidence_weight
            if self.tail_keep_normal_scale:
                tail_residual = tail_residual + normal_scale_loss * tail_confidence_weight
            residual = torch.where(tail_confidence_mask, tail_residual, zero)
        else:
            residual = torch.where(stable_mask, plane_residual, barycenter_residual)
        return {
            "residual": residual,
            "barycenter_residual": barycenter_residual,
            "monitor_residual": monitor_residual,
            "point_to_plane_residual": point_to_plane_residual,
            "plane_residual": plane_residual,
            "normal_residual": normal_residual,
            "tangent_residual": tangent_residual,
            "normal_scale_penalty": normal_scale_penalty,
            "normal_scale_loss": normal_scale_loss,
            "orientation_alignment": orientation_alignment,
            "orientation_alignment_raw": orientation_alignment_raw,
            "orientation_loss": orientation_loss,
            "anisotropic_scale_target": anisotropic_scale_target,
            "anisotropic_scale_target_raw": anisotropic_scale_target_raw,
            "anisotropic_scale_target_loss": anisotropic_scale_target_loss,
            "local_radius": local_radius,
            "target_tangent_scale": target_tangent_scale,
            "target_normal_scale": target_normal_scale,
            "target_tangent_scale_cap": target_tangent_scale_cap,
            "gaussian_tangent_scale": gaussian_tangent_scale,
            "gaussian_normal_scale": gaussian_normal_scale,
            "plane_confidence": plane_confidence,
            "tail_point_to_plane_active": tail_point_to_plane_active,
            "tail_confidence_mask": tail_confidence_mask,
            "tail_confidence_weight": tail_confidence_weight,
            "stable_mask": stable_mask,
            "fallback_mask": ~stable_mask,
        }

    def _global_mining_difficulty_mode(self):
        if self.difficulty_score == "vsurface_mixed":
            return "min_sparse_dist"
        return self.difficulty_score

    def _compute_mid_hard_surface_payload(self, min_sparse_distance, residual_bundle, opacity_values):
        local_radius = residual_bundle["local_radius"].clamp_min(self.knn_eps)
        normalized_distance = (min_sparse_distance / local_radius).clamp(0.0, 4.0)
        normalized_normal = (residual_bundle["normal_residual"] / local_radius).clamp(0.0, 4.0)
        normalized_tangent = (residual_bundle["tangent_residual"] / local_radius).clamp(0.0, 4.0)
        if opacity_values is None:
            opacity_weight = torch.ones_like(normalized_distance)
        else:
            opacity_weight = opacity_values.clamp(0.0, 1.0).pow(self.mid_hard_opacity_weight_power)
        mid_hard_score = opacity_weight * (
            self.mid_hard_normal_weight * normalized_normal
            + self.mid_hard_tangent_weight * normalized_tangent
            + self.mid_hard_orientation_weight * residual_bundle["orientation_alignment_raw"]
            + self.mid_hard_scale_weight * residual_bundle["anisotropic_scale_target_raw"]
        )
        stable_mask = residual_bundle["stable_mask"]
        if bool(stable_mask.any().item()):
            stable_distance = normalized_distance[stable_mask]
            band_quantiles = torch.quantile(
                stable_distance,
                stable_distance.new_tensor([self.mid_hard_distance_q_low, self.mid_hard_distance_q_high]),
            )
            distance_low = band_quantiles[0]
            distance_high = band_quantiles[1]
            mid_hard_mask = stable_mask & (normalized_distance >= distance_low) & (normalized_distance <= distance_high)
        else:
            distance_low = normalized_distance.new_zeros(())
            distance_high = normalized_distance.new_zeros(())
            mid_hard_mask = torch.zeros_like(stable_mask, dtype=torch.bool)
        return {
            "normalized_distance": normalized_distance,
            "normalized_normal": normalized_normal,
            "normalized_tangent": normalized_tangent,
            "mid_hard_score": mid_hard_score,
            "mid_hard_mask": mid_hard_mask,
            "stable_mask": stable_mask,
            "distance_low": distance_low,
            "distance_high": distance_high,
        }

    def _compute_difficulty_scores(
        self,
        query_points,
        sparse_points,
        support_score,
        opacity_values=None,
        gaussian_quats=None,
        gaussian_scales=None,
        difficulty_mode=None,
        residual_bundle=None,
        min_sparse_distance=None,
    ):
        difficulty_mode = self.difficulty_score if difficulty_mode is None else str(difficulty_mode).lower()
        if difficulty_mode == "min_sparse_dist":
            if min_sparse_distance is None:
                min_sparse_distance = self._compute_min_sparse_distance(query_points, sparse_points)
            return min_sparse_distance
        if difficulty_mode == "plane_residual":
            if residual_bundle is None:
                neighborhood = self._build_support_neighborhood(query_points, sparse_points, support_score)
                residual_bundle = self._compute_residual_bundle(query_points, neighborhood, gaussian_quats, gaussian_scales, tail_phase_active=False)
            return residual_bundle["plane_residual"]
        if difficulty_mode == "vsurface_mixed":
            if residual_bundle is None:
                neighborhood = self._build_support_neighborhood(query_points, sparse_points, support_score)
                residual_bundle = self._compute_residual_bundle(query_points, neighborhood, gaussian_quats, gaussian_scales, tail_phase_active=False)
            local_radius = residual_bundle["local_radius"].clamp_min(self.knn_eps)
            if min_sparse_distance is None:
                min_sparse_distance = self._compute_min_sparse_distance(query_points, sparse_points)
            normalized_distance = (min_sparse_distance / local_radius).clamp(0.0, 4.0)
            normalized_normal = (residual_bundle["normal_residual"] / local_radius).clamp(0.0, 4.0)
            return (
                self.difficulty_distance_weight * normalized_distance
                + self.difficulty_orientation_weight * residual_bundle["orientation_alignment_raw"]
                + self.difficulty_scale_weight * residual_bundle["anisotropic_scale_target_raw"]
                + self.difficulty_normal_weight * normalized_normal
            )
        if difficulty_mode == "stable_surface_mixed":
            if residual_bundle is None:
                neighborhood = self._build_support_neighborhood(query_points, sparse_points, support_score)
                residual_bundle = self._compute_residual_bundle(query_points, neighborhood, gaussian_quats, gaussian_scales, tail_phase_active=False)
            local_radius = residual_bundle["local_radius"].clamp_min(self.knn_eps)
            if min_sparse_distance is None:
                min_sparse_distance = self._compute_min_sparse_distance(query_points, sparse_points)
            normalized_distance = (min_sparse_distance / local_radius).clamp(0.0, 4.0)
            normalized_normal = (residual_bundle["normal_residual"] / local_radius).clamp(0.0, 4.0)
            base_score = (
                self.tail_difficulty_distance_weight * normalized_distance
                + self.tail_difficulty_orientation_weight * residual_bundle["orientation_alignment_raw"]
                + self.tail_difficulty_scale_weight * residual_bundle["anisotropic_scale_target_raw"]
                + self.tail_difficulty_normal_weight * normalized_normal
            )
            confidence_term = self._tail_confidence_term(residual_bundle["plane_confidence"])
            stable_gate = residual_bundle["stable_mask"].to(base_score.dtype)
            return base_score * confidence_term * stable_gate
        if difficulty_mode == "mid_hard_surface":
            if residual_bundle is None:
                neighborhood = self._build_support_neighborhood(query_points, sparse_points, support_score)
                residual_bundle = self._compute_residual_bundle(query_points, neighborhood, gaussian_quats, gaussian_scales, tail_phase_active=False)
            if min_sparse_distance is None:
                min_sparse_distance = self._compute_min_sparse_distance(query_points, sparse_points)
            payload = self._compute_mid_hard_surface_payload(min_sparse_distance, residual_bundle, opacity_values)
            return payload["mid_hard_score"]
        raise RuntimeError(f"Unsupported sparse difficulty score: {difficulty_mode}")

    def _mine_global_hard_positions(self, means, active_indices, sparse_points, support_score, context, hard_count, opacity_values=None, quats=None, scales=None):
        step = int(context.get("step", 0))
        if (
            self._cached_hard_positions is not None
            and self._cached_hard_positions_step >= 0
            and step - self._cached_hard_positions_step < self.global_mining_refresh_interval
            and self._cached_hard_active_count == int(active_indices.numel())
            and self._cached_hard_count == int(hard_count)
        ):
            return self._cached_hard_positions

        top_scores = None
        top_positions = None
        active_count = int(active_indices.numel())
        difficulty_mode = self._global_mining_difficulty_mode()
        mid_hard_candidate_count = 0
        mid_hard_distance_sum = 0.0
        mid_hard_score_sum = 0.0
        scan_positions = None
        scan_count = active_count
        if difficulty_mode == "mid_hard_surface":
            scan_count = self._mid_hard_candidate_target_count(active_count, hard_count)
            scan_positions = self._sample_mid_hard_candidate_positions(active_count, scan_count, active_indices.device)
        else:
            scan_positions = torch.arange(active_count, device=active_indices.device, dtype=torch.long)
        for start in range(0, scan_count, self.global_mining_chunk_size):
            end = min(start + self.global_mining_chunk_size, scan_count)
            chunk_positions = scan_positions[start:end]
            chunk_indices = active_indices[chunk_positions]
            with torch.no_grad():
                chunk_means = means[chunk_indices].detach()
                chunk_opacity = None if opacity_values is None else opacity_values[chunk_indices].detach()
                chunk_quats = None if quats is None else quats[chunk_indices].detach()
                chunk_scales = None if scales is None else scales[chunk_indices].detach()
                if difficulty_mode == "mid_hard_surface":
                    neighborhood = self._build_support_neighborhood(chunk_means, sparse_points, support_score)
                    residual_bundle = self._compute_residual_bundle(chunk_means, neighborhood, chunk_quats, chunk_scales, tail_phase_active=False)
                    chunk_min_sparse_distance = self._compute_min_sparse_distance(chunk_means, sparse_points)
                    mid_hard_payload = self._compute_mid_hard_surface_payload(chunk_min_sparse_distance, residual_bundle, chunk_opacity)
                    mid_hard_mask = mid_hard_payload["mid_hard_mask"]
                    if bool(mid_hard_mask.any().item()):
                        candidate_mask = mid_hard_mask
                    elif bool(mid_hard_payload["stable_mask"].any().item()):
                        candidate_mask = mid_hard_payload["stable_mask"]
                    else:
                        candidate_mask = torch.ones_like(mid_hard_mask, dtype=torch.bool)
                    chunk_scores = mid_hard_payload["mid_hard_score"][candidate_mask]
                    chunk_positions = chunk_positions[candidate_mask]
                    mid_hard_count = int(mid_hard_mask.sum().item())
                    mid_hard_candidate_count += mid_hard_count
                    if mid_hard_count > 0:
                        mid_hard_distance_sum += float(mid_hard_payload["normalized_distance"][mid_hard_mask].sum().item())
                        mid_hard_score_sum += float(mid_hard_payload["mid_hard_score"][mid_hard_mask].sum().item())
                else:
                    chunk_scores = self._compute_difficulty_scores(
                        chunk_means,
                        sparse_points,
                        support_score,
                        opacity_values=chunk_opacity,
                        gaussian_quats=chunk_quats,
                        gaussian_scales=chunk_scales,
                        difficulty_mode=difficulty_mode,
                    )

            if top_scores is None:
                keep = min(hard_count, int(chunk_scores.numel()))
                keep_pos = torch.topk(chunk_scores, k=keep, largest=True).indices
                top_scores = chunk_scores[keep_pos]
                top_positions = chunk_positions[keep_pos]
                continue

            combined_scores = torch.cat([top_scores, chunk_scores], dim=0)
            combined_positions = torch.cat([top_positions, chunk_positions], dim=0)
            keep = min(hard_count, int(combined_scores.numel()))
            keep_pos = torch.topk(combined_scores, k=keep, largest=True).indices
            top_scores = combined_scores[keep_pos]
            top_positions = combined_positions[keep_pos]

        if top_positions is None:
            top_positions = torch.zeros((0,), device=active_indices.device, dtype=torch.long)
        payload = {
            "positions": top_positions,
            "mid_hard_candidate_ratio": float(mid_hard_candidate_count / max(1, active_count)) if difficulty_mode == "mid_hard_surface" else 0.0,
            "mid_hard_band_distance_mean": float(mid_hard_distance_sum / max(1, mid_hard_candidate_count)) if difficulty_mode == "mid_hard_surface" else 0.0,
            "mid_hard_score_mean": float(mid_hard_score_sum / max(1, mid_hard_candidate_count)) if difficulty_mode == "mid_hard_surface" else 0.0,
            "mid_hard_scan_candidate_count": float(scan_count if difficulty_mode == "mid_hard_surface" else active_count),
            "mid_hard_scan_candidate_ratio": float(scan_count / max(1, active_count)) if difficulty_mode == "mid_hard_surface" else 1.0,
        }
        self._cached_hard_positions = payload
        self._cached_hard_positions_step = step
        self._cached_hard_active_count = active_count
        self._cached_hard_count = int(hard_count)
        return payload

    def _mine_tail_hard_payload(self, means, active_indices, sparse_points, support_score, context, hard_count, quats=None, scales=None):
        step = int(context.get("step", 0))
        if (
            self._cached_tail_hard_payload is not None
            and self._cached_tail_hard_step >= 0
            and step - self._cached_tail_hard_step < self.tail_global_mining_refresh_interval
            and self._cached_tail_hard_active_count == int(active_indices.numel())
            and self._cached_tail_hard_count == int(hard_count)
        ):
            return self._cached_tail_hard_payload

        top_high_scores = None
        top_high_positions = None
        top_stable_scores = None
        top_stable_positions = None
        high_conf_count = 0
        stable_count = 0
        active_count = int(active_indices.numel())
        candidate_count = self._tail_candidate_target_count(active_count, hard_count)
        candidate_positions = self._sample_tail_candidate_positions(active_count, candidate_count, active_indices.device)
        for start in range(0, candidate_count, self.global_mining_chunk_size):
            end = min(start + self.global_mining_chunk_size, candidate_count)
            chunk_positions = candidate_positions[start:end]
            chunk_indices = active_indices[chunk_positions]
            with torch.no_grad():
                chunk_means = means[chunk_indices].detach()
                chunk_quats = None if quats is None else quats[chunk_indices].detach()
                chunk_scales = None if scales is None else scales[chunk_indices].detach()
                neighborhood = self._build_support_neighborhood(chunk_means, sparse_points, support_score)
                residual_bundle = self._compute_residual_bundle(chunk_means, neighborhood, chunk_quats, chunk_scales, tail_phase_active=False)
                chunk_min_sparse_distance = self._compute_min_sparse_distance(chunk_means, sparse_points)
                chunk_scores = self._compute_difficulty_scores(
                    chunk_means,
                    sparse_points,
                    support_score,
                    gaussian_quats=chunk_quats,
                    gaussian_scales=chunk_scales,
                    difficulty_mode=self.tail_difficulty_score,
                    residual_bundle=residual_bundle,
                    min_sparse_distance=chunk_min_sparse_distance,
                )
                stable_mask = residual_bundle["stable_mask"]
                high_conf_mask = stable_mask & (residual_bundle["plane_confidence"] >= self.tail_min_plane_confidence)
                stable_only_mask = stable_mask & ~high_conf_mask
                high_conf_count += int(high_conf_mask.sum().item())
                stable_count += int(stable_mask.sum().item())
                top_high_scores, top_high_positions = self._update_topk(
                    top_high_scores,
                    top_high_positions,
                    chunk_scores[high_conf_mask],
                    chunk_positions[high_conf_mask],
                    hard_count,
                )
                top_stable_scores, top_stable_positions = self._update_topk(
                    top_stable_scores,
                    top_stable_positions,
                    chunk_scores[stable_only_mask],
                    chunk_positions[stable_only_mask],
                    hard_count,
                )

        if top_high_positions is None:
            top_high_positions = torch.zeros((0,), device=active_indices.device, dtype=torch.long)
        if top_stable_positions is None:
            top_stable_positions = torch.zeros((0,), device=active_indices.device, dtype=torch.long)
        payload = {
            "high_conf_positions": top_high_positions,
            "stable_positions": top_stable_positions,
            "high_conf_candidate_ratio": float(high_conf_count / max(1, candidate_count)),
            "stable_candidate_ratio": float(stable_count / max(1, candidate_count)),
            "scan_candidate_ratio": float(candidate_count / max(1, active_count)),
            "candidate_count": float(candidate_count),
        }
        self._cached_tail_hard_payload = payload
        self._cached_tail_hard_step = step
        self._cached_tail_hard_active_count = active_count
        self._cached_tail_hard_count = int(hard_count)
        return payload

    def _sample_active_indices(self, means, active_indices, sparse_points, support_score, context, opacity_values=None, quats=None, scales=None, target_sample_points=None):
        active_count = int(active_indices.numel())
        sample_points = self.sample_points if target_sample_points is None else int(max(0, target_sample_points))
        effective_sampling_mode = self._effective_sampling_mode(context)
        effective_difficulty_mode = self._effective_difficulty_mode(context, effective_sampling_mode)
        effective_hard_ratio = self.tail_hard_ratio if effective_sampling_mode == "stable_surface_mixed" else self.hard_ratio
        effective_random_sample_fallback = self.tail_random_sample_fallback if effective_sampling_mode == "stable_surface_mixed" else self.random_sample_fallback
        zero_sampling_logs = {
            "tail_stable_candidate_ratio": 0.0,
            "tail_high_conf_candidate_ratio": 0.0,
            "tail_sample_high_conf_ratio": 0.0,
            "tail_scan_candidate_ratio": 0.0,
            "mid_hard_candidate_ratio": 0.0,
            "mid_hard_band_distance_mean": 0.0,
            "mid_hard_score_mean": 0.0,
            "mid_hard_scan_candidate_count": 0.0,
            "mid_hard_scan_candidate_ratio": 0.0,
        }
        if sample_points <= 0:
            return torch.zeros((0,), device=active_indices.device, dtype=active_indices.dtype), {
                "hard_sample_count": 0.0,
                "random_sample_count": 0.0,
                "candidate_count": float(active_count),
                "sampling_mode": effective_sampling_mode,
                "difficulty_score_mode": effective_difficulty_mode,
                "hard_ratio": float(effective_hard_ratio),
                **zero_sampling_logs,
            }

        if active_count <= sample_points:
            return active_indices, {
                "hard_sample_count": 0.0,
                "random_sample_count": float(active_count),
                "candidate_count": float(active_count),
                "sampling_mode": effective_sampling_mode,
                "difficulty_score_mode": effective_difficulty_mode,
                "hard_ratio": float(effective_hard_ratio),
                **zero_sampling_logs,
            }

        if effective_sampling_mode == "random":
            perm = torch.randperm(active_count, device=active_indices.device)[:sample_points]
            return active_indices[perm], {
                "hard_sample_count": 0.0,
                "random_sample_count": float(sample_points),
                "candidate_count": float(sample_points),
                "sampling_mode": effective_sampling_mode,
                "difficulty_score_mode": effective_difficulty_mode,
                "hard_ratio": float(effective_hard_ratio),
                **zero_sampling_logs,
            }

        if effective_sampling_mode == "stable_surface_mixed":
            hard_count = int(round(sample_points * effective_hard_ratio))
            hard_count = min(max(0, hard_count), sample_points, active_count)
            if hard_count <= 0:
                perm = torch.randperm(active_count, device=active_indices.device)[:sample_points]
                return active_indices[perm], {
                    "hard_sample_count": 0.0,
                    "random_sample_count": float(sample_points),
                    "candidate_count": float(sample_points),
                    "sampling_mode": effective_sampling_mode,
                    "difficulty_score_mode": effective_difficulty_mode,
                    "hard_ratio": float(effective_hard_ratio),
                    **zero_sampling_logs,
                }

            stable_target = min(sample_points, max(hard_count, int(round(sample_points * self.tail_stable_sample_ratio_floor))))
            tail_payload = self._mine_tail_hard_payload(
                means,
                active_indices,
                sparse_points,
                support_score,
                context,
                stable_target,
                quats,
                scales,
            )
            selected_high_positions = tail_payload["high_conf_positions"][: min(stable_target, int(tail_payload["high_conf_positions"].numel()))]
            remaining_stable = max(0, stable_target - int(selected_high_positions.numel()))
            selected_stable_positions = tail_payload["stable_positions"][: min(remaining_stable, int(tail_payload["stable_positions"].numel()))]
            hard_positions = torch.cat([selected_high_positions, selected_stable_positions], dim=0)
            hard_indices = active_indices[hard_positions]
            random_count = sample_points - int(hard_indices.numel())
            if random_count > 0 and effective_random_sample_fallback:
                remaining_mask = torch.ones((active_count,), dtype=torch.bool, device=active_indices.device)
                remaining_mask[hard_positions] = False
                remaining_indices = active_indices[remaining_mask]
                random_count = min(random_count, int(remaining_indices.numel()))
                random_perm = torch.randperm(int(remaining_indices.numel()), device=active_indices.device)[:random_count]
                random_indices = remaining_indices[random_perm]
                sampled_indices = torch.cat([hard_indices, random_indices], dim=0)
            else:
                random_count = 0
                sampled_indices = hard_indices
            shuffle_perm = torch.randperm(int(sampled_indices.numel()), device=active_indices.device)
            sampled_indices = sampled_indices[shuffle_perm]
            return sampled_indices, {
                "hard_sample_count": float(int(hard_indices.numel())),
                "random_sample_count": float(random_count),
                "candidate_count": float(tail_payload["candidate_count"]),
                "sampling_mode": effective_sampling_mode,
                "difficulty_score_mode": effective_difficulty_mode,
                "hard_ratio": float(effective_hard_ratio),
                "tail_stable_candidate_ratio": float(tail_payload["stable_candidate_ratio"]),
                "tail_high_conf_candidate_ratio": float(tail_payload["high_conf_candidate_ratio"]),
                "tail_sample_high_conf_ratio": float(int(selected_high_positions.numel()) / max(1, int(sampled_indices.numel()))),
                "tail_scan_candidate_ratio": float(tail_payload.get("scan_candidate_ratio", 1.0)),
                "mid_hard_candidate_ratio": 0.0,
                "mid_hard_band_distance_mean": 0.0,
                "mid_hard_score_mean": 0.0,
                "mid_hard_scan_candidate_count": 0.0,
                "mid_hard_scan_candidate_ratio": 0.0,
            }

        if effective_sampling_mode not in {"hardest_mixed", "hardest_global_mixed"}:
            raise RuntimeError(f"Unsupported sparse sampling mode: {effective_sampling_mode}")

        hard_count = int(round(sample_points * effective_hard_ratio))
        hard_count = min(max(0, hard_count), sample_points, active_count)
        if hard_count <= 0:
            perm = torch.randperm(active_count, device=active_indices.device)[:sample_points]
            return active_indices[perm], {
                "hard_sample_count": 0.0,
                "random_sample_count": float(sample_points),
                "candidate_count": float(sample_points),
                "sampling_mode": effective_sampling_mode,
                "difficulty_score_mode": effective_difficulty_mode,
                "hard_ratio": float(effective_hard_ratio),
                **zero_sampling_logs,
            }

        hard_payload = None
        if effective_sampling_mode == "hardest_global_mixed":
            requested_hard = sample_points if not effective_random_sample_fallback else hard_count
            hard_payload = self._mine_global_hard_positions(
                means,
                active_indices,
                sparse_points,
                support_score,
                context,
                requested_hard,
                opacity_values,
                quats,
                scales,
            )
            hard_positions = hard_payload["positions"]
            hard_indices = active_indices[hard_positions]
            random_count = sample_points - int(hard_indices.numel())
            if random_count > 0 and effective_random_sample_fallback:
                remaining_mask = torch.ones((active_count,), dtype=torch.bool, device=active_indices.device)
                remaining_mask[hard_positions] = False
                remaining_indices = active_indices[remaining_mask]
                random_count = min(random_count, int(remaining_indices.numel()))
                random_perm = torch.randperm(int(remaining_indices.numel()), device=active_indices.device)[:random_count]
                random_indices = remaining_indices[random_perm]
                sampled_indices = torch.cat([hard_indices, random_indices], dim=0)
            else:
                random_count = 0
                sampled_indices = hard_indices
            candidate_count = float(hard_payload.get("mid_hard_scan_candidate_count", active_count))
            hard_sample_count = float(int(hard_indices.numel()))
        else:
            candidate_count = min(active_count, max(sample_points, hard_count * self.hard_candidate_multiplier))
            candidate_perm = torch.randperm(active_count, device=active_indices.device)[:candidate_count]
            candidate_indices = active_indices[candidate_perm]
            with torch.no_grad():
                candidate_quats = None if quats is None else quats[candidate_indices].detach()
                candidate_scales = None if scales is None else scales[candidate_indices].detach()
                candidate_scores = self._compute_difficulty_scores(
                    means[candidate_indices].detach(),
                    sparse_points,
                    support_score,
                    opacity_values=None if opacity_values is None else opacity_values[candidate_indices].detach(),
                    gaussian_quats=candidate_quats,
                    gaussian_scales=candidate_scales,
                    difficulty_mode=effective_difficulty_mode,
                )
            hard_pos = torch.topk(candidate_scores, k=hard_count, largest=True).indices
            hard_indices = candidate_indices[hard_pos]

            remaining_mask = torch.ones((active_count,), dtype=torch.bool, device=active_indices.device)
            remaining_mask[candidate_perm[hard_pos]] = False
            remaining_indices = active_indices[remaining_mask]

            random_count = sample_points - hard_count
            if random_count > 0 and effective_random_sample_fallback:
                random_count = min(random_count, int(remaining_indices.numel()))
                random_perm = torch.randperm(int(remaining_indices.numel()), device=active_indices.device)[:random_count]
                random_indices = remaining_indices[random_perm]
                sampled_indices = torch.cat([hard_indices, random_indices], dim=0)
            elif random_count > 0:
                candidate_keep = min(candidate_count, sample_points)
                keep_pos = torch.topk(candidate_scores, k=candidate_keep, largest=True).indices
                sampled_indices = candidate_indices[keep_pos]
                random_count = 0
            else:
                sampled_indices = hard_indices
            hard_sample_count = float(hard_count)

        shuffle_perm = torch.randperm(int(sampled_indices.numel()), device=active_indices.device)
        sampled_indices = sampled_indices[shuffle_perm]
        return sampled_indices, {
            "hard_sample_count": hard_sample_count,
            "random_sample_count": float(random_count),
            "candidate_count": float(candidate_count),
            "sampling_mode": effective_sampling_mode,
            "difficulty_score_mode": effective_difficulty_mode,
            "hard_ratio": float(effective_hard_ratio),
            "tail_stable_candidate_ratio": 0.0,
            "tail_high_conf_candidate_ratio": 0.0,
            "tail_sample_high_conf_ratio": 0.0,
            "tail_scan_candidate_ratio": 0.0,
            "mid_hard_candidate_ratio": float(0.0 if hard_payload is None else hard_payload.get("mid_hard_candidate_ratio", 0.0)),
            "mid_hard_band_distance_mean": float(0.0 if hard_payload is None else hard_payload.get("mid_hard_band_distance_mean", 0.0)),
            "mid_hard_score_mean": float(0.0 if hard_payload is None else hard_payload.get("mid_hard_score_mean", 0.0)),
            "mid_hard_scan_candidate_count": float(0.0 if hard_payload is None else hard_payload.get("mid_hard_scan_candidate_count", 0.0)),
            "mid_hard_scan_candidate_ratio": float(0.0 if hard_payload is None else hard_payload.get("mid_hard_scan_candidate_ratio", 0.0)),
        }

    def compute(self, context):
        if not self.is_active(context):
            zero = zero_scalar_like(context)
            return zero, {
                "active": 0.0,
                "sampled": 0.0,
                "distance_mean": 0.0,
                "orientation_loss": 0.0,
                "orientation_alignment_mean": 0.0,
                "orientation_alignment_p50": 0.0,
                "orientation_alignment_p90": 0.0,
                "anisotropic_scale_target_loss": 0.0,
                "target_tangent_scale_mean": 0.0,
                "target_tangent_scale_cap_mean": 0.0,
                "target_normal_scale_mean": 0.0,
                "gaussian_tangent_scale_mean": 0.0,
                "gaussian_normal_scale_mean": 0.0,
                "stable_plane_ratio": 0.0,
                "loss_residual_mean": 0.0,
                "normal_scale_loss": 0.0,
                "tail_phase_active": 0.0,
                "point_to_plane_loss_active": 0.0,
                "tail_stable_candidate_ratio": 0.0,
                "tail_high_conf_candidate_ratio": 0.0,
                "tail_sample_high_conf_ratio": 0.0,
                "tail_scan_candidate_ratio": 0.0,
                "tail_point_to_plane_effective_ratio": 0.0,
                "tail_confidence_mask_ratio": 0.0,
                "tail_plane_confidence_mean": 0.0,
                "local_geometry_mask_ratio": 0.0,
                "local_geometry_newborn_ratio": 0.0,
                "local_geometry_low_opacity_ratio": 0.0,
                "local_geometry_low_render_conf_ratio": 0.0,
                "local_geometry_high_mismatch_ratio": 0.0,
                "local_geometry_supplement_count": 0.0,
                "local_geometry_newborn_quota_count": 0.0,
                "local_geometry_low_opacity_quota_count": 0.0,
                "render_confidence_mean": 0.0,
                "render_confidence_p50": 0.0,
                "render_confidence_p90": 0.0,
            }

        sparse_points = context.get("colmap_sparse_points")
        sparse_track_len = context.get("colmap_sparse_track_len")
        sparse_reproj_error = context.get("colmap_sparse_reproj_error")
        sparse_brightness_score = context.get("colmap_sparse_brightness_score")
        sparse_gradient_score = context.get("colmap_sparse_gradient_score")
        means = context.get("gaussian_means")
        quats = context.get("gaussian_quats")
        scales = context.get("gaussian_scales")
        opacities = context.get("gaussian_opacities")
        if sparse_points is None:
            raise RuntimeError("SparsePointRegularizationLoss requires colmap_sparse_points in context.")
        if means is None or opacities is None:
            raise RuntimeError("SparsePointRegularizationLoss requires gaussian_means and gaussian_opacities in context.")
        if sparse_points.ndim != 2 or sparse_points.shape[1] != 3:
            raise RuntimeError(f"SparsePointRegularizationLoss expects colmap_sparse_points with shape [N,3], got {tuple(sparse_points.shape)}")

        total_gaussians = int(means.shape[0])
        opacity_values = torch.sigmoid(opacities.reshape(-1))
        active_mask = opacity_values > self.min_opacity
        active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        if active_indices.numel() == 0:
            zero = zero_scalar_like(context)
            return zero, {
                "active": 1.0,
                "active_count": 0.0,
                "active_ratio": 0.0,
                "sampled": 0.0,
                "sampled_ratio": 0.0,
                "distance_mean": 0.0,
                "orientation_loss": 0.0,
                "orientation_alignment_mean": 0.0,
                "orientation_alignment_p50": 0.0,
                "orientation_alignment_p90": 0.0,
                "anisotropic_scale_target_loss": 0.0,
                "target_tangent_scale_mean": 0.0,
                "target_tangent_scale_cap_mean": 0.0,
                "target_normal_scale_mean": 0.0,
                "gaussian_tangent_scale_mean": 0.0,
                "gaussian_normal_scale_mean": 0.0,
                "stable_plane_ratio": 0.0,
                "loss_residual_mean": 0.0,
                "normal_scale_loss": 0.0,
                "tail_phase_active": 0.0,
                "point_to_plane_loss_active": 0.0,
                "tail_stable_candidate_ratio": 0.0,
                "tail_high_conf_candidate_ratio": 0.0,
                "tail_sample_high_conf_ratio": 0.0,
                "tail_scan_candidate_ratio": 0.0,
                "tail_point_to_plane_effective_ratio": 0.0,
                "tail_confidence_mask_ratio": 0.0,
                "tail_plane_confidence_mean": 0.0,
                "local_geometry_mask_ratio": 0.0,
                "local_geometry_newborn_ratio": 0.0,
                "local_geometry_low_opacity_ratio": 0.0,
                "local_geometry_low_render_conf_ratio": 0.0,
                "local_geometry_high_mismatch_ratio": 0.0,
                "local_geometry_supplement_count": 0.0,
                "local_geometry_newborn_quota_count": 0.0,
                "local_geometry_low_opacity_quota_count": 0.0,
                "render_confidence_mean": 0.0,
                "render_confidence_p50": 0.0,
                "render_confidence_p90": 0.0,
            }

        quality_score, density_score, brightness_score, gradient_score, loss_support_score, prune_support_score = self._get_sparse_support_scores(
            sparse_points,
            sparse_track_len,
            sparse_reproj_error,
            sparse_brightness_score,
            sparse_gradient_score,
        )
        supplement_indices, supplement_logs = self._build_local_geometry_supplement_indices(
            context,
            means,
            opacity_values,
            sparse_points,
            active_mask,
        )
        main_sample_target = max(0, self.sample_points - int(supplement_indices.numel()))
        sampled_active_indices, sampling_logs = self._sample_active_indices(
            means,
            active_indices,
            sparse_points,
            loss_support_score,
            context,
            opacity_values,
            quats,
            scales,
            target_sample_points=main_sample_target,
        )
        if int(supplement_indices.numel()) > 0:
            sampled_indices = torch.cat([sampled_active_indices, supplement_indices], dim=0)
        else:
            sampled_indices = sampled_active_indices
        sample_count = int(sampled_indices.numel())
        sampled_means = means[sampled_indices]
        sampled_quats = None if quats is None else quats[sampled_indices]
        sampled_scales = None if scales is None else scales[sampled_indices]

        neighborhood = self._build_support_neighborhood(sampled_means, sparse_points, loss_support_score)
        tail_phase_active = self._is_tail_phase(context)
        residual_bundle = self._compute_residual_bundle(sampled_means, neighborhood, sampled_quats, sampled_scales, tail_phase_active=tail_phase_active)
        monitor_dist = residual_bundle["monitor_residual"]
        sampled_opacity = opacity_values[sampled_indices]
        sampled_render_confidence = self._resolve_sampled_render_confidence(context, sampled_indices, sampled_opacity)
        local_geometry_mask, newborn_mask, low_opacity_mask, low_render_conf_mask, high_mismatch_mask = self._build_local_geometry_masks(
            context,
            sampled_indices,
            sampled_opacity,
            sampled_render_confidence,
            mismatch_values=monitor_dist,
        )
        stable_local_geometry_mask = residual_bundle["stable_mask"] & local_geometry_mask
        point_to_plane_residual = residual_bundle["point_to_plane_residual"]
        geometry_regularization = (
            residual_bundle["orientation_loss"]
            + residual_bundle["anisotropic_scale_target_loss"]
            + residual_bundle["normal_scale_loss"]
        )
        if tail_phase_active:
            tail_confidence_mask = residual_bundle["tail_confidence_mask"]
            tail_confidence_weight = residual_bundle["tail_confidence_weight"]
            target_dist = torch.zeros_like(point_to_plane_residual)
            if self.tail_keep_point_to_plane and self.tail_point_to_plane_weight_scale > 0.0:
                target_dist = target_dist + self.tail_point_to_plane_weight_scale * point_to_plane_residual * tail_confidence_weight
            tail_local_geometry_mask = tail_confidence_mask & local_geometry_mask
            tail_geometry_regularization = torch.zeros_like(point_to_plane_residual)
            if self.tail_keep_orientation:
                tail_geometry_regularization = tail_geometry_regularization + residual_bundle["orientation_loss"]
            if self.tail_keep_anisotropic_scale:
                tail_geometry_regularization = tail_geometry_regularization + residual_bundle["anisotropic_scale_target_loss"]
            if self.tail_keep_normal_scale:
                tail_geometry_regularization = tail_geometry_regularization + residual_bundle["normal_scale_loss"]
            target_dist = target_dist + torch.where(
                tail_local_geometry_mask,
                tail_geometry_regularization * tail_confidence_weight,
                torch.zeros_like(point_to_plane_residual),
            )
            effective_local_geometry_mask = tail_local_geometry_mask
        else:
            base_point_to_plane = torch.where(residual_bundle["stable_mask"], point_to_plane_residual, residual_bundle["barycenter_residual"])
            target_dist = base_point_to_plane + torch.where(
                stable_local_geometry_mask,
                geometry_regularization,
                torch.zeros_like(point_to_plane_residual),
            )
            effective_local_geometry_mask = stable_local_geometry_mask
        effective_difficulty_mode = sampling_logs.get("difficulty_score_mode", self._effective_difficulty_mode(context))
        sampled_min_sparse_distance = None
        if effective_difficulty_mode in {"min_sparse_dist", "vsurface_mixed", "stable_surface_mixed", "mid_hard_surface"}:
            sampled_min_sparse_distance = self._compute_min_sparse_distance(sampled_means.detach(), sparse_points)
        difficulty_scores = self._compute_difficulty_scores(
            sampled_means.detach(),
            sparse_points,
            loss_support_score,
            opacity_values=sampled_opacity.detach(),
            gaussian_quats=None if sampled_quats is None else sampled_quats.detach(),
            gaussian_scales=None if sampled_scales is None else sampled_scales.detach(),
            difficulty_mode=effective_difficulty_mode,
            residual_bundle=residual_bundle,
            min_sparse_distance=sampled_min_sparse_distance,
        )
        sampled_mid_hard_payload = None
        sampled_mid_hard_score_logs = {"mid_hard_score_p50": 0.0, "mid_hard_score_p90": 0.0}
        sampled_mid_hard_candidate_ratio = 0.0
        sampled_mid_hard_band_distance_mean = 0.0
        sampled_mid_hard_score_mean = 0.0
        if effective_difficulty_mode == "mid_hard_surface" and sampled_min_sparse_distance is not None:
            sampled_mid_hard_payload = self._compute_mid_hard_surface_payload(
                sampled_min_sparse_distance,
                residual_bundle,
                sampled_opacity.detach(),
            )
            sampled_mid_hard_mask = sampled_mid_hard_payload["mid_hard_mask"]
            sampled_mid_hard_candidate_ratio = float(sampled_mid_hard_mask.float().mean().item())
            if bool(sampled_mid_hard_mask.any().item()):
                sampled_mid_hard_band_distance_mean = float(sampled_mid_hard_payload["normalized_distance"][sampled_mid_hard_mask].mean().item())
                sampled_mid_hard_score_mean = float(sampled_mid_hard_payload["mid_hard_score"][sampled_mid_hard_mask].mean().item())
                sampled_mid_hard_score_logs = _quantile_logs(
                    "mid_hard_score",
                    sampled_mid_hard_payload["mid_hard_score"][sampled_mid_hard_mask],
                    [(0.50, "p50"), (0.90, "p90")],
                )
        if self.robust_scale > 0.0:
            robust_loss = torch.sqrt(target_dist.square() + self.robust_scale * self.robust_scale) - self.robust_scale
        else:
            robust_loss = target_dist
        loss = robust_loss.mean()
        logs = {
            "active": 1.0,
            "active_count": float(opacity_values.gt(self.min_opacity).sum().detach().item()),
            "active_ratio": float(opacity_values.gt(self.min_opacity).float().mean().detach().item()) if total_gaussians > 0 else 0.0,
            "sampled": float(sample_count),
            "sampled_ratio": float(sample_count / max(1, int(opacity_values.gt(self.min_opacity).sum().detach().item()))),
            "distance_mean": float(monitor_dist.detach().mean().item()),
            "knn_dist_mean": float(neighborhood["knn_dist"].detach().mean().item()),
            "knn_k": float(neighborhood["neighbor_k"]),
            "robust_mean": float(robust_loss.detach().mean().item()),
            "loss_residual_mean": float(target_dist.detach().mean().item()),
            "robust_scale": float(self.robust_scale),
            "opacity_mean": float(sampled_opacity.detach().mean().item()),
            "quality_score_mean": float(quality_score.detach().mean().item()),
            "density_score_mean": float(density_score.detach().mean().item()),
            "brightness_score_mean": float(brightness_score.detach().mean().item()),
            "gradient_score_mean": float(gradient_score.detach().mean().item()),
            "support_score_mean": float(loss_support_score.detach().mean().item()),
            "loss_support_score_mean": float(loss_support_score.detach().mean().item()),
            "prune_support_score_mean": float(prune_support_score.detach().mean().item()),
            "mode": self.mode,
            "sampling_mode": sampling_logs.get("sampling_mode", self.sampling_mode),
            "hard_ratio": float(sampling_logs.get("hard_ratio", self.hard_ratio)),
            "difficulty_score_mode": effective_difficulty_mode,
            "hard_sample_count": sampling_logs["hard_sample_count"],
            "random_sample_count": sampling_logs["random_sample_count"],
            "candidate_count": sampling_logs["candidate_count"],
            "tail_stable_candidate_ratio": float(sampling_logs.get("tail_stable_candidate_ratio", 0.0)),
            "tail_high_conf_candidate_ratio": float(sampling_logs.get("tail_high_conf_candidate_ratio", 0.0)),
            "tail_sample_high_conf_ratio": float(sampling_logs.get("tail_sample_high_conf_ratio", 0.0)),
            "tail_scan_candidate_ratio": float(sampling_logs.get("tail_scan_candidate_ratio", 0.0)),
            "mid_hard_candidate_ratio": float(sampling_logs.get("mid_hard_candidate_ratio", sampled_mid_hard_candidate_ratio)),
            "mid_hard_distance_q_low": float(self.mid_hard_distance_q_low) if effective_difficulty_mode == "mid_hard_surface" else 0.0,
            "mid_hard_distance_q_high": float(self.mid_hard_distance_q_high) if effective_difficulty_mode == "mid_hard_surface" else 0.0,
            "mid_hard_band_distance_mean": float(sampling_logs.get("mid_hard_band_distance_mean", sampled_mid_hard_band_distance_mean)),
            "mid_hard_score_mean": float(sampling_logs.get("mid_hard_score_mean", sampled_mid_hard_score_mean)),
            "mid_hard_scan_candidate_count": float(sampling_logs.get("mid_hard_scan_candidate_count", 0.0)),
            "mid_hard_scan_candidate_ratio": float(sampling_logs.get("mid_hard_scan_candidate_ratio", 0.0)),
            "difficulty_mean": float(difficulty_scores.detach().mean().item()),
            "normal_residual_mean": float(residual_bundle["normal_residual"].detach().mean().item()),
            "tangent_residual_mean": float(residual_bundle["tangent_residual"].detach().mean().item()),
            "point_to_plane_fallback_ratio": float(residual_bundle["fallback_mask"].float().mean().detach().item()),
            "normal_scale_loss": _masked_mean(residual_bundle["normal_scale_loss"], effective_local_geometry_mask),
            "orientation_loss": _masked_mean(residual_bundle["orientation_loss"], effective_local_geometry_mask),
            "orientation_alignment_mean": _masked_mean(residual_bundle["orientation_alignment"], residual_bundle["stable_mask"]),
            "anisotropic_scale_target_loss": _masked_mean(residual_bundle["anisotropic_scale_target_loss"], effective_local_geometry_mask),
            "target_tangent_scale_mean": _masked_mean(residual_bundle["target_tangent_scale"], residual_bundle["stable_mask"]),
            "target_tangent_scale_cap_mean": _masked_mean(residual_bundle["target_tangent_scale_cap"], residual_bundle["stable_mask"]),
            "target_normal_scale_mean": _masked_mean(residual_bundle["target_normal_scale"], residual_bundle["stable_mask"]),
            "gaussian_tangent_scale_mean": _masked_mean(residual_bundle["gaussian_tangent_scale"], residual_bundle["stable_mask"]),
            "gaussian_normal_scale_mean": _masked_mean(residual_bundle["gaussian_normal_scale"], residual_bundle["stable_mask"]),
            "stable_plane_ratio": float(residual_bundle["stable_mask"].float().mean().detach().item()),
            "tail_phase_active": float(tail_phase_active),
            "point_to_plane_loss_active": float((not tail_phase_active) or residual_bundle["tail_point_to_plane_active"]),
            "tail_point_to_plane_effective_ratio": float(residual_bundle["tail_confidence_mask"].float().mean().detach().item()) if tail_phase_active else 0.0,
            "tail_confidence_mask_ratio": float(residual_bundle["tail_confidence_mask"].float().mean().detach().item()) if tail_phase_active else 0.0,
            "tail_plane_confidence_mean": _masked_mean(residual_bundle["plane_confidence"], residual_bundle["tail_confidence_mask"]) if tail_phase_active else 0.0,
            "local_geometry_mask_ratio": float(local_geometry_mask.float().mean().detach().item()),
            "local_geometry_newborn_ratio": float(newborn_mask.float().mean().detach().item()),
            "local_geometry_low_opacity_ratio": float(low_opacity_mask.float().mean().detach().item()),
            "local_geometry_low_render_conf_ratio": float(low_render_conf_mask.float().mean().detach().item()),
            "local_geometry_high_mismatch_ratio": float(high_mismatch_mask.float().mean().detach().item()),
            "local_geometry_supplement_count": float(supplement_logs["supplement_count"]),
            "local_geometry_newborn_quota_count": float(supplement_logs["newborn_quota_count"]),
            "local_geometry_low_opacity_quota_count": float(supplement_logs["low_opacity_quota_count"]),
            "render_confidence_mean": float(sampled_render_confidence.detach().mean().item()),
        }
        logs.update(_quantile_logs("distance", monitor_dist, [(0.50, "p50"), (0.90, "p90"), (0.99, "p99")]))
        logs.update(_quantile_logs("plane_residual", residual_bundle["plane_residual"], [(0.50, "p50"), (0.90, "p90")]))
        logs.update(_quantile_logs("quality_score", quality_score, [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]))
        logs.update(_quantile_logs("density_score", density_score, [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]))
        logs.update(_quantile_logs("brightness_score", brightness_score, [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]))
        logs.update(_quantile_logs("gradient_score", gradient_score, [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]))
        logs.update(_quantile_logs("support_score", loss_support_score, [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]))
        logs.update(_quantile_logs("loss_support_score", loss_support_score, [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]))
        logs.update(_quantile_logs("prune_support_score", prune_support_score, [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]))
        logs.update(_quantile_logs("difficulty", difficulty_scores, [(0.50, "p50"), (0.90, "p90")]))
        logs.update(sampled_mid_hard_score_logs)
        logs.update(_masked_quantile_logs("orientation_alignment", residual_bundle["orientation_alignment"], residual_bundle["stable_mask"], [(0.50, "p50"), (0.90, "p90")]))
        logs.update(_quantile_logs("render_confidence", sampled_render_confidence, [(0.50, "p50"), (0.90, "p90")]))
        return loss, logs


class LowLightConsistencyLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="low_light", weight=weight, enabled=weight > 0.0, start_step=0)

    def compute(self, context):
        loss = low_light_consistency_loss(context["rendered"], context["reference_hwc"])
        return loss, {}


class ExposureControlLoss(BaseLossModule):
    def __init__(self, weight, input_key="rendered", target_mean_key="target_mean", name="exposure"):
        super().__init__(name=name, weight=weight, enabled=weight > 0.0, start_step=0)
        self.input_key = str(input_key)
        self.target_mean_key = str(target_mean_key)

    def compute(self, context):
        target_mean = context.get(self.target_mean_key)
        if target_mean is None:
            raise RuntimeError(f"ExposureControlLoss requires '{self.target_mean_key}' in context.")
        loss = exposure_control_loss(context[self.input_key], target_mean)
        return loss, {}


def zero_scalar_like(context):
    return torch.zeros((), device=context["rendered"].device, dtype=context["rendered"].dtype)



def pearson_depth_loss(depth_src, depth_target):
    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()
    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)
    return 1.0 - (src * target).mean()



def local_pearson_loss(depth_src, depth_target, box_size, sample_ratio):
    box_size = int(max(4, min(box_size, depth_src.shape[0], depth_src.shape[1])))
    num_box_h = max(1, depth_src.shape[0] // box_size)
    num_box_w = max(1, depth_src.shape[1] // box_size)
    n_corr = max(1, int(sample_ratio * num_box_h * num_box_w))
    max_h = max(1, depth_src.shape[0] - box_size + 1)
    max_w = max(1, depth_src.shape[1] - box_size + 1)

    x_0 = torch.randint(0, max_h, (n_corr,), device=depth_src.device)
    y_0 = torch.randint(0, max_w, (n_corr,), device=depth_src.device)

    total = torch.zeros((), device=depth_src.device, dtype=depth_src.dtype)
    for x0, y0 in zip(x_0.tolist(), y_0.tolist()):
        x1 = x0 + box_size
        y1 = y0 + box_size
        total = total + pearson_depth_loss(
            depth_src[x0:x1, y0:y1].reshape(-1),
            depth_target[x0:x1, y0:y1].reshape(-1),
        )
    return total / n_corr



def squeeze_single_channel(image_tensor, label):
    if image_tensor is None:
        return None
    if image_tensor.dim() == 2:
        return image_tensor
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 1:
        return image_tensor.squeeze(0)
    if image_tensor.dim() == 3 and image_tensor.shape[-1] == 1:
        return image_tensor.squeeze(-1)
    raise RuntimeError(f"Unsupported {label} tensor shape: {tuple(image_tensor.shape)}")



def standardize_map(image_tensor):
    return (image_tensor - image_tensor.mean()) / image_tensor.std().clamp_min(1e-6)



def minmax_normalize_map(image_tensor):
    min_value = image_tensor.min()
    max_value = image_tensor.max()
    return (image_tensor - min_value) / (max_value - min_value).clamp_min(1e-6)


def build_intrinsics_tensor(intrinsics, device, dtype):
    if torch.is_tensor(intrinsics):
        return intrinsics.to(device=device, dtype=dtype)
    return torch.tensor(intrinsics, device=device, dtype=dtype)


def build_c2w_4x4(camtoworld, device, dtype):
    c2w = torch.eye(4, device=device, dtype=dtype)
    c2w[:3, :] = camtoworld.to(device=device, dtype=dtype)
    return c2w


def rgb_to_gray_map(image_tensor):
    if image_tensor is None:
        return None
    if image_tensor.dim() == 2:
        return image_tensor
    if image_tensor.dim() == 3 and image_tensor.shape[-1] == 3:
        return 0.299 * image_tensor[..., 0] + 0.587 * image_tensor[..., 1] + 0.114 * image_tensor[..., 2]
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
        return 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
    raise RuntimeError(f"Unsupported RGB tensor shape for grayscale conversion: {tuple(image_tensor.shape)}")


def sample_image_grid(height, width, stride, device, dtype):
    ys = torch.arange(0, height, stride, device=device, dtype=dtype)
    xs = torch.arange(0, width, stride, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x.reshape(-1), grid_y.reshape(-1)


def normalize_coords(u, v, width, height):
    x_norm = 2.0 * (u / max(width - 1, 1)) - 1.0
    y_norm = 2.0 * (v / max(height - 1, 1)) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def project_world_to_target(world_points, target_c2w, intrinsics, eps):
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=world_points.device, dtype=world_points.dtype))
    rotation = target_c2w[:3, :3]
    translation = target_c2w[:3, 3]
    cam_gl = (world_points - translation) @ rotation
    cam_cv = cam_gl @ flip.T
    z = cam_cv[:, 2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    u = fx * (cam_cv[:, 0] / z.clamp_min(eps)) + cx
    v = fy * (cam_cv[:, 1] / z.clamp_min(eps)) + cy
    return u, v, z


def backproject_to_world(depth_map, camtoworld, intrinsics, stride, eps, min_alpha, alpha_map):
    height, width = depth_map.shape
    device = depth_map.device
    dtype = depth_map.dtype
    grid_x, grid_y = sample_image_grid(height, width, stride, device, dtype)
    pixel_x = grid_x.long()
    pixel_y = grid_y.long()
    depth_values = depth_map[pixel_y, pixel_x]
    alpha_values = alpha_map[pixel_y, pixel_x]

    valid = torch.isfinite(depth_values) & (depth_values > eps) & (alpha_values > min_alpha)
    if not valid.any():
        empty = torch.empty((0,), device=device, dtype=dtype)
        return empty, empty, empty

    grid_x = grid_x[valid]
    grid_y = grid_y[valid]
    depth_values = depth_values[valid]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = (grid_x - cx) / fx * depth_values
    y = (grid_y - cy) / fy * depth_values
    cam_points_cv = torch.stack([x, y, depth_values], dim=-1)

    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype))
    c2w = build_c2w_4x4(camtoworld, device, dtype)
    world_points = (cam_points_cv @ flip.T) @ c2w[:3, :3].T + c2w[:3, 3]
    return world_points, grid_x, grid_y


def backproject_pixels_to_world(depth_values, pixel_u, pixel_v, camtoworld, intrinsics):
    device = depth_values.device
    dtype = depth_values.dtype
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = (pixel_u - cx) / fx * depth_values
    y = (pixel_v - cy) / fy * depth_values
    cam_points_cv = torch.stack([x, y, depth_values], dim=-1)
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype))
    c2w = build_c2w_4x4(camtoworld, device, dtype)
    return (cam_points_cv @ flip.T) @ c2w[:3, :3].T + c2w[:3, 3]


def sample_target_map(target_map, coords_norm):
    sampled = F.grid_sample(
        target_map[None, None],
        coords_norm.view(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(-1)


def sample_target_patch_map(target_map, coords_norm):
    sampled = F.grid_sample(
        target_map[None, None],
        coords_norm.view(1, coords_norm.shape[0], coords_norm.shape[1], 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(coords_norm.shape[0], coords_norm.shape[1])


def compute_gray_gradient_confidence(gray_map):
    grad_x = torch.zeros_like(gray_map)
    grad_y = torch.zeros_like(gray_map)
    grad_x[:, 1:-1] = 0.5 * torch.abs(gray_map[:, 2:] - gray_map[:, :-2])
    grad_y[1:-1, :] = 0.5 * torch.abs(gray_map[2:, :] - gray_map[:-2, :])
    grad = torch.maximum(grad_x, grad_y)
    grad = grad / grad.max().clamp_min(1.0e-6)
    grad[[0, -1], :] = 1.0
    grad[:, [0, -1]] = 1.0
    return grad.clamp(0.0, 1.0)


def build_image_weight_map(gray_map, low_texture_thresh, highlight_thresh, dark_thresh, dark_grad_thresh):
    if gray_map is None:
        return None
    grad_conf = compute_gray_gradient_confidence(gray_map)
    img_weight = grad_conf.square()
    low_texture_mask = grad_conf < low_texture_thresh
    highlight_mask = gray_map > highlight_thresh
    dark_reflective_mask = (gray_map < dark_thresh) & (grad_conf < dark_grad_thresh)
    img_weight[low_texture_mask | highlight_mask | dark_reflective_mask] = 0.0
    return img_weight


def weighted_masked_mean(values, weights, eps=1.0e-6):
    if values.numel() == 0 or weights.numel() == 0:
        return torch.zeros((), device=values.device if values.numel() > 0 else weights.device, dtype=values.dtype if values.numel() > 0 else weights.dtype)
    denom = weights.sum()
    if float(denom.detach().item()) <= eps:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return (values * weights).sum() / denom.clamp_min(eps)


def lncc_loss(ref_patches, target_patches):
    batch_size, total_patch_size = target_patches.shape
    patch_size = int(round(total_patch_size ** 0.5))
    if patch_size * patch_size != total_patch_size:
        raise RuntimeError(f"Invalid patch vector length for LNCC: {total_patch_size}")

    ref = ref_patches.view(batch_size, 1, patch_size, patch_size)
    target = target_patches.view(batch_size, 1, patch_size, patch_size)
    ref_target = ref * target
    ref_sq = ref.pow(2)
    target_sq = target.pow(2)
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device, dtype=ref.dtype)
    padding = patch_size // 2

    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    target_sum = F.conv2d(target, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_sq_sum = F.conv2d(ref_sq, filters, stride=1, padding=padding)[:, :, padding, padding]
    target_sq_sum = F.conv2d(target_sq, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_target_sum = F.conv2d(ref_target, filters, stride=1, padding=padding)[:, :, padding, padding]

    ref_avg = ref_sum / total_patch_size
    target_avg = target_sum / total_patch_size
    cross = ref_target_sum - target_avg * ref_sum
    ref_var = ref_sq_sum - ref_avg * ref_sum
    target_var = target_sq_sum - target_avg * target_sum
    cc = cross * cross / (ref_var * target_var + 1.0e-8)
    ncc = 1.0 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0).mean(dim=1)
    valid = ncc < 0.9
    return ncc, valid


def compute_patch_lncc_loss(
    source_gray,
    target_gray,
    source_depth,
    source_u,
    source_v,
    weights,
    source_camtoworld,
    target_camtoworld,
    intrinsics,
    patch_size,
    patch_samples,
    eps,
):
    if source_gray is None or target_gray is None or source_u.numel() == 0:
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, {"lncc_count": 0.0}

    valid_indices = torch.arange(source_u.shape[0], device=source_u.device)
    if valid_indices.numel() > patch_samples:
        choice = torch.randperm(valid_indices.numel(), device=valid_indices.device)[:patch_samples]
        valid_indices = valid_indices[choice]

    source_u = source_u[valid_indices]
    source_v = source_v[valid_indices]
    weights = weights[valid_indices]
    patch_radius = int(max(1, patch_size))
    offset_values = torch.arange(-patch_radius, patch_radius + 1, device=source_u.device, dtype=source_u.dtype)
    offset_y, offset_x = torch.meshgrid(offset_values, offset_values, indexing="ij")
    offset_x = offset_x.reshape(1, -1)
    offset_y = offset_y.reshape(1, -1)

    source_patch_u = source_u[:, None] + offset_x
    source_patch_v = source_v[:, None] + offset_y
    height = int(source_gray.shape[0])
    width = int(source_gray.shape[1])
    source_inside = (source_patch_u >= 0.0) & (source_patch_u <= float(width - 1))
    source_inside &= (source_patch_v >= 0.0) & (source_patch_v <= float(height - 1))

    source_patch_coords = normalize_coords(source_patch_u.reshape(-1), source_patch_v.reshape(-1), width, height)
    source_patch_coords = source_patch_coords.view(source_patch_u.shape[0], source_patch_u.shape[1], 2)
    source_patch_gray = sample_target_patch_map(source_gray, source_patch_coords)
    source_patch_depth = sample_target_patch_map(source_depth, source_patch_coords)
    patch_depth_valid = torch.isfinite(source_patch_depth) & (source_patch_depth > eps) & source_inside
    if not patch_depth_valid.any():
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, {"lncc_count": 0.0}

    world_points = backproject_pixels_to_world(
        source_patch_depth.reshape(-1),
        source_patch_u.reshape(-1),
        source_patch_v.reshape(-1),
        source_camtoworld,
        intrinsics,
    ).view(source_patch_depth.shape[0], source_patch_depth.shape[1], 3)
    target_c2w = build_c2w_4x4(target_camtoworld, source_depth.device, source_depth.dtype)
    projected_u, projected_v, projected_z = project_world_to_target(
        world_points.reshape(-1, 3),
        target_c2w,
        intrinsics,
        eps,
    )
    projected_u = projected_u.view(source_patch_depth.shape)
    projected_v = projected_v.view(source_patch_depth.shape)
    projected_z = projected_z.view(source_patch_depth.shape)
    target_inside = torch.isfinite(projected_z) & (projected_z > eps)
    target_inside &= (projected_u >= 0.0) & (projected_u <= float(width - 1))
    target_inside &= (projected_v >= 0.0) & (projected_v <= float(height - 1))
    patch_valid = patch_depth_valid & target_inside
    patch_center_valid = patch_valid.all(dim=1)
    if not patch_center_valid.any():
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, {"lncc_count": 0.0}

    target_patch_coords = normalize_coords(projected_u.reshape(-1), projected_v.reshape(-1), width, height)
    target_patch_coords = target_patch_coords.view(projected_u.shape[0], projected_u.shape[1], 2)
    target_patch_gray = sample_target_patch_map(target_gray, target_patch_coords)

    source_patch_gray = source_patch_gray[patch_center_valid]
    target_patch_gray = target_patch_gray[patch_center_valid]
    weights = weights[patch_center_valid]

    ncc_values, ncc_valid = lncc_loss(source_patch_gray, target_patch_gray)
    if not ncc_valid.any():
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, {"lncc_count": 0.0}

    ncc_values = ncc_values[ncc_valid]
    weights = weights[ncc_valid]
    return weighted_masked_mean(ncc_values, weights), {"lncc_count": float(ncc_valid.sum().detach().item())}


def compute_reprojection_directional_loss(
    source_depth,
    source_alpha,
    source_camtoworld,
    target_depth,
    target_alpha,
    target_camtoworld,
    intrinsics,
    stride,
    min_alpha,
    rel_thresh,
    abs_thresh,
    eps,
    pixel_noise_thresh=0.0,
    source_image_weight_map=None,
    target_image_weight_map=None,
    source_gray=None,
    target_gray=None,
    lncc_enabled=False,
    patch_size=3,
    patch_samples=256,
):
    world_points, source_u, source_v = backproject_to_world(
        depth_map=source_depth,
        camtoworld=source_camtoworld,
        intrinsics=intrinsics,
        stride=stride,
        eps=eps,
        min_alpha=min_alpha,
        alpha_map=source_alpha,
    )
    total_samples = max(1, ((source_depth.shape[0] + stride - 1) // stride) * ((source_depth.shape[1] + stride - 1) // stride))
    if world_points.numel() == 0:
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, zero, {
            "valid_ratio": 0.0,
            "geom_valid_ratio": 0.0,
            "occ_valid_ratio": 0.0,
            "valid_count": 0.0,
            "sample_count": float(total_samples),
            "pixel_noise_mean": 0.0,
            "lncc": 0.0,
        }

    target_c2w = build_c2w_4x4(target_camtoworld, source_depth.device, source_depth.dtype)
    u, v, z_proj = project_world_to_target(world_points, target_c2w, intrinsics, eps)

    width = int(target_depth.shape[1])
    height = int(target_depth.shape[0])
    valid = torch.isfinite(z_proj) & (z_proj > eps)
    valid &= (u >= 0.0) & (u <= float(width - 1))
    valid &= (v >= 0.0) & (v <= float(height - 1))
    if not valid.any():
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, zero, {
            "valid_ratio": 0.0,
            "geom_valid_ratio": 0.0,
            "occ_valid_ratio": 0.0,
            "valid_count": 0.0,
            "sample_count": float(total_samples),
            "pixel_noise_mean": 0.0,
            "lncc": 0.0,
        }

    u = u[valid]
    v = v[valid]
    z_proj = z_proj[valid]
    source_u = source_u[valid]
    source_v = source_v[valid]

    coords_norm = normalize_coords(u, v, width, height)

    sampled_target_depth = sample_target_map(target_depth, coords_norm)
    sampled_target_alpha = sample_target_map(target_alpha, coords_norm)

    geom_valid = torch.isfinite(sampled_target_depth) & (sampled_target_depth > eps)
    geom_valid &= sampled_target_alpha > min_alpha
    depth_delta = torch.abs(z_proj - sampled_target_depth)
    geom_valid &= depth_delta <= (abs_thresh + rel_thresh * sampled_target_depth.abs().clamp_min(eps))
    if not geom_valid.any():
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, zero, {
            "valid_ratio": 0.0,
            "geom_valid_ratio": 0.0,
            "occ_valid_ratio": 0.0,
            "valid_count": 0.0,
            "sample_count": float(total_samples),
            "pixel_noise_mean": 0.0,
            "lncc": 0.0,
        }

    z_proj = z_proj[geom_valid]
    sampled_target_depth = sampled_target_depth[geom_valid]
    u = u[geom_valid]
    v = v[geom_valid]
    source_u = source_u[geom_valid]
    source_v = source_v[geom_valid]

    depth_error = torch.abs(z_proj - sampled_target_depth) / sampled_target_depth.abs().clamp_min(eps)
    depth_loss = depth_error.mean()
    valid_ratio = float(geom_valid.sum().detach().item() / float(total_samples))
    zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
    return depth_loss, zero, {
        "valid_ratio": valid_ratio,
        "geom_valid_ratio": valid_ratio,
        "occ_valid_ratio": valid_ratio,
        "valid_count": float(geom_valid.sum().detach().item()),
        "sample_count": float(total_samples),
        "pixel_noise_mean": 0.0,
        "lncc": 0.0,
    }


class DepthPriorLoss(BaseLossModule):
    def __init__(
        self,
        weight,
        start_step=0,
        end_step=None,
        ramp_up_steps=0,
        ramp_down_steps=0,
        start_scale=1.0,
        end_scale=0.0,
        global_weight=1.0,
        local_weight=1.0,
        box_size=128,
        sample_ratio=0.5,
    ):
        super().__init__(
            name="depth_prior",
            weight=weight,
            enabled=weight > 0.0,
            start_step=start_step,
            end_step=end_step,
            ramp_up_steps=ramp_up_steps,
            ramp_down_steps=ramp_down_steps,
            start_scale=start_scale,
            end_scale=end_scale,
        )
        self.global_weight = float(global_weight)
        self.local_weight = float(local_weight)
        self.box_size = int(box_size)
        self.sample_ratio = float(sample_ratio)

    def compute(self, context):
        depth_target = context.get("depth")
        frame_key = context.get("data", {}).get("infos", {}).get("frame_key", "unknown")
        if depth_target is None:
            raise RuntimeError(f"DepthPriorLoss is enabled, but frame '{frame_key}' has no depth prior.")

        if not self.is_active(context):
            zero = zero_scalar_like(context)
            return zero, {"global": 0.0, "local": 0.0}

        rendered_depth = context.get("depth_aux")
        if rendered_depth is None:
            raise RuntimeError("DepthPriorLoss requires depth_aux, but the model did not render the D_r head.")

        rendered_depth = squeeze_single_channel(rendered_depth, "rendered_depth")
        depth_target = squeeze_single_channel(depth_target.to(rendered_depth.device, dtype=rendered_depth.dtype), "depth")

        rendered_depth = standardize_map(rendered_depth)
        depth_target = standardize_map(depth_target)
        global_loss = pearson_depth_loss(rendered_depth.reshape(-1), depth_target.reshape(-1))
        local_loss = local_pearson_loss(rendered_depth, depth_target, self.box_size, self.sample_ratio)
        depth_loss = self.global_weight * global_loss + self.local_weight * local_loss
        return depth_loss, {
            "global": float(global_loss.detach().item()),
            "local": float(local_loss.detach().item()),
        }


class StructurePriorLoss(BaseLossModule):
    def __init__(self, weight, start_step=0):
        super().__init__(name="structure_prior", weight=weight, enabled=weight > 0.0, start_step=start_step)

    def compute(self, context):
        structure_target = context.get("structure")
        if structure_target is None:
            zero = zero_scalar_like(context)
            return zero, {"available": 0.0}

        if not self.is_active(context):
            zero = zero_scalar_like(context)
            return zero, {"available": 1.0}

        rendered_prior = context.get("prior_aux")
        if rendered_prior is None:
            raise RuntimeError("StructurePriorLoss requires prior_aux, but the model did not render the P_r head.")

        rendered_prior = squeeze_single_channel(rendered_prior, "rendered_prior")
        structure_target = squeeze_single_channel(structure_target.to(rendered_prior.device, dtype=rendered_prior.dtype), "structure")
        predicted = minmax_normalize_map(rendered_prior)
        target = minmax_normalize_map(structure_target)
        loss = torch.abs(predicted - target).mean()
        return loss, {"available": 1.0}


class MultiViewReprojectionLoss(BaseLossModule):
    def __init__(
        self,
        weight,
        start_step=0,
        end_step=None,
        ramp_up_steps=0,
        ramp_down_steps=0,
        start_scale=1.0,
        end_scale=0.0,
        sample_stride=4,
        min_alpha=0.2,
        relative_depth_thresh=0.05,
        absolute_depth_thresh=0.02,
        eps=1.0e-4,
        pixel_noise_thresh=0.0,
        low_texture_thresh=-1.0,
        highlight_thresh=2.0,
        dark_thresh=-1.0,
        dark_grad_thresh=-1.0,
        lncc_enabled=False,
        lncc_weight=0.15,
        patch_size=3,
        patch_samples=256,
    ):
        super().__init__(
            name="multiview_reproj",
            weight=weight,
            enabled=weight > 0.0,
            start_step=start_step,
            end_step=end_step,
            ramp_up_steps=ramp_up_steps,
            ramp_down_steps=ramp_down_steps,
            start_scale=start_scale,
            end_scale=end_scale,
        )
        self.sample_stride = int(max(1, sample_stride))
        self.min_alpha = float(min_alpha)
        self.relative_depth_thresh = float(relative_depth_thresh)
        self.absolute_depth_thresh = float(absolute_depth_thresh)
        self.eps = float(eps)
        self.pixel_noise_thresh = float(pixel_noise_thresh)
        self.low_texture_thresh = float(low_texture_thresh)
        self.highlight_thresh = float(highlight_thresh)
        self.dark_thresh = float(dark_thresh)
        self.dark_grad_thresh = float(dark_grad_thresh)
        self.lncc_enabled = bool(lncc_enabled)
        self.lncc_weight = float(lncc_weight)
        self.patch_size = int(max(1, patch_size))
        self.patch_samples = int(max(1, patch_samples))

    def compute(self, context):
        if not self.is_active(context):
            zero = zero_scalar_like(context)
            return zero, {"active": 0.0, "valid_ratio": 0.0, "lncc": 0.0}

        source_depth = context.get("geom_depth")
        target_depth = context.get("neighbor_geom_depth")
        source_alpha = context.get("alphas")
        target_alpha = context.get("neighbor_alphas")
        source_camtoworld = context.get("camtoworld")
        target_camtoworld = context.get("neighbor_camtoworld")
        intrinsics = context.get("intrinsics")
        if source_depth is None or target_depth is None or source_alpha is None or target_alpha is None:
            zero = zero_scalar_like(context)
            return zero, {"active": 0.0, "valid_ratio": 0.0, "lncc": 0.0}
        if source_camtoworld is None or target_camtoworld is None or intrinsics is None:
            zero = zero_scalar_like(context)
            return zero, {"active": 0.0, "valid_ratio": 0.0, "lncc": 0.0}

        source_depth = squeeze_single_channel(source_depth, "geom_depth")
        target_depth = squeeze_single_channel(target_depth, "neighbor_geom_depth")
        source_alpha = squeeze_single_channel(source_alpha, "alphas")
        target_alpha = squeeze_single_channel(target_alpha, "neighbor_alphas")
        intrinsics = build_intrinsics_tensor(intrinsics, source_depth.device, source_depth.dtype)
        src_to_tgt_loss, src_to_tgt_lncc, src_logs = compute_reprojection_directional_loss(
            source_depth=source_depth,
            source_alpha=source_alpha,
            source_camtoworld=source_camtoworld,
            target_depth=target_depth,
            target_alpha=target_alpha,
            target_camtoworld=target_camtoworld,
            intrinsics=intrinsics,
            stride=self.sample_stride,
            min_alpha=self.min_alpha,
            rel_thresh=self.relative_depth_thresh,
            abs_thresh=self.absolute_depth_thresh,
            eps=self.eps,
        )
        tgt_to_src_loss, tgt_to_src_lncc, tgt_logs = compute_reprojection_directional_loss(
            source_depth=target_depth,
            source_alpha=target_alpha,
            source_camtoworld=target_camtoworld,
            target_depth=source_depth,
            target_alpha=source_alpha,
            target_camtoworld=source_camtoworld,
            intrinsics=intrinsics,
            stride=self.sample_stride,
            min_alpha=self.min_alpha,
            rel_thresh=self.relative_depth_thresh,
            abs_thresh=self.absolute_depth_thresh,
            eps=self.eps,
        )
        depth_loss = 0.5 * (src_to_tgt_loss + tgt_to_src_loss)
        loss = depth_loss
        valid_ratio = 0.5 * (src_logs["valid_ratio"] + tgt_logs["valid_ratio"])
        return loss, {
            "active": 1.0,
            "valid_ratio": valid_ratio,
            "src_valid": src_logs["valid_count"],
            "tgt_valid": tgt_logs["valid_count"],
        }



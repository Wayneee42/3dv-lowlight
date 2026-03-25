import random

import torch
import torch.nn.functional as F



def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    try:
        return getattr(cfg, key)
    except AttributeError:
        return default



def gamma_augment(image, gamma):
    return torch.clamp(image, 0.0, 1.0).pow(gamma)



def exposure_match(image, target_mean, min_scale=1.0, max_scale=3.0, eps=1e-6):
    image = torch.clamp(image, 0.0, 1.0)
    current_mean = image.mean().clamp_min(eps)
    scale = torch.clamp(torch.tensor(target_mean, device=image.device) / current_mean, min_scale, max_scale)
    return torch.clamp(image * scale, 0.0, 1.0), float(scale.item())



def _compute_gray(image):
    if image.dim() != 3 or image.shape[0] != 3:
        raise RuntimeError(f"Proxy target expects CHW RGB image, got shape {tuple(image.shape)}")
    image = torch.clamp(image, 0.0, 1.0)
    return 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]



def _compute_proxy_stat_mean(gray, proxy_cfg, eps):
    stat_mode = str(_cfg_get(proxy_cfg, "STAT_MODE", "mean")).lower()

    if stat_mode == "mean":
        effective_mean = gray.mean().clamp_min(eps)
        return effective_mean, stat_mode

    if stat_mode == "clipped_mean":
        clip_percentile = float(_cfg_get(proxy_cfg, "CLIP_PERCENTILE", 90.0))
        clip_percentile = min(max(clip_percentile, 0.0), 100.0)
        clip_value = torch.quantile(gray.reshape(-1), clip_percentile / 100.0)
        effective_mean = torch.minimum(gray, clip_value).mean().clamp_min(eps)
        return effective_mean, f"clipped_mean@p{clip_percentile:.1f}"

    raise RuntimeError(f"Unsupported PROXY_TARGET.STAT_MODE: {stat_mode}")



def _calibrate_proxy_target_mean(gray, stat_mean, base_target_mean, proxy_cfg, eps):
    mode = str(_cfg_get(proxy_cfg, "CALIBRATION_MODE", "fixed")).lower()
    highlight_percentile = float(_cfg_get(proxy_cfg, "CALIBRATION_HIGHLIGHT_PERCENTILE", 95.0))
    highlight_percentile = min(max(highlight_percentile, 0.0), 100.0)
    highlight_value = torch.quantile(gray.reshape(-1), highlight_percentile / 100.0).clamp_min(eps)

    if mode == "fixed":
        calibrated_target = float(base_target_mean)
        return {
            "target_mean": calibrated_target,
            "base_target_mean": float(base_target_mean),
            "calibration_scale": 1.0,
            "stat_scale": 1.0,
            "highlight_scale": 1.0,
            "highlight_value": float(highlight_value.item()),
            "calibration_mode": mode,
        }

    if mode != "stat_ratio":
        raise RuntimeError(f"Unsupported PROXY_TARGET.CALIBRATION_MODE: {mode}")

    reference_stat = float(_cfg_get(proxy_cfg, "CALIBRATION_REFERENCE_STAT", 0.08))
    calibration_power = float(_cfg_get(proxy_cfg, "CALIBRATION_POWER", 0.35))
    calibration_min_scale = float(_cfg_get(proxy_cfg, "CALIBRATION_MIN_SCALE", 0.85))
    calibration_max_scale = float(_cfg_get(proxy_cfg, "CALIBRATION_MAX_SCALE", 1.25))
    highlight_reference = float(_cfg_get(proxy_cfg, "CALIBRATION_HIGHLIGHT_REFERENCE", 0.55))
    highlight_power = float(_cfg_get(proxy_cfg, "CALIBRATION_HIGHLIGHT_POWER", 0.25))
    highlight_min_scale = float(_cfg_get(proxy_cfg, "CALIBRATION_HIGHLIGHT_MIN_SCALE", 0.85))
    highlight_max_scale = float(_cfg_get(proxy_cfg, "CALIBRATION_HIGHLIGHT_MAX_SCALE", 1.05))
    min_target_mean = float(_cfg_get(proxy_cfg, "CALIBRATION_MIN_TARGET_MEAN", 0.32))
    max_target_mean = float(_cfg_get(proxy_cfg, "CALIBRATION_MAX_TARGET_MEAN", 0.46))

    stat_ratio = torch.tensor(reference_stat, device=gray.device, dtype=gray.dtype) / stat_mean.clamp_min(eps)
    stat_scale = torch.clamp(stat_ratio.pow(calibration_power), calibration_min_scale, calibration_max_scale)

    highlight_ratio = torch.tensor(highlight_reference, device=gray.device, dtype=gray.dtype) / highlight_value
    highlight_scale = torch.clamp(highlight_ratio.pow(highlight_power), highlight_min_scale, highlight_max_scale)

    calibration_scale = stat_scale * highlight_scale
    calibrated_target = float(torch.clamp(
        torch.tensor(base_target_mean, device=gray.device, dtype=gray.dtype) * calibration_scale,
        min_target_mean,
        max_target_mean,
    ).item())

    return {
        "target_mean": calibrated_target,
        "base_target_mean": float(base_target_mean),
        "calibration_scale": float(calibration_scale.item()),
        "stat_scale": float(stat_scale.item()),
        "highlight_scale": float(highlight_scale.item()),
        "highlight_value": float(highlight_value.item()),
        "calibration_mode": mode,
    }



def _build_shadow_proxy(image, gamma, target_mean, min_scale, max_scale, proxy_cfg):
    shadow_source = str(_cfg_get(proxy_cfg, "SHADOW_SOURCE", "supervision_like")).lower()
    if shadow_source != "supervision_like":
        raise RuntimeError(f"Unsupported PROXY_TARGET.SHADOW_SOURCE: {shadow_source}")
    shadow_proxy = gamma_augment(image, gamma)
    shadow_proxy, _ = exposure_match(shadow_proxy, target_mean, min_scale=min_scale, max_scale=max_scale)
    return shadow_proxy



def _smooth_shadow_weight(shadow_weight, proxy_cfg):
    kernel_size = int(_cfg_get(proxy_cfg, "SHADOW_WEIGHT_BLUR_KERNEL", 1))
    if kernel_size <= 1:
        return shadow_weight
    if kernel_size % 2 == 0:
        raise RuntimeError("PROXY_TARGET.SHADOW_WEIGHT_BLUR_KERNEL must be an odd integer.")
    padding = kernel_size // 2
    smoothed = F.avg_pool2d(
        shadow_weight.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    return smoothed.squeeze(0).squeeze(0)



def build_proxy_target(
    image,
    proxy_cfg=None,
    fallback_target_mean=0.35,
    fallback_min_scale=1.0,
    fallback_max_scale=3.0,
    gamma=1.0,
):
    image = torch.clamp(image, 0.0, 1.0)
    gray = _compute_gray(image)

    if not bool(_cfg_get(proxy_cfg, "ENABLED", False)):
        proxy_target, proxy_gain = exposure_match(
            image,
            fallback_target_mean,
            min_scale=fallback_min_scale,
            max_scale=fallback_max_scale,
            eps=float(_cfg_get(proxy_cfg, "EPS", 1e-6)),
        )
        zero_weight = torch.zeros_like(gray)
        proxy_mean = float(proxy_target.mean().item())
        return {
            "proxy_target": proxy_target,
            "proxy_global": proxy_target,
            "proxy_shadow": proxy_target,
            "proxy_shadow_weight": zero_weight,
            "proxy_gain": float(proxy_gain),
            "proxy_stat_mean": proxy_mean,
            "proxy_stat_mode": "fallback_exposure_match",
            "proxy_form": "fallback_exposure_match",
            "proxy_global_mean": proxy_mean,
            "proxy_shadow_mean": proxy_mean,
            "proxy_blend_mean": proxy_mean,
            "proxy_shadow_weight_mean": 0.0,
            "proxy_target_mean": float(fallback_target_mean),
            "proxy_base_target_mean": float(fallback_target_mean),
            "proxy_calibration_scale": 1.0,
            "proxy_stat_scale": 1.0,
            "proxy_highlight_scale": 1.0,
            "proxy_highlight_value": float(gray.mean().item()),
            "proxy_calibration_mode": "fallback_exposure_match",
        }

    eps = float(_cfg_get(proxy_cfg, "EPS", 1e-6))
    base_target_mean = float(_cfg_get(proxy_cfg, "TARGET_MEAN", fallback_target_mean))
    min_gain = float(_cfg_get(proxy_cfg, "MIN_GAIN", 1.0))
    max_gain = float(_cfg_get(proxy_cfg, "MAX_GAIN", 32.0))

    stat_mean, stat_label = _compute_proxy_stat_mean(gray, proxy_cfg, eps)
    calibration_info = _calibrate_proxy_target_mean(gray, stat_mean, base_target_mean, proxy_cfg, eps)
    target_mean = float(calibration_info["target_mean"])
    global_gain = torch.clamp(torch.tensor(target_mean, device=image.device) / stat_mean, min_gain, max_gain)
    proxy_global = torch.clamp(image * global_gain, 0.0, 1.0)

    form = str(_cfg_get(proxy_cfg, "FORM", "global_linear")).lower()
    if form == "global_linear":
        zero_weight = torch.zeros_like(gray)
        proxy_mean = float(proxy_global.mean().item())
        return {
            "proxy_target": proxy_global,
            "proxy_global": proxy_global,
            "proxy_shadow": proxy_global,
            "proxy_shadow_weight": zero_weight,
            "proxy_gain": float(global_gain.item()),
            "proxy_stat_mean": float(stat_mean.item()),
            "proxy_stat_mode": stat_label,
            "proxy_form": form,
            "proxy_global_mean": proxy_mean,
            "proxy_shadow_mean": proxy_mean,
            "proxy_blend_mean": proxy_mean,
            "proxy_shadow_weight_mean": 0.0,
            "proxy_target_mean": target_mean,
            "proxy_base_target_mean": float(calibration_info["base_target_mean"]),
            "proxy_calibration_scale": float(calibration_info["calibration_scale"]),
            "proxy_stat_scale": float(calibration_info["stat_scale"]),
            "proxy_highlight_scale": float(calibration_info["highlight_scale"]),
            "proxy_highlight_value": float(calibration_info["highlight_value"]),
            "proxy_calibration_mode": str(calibration_info["calibration_mode"]),
        }

    if form == "shadow_blend":
        shadow_proxy = _build_shadow_proxy(
            image,
            gamma=gamma,
            target_mean=target_mean,
            min_scale=fallback_min_scale,
            max_scale=fallback_max_scale,
            proxy_cfg=proxy_cfg,
        )
        shadow_proxy = torch.maximum(shadow_proxy, proxy_global)
        shadow_threshold = max(float(_cfg_get(proxy_cfg, "SHADOW_THRESHOLD", 0.20)), 1e-6)
        shadow_power = float(_cfg_get(proxy_cfg, "SHADOW_POWER", 2.0))
        shadow_weight = torch.clamp((shadow_threshold - gray) / shadow_threshold, 0.0, 1.0)
        if shadow_power != 1.0:
            shadow_weight = shadow_weight.pow(shadow_power)
        shadow_weight = _smooth_shadow_weight(shadow_weight, proxy_cfg)
        proxy_target = torch.clamp(
            (1.0 - shadow_weight.unsqueeze(0)) * proxy_global + shadow_weight.unsqueeze(0) * shadow_proxy,
            0.0,
            1.0,
        )
        return {
            "proxy_target": proxy_target,
            "proxy_global": proxy_global,
            "proxy_shadow": shadow_proxy,
            "proxy_shadow_weight": shadow_weight,
            "proxy_gain": float(global_gain.item()),
            "proxy_stat_mean": float(stat_mean.item()),
            "proxy_stat_mode": stat_label,
            "proxy_form": form,
            "proxy_global_mean": float(proxy_global.mean().item()),
            "proxy_shadow_mean": float(shadow_proxy.mean().item()),
            "proxy_blend_mean": float(proxy_target.mean().item()),
            "proxy_shadow_weight_mean": float(shadow_weight.mean().item()),
            "proxy_target_mean": target_mean,
            "proxy_base_target_mean": float(calibration_info["base_target_mean"]),
            "proxy_calibration_scale": float(calibration_info["calibration_scale"]),
            "proxy_stat_scale": float(calibration_info["stat_scale"]),
            "proxy_highlight_scale": float(calibration_info["highlight_scale"]),
            "proxy_highlight_value": float(calibration_info["highlight_value"]),
            "proxy_calibration_mode": str(calibration_info["calibration_mode"]),
        }

    raise RuntimeError(f"Unsupported PROXY_TARGET.FORM: {form}")



def prepare_low_light_batch(image, aug_cfg=None, training=True, proxy_cfg=None):
    enabled = bool(_cfg_get(aug_cfg, "ENABLED", True))
    mode = str(_cfg_get(aug_cfg, "MODE", "gamma")).lower()

    base_target_mean = float(_cfg_get(aug_cfg, "TARGET_MEAN", 0.35))
    target_mean_jitter = float(_cfg_get(aug_cfg, "TARGET_MEAN_JITTER", 0.0))
    min_scale = float(_cfg_get(aug_cfg, "MIN_SCALE", 1.0))
    max_scale = float(_cfg_get(aug_cfg, "MAX_SCALE", 3.0))
    gamma_range = _cfg_get(aug_cfg, "GAMMA_RANGE", [0.5, 0.5])
    eval_gamma = float(_cfg_get(aug_cfg, "EVAL_GAMMA", 0.5))

    if training:
        gamma = random.uniform(float(gamma_range[0]), float(gamma_range[1]))
        jitter = random.uniform(-target_mean_jitter, target_mean_jitter)
        target_mean = min(max(base_target_mean * (1.0 + jitter), 0.05), 0.95)
    else:
        gamma = eval_gamma
        target_mean = base_target_mean

    proxy_info = build_proxy_target(
        image,
        proxy_cfg=proxy_cfg,
        fallback_target_mean=target_mean,
        fallback_min_scale=min_scale,
        fallback_max_scale=max_scale,
        gamma=float(gamma),
    )
    proxy_target = proxy_info["proxy_target"]

    if not enabled:
        return {
            "supervision": image,
            "reference": image,
            "proxy_target": proxy_target,
            "proxy_global": proxy_info["proxy_global"],
            "proxy_shadow": proxy_info["proxy_shadow"],
            "proxy_shadow_weight": proxy_info["proxy_shadow_weight"],
            "target_mean": float(target_mean),
            "gamma": 1.0,
            "scale": 1.0,
            "proxy_scale": float(proxy_info["proxy_gain"]),
            "proxy_mean": float(proxy_target.mean().item()),
            "proxy_stat_mean": float(proxy_info["proxy_stat_mean"]),
            "proxy_stat_mode": proxy_info["proxy_stat_mode"],
            "proxy_form": proxy_info["proxy_form"],
            "proxy_global_mean": float(proxy_info["proxy_global_mean"]),
            "proxy_shadow_mean": float(proxy_info["proxy_shadow_mean"]),
            "proxy_blend_mean": float(proxy_info["proxy_blend_mean"]),
            "proxy_shadow_weight_mean": float(proxy_info["proxy_shadow_weight_mean"]),
            "proxy_target_mean": float(proxy_info["proxy_target_mean"]),
            "proxy_base_target_mean": float(proxy_info["proxy_base_target_mean"]),
            "proxy_calibration_scale": float(proxy_info["proxy_calibration_scale"]),
            "proxy_stat_scale": float(proxy_info["proxy_stat_scale"]),
            "proxy_highlight_scale": float(proxy_info["proxy_highlight_scale"]),
            "proxy_highlight_value": float(proxy_info["proxy_highlight_value"]),
            "proxy_calibration_mode": str(proxy_info["proxy_calibration_mode"]),
            "low_mean": float(image.mean().item()),
            "mode": "identity",
        }

    supervision = image
    scale = 1.0

    if mode in ("gamma", "hybrid"):
        supervision = gamma_augment(supervision, gamma)
    if mode in ("exposure_match", "hybrid"):
        supervision, scale = exposure_match(supervision, target_mean, min_scale=min_scale, max_scale=max_scale)

    return {
        "supervision": supervision,
        "reference": image,
        "proxy_target": proxy_target,
        "proxy_global": proxy_info["proxy_global"],
        "proxy_shadow": proxy_info["proxy_shadow"],
        "proxy_shadow_weight": proxy_info["proxy_shadow_weight"],
        "target_mean": float(target_mean),
        "gamma": float(gamma),
        "scale": float(scale),
        "proxy_scale": float(proxy_info["proxy_gain"]),
        "proxy_mean": float(proxy_target.mean().item()),
        "proxy_stat_mean": float(proxy_info["proxy_stat_mean"]),
        "proxy_stat_mode": proxy_info["proxy_stat_mode"],
        "proxy_form": proxy_info["proxy_form"],
        "proxy_global_mean": float(proxy_info["proxy_global_mean"]),
        "proxy_shadow_mean": float(proxy_info["proxy_shadow_mean"]),
        "proxy_blend_mean": float(proxy_info["proxy_blend_mean"]),
        "proxy_shadow_weight_mean": float(proxy_info["proxy_shadow_weight_mean"]),
        "proxy_target_mean": float(proxy_info["proxy_target_mean"]),
        "proxy_base_target_mean": float(proxy_info["proxy_base_target_mean"]),
        "proxy_calibration_scale": float(proxy_info["proxy_calibration_scale"]),
        "proxy_stat_scale": float(proxy_info["proxy_stat_scale"]),
        "proxy_highlight_scale": float(proxy_info["proxy_highlight_scale"]),
        "proxy_highlight_value": float(proxy_info["proxy_highlight_value"]),
        "proxy_calibration_mode": str(proxy_info["proxy_calibration_mode"]),
        "low_mean": float(image.mean().item()),
        "mode": mode,
    }

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


def _compute_proxy_image_stats(proxy_image, scene_sat_thresh=0.95):
    proxy_image = torch.clamp(proxy_image, 0.0, 1.0)
    gray = _compute_gray(proxy_image)
    rgb_max = proxy_image.max(dim=0).values
    return {
        "y_mean": float(gray.mean().item()),
        "y_p95": float(torch.quantile(gray.reshape(-1), 0.95).item()),
        "sat_ratio": float((rgb_max > float(scene_sat_thresh)).to(dtype=gray.dtype).mean().item()),
    }


def _compute_proxy_effective_target_mean(proxy_pre, raw_target_mean, proxy_gain, proxy_cfg):
    max_gain = float(max(_cfg_get(proxy_cfg, "MAX_GAIN", 1.0), 1.0))
    gain_soft_ratio = float(_cfg_get(proxy_cfg, "POST_MATCH_GAIN_SOFT_RATIO", 0.75))
    gain_hard_ratio = float(_cfg_get(proxy_cfg, "POST_MATCH_GAIN_HARD_RATIO", 0.95))
    scene_sat_thresh = float(_cfg_get(proxy_cfg, "POST_MATCH_SCENE_SAT_THRESH", 0.95))
    scene_sat_soft_ratio = float(_cfg_get(proxy_cfg, "POST_MATCH_SCENE_SAT_SOFT_RATIO", 0.18))
    scene_sat_hard_ratio = float(_cfg_get(proxy_cfg, "POST_MATCH_SCENE_SAT_HARD_RATIO", 0.30))

    stats = _compute_proxy_image_stats(proxy_pre, scene_sat_thresh=scene_sat_thresh)
    gain_soft = gain_soft_ratio * max_gain
    gain_hard = gain_hard_ratio * max_gain
    gain_denom = max(gain_hard - gain_soft, 1.0e-6)
    sat_denom = max(scene_sat_hard_ratio - scene_sat_soft_ratio, 1.0e-6)

    gain_safety = min(max((gain_hard - float(proxy_gain)) / gain_denom, 0.0), 1.0)
    sat_safety = min(max((scene_sat_hard_ratio - float(stats["sat_ratio"])) / sat_denom, 0.0), 1.0)
    # Tail safety only applies when gain pressure is already high; low-gain dark scenes keep their push budget.
    push_safety = gain_safety + (1.0 - gain_safety) * sat_safety
    effective_target_mean = float(stats["y_mean"] + push_safety * (float(raw_target_mean) - float(stats["y_mean"])))

    return {
        "proxy_pre_y_mean": float(stats["y_mean"]),
        "proxy_pre_y_p95": float(stats["y_p95"]),
        "proxy_pre_sat_ratio": float(stats["sat_ratio"]),
        "proxy_gain_safety": float(gain_safety),
        "proxy_sat_safety": float(sat_safety),
        "proxy_push_safety": float(push_safety),
        "proxy_effective_y_target_mean": float(effective_target_mean),
    }


def _match_proxy_luminance_mean_global(proxy_pre, target_mean, proxy_cfg, eps, search_steps=12):
    proxy_pre = torch.clamp(proxy_pre, 0.0, 1.0)
    target_mean = float(min(max(target_mean, 0.0), 1.0))
    current_y_mean = float(_compute_gray(proxy_pre).mean().item())
    if abs(current_y_mean - target_mean) <= 1.0e-4:
        return {
            "proxy_target": proxy_pre,
            "proxy_post_applied": 0,
            "proxy_post_delta": 0.0,
            "proxy_post_match_scale": 1.0,
            "proxy_y_mean": float(current_y_mean),
            "proxy_post_route": "global",
        }

    max_gain = float(max(_cfg_get(proxy_cfg, "POST_MATCH_MAX_GAIN", _cfg_get(proxy_cfg, "MAX_GAIN", 1.0)), 1.0))
    search_steps = int(max(1, search_steps))

    def _apply_gain(scale):
        scaled = torch.clamp(proxy_pre * float(scale), 0.0, 1.0)
        scaled_y_mean = float(_compute_gray(scaled).mean().item())
        return scaled, scaled_y_mean

    if current_y_mean < target_mean:
        low_scale = 1.0
        high_scale = max_gain
        low_image, low_y_mean = proxy_pre, current_y_mean
        high_image, high_y_mean = _apply_gain(high_scale)
        if high_y_mean <= target_mean + 1.0e-4:
            return {
                "proxy_target": high_image,
                "proxy_post_applied": 1,
                "proxy_post_delta": float(high_scale - 1.0),
                "proxy_post_match_scale": float(high_scale),
                "proxy_y_mean": float(high_y_mean),
                "proxy_post_route": "global",
            }
    else:
        low_scale = 0.0
        high_scale = 1.0
        low_image, low_y_mean = _apply_gain(low_scale)
        high_image, high_y_mean = proxy_pre, current_y_mean

    for _ in range(search_steps):
        mid_scale = 0.5 * (low_scale + high_scale)
        mid_image, mid_y_mean = _apply_gain(mid_scale)
        if mid_y_mean < target_mean:
            low_scale, low_image, low_y_mean = mid_scale, mid_image, mid_y_mean
        else:
            high_scale, high_image, high_y_mean = mid_scale, mid_image, mid_y_mean

    if abs(low_y_mean - target_mean) <= abs(high_y_mean - target_mean):
        chosen_scale, chosen_image, chosen_y = low_scale, low_image, low_y_mean
    else:
        chosen_scale, chosen_image, chosen_y = high_scale, high_image, high_y_mean
    return {
        "proxy_target": chosen_image,
        "proxy_post_applied": int(abs(chosen_scale - 1.0) > 1.0e-6),
        "proxy_post_delta": float(chosen_scale - 1.0),
        "proxy_post_match_scale": float(chosen_scale),
        "proxy_y_mean": float(chosen_y),
        "proxy_post_route": "global",
    }


def _match_proxy_luminance_mean(proxy_pre, target_mean, proxy_push_safety, proxy_cfg, eps, search_steps=12):
    proxy_pre = torch.clamp(proxy_pre, 0.0, 1.0)
    target_mean = float(min(max(target_mean, 0.0), 1.0))
    gray_pre = _compute_gray(proxy_pre)
    current_y_mean = float(gray_pre.mean().item())
    search_steps = int(max(1, search_steps))
    max_delta = float(max(_cfg_get(proxy_cfg, "POST_MATCH_MAX_DELTA", 4.0), 0.0))
    headroom_power = float(max(_cfg_get(proxy_cfg, "POST_MATCH_HEADROOM_POWER", 1.5), 1.0e-6))
    pixel_sat_soft = float(_cfg_get(proxy_cfg, "POST_MATCH_PIXEL_SAT_SOFT", 0.90))
    pixel_sat_hard = float(_cfg_get(proxy_cfg, "POST_MATCH_PIXEL_SAT_HARD", 0.98))
    global_route_thresh = float(_cfg_get(proxy_cfg, "POST_MATCH_GLOBAL_ROUTE_PUSH_SAFETY_THRESH", 0.90))

    if float(proxy_push_safety) >= global_route_thresh:
        return _match_proxy_luminance_mean_global(proxy_pre, target_mean, proxy_cfg, eps, search_steps=search_steps)

    rgb_max = proxy_pre.max(dim=0).values
    headroom = torch.clamp(1.0 - gray_pre, 0.0, 1.0).pow(headroom_power)
    sat_gate = 1.0 - torch.clamp((rgb_max - pixel_sat_soft) / max(pixel_sat_hard - pixel_sat_soft, 1.0e-6), 0.0, 1.0)
    lift_weight = headroom * sat_gate

    if target_mean <= current_y_mean + 1.0e-4 or float(lift_weight.max().item()) <= 1.0e-6 or max_delta <= 0.0:
        return {
            "proxy_target": proxy_pre,
            "proxy_post_applied": 0,
            "proxy_post_delta": 0.0,
            "proxy_post_match_scale": 1.0,
            "proxy_y_mean": float(current_y_mean),
            "proxy_post_route": "local",
        }

    def _apply_delta(delta):
        delta = float(max(delta, 0.0))
        y_post = torch.clamp(gray_pre + delta * lift_weight * (1.0 - gray_pre), 0.0, 1.0)
        scale = torch.where(
            gray_pre > eps,
            y_post / gray_pre.clamp_min(eps),
            torch.ones_like(y_post),
        )
        proxy_post = torch.clamp(proxy_pre * scale.unsqueeze(0), 0.0, 1.0)
        proxy_post_y_mean = float(_compute_gray(proxy_post).mean().item())
        scale_mean = float(scale.mean().item())
        return proxy_post, proxy_post_y_mean, scale_mean

    low_delta = 0.0
    high_delta = max_delta
    low_image, low_y_mean, low_scale = proxy_pre, current_y_mean, 1.0
    high_image, high_y_mean, high_scale = _apply_delta(high_delta)
    if high_y_mean <= target_mean + 1.0e-4:
        return {
            "proxy_target": high_image,
            "proxy_post_applied": 1,
            "proxy_post_delta": float(high_delta),
            "proxy_post_match_scale": float(high_scale),
            "proxy_y_mean": float(high_y_mean),
            "proxy_post_route": "local",
        }

    for _ in range(search_steps):
        mid_delta = 0.5 * (low_delta + high_delta)
        mid_image, mid_y_mean, mid_scale = _apply_delta(mid_delta)
        if mid_y_mean < target_mean:
            low_delta, low_image, low_y_mean, low_scale = mid_delta, mid_image, mid_y_mean, mid_scale
        else:
            high_delta, high_image, high_y_mean, high_scale = mid_delta, mid_image, mid_y_mean, mid_scale

    if abs(low_y_mean - target_mean) <= abs(high_y_mean - target_mean):
        return {
            "proxy_target": low_image,
            "proxy_post_applied": int(low_delta > 1.0e-6),
            "proxy_post_delta": float(low_delta),
            "proxy_post_match_scale": float(low_scale),
            "proxy_y_mean": float(low_y_mean),
            "proxy_post_route": "local",
        }
    return {
        "proxy_target": high_image,
        "proxy_post_applied": int(high_delta > 1.0e-6),
        "proxy_post_delta": float(high_delta),
        "proxy_post_match_scale": float(high_scale),
        "proxy_y_mean": float(high_y_mean),
        "proxy_post_route": "local",
    }


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
        post_match_enabled = bool(_cfg_get(proxy_cfg, "POST_MATCH_Y_MEAN_ENABLED", True))
        post_match_steps = int(_cfg_get(proxy_cfg, "POST_MATCH_SEARCH_STEPS", 12))
        effective_target_info = _compute_proxy_effective_target_mean(
            proxy_target,
            fallback_target_mean,
            proxy_gain,
            proxy_cfg,
        )
        post_match_info = {
            "proxy_target": proxy_target,
            "proxy_post_applied": 0,
            "proxy_post_delta": 0.0,
            "proxy_post_match_scale": 1.0,
            "proxy_y_mean": float(_compute_gray(proxy_target).mean().item()),
            "proxy_post_route": "local",
        }
        if post_match_enabled:
            post_match_info = _match_proxy_luminance_mean(
                proxy_target,
                effective_target_info["proxy_effective_y_target_mean"],
                effective_target_info["proxy_push_safety"],
                proxy_cfg=proxy_cfg,
                eps=float(_cfg_get(proxy_cfg, "EPS", 1e-6)),
                search_steps=post_match_steps,
            )
            proxy_target = post_match_info["proxy_target"]
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
            "proxy_y_mean": float(post_match_info["proxy_y_mean"]),
            "proxy_global_y_mean": float(effective_target_info["proxy_pre_y_mean"]),
            "proxy_shadow_y_mean": float(effective_target_info["proxy_pre_y_mean"]),
            "proxy_blend_y_mean": float(effective_target_info["proxy_pre_y_mean"]),
            "proxy_pre_y_mean": float(effective_target_info["proxy_pre_y_mean"]),
            "proxy_pre_y_p95": float(effective_target_info["proxy_pre_y_p95"]),
            "proxy_pre_sat_ratio": float(effective_target_info["proxy_pre_sat_ratio"]),
            "proxy_gain_safety": float(effective_target_info["proxy_gain_safety"]),
            "proxy_sat_safety": float(effective_target_info["proxy_sat_safety"]),
            "proxy_push_safety": float(effective_target_info["proxy_push_safety"]),
            "proxy_effective_y_target_mean": float(effective_target_info["proxy_effective_y_target_mean"]),
            "proxy_post_match_scale": float(post_match_info["proxy_post_match_scale"]),
            "proxy_post_applied": int(post_match_info["proxy_post_applied"]),
            "proxy_post_delta": float(post_match_info["proxy_post_delta"]),
            "proxy_post_route": str(post_match_info["proxy_post_route"]),
            "proxy_post_match_enabled": int(post_match_enabled),
            "proxy_target_mean_raw": float(fallback_target_mean),
            "proxy_target_mean": float(effective_target_info["proxy_effective_y_target_mean"]),
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
    proxy_global_y_mean = float(_compute_gray(proxy_global).mean().item())
    post_match_enabled = bool(_cfg_get(proxy_cfg, "POST_MATCH_Y_MEAN_ENABLED", True))
    post_match_steps = int(_cfg_get(proxy_cfg, "POST_MATCH_SEARCH_STEPS", 12))

    form = str(_cfg_get(proxy_cfg, "FORM", "global_linear")).lower()
    if form == "global_linear":
        zero_weight = torch.zeros_like(gray)
        proxy_pre = proxy_global
        effective_target_info = _compute_proxy_effective_target_mean(
            proxy_pre,
            target_mean,
            float(global_gain.item()),
            proxy_cfg,
        )
        post_match_info = {
            "proxy_target": proxy_pre,
            "proxy_post_applied": 0,
            "proxy_post_delta": 0.0,
            "proxy_post_match_scale": 1.0,
            "proxy_y_mean": float(effective_target_info["proxy_pre_y_mean"]),
            "proxy_post_route": "local",
        }
        if post_match_enabled:
            post_match_info = _match_proxy_luminance_mean(
                proxy_pre,
                effective_target_info["proxy_effective_y_target_mean"],
                effective_target_info["proxy_push_safety"],
                proxy_cfg=proxy_cfg,
                eps=eps,
                search_steps=post_match_steps,
            )
        proxy_target = post_match_info["proxy_target"]
        proxy_mean = float(proxy_target.mean().item())
        return {
            "proxy_target": proxy_target,
            "proxy_global": proxy_global,
            "proxy_shadow": proxy_global,
            "proxy_shadow_weight": zero_weight,
            "proxy_gain": float(global_gain.item()),
            "proxy_stat_mean": float(stat_mean.item()),
            "proxy_stat_mode": stat_label,
            "proxy_form": form,
            "proxy_global_mean": float(proxy_global.mean().item()),
            "proxy_shadow_mean": float(proxy_global.mean().item()),
            "proxy_blend_mean": float(proxy_pre.mean().item()),
            "proxy_shadow_weight_mean": 0.0,
            "proxy_y_mean": float(post_match_info["proxy_y_mean"]),
            "proxy_global_y_mean": proxy_global_y_mean,
            "proxy_shadow_y_mean": proxy_global_y_mean,
            "proxy_blend_y_mean": float(effective_target_info["proxy_pre_y_mean"]),
            "proxy_pre_y_mean": float(effective_target_info["proxy_pre_y_mean"]),
            "proxy_pre_y_p95": float(effective_target_info["proxy_pre_y_p95"]),
            "proxy_pre_sat_ratio": float(effective_target_info["proxy_pre_sat_ratio"]),
            "proxy_gain_safety": float(effective_target_info["proxy_gain_safety"]),
            "proxy_sat_safety": float(effective_target_info["proxy_sat_safety"]),
            "proxy_push_safety": float(effective_target_info["proxy_push_safety"]),
            "proxy_effective_y_target_mean": float(effective_target_info["proxy_effective_y_target_mean"]),
            "proxy_post_match_scale": float(post_match_info["proxy_post_match_scale"]),
            "proxy_post_applied": int(post_match_info["proxy_post_applied"]),
            "proxy_post_delta": float(post_match_info["proxy_post_delta"]),
            "proxy_post_route": str(post_match_info["proxy_post_route"]),
            "proxy_post_match_enabled": int(post_match_enabled),
            "proxy_target_mean_raw": target_mean,
            "proxy_target_mean": float(effective_target_info["proxy_effective_y_target_mean"]),
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
        proxy_pre = torch.clamp(
            (1.0 - shadow_weight.unsqueeze(0)) * proxy_global + shadow_weight.unsqueeze(0) * shadow_proxy,
            0.0,
            1.0,
        )
        effective_target_info = _compute_proxy_effective_target_mean(
            proxy_pre,
            target_mean,
            float(global_gain.item()),
            proxy_cfg,
        )
        proxy_shadow_y_mean = float(_compute_gray(shadow_proxy).mean().item())
        post_match_info = {
            "proxy_target": proxy_pre,
            "proxy_post_applied": 0,
            "proxy_post_delta": 0.0,
            "proxy_post_match_scale": 1.0,
            "proxy_y_mean": float(effective_target_info["proxy_pre_y_mean"]),
            "proxy_post_route": "local",
        }
        if post_match_enabled:
            post_match_info = _match_proxy_luminance_mean(
                proxy_pre,
                effective_target_info["proxy_effective_y_target_mean"],
                effective_target_info["proxy_push_safety"],
                proxy_cfg=proxy_cfg,
                eps=eps,
                search_steps=post_match_steps,
            )
        proxy_target = post_match_info["proxy_target"]
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
            "proxy_blend_mean": float(proxy_pre.mean().item()),
            "proxy_shadow_weight_mean": float(shadow_weight.mean().item()),
            "proxy_y_mean": float(post_match_info["proxy_y_mean"]),
            "proxy_global_y_mean": proxy_global_y_mean,
            "proxy_shadow_y_mean": proxy_shadow_y_mean,
            "proxy_blend_y_mean": float(effective_target_info["proxy_pre_y_mean"]),
            "proxy_pre_y_mean": float(effective_target_info["proxy_pre_y_mean"]),
            "proxy_pre_y_p95": float(effective_target_info["proxy_pre_y_p95"]),
            "proxy_pre_sat_ratio": float(effective_target_info["proxy_pre_sat_ratio"]),
            "proxy_gain_safety": float(effective_target_info["proxy_gain_safety"]),
            "proxy_sat_safety": float(effective_target_info["proxy_sat_safety"]),
            "proxy_push_safety": float(effective_target_info["proxy_push_safety"]),
            "proxy_effective_y_target_mean": float(effective_target_info["proxy_effective_y_target_mean"]),
            "proxy_post_match_scale": float(post_match_info["proxy_post_match_scale"]),
            "proxy_post_applied": int(post_match_info["proxy_post_applied"]),
            "proxy_post_delta": float(post_match_info["proxy_post_delta"]),
            "proxy_post_route": str(post_match_info["proxy_post_route"]),
            "proxy_post_match_enabled": int(post_match_enabled),
            "proxy_target_mean_raw": target_mean,
            "proxy_target_mean": float(effective_target_info["proxy_effective_y_target_mean"]),
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
            "proxy_y_mean": float(proxy_info["proxy_y_mean"]),
            "proxy_global_y_mean": float(proxy_info["proxy_global_y_mean"]),
            "proxy_shadow_y_mean": float(proxy_info["proxy_shadow_y_mean"]),
            "proxy_blend_y_mean": float(proxy_info["proxy_blend_y_mean"]),
            "proxy_pre_y_mean": float(proxy_info["proxy_pre_y_mean"]),
            "proxy_pre_y_p95": float(proxy_info["proxy_pre_y_p95"]),
            "proxy_pre_sat_ratio": float(proxy_info["proxy_pre_sat_ratio"]),
            "proxy_gain_safety": float(proxy_info["proxy_gain_safety"]),
            "proxy_sat_safety": float(proxy_info["proxy_sat_safety"]),
            "proxy_push_safety": float(proxy_info["proxy_push_safety"]),
            "proxy_effective_y_target_mean": float(proxy_info["proxy_effective_y_target_mean"]),
            "proxy_post_match_scale": float(proxy_info["proxy_post_match_scale"]),
            "proxy_post_applied": int(proxy_info["proxy_post_applied"]),
            "proxy_post_delta": float(proxy_info["proxy_post_delta"]),
            "proxy_post_route": str(proxy_info["proxy_post_route"]),
            "proxy_post_match_enabled": int(proxy_info["proxy_post_match_enabled"]),
            "proxy_target_mean_raw": float(proxy_info["proxy_target_mean_raw"]),
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
        "proxy_y_mean": float(proxy_info["proxy_y_mean"]),
        "proxy_global_y_mean": float(proxy_info["proxy_global_y_mean"]),
        "proxy_shadow_y_mean": float(proxy_info["proxy_shadow_y_mean"]),
        "proxy_blend_y_mean": float(proxy_info["proxy_blend_y_mean"]),
        "proxy_pre_y_mean": float(proxy_info["proxy_pre_y_mean"]),
        "proxy_pre_y_p95": float(proxy_info["proxy_pre_y_p95"]),
        "proxy_pre_sat_ratio": float(proxy_info["proxy_pre_sat_ratio"]),
        "proxy_gain_safety": float(proxy_info["proxy_gain_safety"]),
        "proxy_sat_safety": float(proxy_info["proxy_sat_safety"]),
        "proxy_push_safety": float(proxy_info["proxy_push_safety"]),
        "proxy_effective_y_target_mean": float(proxy_info["proxy_effective_y_target_mean"]),
        "proxy_post_match_scale": float(proxy_info["proxy_post_match_scale"]),
        "proxy_post_applied": int(proxy_info["proxy_post_applied"]),
        "proxy_post_delta": float(proxy_info["proxy_post_delta"]),
        "proxy_post_route": str(proxy_info["proxy_post_route"]),
        "proxy_post_match_enabled": int(proxy_info["proxy_post_match_enabled"]),
        "proxy_target_mean_raw": float(proxy_info["proxy_target_mean_raw"]),
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

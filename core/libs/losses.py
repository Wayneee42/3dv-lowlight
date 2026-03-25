import torch

from .utils import ssim



def rgb_to_ycbcr_hwc(rgb_hwc):
    rgb = torch.clamp(rgb_hwc, 0.0, 1.0)
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    cb = (rgb[..., 2] - y) / 1.772
    cr = (rgb[..., 0] - y) / 1.402
    return torch.stack([y, cb, cr], dim=-1)


def ycbcr_to_rgb_hwc(ycbcr_hwc):
    y = ycbcr_hwc[..., 0]
    cb = ycbcr_hwc[..., 1]
    cr = ycbcr_hwc[..., 2]
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], dim=-1)


def chroma_factor_from_aux(chroma_aux, scale):
    return 1.0 + float(scale) * torch.tanh(chroma_aux)


def chroma_delta_from_aux(chroma_aux, scale):
    return float(scale) * torch.tanh(chroma_aux)


def apply_ycbcr_chroma_residual(base_lit_rgb, chroma_aux, scale, mode="multiplicative"):
    ycbcr = rgb_to_ycbcr_hwc(base_lit_rgb)
    y = ycbcr[..., 0]
    chroma_factor = None
    chroma_delta = None
    normalized_mode = str(mode).lower()
    if normalized_mode == "multiplicative":
        chroma_factor = chroma_factor_from_aux(chroma_aux, scale)
        cb = ycbcr[..., 1] * chroma_factor[..., 0]
        cr = ycbcr[..., 2] * chroma_factor[..., 1]
    elif normalized_mode == "additive":
        chroma_delta = chroma_delta_from_aux(chroma_aux, scale)
        cb = ycbcr[..., 1] + chroma_delta[..., 0]
        cr = ycbcr[..., 2] + chroma_delta[..., 1]
    else:
        raise RuntimeError(f"Unsupported YCbCr chroma residual mode: {mode}")
    adjusted_ycbcr = torch.stack([y, cb, cr], dim=-1)
    recon_rgb = torch.clamp(ycbcr_to_rgb_hwc(adjusted_ycbcr), 0.0, 1.0)
    return recon_rgb, chroma_factor, chroma_delta


def rgb_reconstruction_loss(rendered, target_hwc, lambda_ssim, weight_map=None):
    if weight_map is None:
        l1_loss = torch.abs(rendered - target_hwc).mean()
    else:
        if weight_map.dim() == 2:
            weight_map = weight_map.unsqueeze(-1)
        if weight_map.dim() != 3:
            raise RuntimeError(f"rgb_reconstruction_loss expects weight_map with 2 or 3 dims, got {tuple(weight_map.shape)}")
        weight_map = weight_map.to(device=rendered.device, dtype=rendered.dtype)
        if weight_map.shape[-1] == 1:
            weight_map = weight_map.expand(-1, -1, rendered.shape[-1])
        if weight_map.shape != rendered.shape:
            raise RuntimeError(
                f"rgb_reconstruction_loss weight_map shape {tuple(weight_map.shape)} does not match rendered {tuple(rendered.shape)}"
            )
        l1_loss = (torch.abs(rendered - target_hwc) * weight_map).sum() / weight_map.sum().clamp_min(1.0e-6)
    ssim_value = ssim(rendered, target_hwc)
    rgb_loss = (1.0 - lambda_ssim) * l1_loss + lambda_ssim * (1.0 - ssim_value)
    return {
        "total": rgb_loss,
        "l1": l1_loss,
        "ssim": ssim_value,
    }



def luminance(image_hwc):
    return 0.299 * image_hwc[..., 0] + 0.587 * image_hwc[..., 1] + 0.114 * image_hwc[..., 2]



def low_light_consistency_loss(rendered, reference_hwc, eps=1e-6):
    rendered_luma = luminance(rendered)
    reference_luma = luminance(reference_hwc)
    rendered_norm = rendered_luma / rendered_luma.mean().clamp_min(eps)
    reference_norm = reference_luma / reference_luma.mean().clamp_min(eps)
    return torch.abs(rendered_norm - reference_norm).mean()



def exposure_control_loss(rendered, target_mean):
    rendered_luma = luminance(rendered)
    target = torch.tensor(float(target_mean), device=rendered.device, dtype=rendered.dtype)
    return torch.abs(rendered_luma.mean() - target)



def robust_exposure_control_loss(rendered, target_median, target_p75, mask_low=0.05, mask_high=0.95):
    rendered_luma = luminance(rendered)
    valid_mask = (rendered_luma > float(mask_low)) & (rendered_luma < float(mask_high))
    valid_values = rendered_luma[valid_mask]
    if valid_values.numel() == 0:
        valid_values = rendered_luma.reshape(-1)

    median_value = torch.median(valid_values)
    p75_value = torch.quantile(valid_values, 0.75)
    target_median = torch.tensor(float(target_median), device=rendered.device, dtype=rendered.dtype)
    target_p75 = torch.tensor(float(target_p75), device=rendered.device, dtype=rendered.dtype)
    return torch.abs(median_value - target_median) + 0.5 * torch.abs(p75_value - target_p75)

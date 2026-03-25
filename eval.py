#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import argparse
import json
import os
import warnings

import torch
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from core.data import Blender
from core.libs import ConfigDict, ssim
from core.model import Simple3DGS



def psnr(rendered, target, eps=1e-10):
    mse = ((rendered - target) ** 2).mean().clamp_min(eps)
    return float((-10.0 * torch.log10(mse)).item())



def try_build_lpips(device):
    try:
        import lpips
    except ImportError:
        return None
    model = lpips.LPIPS(net="vgg").to(device)
    model.eval()
    return model



def lpips_score(lpips_model, rendered, target):
    if lpips_model is None:
        return None
    rendered_nchw = rendered.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    target_nchw = target.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    return float(lpips_model(rendered_nchw, target_nchw).mean().item())



def can_compute_metrics(dataset):
    if len(dataset._records_keys) == 0:
        return False
    return all(os.path.exists(dataset._records[key]["file_path"]) for key in dataset._records_keys)



def write_metric_outputs(ckpt_dir, summary, per_view):
    summary_path = os.path.join(ckpt_dir, "results.json")
    per_view_path = os.path.join(ckpt_dir, "per_view.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(per_view_path, "w", encoding="utf-8") as handle:
        json.dump(per_view, handle, indent=2)
    print(f"Metric summary written to {summary_path}")
    print(f"Per-view metrics written to {per_view_path}")



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



def build_eval_heads(meta_cfg):
    loss_cfg = _cfg_get(meta_cfg, "LOSS", None)
    model_cfg = _cfg_get(meta_cfg, "MODEL", None)
    heads = []
    priors_cfg = _cfg_get(meta_cfg, "PRIORS", None)
    depth_cfg = _cfg_get(priors_cfg, "DEPTH", None)
    structure_cfg = _cfg_get(priors_cfg, "STRUCTURE", None)
    if bool(_cfg_get(depth_cfg, "ENABLED", False)):
        heads.append("depth")
    if bool(_cfg_get(structure_cfg, "ENABLED", False)):
        heads.append("prior")
    if float(_cfg_get(loss_cfg, "LAMBDA_RECONSTRUCTION", 0.0)) > 0.0:
        heads.append("illum")
    if bool(_cfg_get(model_cfg, "CHROMA_RESIDUAL_ENABLED", False)):
        heads.append("chroma")
    return tuple(heads)



def _build_chroma_preview_maps(render_outputs):
    chroma_delta = render_outputs.get("chroma_delta")
    chroma_factor = render_outputs.get("chroma_factor")
    chroma_scale = float(render_outputs.get("chroma_scale", 0.10))
    if chroma_delta is not None:
        denom = max(2.0 * chroma_scale, 1.0e-6)
        cb = (0.5 + chroma_delta[..., 0] / denom).clamp(0.0, 1.0).unsqueeze(0).repeat(3, 1, 1)
        cr = (0.5 + chroma_delta[..., 1] / denom).clamp(0.0, 1.0).unsqueeze(0).repeat(3, 1, 1)
        return cb, cr
    if chroma_factor is None:
        return None, None
    min_factor = 1.0 - chroma_scale
    denom = max(2.0 * chroma_scale, 1.0e-6)
    cb = ((chroma_factor[..., 0] - min_factor) / denom).clamp(0.0, 1.0).unsqueeze(0).repeat(3, 1, 1)
    cr = ((chroma_factor[..., 1] - min_factor) / denom).clamp(0.0, 1.0).unsqueeze(0).repeat(3, 1, 1)
    return cb, cr


def save_render_outputs(render_outputs, frame_key, output_dir):
    final_image = render_outputs["recon_rgb"]
    save_image(final_image.permute(2, 0, 1).clamp(0, 1), os.path.join(output_dir, f"{frame_key}.png"))

    illum_aux = render_outputs.get("illum_aux")
    if illum_aux is None:
        return final_image

    base_dir = os.path.join(output_dir, "base")
    illum_dir = os.path.join(output_dir, "illum")
    recon_dir = os.path.join(output_dir, "recon")
    chroma_dir = os.path.join(output_dir, "chroma")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(illum_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)

    base_image = render_outputs.get("base_lit_rgb", render_outputs["rgb"])
    save_image(base_image.permute(2, 0, 1).clamp(0, 1), os.path.join(base_dir, f"{frame_key}.png"))
    save_image((torch.clamp(2.0 * torch.sigmoid(illum_aux), 0.0, 2.0) / 2.0).permute(2, 0, 1), os.path.join(illum_dir, f"{frame_key}.png"))
    save_image(final_image.permute(2, 0, 1).clamp(0, 1), os.path.join(recon_dir, f"{frame_key}.png"))
    cb_preview, cr_preview = _build_chroma_preview_maps(render_outputs)
    if cb_preview is not None and cr_preview is not None:
        save_image(cb_preview, os.path.join(chroma_dir, f"cb_factor_{frame_key}.png"))
        save_image(cr_preview, os.path.join(chroma_dir, f"cr_factor_{frame_key}.png"))
    return final_image


@torch.no_grad()
def evaluate(checkpoint_path, device="cuda"):
    ckpt_dir = os.path.dirname(checkpoint_path)
    config_path = resolve_config_path(ckpt_dir)
    with open(config_path) as handle:
        config_dict = yaml.load(handle, Loader=yaml.Loader)
    config_dict["EXP_STR"] = ""
    config_dict["TIME_STR"] = ""
    meta_cfg = ConfigDict(config_path=config_dict)
    cfg = meta_cfg.MODEL
    render_heads = build_eval_heads(meta_cfg)

    test_dataset = Blender(meta_cfg.DATASET, split="test", load_images=False)
    metric_dataset = Blender(meta_cfg.DATASET, split="test", load_images=True) if can_compute_metrics(test_dataset) else None
    H, W = test_dataset._data_info["img_h"], test_dataset._data_info["img_w"]

    model = Simple3DGS(cfg, test_dataset._data_info).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for key, value in ckpt.items():
        model.splats[key] = torch.nn.Parameter(value)
    model.sh_degree = model.sh_degree_max
    model.eval()

    output_dir = os.path.join(ckpt_dir, "test")
    os.makedirs(output_dir, exist_ok=True)
    num_test = len(test_dataset._records_keys)

    lpips_model = try_build_lpips(device)
    metric_values = {"PSNR": [], "SSIM": []}
    per_view = {"PSNR": {}, "SSIM": {}}
    if lpips_model is not None:
        metric_values["LPIPS"] = []
        per_view["LPIPS"] = {}

    for index in tqdm(range(num_test), desc="Rendering"):
        data = test_dataset[index]
        camtoworld = data["transforms"].to(device)
        render_outputs = model(camtoworld, H, W, render_heads=render_heads)
        frame_key = data["infos"]["frame_key"]
        rendered = save_render_outputs(render_outputs, frame_key, output_dir)

        if metric_dataset is not None:
            gt_data = metric_dataset[index]
            gt_hwc = gt_data["images"].to(device).permute(1, 2, 0)
            psnr_value = psnr(rendered, gt_hwc)
            ssim_value = float(ssim(rendered, gt_hwc).item())
            metric_values["PSNR"].append(psnr_value)
            metric_values["SSIM"].append(ssim_value)
            per_view["PSNR"][frame_key] = psnr_value
            per_view["SSIM"][frame_key] = ssim_value

            if lpips_model is not None:
                lpips_value = lpips_score(lpips_model, rendered, gt_hwc)
                metric_values["LPIPS"].append(lpips_value)
                per_view["LPIPS"][frame_key] = lpips_value

    print(f"Rendered {num_test} images to {output_dir}/ | {model.num_gaussians} Gaussians")

    if metric_dataset is not None:
        summary = {metric_name: float(sum(values) / len(values)) for metric_name, values in metric_values.items() if values}
        write_metric_outputs(ckpt_dir, summary, per_view)
        for metric_name, value in summary.items():
            print(f"{metric_name}: {value:.6f}")
    else:
        print("Ground-truth test images not found; skipped metric computation.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-w", required=True, type=str)
    args = parser.parse_args()
    evaluate(args.checkpoint)

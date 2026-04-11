#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import argparse
import json
import os
import warnings
from pathlib import Path

import torch
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from core.data import Blender
from core.data.blender import load_img
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



def resolve_metric_gt_paths(dataset, scene_name):
    if len(dataset._records_keys) == 0:
        return None

    repo_root = Path(__file__).resolve().parent
    gt_paths = []
    for key in dataset._records_keys:
        record = dataset._records[key]
        candidates = []

        file_path = record.get("file_path", None)
        if file_path is not None:
            candidates.append(Path(file_path))

        relative_path = record.get("relative_path", None)
        if scene_name and relative_path:
            candidates.append(repo_root / "lowlight" / scene_name / relative_path.replace("/", os.sep))

        resolved_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if resolved_path is None:
            return None
        gt_paths.append(str(resolved_path))
    return gt_paths



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



def resolve_train_export_options(meta_cfg):
    eval_cfg = _cfg_get(meta_cfg, "EVAL", None)
    export_train_views = bool(_cfg_get(eval_cfg, "EXPORT_TRAIN_VIEWS", False))
    train_render_dir = str(_cfg_get(eval_cfg, "TRAIN_RENDER_DIR", "train_render"))
    if not train_render_dir:
        train_render_dir = "train_render"
    return export_train_views, train_render_dir


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
    lambda_recon_y = float(_cfg_get(loss_cfg, "LAMBDA_RECON_Y", 0.0))
    lambda_recon_cbcr = float(_cfg_get(loss_cfg, "LAMBDA_RECON_CBCR", 0.0))
    if lambda_recon_y > 0.0 or lambda_recon_cbcr > 0.0 or float(_cfg_get(loss_cfg, "LAMBDA_ILLUM_REG", 0.0)) > 0.0:
        heads.append("illum")
    if bool(_cfg_get(model_cfg, "CHROMA_RESIDUAL_ENABLED", False)) or lambda_recon_cbcr > 0.0 or float(_cfg_get(loss_cfg, "LAMBDA_CHROMA_REG", 0.0)) > 0.0:
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


def load_rendered_output(output_dir, frame_key, device):
    rendered_path = os.path.join(output_dir, f"{frame_key}.png")
    if not os.path.exists(rendered_path):
        raise FileNotFoundError(f"Rendered image not found for frame '{frame_key}': {rendered_path}")
    return load_img(rendered_path, channel=3).float().to(device)[:3].permute(1, 2, 0) / 255.0


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
    export_train_views, train_render_dir = resolve_train_export_options(meta_cfg)

    test_dataset = Blender(meta_cfg.DATASET, split="test", load_images=False)
    scene_name = _cfg_get(meta_cfg.DATASET, "NAME", None)
    metric_gt_paths = resolve_metric_gt_paths(test_dataset, scene_name)
    H, W = test_dataset._data_info["img_h"], test_dataset._data_info["img_w"]

    output_dir = os.path.join(ckpt_dir, "test")
    checkpoint_exists = os.path.exists(checkpoint_path)
    model = None
    if checkpoint_exists:
        model = Simple3DGS(cfg, test_dataset._data_info).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        for key, value in ckpt.items():
            model.splats[key] = torch.nn.Parameter(value)
        model.sh_degree = model.sh_degree_max
        model.eval()
        os.makedirs(output_dir, exist_ok=True)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print(f"Skip rendering and evaluate existing outputs in: {output_dir}")
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(f"Rendered output directory not found: {output_dir}")

    num_test = len(test_dataset._records_keys)

    lpips_model = try_build_lpips(device)
    metric_values = {"PSNR": [], "SSIM": []}
    per_view = {"PSNR": {}, "SSIM": {}}
    if lpips_model is not None:
        metric_values["LPIPS"] = []
        per_view["LPIPS"] = {}

    loop_desc = "Rendering" if checkpoint_exists else "Evaluating"
    for index in tqdm(range(num_test), desc=loop_desc):
        data = test_dataset[index]
        frame_key = data["infos"]["frame_key"]
        if checkpoint_exists:
            camtoworld = data["transforms"].to(device)
            render_outputs = model(camtoworld, H, W, render_heads=render_heads)
            rendered = save_render_outputs(render_outputs, frame_key, output_dir)
        else:
            rendered = load_rendered_output(output_dir, frame_key, device)

        if metric_gt_paths is not None:
            gt_hwc = load_img(metric_gt_paths[index], channel=3).float().to(device)[:3].permute(1, 2, 0) / 255.0
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

    if checkpoint_exists and export_train_views:
        train_dataset = Blender(meta_cfg.DATASET, split="train", load_images=False)
        train_output_dir = os.path.join(ckpt_dir, train_render_dir)
        train_h, train_w = train_dataset._data_info["img_h"], train_dataset._data_info["img_w"]
        os.makedirs(train_output_dir, exist_ok=True)
        num_train = len(train_dataset._records_keys)
        for index in tqdm(range(num_train), desc="Rendering train views"):
            data = train_dataset[index]
            frame_key = data["infos"]["frame_key"]
            camtoworld = data["transforms"].to(device)
            render_outputs = model(camtoworld, train_h, train_w, render_heads=render_heads)
            save_render_outputs(render_outputs, frame_key, train_output_dir)
        print(f"Rendered {num_train} train-view images to {train_output_dir}/")

    if checkpoint_exists:
        print(f"Rendered {num_test} images to {output_dir}/ | {model.num_gaussians} Gaussians")
    else:
        print(f"Evaluated {num_test} existing rendered images from {output_dir}/")

    if metric_gt_paths is not None:
        summary = {metric_name: float(sum(values) / len(values)) for metric_name, values in metric_values.items() if values}
        write_metric_outputs(ckpt_dir, summary, per_view)
        for metric_name, value in summary.items():
            print(f"{metric_name}: {value:.6f}")
    else:
        print("Ground-truth test images not found in dataset paths or lowlight/<scene>/test; skipped metric computation.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-w", required=True, type=str)
    args = parser.parse_args()
    evaluate(args.checkpoint)

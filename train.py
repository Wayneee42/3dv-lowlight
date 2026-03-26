#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import argparse
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


def load_colmap_sparse_points(meta_cfg, sparse_cfg, device):
    if not bool(_cfg_get(sparse_cfg, "ENABLED", False)):
        return None
    scene_root = Path(str(meta_cfg.DATASET.DATA_PATH))
    colmap_dir = str(_cfg_get(sparse_cfg, "COLMAP_DIR", "auxiliaries/colmap_sparse"))
    colmap_root = Path(colmap_dir) if os.path.isabs(colmap_dir) else scene_root / colmap_dir
    points_path = colmap_root / "points.npy"
    if not points_path.exists():
        raise RuntimeError(f"Sparse prior is enabled, but COLMAP points were not found: {points_path}")
    points = np.load(points_path).astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise RuntimeError(f"Sparse prior requires non-empty [N,3] points.npy, got {points.shape}: {points_path}")
    points_tensor = torch.from_numpy(points).to(device)
    print(f"[Sparse] loaded {points_tensor.shape[0]} COLMAP sparse points from {points_path}")
    return points_tensor


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
    colmap_sparse_points = load_colmap_sparse_points(meta_cfg, sparse_cfg, device)

    model = Simple3DGS(cfg, train_dataset._data_info, init_context=init_context).to(device)
    warmstart_checkpoint = _cfg_get(cfg, "WARMSTART_CHECKPOINT", None)
    if warmstart_checkpoint:
        load_warmstart_checkpoint(model, warmstart_checkpoint, device)
    print(f"Initialized {model.num_gaussians} Gaussians")

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
    optimizers = {}
    for name, param in model.splats.items():
        optimizers[name] = torch.optim.Adam([param], lr=lr_map[name], eps=1e-15)

    total_steps = int(cfg.TRAIN_TOTAL_STEP)
    lr_final_factor = cfg.LR_MEANS_FINAL / cfg.LR_MEANS
    schedulers = {
        "means": torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=lr_final_factor ** (1.0 / total_steps)
        )
    }

    strategy = None
    strategy_state = None
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

        strategy.step_pre_backward(model.splats, optimizers, strategy_state, step, render_outputs["info"])
        loss.backward()
        strategy.step_post_backward(model.splats, optimizers, strategy_state, step, render_outputs["info"], packed=False)

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers.values():
            scheduler.step()
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


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", required=True, type=str)
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))

    train(args.config_path)

# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision


STRUCTURE_MODALITY_ALIASES = ("prior",)



def build_frame_key(file_path):
    split_name, file_name = file_path.split("/")[-2:]
    return f"{split_name}_{Path(file_name).stem}"


def camera_center_from_c2w(transform_matrix):
    c2w = torch.eye(4, dtype=transform_matrix.dtype)
    c2w[:3, :] = transform_matrix
    return c2w[:3, 3].clone()


def camera_forward_from_c2w(transform_matrix):
    c2w = torch.eye(4, dtype=transform_matrix.dtype)
    c2w[:3, :] = transform_matrix
    forward = c2w[:3, :3] @ torch.tensor([0.0, 0.0, -1.0], dtype=transform_matrix.dtype)
    return torch.nn.functional.normalize(forward, dim=0)


class Blender(torch.utils.data.Dataset):
    def __init__(self, data_cfg, split, load_images=True, neighbor_cfg=None):
        super().__init__()
        assert split in ["train", "val", "test"]

        self._scene_root = data_cfg.DATA_PATH
        self._bg_color = data_cfg.BACKGROUND_COLOR / 255.0
        self._load_images = load_images
        self._requested_split = split
        self._meta_split = "test" if split == "val" else split
        self._render_split = split
        self._auxiliary_dir = getattr(data_cfg, "AUXILIARY_DIR", "auxiliaries")

        self._img_path_base = os.path.join(self._scene_root, self._meta_split)
        self._meta_path_base = os.path.join(self._scene_root, f"transforms_{self._meta_split}.json")
        self._records, self._data_info = self._load_data()

        if split == "val":
            first_four_keys = list(self._records.keys())[:4]
            self._records = {key: self._records[key] for key in first_four_keys}

        if load_images:
            self._pre_loading_data()

        self._records_keys = list(self._records.keys())
        self._length = len(self._records_keys)
        self._pose_neighbors = self._build_pose_neighbors()

    def __getitem__(self, index):
        frame_key = self._records_keys[index % len(self._records_keys)]
        return self._load_one_record(self._records[frame_key])

    def __len__(self):
        return self._length

    def get_pose_neighbor(self, frame_key):
        neighbor_key = self._pose_neighbors.get(frame_key, None)
        if neighbor_key is None:
            return None
        return dict(self._records[neighbor_key])

    def _load_one_record(self, record):
        one_record_data = {
            "transforms": record["transform_matrix"],
            "infos": {
                "frame_key": record["frame_key"],
                "frame_name": record["frame_name"],
                "frame_stem": record["frame_stem"],
                "split": record["split"],
                "relative_path": record["relative_path"],
            },
            "low_light_image": None,
            "depth": None,
            "structure": None,
            "prior": None,
        }
        if self._load_images:
            one_record_data["images"] = record["img_tensor"]
            one_record_data["low_light_image"] = record["low_light_tensor"]
            one_record_data["depth"] = record["depth_tensor"]
            one_record_data["structure"] = record["structure_tensor"]
            one_record_data["prior"] = record["structure_tensor"]
        return one_record_data

    def _load_data(self):
        with open(self._meta_path_base, "rb") as f:
            json_data = json.load(f)
        meta_info = {
            "bg_color": self._bg_color,
            "img_h": int(json_data["h"]),
            "img_w": int(json_data["w"]),
            "fl_x": json_data["fl_x"],
            "fl_y": json_data["fl_y"],
            "cx": json_data["cx"],
            "cy": json_data["cy"],
            "camera_convention": "blender_nerf_synthetic_c2w_opengl",
            "renderer_camera_convention": "opencv_w2c_after_yz_flip",
        }
        records = {}
        for frame in json_data["frames"]:
            relative_path = frame["file_path"]
            frame_key = build_frame_key(relative_path)
            frame_name = os.path.basename(relative_path)
            frame_stem = Path(frame_name).stem
            file_path = os.path.join(self._scene_root, relative_path.replace("/", os.sep))
            transform_matrix = torch.tensor(frame["transform_matrix"]).float()[:3]
            camera_center = camera_center_from_c2w(transform_matrix)
            camera_forward = camera_forward_from_c2w(transform_matrix)
            records[frame_key] = {
                "frame_key": frame_key,
                "frame_name": frame_name,
                "frame_stem": frame_stem,
                "split": relative_path.split("/")[-2],
                "relative_path": relative_path,
                "file_path": file_path,
                "img_tensor": None,
                "low_light_tensor": None,
                "depth_tensor": None,
                "structure_tensor": None,
                "transform_matrix": transform_matrix,
                "camera_center": camera_center,
                "camera_forward": camera_forward,
            }
        return records, meta_info

    def _build_pose_neighbors(self):
        if self._render_split != "train":
            return {}
        if len(self._records) < 2:
            return {}

        frame_keys = list(self._records.keys())
        centers = torch.stack([self._records[key]["camera_center"] for key in frame_keys], dim=0)
        forwards = torch.stack([self._records[key]["camera_forward"] for key in frame_keys], dim=0)
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

    def _build_pose_only_topk_neighbors(self, frame_keys, centers, forwards):
        neighbors = {}
        for index, frame_key in enumerate(frame_keys):
            center = centers[index:index + 1]
            forward = forwards[index:index + 1]
            distances = torch.norm(centers - center, dim=1)
            alignment = torch.sum(forwards * forward, dim=1).clamp(-1.0, 1.0)
            scores = distances + 0.25 * (1.0 - alignment)
            scores[index] = float("inf")
            order = torch.argsort(scores)[: self._neighbor_top_k]
            neighbors[frame_key] = [
                {
                    "key": frame_keys[int(neighbor_index.item())],
                    "score": float(-scores[int(neighbor_index.item())].item()),
                    "overlap": 0.0,
                }
                for neighbor_index in order
            ]
        return neighbors

    def _load_overlap_points(self):
        colmap_points_path = Path(self._scene_root) / self._auxiliary_dir / "colmap_sparse" / "points.npy"
        if colmap_points_path.exists():
            points = np.load(colmap_points_path).astype(np.float32)
            if points.ndim == 2 and points.shape[1] == 3 and points.shape[0] > 0:
                if points.shape[0] > 8000:
                    choice = np.random.choice(points.shape[0], size=8000, replace=False)
                    points = points[choice]
                return points
        return None

    def _project_points_visibility_mask(self, points_world, transform_matrix):
        if points_world is None or points_world.shape[0] == 0:
            return np.zeros((0,), dtype=bool)

        width = int(self._data_info["img_w"])
        height = int(self._data_info["img_h"])
        fx = float(self._data_info["fl_x"])
        fy = float(self._data_info["fl_y"])
        cx = float(self._data_info["cx"])
        cy = float(self._data_info["cy"])

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :] = transform_matrix.cpu().numpy().astype(np.float32)
        rotation = c2w[:3, :3]
        translation = c2w[:3, 3]

        cam_gl = (points_world - translation[None, :]) @ rotation
        cam_cv = cam_gl * np.array([1.0, -1.0, -1.0], dtype=np.float32)
        z = cam_cv[:, 2]
        valid = np.isfinite(z) & (z > 1.0e-4)
        if not np.any(valid):
            return np.zeros((points_world.shape[0],), dtype=bool)

        u = fx * (cam_cv[:, 0] / np.clip(z, 1.0e-4, None)) + cx
        v = fy * (cam_cv[:, 1] / np.clip(z, 1.0e-4, None)) + cy
        valid &= np.isfinite(u) & np.isfinite(v)
        valid &= (u >= 0.0) & (u <= float(width - 1)) & (v >= 0.0) & (v <= float(height - 1))
        return valid

    def _pre_loading_data(self):
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor

        def _load_record_assets(key, record):
            image_tensor = load_img(record["file_path"], channel=3).float() / 255.0
            low_light_tensor = self._load_optional_auxiliary(record["frame_key"], "lowlight", channel=3)
            depth_tensor = self._load_optional_auxiliary(record["frame_key"], "depth", channel=1)
            structure_tensor = self._load_optional_auxiliary(
                record["frame_key"],
                "structure",
                channel=1,
                aliases=STRUCTURE_MODALITY_ALIASES,
            )
            return key, image_tensor[:3], low_light_tensor, depth_tensor, structure_tensor

        print(f"Load data [{self._requested_split}]: [{len(self._records)}].")
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 16)) as executor:
            all_records = list(executor.map(lambda item: _load_record_assets(item[0], item[1]), self._records.items()))
        for key, image, low_light, depth, structure in all_records:
            self._records[key]["img_tensor"] = image
            self._records[key]["low_light_tensor"] = low_light
            self._records[key]["depth_tensor"] = depth
            self._records[key]["structure_tensor"] = structure

    def _load_optional_auxiliary(self, frame_key, modality, channel, aliases=()):
        aux_path = resolve_auxiliary_path(
            self._scene_root,
            self._auxiliary_dir,
            modality,
            frame_key,
            aliases=aliases,
        )
        if aux_path is None:
            return None
        aux_path = str(aux_path)
        if aux_path.lower().endswith('.npy'):
            return load_npy_auxiliary(aux_path, channel=channel, size=(self._data_info['img_h'], self._data_info['img_w']))
        return load_img(aux_path, channel=channel).float() / 255.0



def resolve_auxiliary_path(scene_root, auxiliary_dir, modality, frame_key, aliases=()):
    modality_names = (modality,) + tuple(aliases)
    for modality_name in modality_names:
        modality_root = Path(scene_root) / auxiliary_dir / modality_name
        if not modality_root.exists():
            continue
        for extension in (".npy", ".png", ".jpg", ".jpeg", ".JPG", ".JPEG"):
            candidate = modality_root / f"{frame_key}{extension}"
            if candidate.exists():
                return str(candidate)
    return None



def load_npy_auxiliary(file_name, channel=1, size=None):
    array = np.load(file_name).astype(np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3 and array.shape[0] not in (1, 3, 4):
        array = np.transpose(array, (2, 0, 1))
    if array.ndim != 3:
        raise RuntimeError(f"Unsupported npy auxiliary shape {array.shape} for {file_name}")
    tensor = torch.from_numpy(array)
    if size is not None and tuple(tensor.shape[-2:]) != tuple(size):
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    if channel == 1 and tensor.shape[0] != 1:
        tensor = tensor[:1]
    elif channel == 3 and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor



def load_img(file_name, channel=3):
    if channel == 3:
        mode = torchvision.io.ImageReadMode.RGB
    elif channel == 4:
        mode = torchvision.io.ImageReadMode.RGB_ALPHA
    else:
        mode = torchvision.io.ImageReadMode.GRAY
    image = torchvision.io.read_image(file_name, mode=mode)
    assert image is not None, file_name
    return image


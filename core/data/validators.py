from pathlib import Path
import json

from .blender import build_frame_key, resolve_auxiliary_path


REQUIRED_INTRINSICS = ("h", "w", "fl_x", "fl_y", "cx", "cy")
REQUIRED_TRANSFORM_KEYS = ("transform_matrix", "file_path")
OPTIONAL_MODALITIES = ("lowlight", "depth", "structure")
MODALITY_ALIASES = {
    "structure": ("prior",),
}



def validate_blender_scene(scene_root, auxiliary_dir="auxiliaries"):
    scene_root = Path(scene_root)
    results = {
        "scene_root": str(scene_root),
        "errors": [],
        "warnings": [],
        "frames": {},
    }

    available_splits = []
    for split in ("train", "val", "test"):
        json_path = scene_root / f"transforms_{split}.json"
        if json_path.exists():
            available_splits.append(split)
            split_result = validate_split(scene_root, json_path, split, auxiliary_dir)
            results["frames"][split] = split_result
            results["errors"].extend(split_result["errors"])
            results["warnings"].extend(split_result["warnings"])

    if not available_splits:
        results["errors"].append(f"No transforms_*.json files found under {scene_root}")

    return results



def validate_split(scene_root, json_path, split, auxiliary_dir):
    with json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    errors = []
    warnings = []
    frame_keys = []
    image_root = scene_root / split
    if split == "val" and not image_root.exists():
        warnings.append(f"{split}: image directory missing, acceptable if this split is render-only")

    for key in REQUIRED_INTRINSICS:
        if key not in metadata:
            errors.append(f"{json_path.name}: missing intrinsic key '{key}'")

    for frame in metadata.get("frames", []):
        for key in REQUIRED_TRANSFORM_KEYS:
            if key not in frame:
                errors.append(f"{json_path.name}: frame missing key '{key}'")
                continue
        relative_path = frame.get("file_path")
        if relative_path is None:
            continue
        frame_key = build_frame_key(relative_path)
        frame_keys.append(frame_key)
        image_path = scene_root / relative_path.replace("/", "/")
        if split == "train" and not image_path.exists():
            errors.append(f"missing image for frame {frame_key}: {image_path}")
        for modality in OPTIONAL_MODALITIES:
            aliases = MODALITY_ALIASES.get(modality, ())
            aux_path = resolve_auxiliary_path(scene_root, auxiliary_dir, modality, frame_key, aliases=aliases)
            modality_roots = [scene_root / auxiliary_dir / modality]
            modality_roots.extend(scene_root / auxiliary_dir / alias for alias in aliases)
            if any(root.exists() for root in modality_roots) and aux_path is None:
                warnings.append(f"missing optional {modality} for frame {frame_key}")

    return {
        "json_path": str(json_path),
        "split": split,
        "num_frames": len(frame_keys),
        "frame_keys": frame_keys,
        "errors": errors,
        "warnings": warnings,
    }

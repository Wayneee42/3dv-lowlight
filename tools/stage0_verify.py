"""Validate stage-0 baseline configs against the official Blender dataset layout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import yaml


@dataclass(frozen=True)
class SceneSpec:
    name: str
    config_name: str
    data_path: str
    track: str
    required_files: tuple[str, ...]
    required_dirs: tuple[str, ...]


SCENES = (
    SceneSpec(
        name="Chocolate",
        config_name="chocolate.yaml",
        data_path="dataset/Development/lowlight_development/development/Chocolate",
        track="development",
        required_files=("transforms_train.json", "transforms_test.json"),
        required_dirs=("train",),
    ),
    SceneSpec(
        name="Cupcake",
        config_name="cupcake.yaml",
        data_path="dataset/Development/lowlight_development/development/Cupcake",
        track="development",
        required_files=("transforms_train.json", "transforms_test.json"),
        required_dirs=("train",),
    ),
    SceneSpec(
        name="GearWorks",
        config_name="gearworks.yaml",
        data_path="dataset/Development/lowlight_development/development/GearWorks",
        track="development",
        required_files=("transforms_train.json", "transforms_test.json"),
        required_dirs=("train",),
    ),
    SceneSpec(
        name="Laboratory",
        config_name="laboratory.yaml",
        data_path="dataset/Development/lowlight_development/development/Laboratory",
        track="development",
        required_files=("transforms_train.json", "transforms_test.json"),
        required_dirs=("train",),
    ),
    SceneSpec(
        name="BlueHawaii",
        config_name="bluehawaii.yaml",
        data_path="dataset/Validation/lowlight_validation/validation/BlueHawaii",
        track="validation-reference",
        required_files=("transforms_train.json", "transforms_val.json", "transforms_test.json"),
        required_dirs=("train", "val", "test"),
    ),
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_scene(repo_root: Path, scene: SceneSpec) -> list[str]:
    errors: list[str] = []
    config_path = repo_root / "config" / "stage0" / scene.config_name
    if not config_path.exists():
        return [f"missing config: {config_path}"]

    config = load_yaml(config_path)
    actual_name = config.get("DATASET", {}).get("NAME")
    actual_path = config.get("DATASET", {}).get("DATA_PATH")
    if actual_name != scene.name:
        errors.append(f"config NAME mismatch: expected {scene.name}, got {actual_name}")
    if actual_path != scene.data_path:
        errors.append(f"config DATA_PATH mismatch: expected {scene.data_path}, got {actual_path}")

    scene_root = repo_root / scene.data_path
    if not scene_root.exists():
        errors.append(f"missing scene directory: {scene_root}")
        return errors

    for filename in scene.required_files:
        if not (scene_root / filename).exists():
            errors.append(f"missing metadata file: {scene_root / filename}")
    for dirname in scene.required_dirs:
        split_dir = scene_root / dirname
        if not split_dir.exists():
            errors.append(f"missing image split dir: {split_dir}")
        elif not any(split_dir.iterdir()):
            errors.append(f"empty image split dir: {split_dir}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate stage-0 baseline configs and official data layout.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the 3DRR_codebase repository root.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    print(f"[stage0] repo root: {repo_root}")
    failed = False

    for scene in SCENES:
        errors = validate_scene(repo_root, scene)
        config_rel = Path("config") / "stage0" / scene.config_name
        print(f"\n[{scene.name}] ({scene.track})")
        print(f"  config: {config_rel}")
        print(f"  data:   {scene.data_path}")
        print(f"  train:  python train.py -c {config_rel.as_posix()}")
        if scene.track == "development":
            print("  note:   official development scenes provide train images plus transforms_test.json for render-only evaluation")
        if errors:
            failed = True
            for error in errors:
                print(f"  ERROR: {error}")
        else:
            print("  status: OK")

    if failed:
        print("\n[stage0] validation failed")
        return 1

    print("\n[stage0] validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

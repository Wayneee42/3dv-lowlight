#!/usr/bin/env python

import argparse
import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_AUGMENT_SPEC = spec_from_file_location("colmap_fixed_pose_augment", REPO_ROOT / "core" / "libs" / "augment.py")
if _AUGMENT_SPEC is None or _AUGMENT_SPEC.loader is None:
    raise RuntimeError("Failed to load core/libs/augment.py")
_AUGMENT_MODULE = module_from_spec(_AUGMENT_SPEC)
_AUGMENT_SPEC.loader.exec_module(_AUGMENT_MODULE)
prepare_low_light_batch = _AUGMENT_MODULE.prepare_low_light_batch


def dict_to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: dict_to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [dict_to_namespace(item) for item in value]
    return value


def load_config_yaml(config_path: Path):
    with config_path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Config must parse to a mapping: {config_path}")
    return data


def resolve_colmap_binary(colmap_bin: str) -> str:
    if os.path.isabs(colmap_bin) or any(sep in colmap_bin for sep in ('/', '\\')):
        if Path(colmap_bin).exists():
            return str(Path(colmap_bin))
        raise FileNotFoundError(f"COLMAP binary was not found: {colmap_bin}")
    resolved = shutil.which(colmap_bin)
    if resolved is None:
        raise FileNotFoundError(
            f"COLMAP binary '{colmap_bin}' was not found on PATH. Install COLMAP or pass --colmap-bin explicitly."
        )
    return resolved


def run_command(cmd, cwd=None):
    print('[COLMAP]', ' '.join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def get_colmap_version(colmap_exec: str):
    version_cmd = [colmap_exec, 'version']
    result = subprocess.run(version_cmd, check=False, capture_output=True, text=True)
    version_text = ((result.stdout or '') + '\n' + (result.stderr or '')).strip()
    match = re.search(r'(\d+)\.(\d+)(?:\.(\d+))?', version_text)
    if match is None:
        return None, version_text
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3)) if match.group(3) is not None else 0
    return (major, minor, patch), version_text


def ensure_supported_colmap_version(colmap_exec: str):
    version_tuple, version_text = get_colmap_version(colmap_exec)
    if version_tuple is None:
        print(f"[COLMAP] Warning: could not parse COLMAP version from output: {version_text!r}")
        return version_text

    major, minor, _ = version_tuple
    if major == 3 and minor == 10:
        raise RuntimeError(
            "Detected COLMAP 3.10.x. This release has a known point_triangulator crash in fixed-pose / known-pose "
            "workflows due to an observation manager bug, which matches the observed "
            "'Reconstruction::TranscribeImageIdsToDatabase -> std::out_of_range' failure. "
            "Upgrade COLMAP to 3.11.0+ (preferably 3.12.1) or downgrade to a pre-3.10 build such as 3.9/3.8."
        )
    return version_text


def resolve_option_name(colmap_exec: str, command_name: str, candidates):
    help_cmd = [colmap_exec, command_name, '-h']
    result = subprocess.run(help_cmd, check=False, capture_output=True, text=True)
    help_text = (result.stdout or '') + '\n' + (result.stderr or '')
    for candidate in candidates:
        if candidate in help_text:
            return candidate
    raise RuntimeError(
        f"Could not resolve option name for {command_name}. Tried {candidates}.\nCOLMAP help output was:\n{help_text}"
    )


def load_train_metadata(scene_root: Path):
    meta_path = scene_root / 'transforms_train.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing transforms_train.json: {meta_path}")
    with meta_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def image_to_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert('RGB')
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_tensor_image(chw: torch.Tensor, output_path: Path):
    array = chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    image = Image.fromarray(np.asarray(np.round(array * 255.0), dtype=np.uint8), mode='RGB')
    image.save(output_path)


def export_supervision_images(scene_root: Path, augmentation_cfg, workspace_dir: Path):
    metadata = load_train_metadata(scene_root)
    images_dir = workspace_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for frame in metadata.get('frames', []):
        relative_path = frame['file_path']
        image_path = scene_root / relative_path.replace('/', os.sep)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing train image: {image_path}")
        output_name = Path(relative_path).name
        output_path = images_dir / output_name
        image_tensor = image_to_tensor(image_path)
        batch = prepare_low_light_batch(image_tensor, augmentation_cfg, training=False, proxy_cfg=None)
        supervision = batch['supervision']
        save_tensor_image(supervision, output_path)
        exported.append(
            {
                'name': output_name,
                'relative_path': relative_path,
                'image_path': str(image_path),
                'output_path': str(output_path),
                'transform_matrix': frame['transform_matrix'],
            }
        )
    return metadata, exported


def rotation_matrix_to_qvec(rotation: np.ndarray) -> np.ndarray:
    m = rotation.astype(np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    else:
        diag = np.diag(m)
        if diag[0] > diag[1] and diag[0] > diag[2]:
            s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12))
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif diag[1] > diag[2]:
            s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12))
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12))
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    if qvec[0] < 0.0:
        qvec = -qvec
    qvec /= max(np.linalg.norm(qvec), 1e-12)
    return qvec


def c2w_gl_to_colmap_pose(transform_matrix) -> tuple[np.ndarray, np.ndarray]:
    c2w = np.asarray(transform_matrix, dtype=np.float64)
    if c2w.shape == (3, 4):
        full = np.eye(4, dtype=np.float64)
        full[:3, :] = c2w
        c2w = full
    if c2w.shape != (4, 4):
        raise RuntimeError(f"Expected 4x4 c2w transform, got {c2w.shape}")

    r_wc_gl = c2w[:3, :3]
    t_wc = c2w[:3, 3]
    r_cw_gl = r_wc_gl.T
    t_cw_gl = -r_cw_gl @ t_wc

    convention = np.diag([1.0, -1.0, -1.0])
    r_cw_cv = convention @ r_cw_gl
    t_cw_cv = convention @ t_cw_gl
    qvec = rotation_matrix_to_qvec(r_cw_cv)
    return qvec, t_cw_cv.astype(np.float64)


def read_database_state(database_path: Path):
    connection = sqlite3.connect(str(database_path))
    try:
        camera_rows = connection.execute(
            'SELECT camera_id, model, width, height FROM cameras ORDER BY camera_id'
        ).fetchall()
        image_rows = connection.execute(
            'SELECT image_id, name, camera_id FROM images ORDER BY image_id'
        ).fetchall()
        two_view_count = int(connection.execute('SELECT COUNT(*) FROM two_view_geometries').fetchone()[0])
    finally:
        connection.close()
    return camera_rows, image_rows, two_view_count


def parse_manual_images_txt(images_txt_path: Path):
    rows = []
    with images_txt_path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.rstrip('\r\n')
            if line.startswith('#'):
                continue
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) < 10:
                continue
            try:
                image_id = int(fields[0])
                camera_id = int(fields[8])
            except ValueError:
                continue
            name = fields[9]
            rows.append((image_id, name, camera_id))
    return rows


def validate_manual_model_against_database(manual_dir: Path, database_path: Path):
    images_txt_path = manual_dir / 'images.txt'
    if not images_txt_path.exists():
        raise FileNotFoundError(f"Missing manual images.txt: {images_txt_path}")

    _, image_rows, _ = read_database_state(database_path)
    manual_rows = parse_manual_images_txt(images_txt_path)
    if len(manual_rows) != len(image_rows):
        raise RuntimeError(
            f"Manual model image count mismatch: manual has {len(manual_rows)} rows, database has {len(image_rows)} rows"
        )

    db_by_name = {}
    for image_id, name, camera_id in image_rows:
        if name in db_by_name:
            raise RuntimeError(f"Duplicate image name in database: {name}")
        db_by_name[name] = (int(image_id), int(camera_id))

    manual_by_name = {}
    for image_id, name, camera_id in manual_rows:
        if name in manual_by_name:
            raise RuntimeError(f"Duplicate image name in manual images.txt: {name}")
        manual_by_name[name] = (int(image_id), int(camera_id))

    db_names = set(db_by_name.keys())
    manual_names = set(manual_by_name.keys())
    missing_in_db = sorted(manual_names - db_names)
    missing_in_manual = sorted(db_names - manual_names)
    if missing_in_db or missing_in_manual:
        raise RuntimeError(
            "Manual model image names do not match database image names. "
            f"Missing in database: {missing_in_db[:10]}; missing in manual: {missing_in_manual[:10]}"
        )

    mismatched_rows = []
    for name in sorted(db_names):
        db_image_id, db_camera_id = db_by_name[name]
        manual_image_id, manual_camera_id = manual_by_name[name]
        if db_image_id != manual_image_id or db_camera_id != manual_camera_id:
            mismatched_rows.append((name, db_image_id, manual_image_id, db_camera_id, manual_camera_id))
    if mismatched_rows:
        sample = mismatched_rows[:10]
        raise RuntimeError(
            "Manual model IDs do not match database IDs for some images. "
            f"Examples: {sample}"
        )


def write_manual_model(manual_dir: Path, metadata: dict, image_rows):
    manual_dir.mkdir(parents=True, exist_ok=True)
    width = int(round(float(metadata['w'])))
    height = int(round(float(metadata['h'])))
    fx = float(metadata['fl_x'])
    fy = float(metadata['fl_y'])
    cx = float(metadata['cx'])
    cy = float(metadata['cy'])

    camera_ids = {int(row[2]) for row in image_rows}
    if len(camera_ids) != 1:
        raise RuntimeError(f"Expected a single COLMAP camera id, got {sorted(camera_ids)}")
    camera_id = int(next(iter(camera_ids)))

    frame_map = {Path(frame['file_path']).name: frame for frame in metadata.get('frames', [])}
    cameras_txt = manual_dir / 'cameras.txt'
    rigs_txt = manual_dir / 'rigs.txt'
    frames_txt = manual_dir / 'frames.txt'
    images_txt = manual_dir / 'images.txt'
    points_txt = manual_dir / 'points3D.txt'

    cameras_lines = [
        '# Camera list with one line of data per camera:',
        '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]',
        '# Number of cameras: 1',
        f'{camera_id} PINHOLE {width} {height} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}',
        '',
    ]
    cameras_txt.write_text('\n'.join(cameras_lines), encoding='utf-8')

    rigs_lines = [
        '# Rig list with one line of data per rig:',
        '#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[]',
        '# Number of rigs: 1',
        f'{camera_id} 1 CAMERA {camera_id}',
        '',
    ]
    rigs_txt.write_text('\n'.join(rigs_lines), encoding='utf-8')

    image_lines = [
        '# Image list with two lines of data per image:',
        '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME',
        '#   POINTS2D[] as (X, Y, POINT3D_ID)',
        f'# Number of images: {len(image_rows)}',
    ]
    frame_lines = [
        '# Frame list with one line of data per frame:',
        '#   FRAME_ID, RIG_ID, QW, QX, QY, QZ, TX, TY, TZ, NUM_DATA_IDS, DATA_IDS[]',
        '#   DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)',
        f'# Number of frames: {len(image_rows)}',
    ]
    for image_id, name, db_camera_id in image_rows:
        if name not in frame_map:
            raise RuntimeError(f"Database image '{name}' is not present in transforms_train.json")
        qvec, tvec = c2w_gl_to_colmap_pose(frame_map[name]['transform_matrix'])
        frame_lines.append(
            f"{int(image_id)} {int(db_camera_id)} "
            f"{qvec[0]:.17g} {qvec[1]:.17g} {qvec[2]:.17g} {qvec[3]:.17g} "
            f"{tvec[0]:.17g} {tvec[1]:.17g} {tvec[2]:.17g} "
            f"1 CAMERA {int(db_camera_id)} {int(image_id)}"
        )
        image_lines.append(
            f"{int(image_id)} {qvec[0]:.17g} {qvec[1]:.17g} {qvec[2]:.17g} {qvec[3]:.17g} "
            f"{tvec[0]:.17g} {tvec[1]:.17g} {tvec[2]:.17g} {int(db_camera_id)} {name}"
        )
        image_lines.append('')
    frame_lines.append('')
    frames_txt.write_text('\n'.join(frame_lines), encoding='utf-8')
    images_txt.write_text('\n'.join(image_lines), encoding='utf-8')
    points_txt.write_text('', encoding='utf-8')
    return camera_id


def parse_points3d_bin(points_path: Path):
    points = []
    track_lengths = []
    reproj_errors = []
    with points_path.open('rb') as handle:
        num_points = struct.unpack('<Q', handle.read(8))[0]
        for _ in range(num_points):
            handle.read(8)  # point3D_id
            xyz = struct.unpack('<ddd', handle.read(24))
            handle.read(3)  # rgb
            error = struct.unpack('<d', handle.read(8))[0]
            track_len = struct.unpack('<Q', handle.read(8))[0]
            handle.read(8 * track_len)
            points.append(xyz)
            track_lengths.append(float(track_len))
            reproj_errors.append(float(error))
    if not points:
        empty_xyz = np.zeros((0, 3), dtype=np.float32)
        empty_meta = np.zeros((0,), dtype=np.float32)
        return {
            'xyz': empty_xyz,
            'track_len': empty_meta,
            'reproj_error': empty_meta,
        }
    return {
        'xyz': np.asarray(points, dtype=np.float32),
        'track_len': np.asarray(track_lengths, dtype=np.float32),
        'reproj_error': np.asarray(reproj_errors, dtype=np.float32),
    }


def parse_points3d_txt(points_path: Path):
    points = []
    with points_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) < 7:
                continue
            points.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not points:
        empty_xyz = np.zeros((0, 3), dtype=np.float32)
        empty_meta = np.zeros((0,), dtype=np.float32)
        return {
            'xyz': empty_xyz,
            'track_len': empty_meta,
            'reproj_error': empty_meta,
        }
    xyz = np.asarray(points, dtype=np.float32)
    count = int(xyz.shape[0])
    return {
        'xyz': xyz,
        'track_len': np.ones((count,), dtype=np.float32),
        'reproj_error': np.ones((count,), dtype=np.float32),
    }


def load_sparse_points(model_dir: Path):
    bin_path = model_dir / 'points3D.bin'
    txt_path = model_dir / 'points3D.txt'
    if bin_path.exists():
        return parse_points3d_bin(bin_path)
    if txt_path.exists():
        return parse_points3d_txt(txt_path)
    raise FileNotFoundError(f"Neither points3D.bin nor points3D.txt exists in {model_dir}")


def build_sparse_model(scene_root: Path, config_path: Path, colmap_bin: str, workspace_dir: Path, overwrite: bool, use_gpu: bool, gpu_index: int):
    if workspace_dir.exists() and overwrite:
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    images_dir = workspace_dir / 'images'
    manual_dir = workspace_dir / 'manual'
    sparse_raw_dir = workspace_dir / 'sparse_raw'
    sparse_dir = workspace_dir / 'sparse'
    database_path = workspace_dir / 'database.db'
    report_path = workspace_dir / 'report.json'
    points_path = workspace_dir / 'points.npy'
    points_meta_path = workspace_dir / 'points_meta.npz'

    colmap_exec = resolve_colmap_binary(colmap_bin)
    colmap_version = ensure_supported_colmap_version(colmap_exec)
    config_data = load_config_yaml(config_path)
    augmentation_cfg = dict_to_namespace(config_data.get("AUGMENTATION", {}))
    metadata, exported = export_supervision_images(scene_root, augmentation_cfg, workspace_dir)

    camera_params = ','.join(
        [
            f"{float(metadata['fl_x']):.10f}",
            f"{float(metadata['fl_y']):.10f}",
            f"{float(metadata['cx']):.10f}",
            f"{float(metadata['cy']):.10f}",
        ]
    )

    feature_use_gpu_opt = resolve_option_name(
        colmap_exec,
        'feature_extractor',
        ['--FeatureExtraction.use_gpu', '--SiftExtraction.use_gpu'],
    )
    feature_gpu_index_opt = resolve_option_name(
        colmap_exec,
        'feature_extractor',
        ['--FeatureExtraction.gpu_index', '--SiftExtraction.gpu_index'],
    )
    feature_cmd = [
        colmap_exec,
        'feature_extractor',
        '--database_path', str(database_path),
        '--image_path', str(images_dir),
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', 'PINHOLE',
        '--ImageReader.camera_params', camera_params,
        feature_use_gpu_opt, '1' if use_gpu else '0',
        feature_gpu_index_opt, str(gpu_index),
    ]
    run_command(feature_cmd, cwd=workspace_dir)

    camera_rows, image_rows, _ = read_database_state(database_path)
    if len(camera_rows) != 1:
        raise RuntimeError(f"Expected a single camera row after feature extraction, got {len(camera_rows)}")
    if len(image_rows) != len(exported):
        raise RuntimeError(
            f"Database image count mismatch after feature extraction: {len(image_rows)} vs exported {len(exported)}"
        )
    write_manual_model(manual_dir, metadata, image_rows)
    validate_manual_model_against_database(manual_dir, database_path)

    matching_use_gpu_opt = resolve_option_name(
        colmap_exec,
        'exhaustive_matcher',
        ['--FeatureMatching.use_gpu', '--SiftMatching.use_gpu'],
    )
    matching_gpu_index_opt = resolve_option_name(
        colmap_exec,
        'exhaustive_matcher',
        ['--FeatureMatching.gpu_index', '--SiftMatching.gpu_index'],
    )
    matcher_cmd = [
        colmap_exec,
        'exhaustive_matcher',
        '--database_path', str(database_path),
        matching_use_gpu_opt, '1' if use_gpu else '0',
        matching_gpu_index_opt, str(gpu_index),
    ]
    run_command(matcher_cmd, cwd=workspace_dir)

    _, _, two_view_count = read_database_state(database_path)

    sparse_raw_dir.mkdir(parents=True, exist_ok=True)
    triangulator_cmd = [
        colmap_exec,
        'point_triangulator',
        '--database_path', str(database_path),
        '--image_path', str(images_dir),
        '--input_path', str(manual_dir),
        '--output_path', str(sparse_raw_dir),
        '--clear_points', '0',
    ]
    run_command(triangulator_cmd, cwd=workspace_dir)

    sparse_dir.mkdir(parents=True, exist_ok=True)
    filtering_cmd = [
        colmap_exec,
        'point_filtering',
        '--input_path', str(sparse_raw_dir),
        '--output_path', str(sparse_dir),
    ]
    run_command(filtering_cmd, cwd=workspace_dir)

    raw_sparse = load_sparse_points(sparse_raw_dir)
    filtered_sparse = load_sparse_points(sparse_dir)
    raw_points = raw_sparse['xyz']
    filtered_points = filtered_sparse['xyz']

    model_cfg = dict_to_namespace(config_data.get('MODEL', {}))
    min_points = int(getattr(model_cfg, 'INIT_COLMAP_MIN_POINTS', 1000))
    if filtered_points.shape[0] < min_points:
        raise RuntimeError(
            f"Filtered COLMAP sparse points ({filtered_points.shape[0]}) below INIT_COLMAP_MIN_POINTS ({min_points})"
        )

    np.save(points_path, filtered_points.astype(np.float32))
    np.savez(
        points_meta_path,
        xyz=filtered_sparse['xyz'].astype(np.float32),
        track_len=filtered_sparse['track_len'].astype(np.float32),
        reproj_error=filtered_sparse['reproj_error'].astype(np.float32),
    )

    filtered_track_len = filtered_sparse['track_len']
    filtered_reproj_error = filtered_sparse['reproj_error']
    report = {
        'scene_root': str(scene_root),
        'config_path': str(config_path),
        'colmap_bin': str(colmap_exec),
        'colmap_version': colmap_version,
        'use_gpu': bool(use_gpu),
        'gpu_index': int(gpu_index),
        'image_count': len(exported),
        'two_view_geometry_count': int(two_view_count),
        'triangulated_points_raw': int(raw_points.shape[0]),
        'triangulated_points_filtered': int(filtered_points.shape[0]),
        'num_sparse_points': int(filtered_points.shape[0]),
        'track_len_mean': float(filtered_track_len.mean()) if filtered_track_len.size > 0 else 0.0,
        'track_len_median': float(np.median(filtered_track_len)) if filtered_track_len.size > 0 else 0.0,
        'reproj_error_mean': float(filtered_reproj_error.mean()) if filtered_reproj_error.size > 0 else 0.0,
        'reproj_error_median': float(np.median(filtered_reproj_error)) if filtered_reproj_error.size > 0 else 0.0,
        'points_path': str(points_path),
        'points_meta_path': str(points_meta_path),
        'workspace_dir': str(workspace_dir),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Build fixed-pose COLMAP sparse initialization assets for a scene.')
    parser.add_argument('scene_root', type=str, help='Path to scene root containing transforms_train.json')
    parser.add_argument('--config-path', type=str, required=True, help='Training config used to generate deterministic supervision images')
    parser.add_argument('--colmap-bin', type=str, default='colmap', help='Path or name of the COLMAP executable')
    parser.add_argument(
        '--workspace-dir',
        type=str,
        default=None,
        help='Output workspace directory. Defaults to <scene_root>/auxiliaries/colmap_sparse',
    )
    parser.add_argument('--overwrite', action='store_true', help='Delete the existing workspace before rebuilding it')
    parser.add_argument('--use-gpu', type=int, choices=[0, 1], default=1, help='Whether to enable GPU SIFT extraction/matching for COLMAP')
    parser.add_argument('--gpu-index', type=int, default=0, help='CUDA device index passed to COLMAP SIFT extraction/matching')
    args = parser.parse_args()

    scene_root = Path(args.scene_root).resolve()
    config_path = Path(args.config_path).resolve()
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else scene_root / 'auxiliaries' / 'colmap_sparse'
    build_sparse_model(
        scene_root,
        config_path,
        args.colmap_bin,
        workspace_dir,
        overwrite=bool(args.overwrite),
        use_gpu=bool(args.use_gpu),
        gpu_index=int(args.gpu_index),
    )


if __name__ == '__main__':
    main()






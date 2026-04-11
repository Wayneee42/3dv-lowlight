#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[Error] Neither 'python' nor 'python3' was found in PATH." >&2
    exit 1
  fi
fi

DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-test}"
STRIDE="${STRIDE:-4}"
MIN_ALPHA="${MIN_ALPHA:-0.2}"
REL_THRESH="${REL_THRESH:-0.05}"
ABS_THRESH="${ABS_THRESH:-0.02}"
EPS="${EPS:-1e-4}"
STEP="${STEP:-5000}"
MAX_VIEWS="${MAX_VIEWS:-}"
OUT_DIR="${OUT_DIR:-outputs/geometry_consistency_stage5_sparse_ablation/${SPLIT}}"

if [[ -n "${SCENES:-}" ]]; then
  read -r -a SCENE_LIST <<< "${SCENES}"
else
  SCENE_LIST=(
    Chocolate
    Cupcake
    GearWorks
    Laboratory
    MilkCookie
    Popcorn
    Sculpture
    Ujikintoki
  )
fi

if [[ -n "${VARIANTS:-}" ]]; then
  read -r -a VARIANT_LIST <<< "${VARIANTS}"
else
  VARIANT_LIST=(
    stage5b_ft_off
    stage5b_ft_on
  )
fi

mkdir -p "${OUT_DIR}"

echo "[Info] REPO_ROOT=${REPO_ROOT}"
echo "[Info] PYTHON_BIN=${PYTHON_BIN}"
echo "[Info] DEVICE=${DEVICE} SPLIT=${SPLIT} STEP=${STEP} STRIDE=${STRIDE}"
echo "[Info] OUT_DIR=${OUT_DIR}"

for variant in "${VARIANT_LIST[@]}"; do
  for scene in "${SCENE_LIST[@]}"; do
    ckpt="outputs/${variant}/${scene}/step_${STEP}/step_${STEP}.pt"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[Skip] Missing checkpoint: ${ckpt}" >&2
      continue
    fi

    summary_json="${OUT_DIR}/${variant}_${scene}.json"
    per_view_json="${OUT_DIR}/${variant}_${scene}_per_view.json"

    echo "[Run] ${variant} | ${scene}"
    cmd=(
      "${PYTHON_BIN}" tools/eval_geometry_consistency.py
      --checkpoint "${ckpt}"
      --device "${DEVICE}"
      --split "${SPLIT}"
      --sample-stride "${STRIDE}"
      --min-alpha "${MIN_ALPHA}"
      --relative-depth-thresh "${REL_THRESH}"
      --absolute-depth-thresh "${ABS_THRESH}"
      --eps "${EPS}"
      --tag "${variant}"
      --summary-json "${summary_json}"
      --per-view-json "${per_view_json}"
    )
    if [[ -n "${MAX_VIEWS}" ]]; then
      cmd+=(--max-views "${MAX_VIEWS}")
    fi
    "${cmd[@]}"
  done
done

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
json_paths = sorted(
    path for path in out_dir.glob("*.json")
    if not path.name.endswith("_per_view.json")
)

rows = []
for path in json_paths:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows.append(
        {
            "variant": data.get("tag", ""),
            "scene": data.get("scene", ""),
            "checkpoint": data.get("checkpoint", ""),
            "split": data.get("split", ""),
            "sample_stride": data.get("sample_stride", ""),
            "reproj_depth_error": data.get("reproj_depth_error", ""),
            "consistency_ratio": data.get("consistency_ratio", ""),
            "overlap_ratio": data.get("overlap_ratio", ""),
            "overlap_ratio_total": data.get("overlap_ratio_total", ""),
            "source_valid_ratio": data.get("source_valid_ratio", ""),
            "valid_pairs": data.get("valid_pairs", ""),
            "num_pairs": data.get("num_pairs", ""),
            "overlap_count": data.get("overlap_count", ""),
            "consistent_count": data.get("consistent_count", ""),
        }
    )

tsv_path = out_dir / "geometry_consistency_summary.tsv"
with open(tsv_path, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "variant",
            "scene",
            "checkpoint",
            "split",
            "sample_stride",
            "reproj_depth_error",
            "consistency_ratio",
            "overlap_ratio",
            "overlap_ratio_total",
            "source_valid_ratio",
            "valid_pairs",
            "num_pairs",
            "overlap_count",
            "consistent_count",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"[Done] Wrote {len(rows)} rows to {tsv_path}")
PY

# 3DRR Low-Light

This repository contains the cleaned code for the final low-light 3DGS pipeline:

1. `stage4_tuned_colmap`
2. `stage5b_ft`
3. `stage6_adaptive_chroma_ycbcr_additive`

The retained method focuses on:

- fixed-pose COLMAP sparse initialization
- staged geometry-to-appearance training
- weak sparse-guided geometry regularization in stage 5
- scalar illumination with additive YCbCr chroma residual in stage 6

## Repository Layout

```text
3DRR_low_light/
|-- config/
|   |-- stage4_tuned_colmap/
|   |-- stage5b_ft/
|   `-- stage6_adaptive_chroma_ycbcr_additive/
|-- core/
|-- tools/
|-- dataset/
|-- marigold/
|-- colmap/
|-- train.py
|-- eval.py
|-- environment.yml
`-- requirements.txt
```

## Environment

Recommended:

```bash
conda env create -f environment.yml
conda activate 3drr-lowlight
```

If you already have a working PyTorch/CUDA environment, you can also install:

```bash
pip install -r requirements.txt
```

Notes:

- `gsplat` must match your local CUDA / PyTorch setup.
- COLMAP is required for sparse initialization, but it is installed separately from `pip`.
- `lpips` is optional and only used by `eval.py` when available.

## Data

The code expects Blender-style scene folders with:

- `transforms_train.json`
- `transforms_val.json`
- `transforms_test.json`
- RGB images referenced by those JSON files

Additional scene priors are stored under:

- `auxiliaries/depth/`
- `auxiliaries/structure/`
- `auxiliaries/colmap_sparse/`

## Preprocessing

### 1. Marigold depth

```bash
python tools/extract_marigold_depth.py dataset/.../YourScene
```

### 2. Structure prior

```bash
python tools/extract_structure_prior.py dataset/.../YourScene
```

### 3. Fixed-pose COLMAP sparse points

```bash
python tools/build_fixed_pose_colmap_sparse_init.py \
  dataset/.../YourScene \
  --config-path config/stage4_tuned_colmap/yourscene.yaml \
  --colmap-bin colmap \
  --overwrite
```

This stage:

- exports deterministic supervision images
- writes official poses into a manual COLMAP model
- runs feature extraction, matching, and triangulation
- saves sparse points to `auxiliaries/colmap_sparse/points.npy`

## Training

### Stage 1: geometry bootstrap

```bash
python train.py -c config/stage4_tuned_colmap/yourscene.yaml
```

### Stage 2: structure refinement

```bash
python train.py -c config/stage5b_ft/yourscene.yaml
```

This stage warm-starts from `stage4_tuned_colmap` and uses:

- no densification
- structure prior from step 0
- weak depth prior
- weak sparse-guided geometry regularization

### Stage 3: appearance refinement

```bash
python train.py -c config/stage6_adaptive_chroma_ycbcr_additive/yourscene.yaml
```

This stage warm-starts from `stage5b_ft` and uses:

- adaptive proxy brightness calibration
- confidence-aware reconstruction weighting
- scalar illumination head
- additive YCbCr chroma residual

## Evaluation

Render a saved checkpoint:

```bash
python eval.py -w outputs/stage6_adaptive_chroma_ycbcr_additive/YourScene/step_5000/step_5000.pt
```

Outputs are written to:

- `outputs/<stage>/<scene>/step_<N>/test/`

When ground-truth test images exist, `eval.py` also writes:

- `results.json`
- `per_view.json`

## Current Best Pipeline

The recommended full pipeline is:

```text
stage4_tuned_colmap
  -> stage5b_ft
  -> stage6_adaptive_chroma_ycbcr_additive
```

This is the strongest retained version in this cleaned repository. Historical branches such as canonical calibration, color-checker refinement, occupancy initialization, and merged stage-4/5 training were intentionally removed from the active code path.

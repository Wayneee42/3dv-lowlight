# Repository Guidelines

## Project Structure & Module Organization
This repository implements a 3-stage low-light 3DGS pipeline:
- `config/stage4_tuned_colmap/`, `config/stage5b_ft/`, `config/stage6_adaptive_chroma_ycbcr_additive/`: stage-specific YAML configs (one file per scene, e.g. `chocolate.yaml`).
- `core/`: training/evaluation internals (`data`, `model`, `losses`, `libs`).
- `tools/`: preprocessing scripts for depth, structure, and fixed-pose COLMAP sparse priors.
- `train.py`, `eval.py`: main entry points.
- `dataset/`, `lowlight/`: scene data and test-image references.
- `outputs/`: checkpoints and rendered evaluation results.

## Build, Test, and Development Commands
- `conda env create -f environment.yml && conda activate 3drr-lowlight`: create recommended environment.
- `pip install -r requirements.txt`: lightweight install if your CUDA/PyTorch stack already works.
- `python tools/extract_marigold_depth.py <scene_dir>`: generate depth priors.
- `python tools/extract_structure_prior.py <scene_dir>`: generate structure priors.
- `python tools/build_fixed_pose_colmap_sparse_init.py <scene_dir> --config-path config/stage4_tuned_colmap/<scene>.yaml --colmap-bin colmap --overwrite`: export fixed-pose sparse priors.
- `python train.py -c config/<stage>/<scene>.yaml`: run stage training.
- `python eval.py -w outputs/<stage>/<Scene>/step_<N>/step_<N>.pt`: render test outputs and metrics.

## Coding Style & Naming Conventions
- Python 3.10, 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Keep configs scene-specific and lowercase (e.g. `gearworks.yaml`).
- Prefer explicit config-driven behavior over hard-coded scene paths.
- Use concise docstrings/comments only where logic is non-obvious.

## Testing Guidelines
- There is no dedicated unit-test suite yet; validation is pipeline-based.
- For each code change, run at least one representative stage command and one `eval.py` checkpoint render.
- Verify outputs in `outputs/<stage>/<Scene>/step_<N>/test/`; when GT is available, inspect `results.json` and `per_view.json`.

## Commit & Pull Request Guidelines
- Recent history includes inconsistent short subjects (`-`, `~`); do not follow that pattern.
- Use imperative, scoped commit messages (e.g. `stage5: tighten sparse support weighting`).
- PRs should include: purpose, affected stage(s), config(s) changed, exact commands run, and key metric/visual deltas.
- Link related issues and attach representative output paths or screenshots for rendering changes.

## Security & Configuration Tips
- Do not commit dataset contents, checkpoints, or large rendered artifacts.
- Keep machine-local paths (COLMAP binaries, checkpoints) in local command invocations, not committed configs unless portable.

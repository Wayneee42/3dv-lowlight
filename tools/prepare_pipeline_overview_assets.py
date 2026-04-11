#!/usr/bin/env python

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from matplotlib.mathtext import math_to_image
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "report" / "Pipeline_Overview"
OUT_DIR = ASSET_ROOT / "generated"


def crop_first_tile(src: Path, dst: Path) -> None:
    image = Image.open(src).convert("RGB")
    width, height = image.size
    tile = image.crop((0, 0, width // 2, height // 2))
    tile.save(dst)


def crop_transparent_margin(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    return image.crop(bbox) if bbox else image


def render_formula(tex: str, dst: Path, *, dpi: int = 260, pad: int = 8) -> None:
    buffer = BytesIO()
    math_to_image(
        rf"${tex}$",
        buffer,
        dpi=dpi,
        format="png",
        color="black",
    )
    buffer.seek(0)
    image = Image.open(buffer).convert("RGBA")
    image = crop_transparent_margin(image)
    if pad > 0:
        padded = Image.new("RGBA", (image.width + 2 * pad, image.height + 2 * pad), (255, 255, 255, 0))
        padded.alpha_composite(image, (pad, pad))
        image = padded
    image.save(dst)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    crop_first_tile(ASSET_ROOT / "stage6" / "train_aug.jpg", OUT_DIR / "train_aug_tile.png")
    crop_first_tile(ASSET_ROOT / "stage6" / "proxy_target.jpg", OUT_DIR / "proxy_target_tile.png")
    crop_first_tile(ASSET_ROOT / "stage6" / "cb_factor.JPG", OUT_DIR / "cb_factor_tile.png")
    crop_first_tile(ASSET_ROOT / "stage6" / "cr_factor.JPG", OUT_DIR / "cr_factor_tile.png")

    formulas = {
        "stage1_loss.png": r"\mathcal{L}^{(1)}=\mathcal{L}_{rgb}+\lambda_d\mathcal{L}_{depth}+\lambda_{mv}\mathcal{L}_{mv}+\lambda_{exp}\mathcal{L}_{exp}",
        "stage2_barycenter.png": r"\bar{p}_i=\sum_{j=1}^{K} w_{ij}p_{ij}",
        "stage2_sparse_loss.png": r"\mathcal{L}_{sparse}=\frac{1}{M}\sum_i\rho(\|\mu_i-\bar{p}_i\|_2)",
        "stage3_y_gain.png": r"Y'=\operatorname{clip}(2\sigma(A_Y)\odot Y,0,1)",
        "stage3_chroma.png": r"\Delta_C=\alpha_c\tanh(A_C)",
        "stage3_weight_y.png": r"\mathcal{L}_{Y}=\operatorname{WeightedL1}(Y_{rec},Y_{proxy};W_Y)",
        "stage3_weight_cbcr.png": r"\mathcal{L}_{CbCr}=\operatorname{WeightedL1}(\tilde{C}_{rec},\tilde{C}_{tgt};W_C)",
        "stage3_objective.png": r"\mathcal{L}^{(3)}=\mathcal{L}_{rgb}+\lambda_Y\mathcal{L}_Y+\lambda_c\mathcal{L}_{CbCr}+\lambda_{cr}\|\Delta_C\|_1",
    }
    for filename, tex in formulas.items():
        render_formula(tex, OUT_DIR / filename)

    print(f"[PipelineAssets] prepared assets in {OUT_DIR}")


if __name__ == "__main__":
    main()

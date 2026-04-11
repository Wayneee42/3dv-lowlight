#!/usr/bin/env python

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from matplotlib.mathtext import math_to_image
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "report" / "Pipeline_Overview"
DEFAULT_OUT = ASSET_ROOT / "Pipeline_Overview.png"


PALETTE = {
    "canvas": (250, 248, 245, 255),
    "ink": (47, 47, 45, 255),
    "muted": (118, 115, 109, 255),
    "data": (35, 35, 35, 255),
    "grad": (183, 63, 67, 255),
    "stage1_fill": (220, 229, 231, 255),
    "stage1_line": (126, 148, 154, 255),
    "stage2_fill": (231, 216, 208, 255),
    "stage2_line": (161, 132, 122, 255),
    "stage3_fill": (214, 222, 230, 255),
    "stage3_line": (117, 136, 154, 255),
    "prior_fill": (216, 222, 207, 255),
    "prior_line": (123, 140, 114, 255),
    "input_fill": (234, 229, 223, 255),
    "input_line": (140, 132, 125, 255),
    "inner_fill": (254, 253, 251, 255),
    "card_line": (174, 170, 165, 255),
    "legend_fill": (244, 241, 237, 255),
}


CANVAS_W = 3000
CANVAS_H = 1720


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def xyxy(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y + self.h


def load_font(size: int, *, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold and italic:
        candidates += ["timesbi.ttf", "arialbi.ttf", "cambriaz.ttf"]
    elif bold:
        candidates += ["timesbd.ttf", "arialbd.ttf", "cambrib.ttf"]
    elif italic:
        candidates += ["timesi.ttf", "ariali.ttf", "cambriai.ttf"]
    else:
        candidates += ["times.ttf", "arial.ttf", "cambria.ttc"]

    candidates += ["DejaVuSerif.ttf", "DejaVuSans.ttf"]
    font_dirs = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts/truetype/dejavu"),
    ]
    for name in candidates:
        for font_dir in font_dirs:
            font_path = font_dir / name
            if font_path.exists():
                return ImageFont.truetype(str(font_path), size=size)
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = load_font(40, bold=True)
FONT_SUBTITLE = load_font(22, italic=True)
FONT_STAGE = load_font(32, bold=True)
FONT_LABEL = load_font(24, bold=True)
FONT_BODY = load_font(21)
FONT_SMALL = load_font(18)
FONT_TINY = load_font(16)


def crop_transparent_margin(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    return image.crop(bbox) if bbox else image


def render_formula(tex: str, *, color: str = "black", dpi: int = 260, pad: int = 8) -> Image.Image:
    buffer = BytesIO()
    math_to_image(
        rf"${tex}$",
        buffer,
        dpi=dpi,
        format="png",
        color=color,
    )
    buffer.seek(0)
    image = Image.open(buffer).convert("RGBA")
    image = crop_transparent_margin(image)
    return ImageOps.expand(image, border=pad, fill=(255, 255, 255, 0))


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return mask


def add_shadow(base: Image.Image, box: Box, radius: int = 26, opacity: int = 70, blur: int = 18, offset: tuple[int, int] = (0, 6)):
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    x1, y1, x2, y2 = box.xyxy
    ox, oy = offset
    draw.rounded_rectangle((x1 + ox, y1 + oy, x2 + ox, y2 + oy), radius=radius, fill=(0, 0, 0, opacity))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    base.alpha_composite(shadow)


def draw_dashed_path(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color, width: int, dash: float = 16.0, gap: float = 10.0):
    if len(points) < 2:
        return
    pattern = dash + gap
    for idx in range(len(points) - 1):
        x1, y1 = points[idx]
        x2, y2 = points[idx + 1]
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist <= 1.0e-6:
            continue
        ux = dx / dist
        uy = dy / dist
        pos = 0.0
        while pos < dist:
            seg_end = min(pos + dash, dist)
            start = (x1 + ux * pos, y1 + uy * pos)
            end = (x1 + ux * seg_end, y1 + uy * seg_end)
            draw.line((start, end), fill=color, width=width)
            pos += pattern


def build_rounded_rect_path(box: Box, radius: int, steps: int = 14) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = box.xyxy
    r = min(radius, box.w // 2, box.h // 2)
    pts: list[tuple[float, float]] = []

    def arc(cx, cy, start_deg, end_deg):
        for i in range(steps + 1):
            t = start_deg + (end_deg - start_deg) * i / steps
            rad = math.radians(t)
            pts.append((cx + r * math.cos(rad), cy + r * math.sin(rad)))

    pts.append((x1 + r, y1))
    pts.append((x2 - r, y1))
    arc(x2 - r, y1 + r, -90, 0)
    pts.append((x2, y2 - r))
    arc(x2 - r, y2 - r, 0, 90)
    pts.append((x1 + r, y2))
    arc(x1 + r, y2 - r, 90, 180)
    pts.append((x1, y1 + r))
    arc(x1 + r, y1 + r, 180, 270)
    pts.append((x1 + r, y1))
    return pts


def draw_dashed_rounded_rect(base: Image.Image, box: Box, *, fill, outline, radius: int = 30, width: int = 4, dash: float = 20.0, gap: float = 12.0):
    add_shadow(base, box, radius=radius)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(box.xyxy, radius=radius, fill=fill)
    path = build_rounded_rect_path(box, radius)
    draw_dashed_path(draw, path, outline, width=width, dash=dash, gap=gap)
    base.alpha_composite(overlay)


def draw_solid_rounded_rect(base: Image.Image, box: Box, *, fill, outline, radius: int = 24, width: int = 2):
    add_shadow(base, box, radius=radius, opacity=44, blur=10, offset=(0, 4))
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(box.xyxy, radius=radius, fill=fill, outline=outline, width=width)
    base.alpha_composite(overlay)


def fit_image(path: Path, size: tuple[int, int], *, crop=None, radius: int = 18, add_border: bool = True) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if crop is not None:
        image = image.crop(crop)
    image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    rgba = image.convert("RGBA")
    mask = rounded_mask(size, radius)
    rgba.putalpha(mask)
    if not add_border:
        return rgba
    card = Image.new("RGBA", size, (0, 0, 0, 0))
    card.alpha_composite(rgba)
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, outline=PALETTE["card_line"], width=2)
    return card


def crop_first_tile(path: Path) -> tuple[int, int, int, int]:
    image = Image.open(path)
    w, h = image.size
    return (0, 0, w // 2, h // 2)


def paste(base: Image.Image, image: Image.Image, xy: tuple[int, int]):
    base.alpha_composite(image, xy)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill, *, anchor: str = "la"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_wrapped_text(draw: ImageDraw.ImageDraw, box: Box, text: str, font, fill, *, line_spacing: int = 6, align: str = "left"):
    avg_char = max(font.size * 0.58, 7)
    max_chars = max(10, int(box.w / avg_char))
    lines = textwrap.wrap(text, width=max_chars)
    bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=line_spacing, align=align)
    text_h = bbox[3] - bbox[1]
    y = box.y + (box.h - text_h) // 2
    draw.multiline_text((box.x, y), "\n".join(lines), font=font, fill=fill, spacing=line_spacing, align=align)


def draw_wrapped_text_top(draw: ImageDraw.ImageDraw, box: Box, text: str, font, fill, *, line_spacing: int = 6, align: str = "left"):
    avg_char = max(font.size * 0.58, 7)
    max_chars = max(10, int(box.w / avg_char))
    lines = textwrap.wrap(text, width=max_chars)
    draw.multiline_text((box.x, box.y), "\n".join(lines), font=font, fill=fill, spacing=line_spacing, align=align)


def draw_pill(base: Image.Image, box: Box, text: str, *, fill, outline=None, font=FONT_TINY, text_fill=None):
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(box.xyxy, radius=box.h // 2, fill=fill, outline=outline, width=2 if outline else 1)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = box.x + (box.w - tw) // 2
    ty = box.y + (box.h - th) // 2 - 1
    draw.text((tx, ty), text, font=font, fill=text_fill or PALETTE["ink"])
    base.alpha_composite(overlay)


def draw_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color, width: int = 5, arrow_size: int = 16):
    draw.line(points, fill=color, width=width, joint="curve")
    if len(points) < 2:
        return
    (x1, y1), (x2, y2) = points[-2], points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    left = (
        x2 - arrow_size * math.cos(angle) + arrow_size * 0.55 * math.sin(angle),
        y2 - arrow_size * math.sin(angle) - arrow_size * 0.55 * math.cos(angle),
    )
    right = (
        x2 - arrow_size * math.cos(angle) - arrow_size * 0.55 * math.sin(angle),
        y2 - arrow_size * math.sin(angle) + arrow_size * 0.55 * math.cos(angle),
    )
    draw.polygon([(x2, y2), left, right], fill=color)


def paste_stack(base: Image.Image, image_paths: list[Path], anchor: tuple[int, int], size: tuple[int, int]):
    x, y = anchor
    offsets = [(22, 22), (11, 11), (0, 0)]
    for path, (ox, oy) in zip(image_paths, offsets):
        card = fit_image(path, size, radius=16)
        shadow_box = Box(x + ox, y + oy, size[0], size[1])
        add_shadow(base, shadow_box, radius=18, blur=10, opacity=48, offset=(0, 4))
        paste(base, card, (x + ox, y + oy))


def build_overview(out_path: Path):
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), PALETTE["canvas"])
    draw = ImageDraw.Draw(canvas)

    input_box = Box(40, 170, 300, 840)
    stage1_box = Box(380, 140, 520, 520)
    stage2_box = Box(950, 140, 620, 520)
    stage3_box = Box(1630, 80, 1320, 1040)
    prior_box = Box(380, 1010, 1190, 430)
    legend_box = Box(640, 1570, 1680, 92)

    draw_text(draw, (CANVAS_W // 2, 44), "Reliability-Aware Staged Low-Light 3DGS", FONT_TITLE, PALETTE["ink"], anchor="ma")
    draw_text(
        draw,
        (CANVAS_W // 2, 86),
        "fixed-pose sparse bootstrap  →  sparse-guided geometry refinement  →  appearance-only decoupling",
        FONT_SUBTITLE,
        PALETTE["muted"],
        anchor="ma",
    )

    draw_dashed_rounded_rect(canvas, input_box, fill=PALETTE["input_fill"], outline=PALETTE["input_line"], radius=32)
    draw_dashed_rounded_rect(canvas, stage1_box, fill=PALETTE["stage1_fill"], outline=PALETTE["stage1_line"], radius=34)
    draw_dashed_rounded_rect(canvas, stage2_box, fill=PALETTE["stage2_fill"], outline=PALETTE["stage2_line"], radius=34)
    draw_dashed_rounded_rect(canvas, stage3_box, fill=PALETTE["stage3_fill"], outline=PALETTE["stage3_line"], radius=36)
    draw_dashed_rounded_rect(canvas, prior_box, fill=PALETTE["prior_fill"], outline=PALETTE["prior_line"], radius=34)
    draw_dashed_rounded_rect(canvas, legend_box, fill=PALETTE["legend_fill"], outline=PALETTE["card_line"], radius=24, width=3, dash=14, gap=10)

    # Titles
    draw_text(draw, (input_box.x + 22, input_box.y + 24), "Input Views & Supervision", FONT_STAGE, PALETTE["ink"])
    draw_text(draw, (stage1_box.x + 22, stage1_box.y + 24), "Stage 1  Geometry Bootstrap", FONT_STAGE, PALETTE["ink"])
    draw_text(draw, (stage2_box.x + 22, stage2_box.y + 24), "Stage 2  Sparse-guided Geometry Refinement", FONT_STAGE, PALETTE["ink"])
    draw_text(draw, (stage3_box.x + 22, stage3_box.y + 24), "Stage 3  Appearance-only Decoupling", FONT_STAGE, PALETTE["ink"])
    draw_text(draw, (prior_box.x + 22, prior_box.y + 24), "Prior Preparation", FONT_STAGE, PALETTE["ink"])

    draw_text(draw, (stage1_box.x + 26, stage1_box.y + 64), "stable sparse initialization + early densification", FONT_SMALL, PALETTE["muted"])
    draw_text(draw, (stage2_box.x + 26, stage2_box.y + 64), "no densify, soft geometric anchoring with sparse support", FONT_SMALL, PALETTE["muted"])
    draw_text(draw, (stage3_box.x + 26, stage3_box.y + 64), "freeze geometry, decouple Y restoration and Cb/Cr correction", FONT_SMALL, PALETTE["muted"])
    draw_text(draw, (prior_box.x + 26, prior_box.y + 64), "offline cues reused conservatively across stages", FONT_SMALL, PALETTE["muted"])

    # Input module
    stack_anchor = (input_box.x + 38, input_box.y + 94)
    stack_size = (210, 300)
    stack_paths = [
        ASSET_ROOT / "Low_Light_3.JPG",
        ASSET_ROOT / "Low_Light_2.JPG",
        ASSET_ROOT / "Low_Light_1.JPG",
    ]
    paste_stack(canvas, stack_paths, stack_anchor, stack_size)
    draw_text(draw, (input_box.center[0], input_box.y + 430), "Low-light training views I_in", FONT_LABEL, PALETTE["ink"], anchor="ma")

    sup_box = Box(input_box.x + 38, input_box.y + 492, 224, 168)
    add_shadow(canvas, sup_box, radius=20, opacity=55, blur=14, offset=(0, 5))
    sup_tile = fit_image(ASSET_ROOT / "stage6" / "train_aug.jpg", (sup_box.w, sup_box.h), crop=crop_first_tile(ASSET_ROOT / "stage6" / "train_aug.jpg"), radius=20)
    paste(canvas, sup_tile, (sup_box.x, sup_box.y))
    draw_text(draw, (sup_box.center[0], sup_box.bottom + 32), "Enhanced supervision I_sup", FONT_LABEL, PALETTE["ink"], anchor="ma")
    draw_pill(canvas, Box(input_box.x + 44, input_box.bottom - 92, 214, 34), "gamma + exposure match", fill=(255, 255, 255, 210), outline=PALETTE["input_line"])
    draw_pill(canvas, Box(input_box.x + 74, input_box.bottom - 46, 150, 34), "shared RGB target", fill=(255, 255, 255, 210), outline=PALETTE["input_line"])

    # Prior module cards
    sparse_card = Box(prior_box.x + 36, prior_box.y + 108, 322, 214)
    depth_card = Box(prior_box.x + 434, prior_box.y + 108, 322, 214)
    structure_card = Box(prior_box.x + 832, prior_box.y + 108, 322, 214)
    for box, path in [
        (sparse_card, ASSET_ROOT / "stage5" / "cupcake_sparse_base.png"),
        (depth_card, ASSET_ROOT / "Depth.JPG"),
        (structure_card, ASSET_ROOT / "Structure.JPG"),
    ]:
        add_shadow(canvas, box, radius=24, opacity=55, blur=14, offset=(0, 5))
        paste(canvas, fit_image(path, (box.w, box.h), radius=22), (box.x, box.y))

    draw_text(draw, (sparse_card.center[0], prior_box.y + 95), "Fixed-pose COLMAP sparse", FONT_LABEL, PALETTE["ink"], anchor="ma")
    draw_text(draw, (depth_card.center[0], prior_box.y + 95), "Marigold depth", FONT_LABEL, PALETTE["ink"], anchor="ma")
    draw_text(draw, (structure_card.center[0], prior_box.y + 95), "Structure prior", FONT_LABEL, PALETTE["ink"], anchor="ma")
    draw_pill(canvas, Box(sparse_card.x + 28, sparse_card.bottom + 26, 266, 34), "point filtering + voxel dedup", fill=(255, 255, 255, 215), outline=PALETTE["prior_line"])
    draw_pill(canvas, Box(depth_card.x + 84, depth_card.bottom + 26, 154, 34), "weak trend prior", fill=(255, 255, 255, 215), outline=PALETTE["prior_line"])
    draw_pill(canvas, Box(structure_card.x + 46, structure_card.bottom + 26, 232, 34), "illumination-invariant edges", fill=(255, 255, 255, 215), outline=PALETTE["prior_line"])

    # Stage 1
    stage1_sparse = Box(stage1_box.x + 34, stage1_box.y + 110, 190, 140)
    stage1_render = Box(stage1_box.x + 258, stage1_box.y + 110, 228, 178)
    add_shadow(canvas, stage1_sparse, radius=20, opacity=52, blur=12, offset=(0, 4))
    add_shadow(canvas, stage1_render, radius=22, opacity=52, blur=12, offset=(0, 4))
    paste(canvas, fit_image(ASSET_ROOT / "stage5" / "cupcake_sparse_base.png", (stage1_sparse.w, stage1_sparse.h), radius=18), (stage1_sparse.x, stage1_sparse.y))
    paste(canvas, fit_image(ASSET_ROOT / "stage4" / "rendering.JPG", (stage1_render.w, stage1_render.h), radius=20), (stage1_render.x, stage1_render.y))
    draw_text(draw, (stage1_sparse.center[0], stage1_sparse.bottom + 26), "sparse init", FONT_SMALL, PALETTE["ink"], anchor="ma")
    draw_text(draw, (stage1_render.center[0], stage1_render.bottom + 26), "rendered geometry base", FONT_SMALL, PALETTE["ink"], anchor="ma")
    draw_arrow(draw, [(stage1_sparse.right + 16, stage1_sparse.center[1]), (stage1_render.x - 16, stage1_render.center[1])], PALETTE["data"], width=5)

    stage1_formula = render_formula(r"\mathcal{L}^{(1)}=\mathcal{L}_{rgb}+\lambda_d\mathcal{L}_{depth}+\lambda_{mv}\mathcal{L}_{mv}+\lambda_{exp}\mathcal{L}_{exp}")
    formula_w = min(stage1_formula.width, stage1_box.w - 70)
    formula_h = int(stage1_formula.height * formula_w / stage1_formula.width)
    stage1_formula = stage1_formula.resize((formula_w, formula_h), Image.Resampling.LANCZOS)
    paste(canvas, stage1_formula, (stage1_box.x + (stage1_box.w - formula_w) // 2, stage1_box.y + 326))
    draw_pill(canvas, Box(stage1_box.x + 42, stage1_box.bottom - 74, 128, 34), "only densify stage", fill=(255, 255, 255, 215), outline=PALETTE["stage1_line"])
    draw_pill(canvas, Box(stage1_box.x + 186, stage1_box.bottom - 74, 98, 34), "weak depth", fill=(255, 255, 255, 215), outline=PALETTE["stage1_line"])
    draw_pill(canvas, Box(stage1_box.x + 298, stage1_box.bottom - 74, 118, 34), "weak multiview", fill=(255, 255, 255, 215), outline=PALETTE["stage1_line"])
    draw_pill(canvas, Box(stage1_box.x + 428, stage1_box.bottom - 74, 52, 34), "+ exp", fill=(255, 255, 255, 215), outline=PALETTE["stage1_line"])
    draw_pill(canvas, Box(stage1_box.x + 44, stage1_box.y + 274, 180, 32), "all sparse + 5K random GS", fill=(255, 255, 255, 210), outline=PALETTE["stage1_line"], font=FONT_TINY)

    # Stage 2
    stage2_sparse = Box(stage2_box.x + 34, stage2_box.y + 108, 190, 140)
    stage2_render = Box(stage2_box.x + 358, stage2_box.y + 108, 228, 178)
    add_shadow(canvas, stage2_sparse, radius=20, opacity=52, blur=12, offset=(0, 4))
    add_shadow(canvas, stage2_render, radius=22, opacity=52, blur=12, offset=(0, 4))
    paste(canvas, fit_image(ASSET_ROOT / "stage5" / "cupcake_sparse_base.png", (stage2_sparse.w, stage2_sparse.h), radius=18), (stage2_sparse.x, stage2_sparse.y))
    paste(canvas, fit_image(ASSET_ROOT / "stage5" / "rendering.JPG", (stage2_render.w, stage2_render.h), radius=20), (stage2_render.x, stage2_render.y))
    draw_text(draw, (stage2_sparse.center[0], stage2_sparse.bottom + 24), "persistent sparse support", FONT_SMALL, PALETTE["ink"], anchor="ma")
    draw_text(draw, (stage2_render.center[0], stage2_render.bottom + 24), "refined geometry", FONT_SMALL, PALETTE["ink"], anchor="ma")
    draw_arrow(draw, [(stage2_sparse.right + 18, stage2_sparse.center[1]), (stage2_render.x - 16, stage2_render.center[1])], PALETTE["data"], width=5)

    sparse_formula_1 = render_formula(r"\bar{p}_i=\sum_{j=1}^{K} w_{ij}p_{ij}")
    sparse_formula_2 = render_formula(r"\mathcal{L}_{sparse}=\frac{1}{M}\sum_i \rho(\|\mu_i-\bar{p}_i\|_2)")
    paste(canvas, sparse_formula_1.resize((300, int(sparse_formula_1.height * 300 / sparse_formula_1.width)), Image.Resampling.LANCZOS), (stage2_box.x + 26, stage2_box.y + 296))
    paste(canvas, sparse_formula_2.resize((370, int(sparse_formula_2.height * 370 / sparse_formula_2.width)), Image.Resampling.LANCZOS), (stage2_box.x + 18, stage2_box.y + 354))
    draw_text(draw, (stage2_box.x + 34, stage2_box.y + 444), "w_ij: track length + reprojection error + local density + distance", FONT_TINY, PALETTE["muted"])
    draw_pill(canvas, Box(stage2_box.x + 356, stage2_box.bottom - 72, 102, 34), "no densify", fill=(255, 255, 255, 215), outline=PALETTE["stage2_line"])
    draw_pill(canvas, Box(stage2_box.x + 470, stage2_box.bottom - 72, 102, 34), "weak depth", fill=(255, 255, 255, 215), outline=PALETTE["stage2_line"])
    draw_pill(canvas, Box(stage2_box.x + 356, stage2_box.bottom - 32, 98, 34), "structure", fill=(255, 255, 255, 215), outline=PALETTE["stage2_line"])
    draw_pill(canvas, Box(stage2_box.x + 468, stage2_box.bottom - 32, 116, 34), "active GS + KNN", fill=(255, 255, 255, 215), outline=PALETTE["stage2_line"])

    # Stage 3 layout
    sup3_box = Box(stage3_box.x + 36, stage3_box.y + 118, 246, 176)
    proxy3_box = Box(stage3_box.x + 36, stage3_box.y + 352, 246, 176)
    core_box = Box(stage3_box.x + 316, stage3_box.y + 118, 948, 720)
    draw_dashed_rounded_rect(canvas, core_box, fill=(248, 246, 242, 220), outline=PALETTE["stage3_line"], radius=30, width=3, dash=16, gap=10)
    draw_text(draw, (core_box.x + 24, core_box.y + 20), "YCbCr appearance-only refinement", FONT_LABEL, PALETTE["ink"])

    for box, path, crop in [
        (sup3_box, ASSET_ROOT / "stage6" / "train_aug.jpg", crop_first_tile(ASSET_ROOT / "stage6" / "train_aug.jpg")),
        (proxy3_box, ASSET_ROOT / "stage6" / "proxy_target.jpg", crop_first_tile(ASSET_ROOT / "stage6" / "proxy_target.jpg")),
    ]:
        add_shadow(canvas, box, radius=20, opacity=50, blur=12, offset=(0, 4))
        paste(canvas, fit_image(path, (box.w, box.h), crop=crop, radius=20), (box.x, box.y))

    draw_text(draw, (sup3_box.center[0], sup3_box.bottom + 28), "I_sup", FONT_LABEL, PALETTE["ink"], anchor="ma")
    draw_text(draw, (proxy3_box.center[0], proxy3_box.bottom + 28), "I_proxy", FONT_LABEL, PALETTE["ink"], anchor="ma")
    draw_pill(canvas, Box(stage3_box.x + 44, stage3_box.y + 548, 234, 34), "shadow-aware calibrated target", fill=(255, 255, 255, 215), outline=PALETTE["stage3_line"])
    draw_pill(canvas, Box(stage3_box.x + 60, stage3_box.y + 592, 202, 34), "Y target only in Stage 3", fill=(255, 255, 255, 215), outline=PALETTE["stage3_line"])
    draw_pill(canvas, Box(stage3_box.x + 36, stage3_box.bottom - 120, 248, 34), "freeze geometry: μ, q, s", fill=(255, 255, 255, 215), outline=PALETTE["stage3_line"])
    draw_pill(canvas, Box(stage3_box.x + 44, stage3_box.bottom - 76, 232, 34), "optimize opacity / SH / illum / chroma", fill=(255, 255, 255, 215), outline=PALETTE["stage3_line"])

    base_box = Box(core_box.x + 20, core_box.y + 82, 224, 182)
    y_box = Box(core_box.x + 274, core_box.y + 72, 190, 264)
    c_box = Box(core_box.x + 498, core_box.y + 72, 214, 288)
    final_box = Box(core_box.x + 746, core_box.y + 80, 170, 182)

    add_shadow(canvas, base_box, radius=20, opacity=50, blur=12, offset=(0, 4))
    paste(canvas, fit_image(ASSET_ROOT / "stage6" / "base.JPG", (base_box.w, base_box.h), radius=20), (base_box.x, base_box.y))
    draw_text(draw, (base_box.center[0], base_box.bottom + 28), "Base RGB I_base", FONT_SMALL, PALETTE["ink"], anchor="ma")

    draw_dashed_rounded_rect(canvas, y_box, fill=(255, 253, 250, 225), outline=PALETTE["stage3_line"], radius=22, width=3, dash=12, gap=9)
    draw_text(draw, (y_box.center[0], y_box.y + 24), "Y-only luminance gain", FONT_LABEL, PALETTE["ink"], anchor="ma")
    illum_img = fit_image(ASSET_ROOT / "stage6" / "illum.JPG", (160, 110), radius=18)
    paste(canvas, illum_img, (y_box.x + (y_box.w - 160) // 2, y_box.y + 54))
    draw_text(draw, (y_box.center[0], y_box.y + 178), "illum head A_Y", FONT_SMALL, PALETTE["ink"], anchor="ma")
    y_formula = render_formula(r"Y'=\operatorname{clip}(2\sigma(A_Y)\odot Y,\,0,\,1)")
    y_formula = y_formula.resize((y_box.w - 28, int(y_formula.height * (y_box.w - 28) / y_formula.width)), Image.Resampling.LANCZOS)
    paste(canvas, y_formula, (y_box.x + 14, y_box.y + 198))
    draw_pill(canvas, Box(y_box.x + 38, y_box.bottom - 42, 150, 30), "proxy-guided Y", fill=(246, 238, 236, 255), outline=PALETTE["stage3_line"])

    draw_dashed_rounded_rect(canvas, c_box, fill=(255, 253, 250, 225), outline=PALETTE["stage3_line"], radius=22, width=3, dash=12, gap=9)
    draw_text(draw, (c_box.center[0], c_box.y + 24), "Additive chroma residual", FONT_LABEL, PALETTE["ink"], anchor="ma")
    cb_img = fit_image(ASSET_ROOT / "stage6" / "cb_factor.JPG", (78, 60), radius=14)
    cr_img = fit_image(ASSET_ROOT / "stage6" / "cr_factor.JPG", (78, 60), radius=14)
    paste(canvas, cb_img, (c_box.x + 20, c_box.y + 62))
    paste(canvas, cr_img, (c_box.x + 116, c_box.y + 62))
    draw_text(draw, (c_box.x + 59, c_box.y + 132), "ΔC_b", FONT_SMALL, PALETTE["ink"], anchor="ma")
    draw_text(draw, (c_box.x + 155, c_box.y + 132), "ΔC_r", FONT_SMALL, PALETTE["ink"], anchor="ma")
    c_formula = render_formula(r"\Delta_C=\alpha_c\tanh(A_C)")
    c_formula = c_formula.resize((c_box.w - 32, int(c_formula.height * (c_box.w - 32) / c_formula.width)), Image.Resampling.LANCZOS)
    paste(canvas, c_formula, (c_box.x + 16, c_box.y + 176))
    draw_wrapped_text(
        draw,
        Box(c_box.x + 26, c_box.y + 238, c_box.w - 52, 40),
        "Cb/Cr aligns with the more reliable supervision image under an independent chroma weight map.",
        FONT_TINY,
        PALETTE["muted"],
        line_spacing=4,
    )

    add_shadow(canvas, final_box, radius=20, opacity=54, blur=12, offset=(0, 4))
    paste(canvas, fit_image(ASSET_ROOT / "stage6" / "rendering.JPG", (final_box.w, final_box.h), radius=20), (final_box.x, final_box.y))
    draw_text(draw, (final_box.center[0], final_box.bottom + 28), "Final render I_rec", FONT_SMALL, PALETTE["ink"], anchor="ma")

    alloc_box = Box(core_box.x + 26, core_box.y + 430, 226, 176)
    weight_box = Box(core_box.x + 280, core_box.y + 430, 300, 176)
    objective_box = Box(core_box.x + 608, core_box.y + 430, 304, 176)
    draw_solid_rounded_rect(canvas, alloc_box, fill=(255, 255, 255, 220), outline=PALETTE["stage3_line"], radius=22)
    draw_solid_rounded_rect(canvas, weight_box, fill=(255, 255, 255, 220), outline=PALETTE["stage3_line"], radius=22)
    draw_solid_rounded_rect(canvas, objective_box, fill=(255, 255, 255, 220), outline=PALETTE["stage3_line"], radius=22)
    draw_text(draw, (alloc_box.x + 18, alloc_box.y + 18), "Target Allocation", FONT_LABEL, PALETTE["ink"])
    draw_text(draw, (weight_box.x + 18, weight_box.y + 18), "Branch-specific Weight Maps", FONT_LABEL, PALETTE["ink"])
    draw_text(draw, (objective_box.x + 18, objective_box.y + 18), "Stage 3 Objective", FONT_LABEL, PALETTE["ink"])
    draw.multiline_text(
        (alloc_box.x + 18, alloc_box.y + 62),
        "RGB base  ←  I_sup\nY(I_rec)   ←  I_proxy\nCb/Cr(I_rec)  ←  I_sup",
        font=FONT_BODY,
        fill=PALETTE["muted"],
        spacing=7,
    )
    weight_formula_y = render_formula(r"\mathcal{L}_{Y}=\operatorname{WeightedL1}(Y_{rec},Y_{proxy};W_Y)")
    weight_formula_c = render_formula(r"\mathcal{L}_{CbCr}=\operatorname{WeightedL1}(\tilde{C}_{rec},\tilde{C}_{tgt};W_C)")
    weight_formula_y = weight_formula_y.resize((weight_box.w - 28, int(weight_formula_y.height * (weight_box.w - 28) / weight_formula_y.width)), Image.Resampling.LANCZOS)
    weight_formula_c = weight_formula_c.resize((weight_box.w - 28, int(weight_formula_c.height * (weight_box.w - 28) / weight_formula_c.width)), Image.Resampling.LANCZOS)
    paste(canvas, weight_formula_y, (weight_box.x + 14, weight_box.y + 50))
    paste(canvas, weight_formula_c, (weight_box.x + 14, weight_box.y + 96))
    draw_text(draw, (weight_box.x + 18, weight_box.bottom - 28), "W_Y boosts dark informative regions; W_C is more conservative in deep shadows.", FONT_TINY, PALETTE["muted"])
    objective_formula = render_formula(r"\mathcal{L}^{(3)}=\mathcal{L}_{rgb}+\lambda_Y\mathcal{L}_Y+\lambda_c\mathcal{L}_{CbCr}+\lambda_{cr}\|\Delta_C\|_1")
    objective_formula = objective_formula.resize((objective_box.w - 24, int(objective_formula.height * (objective_box.w - 24) / objective_formula.width)), Image.Resampling.LANCZOS)
    paste(canvas, objective_formula, (objective_box.x + 12, objective_box.y + 58))
    draw.multiline_text(
        (objective_box.x + 18, objective_box.y + 118),
        "freeze μ, q, s\noptimize opacity / SH / illum / chroma only",
        font=FONT_TINY,
        fill=PALETTE["muted"],
        spacing=4,
    )

    # Stage 3 internal arrows
    draw_arrow(draw, [(base_box.right + 14, base_box.center[1]), (y_box.x - 14, base_box.center[1])], PALETTE["data"], width=5)
    draw_arrow(draw, [(y_box.right + 12, base_box.center[1]), (c_box.x - 12, base_box.center[1])], PALETTE["data"], width=5)
    draw_arrow(draw, [(c_box.right + 12, base_box.center[1]), (final_box.x - 12, base_box.center[1])], PALETTE["data"], width=5)

    rgb_loss_x = base_box.x + 28
    chroma_loss_x = c_box.x + c_box.w // 2
    y_loss_x = y_box.x + y_box.w // 2
    draw_arrow(draw, [(sup3_box.right + 16, sup3_box.center[1]), (sup3_box.right + 60, sup3_box.center[1]), (sup3_box.right + 60, base_box.y - 18), (rgb_loss_x, base_box.y - 18), (rgb_loss_x, base_box.y - 2)], PALETTE["grad"], width=5)
    draw_arrow(draw, [(sup3_box.right + 16, sup3_box.center[1]), (sup3_box.right + 90, sup3_box.center[1]), (sup3_box.right + 90, c_box.bottom + 22), (chroma_loss_x, c_box.bottom + 22), (chroma_loss_x, c_box.bottom - 2)], PALETTE["grad"], width=5)
    draw_arrow(draw, [(proxy3_box.right + 16, proxy3_box.center[1]), (proxy3_box.right + 62, proxy3_box.center[1]), (proxy3_box.right + 62, y_box.bottom + 18), (y_loss_x, y_box.bottom + 18), (y_loss_x, y_box.bottom - 2)], PALETTE["grad"], width=5)
    draw_text(draw, (rgb_loss_x + 4, base_box.y - 34), "L_rgb", FONT_SMALL, PALETTE["grad"])
    draw_text(draw, (chroma_loss_x - 34, c_box.bottom + 26), "L_CbCr", FONT_SMALL, PALETTE["grad"])
    draw_text(draw, (y_loss_x - 10, y_box.bottom + 22), "L_Y", FONT_SMALL, PALETTE["grad"])

    # Main data flow arrows
    input_mid_y = input_box.y + 290
    draw_arrow(draw, [(input_box.right + 12, input_mid_y), (stage1_box.x - 12, input_mid_y)], PALETTE["data"], width=6, arrow_size=18)
    draw_arrow(draw, [(stage1_box.right + 14, stage1_box.center[1]), (stage2_box.x - 14, stage2_box.center[1])], PALETTE["data"], width=6, arrow_size=18)
    draw_arrow(draw, [(stage2_box.right + 14, stage2_box.center[1]), (stage3_box.x - 14, stage2_box.center[1])], PALETTE["data"], width=6, arrow_size=18)
    draw_arrow(draw, [(input_box.right - 12, input_box.bottom - 120), (input_box.right + 30, input_box.bottom - 120), (input_box.right + 30, prior_box.y + 120), (prior_box.x - 12, prior_box.y + 120)], PALETTE["data"], width=5, arrow_size=16)

    # Prior to stage arrows
    draw_arrow(draw, [(sparse_card.center[0], sparse_card.y - 14), (sparse_card.center[0], stage1_box.bottom + 16), (stage1_box.x + 110, stage1_box.bottom + 16), (stage1_box.x + 110, stage1_box.bottom - 2)], PALETTE["data"], width=5)
    draw_text(draw, (stage1_box.x + 64, stage1_box.bottom + 20), "init", FONT_SMALL, PALETTE["data"])

    draw_arrow(draw, [(depth_card.center[0], depth_card.y - 10), (depth_card.center[0], stage1_box.bottom + 58), (stage1_box.center[0], stage1_box.bottom + 58), (stage1_box.center[0], stage1_box.bottom - 4)], PALETTE["grad"], width=5)
    draw_text(draw, (stage1_box.center[0] - 34, stage1_box.bottom + 62), "L_depth", FONT_SMALL, PALETTE["grad"])

    draw_arrow(draw, [(depth_card.center[0], depth_card.y - 10), (depth_card.center[0], stage2_box.bottom + 58), (stage2_box.x + 408, stage2_box.bottom + 58), (stage2_box.x + 408, stage2_box.bottom - 4)], PALETTE["grad"], width=5)
    draw_text(draw, (stage2_box.x + 366, stage2_box.bottom + 62), "L_depth", FONT_SMALL, PALETTE["grad"])

    draw_arrow(draw, [(structure_card.center[0], structure_card.y - 10), (structure_card.center[0], stage2_box.bottom + 106), (stage2_box.x + 508, stage2_box.bottom + 106), (stage2_box.x + 508, stage2_box.bottom - 4)], PALETTE["grad"], width=5)
    draw_text(draw, (stage2_box.x + 462, stage2_box.bottom + 110), "L_struct", FONT_SMALL, PALETTE["grad"])

    draw_arrow(draw, [(sparse_card.center[0], sparse_card.y - 8), (sparse_card.center[0], stage2_box.bottom + 154), (stage2_box.x + 160, stage2_box.bottom + 154), (stage2_box.x + 160, stage2_box.bottom - 4)], PALETTE["grad"], width=5)
    draw_text(draw, (stage2_box.x + 112, stage2_box.bottom + 158), "L_sparse", FONT_SMALL, PALETTE["grad"])

    draw_arrow(draw, [(sup_box.right + 8, sup_box.center[1]), (sup_box.right + 34, sup_box.center[1]), (sup_box.right + 34, stage1_box.y - 20), (stage1_box.center[0], stage1_box.y - 20), (stage1_box.center[0], stage1_box.y - 2)], PALETTE["grad"], width=5)
    draw_arrow(draw, [(sup_box.right + 8, sup_box.center[1]), (sup_box.right + 62, sup_box.center[1]), (sup_box.right + 62, stage2_box.y - 20), (stage2_box.center[0], stage2_box.y - 20), (stage2_box.center[0], stage2_box.y - 2)], PALETTE["grad"], width=5)
    draw_text(draw, (stage1_box.center[0] - 24, stage1_box.y - 54), "L_rgb", FONT_SMALL, PALETTE["grad"])
    draw_text(draw, (stage2_box.center[0] - 24, stage2_box.y - 54), "L_rgb", FONT_SMALL, PALETTE["grad"])

    # Legend
    draw_text(draw, (legend_box.x + 30, legend_box.y + 30), "Legend", FONT_LABEL, PALETTE["ink"])
    draw_arrow(draw, [(legend_box.x + 162, legend_box.y + 45), (legend_box.x + 258, legend_box.y + 45)], PALETTE["data"], width=5, arrow_size=15)
    draw_text(draw, (legend_box.x + 274, legend_box.y + 45), "data flow", FONT_BODY, PALETTE["ink"], anchor="lm")
    draw_arrow(draw, [(legend_box.x + 494, legend_box.y + 45), (legend_box.x + 590, legend_box.y + 45)], PALETTE["grad"], width=5, arrow_size=15)
    draw_text(draw, (legend_box.x + 606, legend_box.y + 45), "gradient flow / supervision", FONT_BODY, PALETTE["ink"], anchor="lm")
    draw_pill(canvas, Box(legend_box.x + 1038, legend_box.y + 28, 248, 34), "rounded dashed module", fill=(255, 255, 255, 210), outline=PALETTE["card_line"], font=FONT_SMALL)
    draw_pill(canvas, Box(legend_box.x + 1324, legend_box.y + 28, 256, 34), "Morandi module palette", fill=(255, 255, 255, 210), outline=PALETTE["card_line"], font=FONT_SMALL)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, quality=96)
    print(f"[PipelineOverview] saved to {out_path}")


if __name__ == "__main__":
    build_overview(DEFAULT_OUT)

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


SCENES = [
    "Chocolate",
    "Cupcake",
    "GearWorks",
    "Laboratory",
    "MilkCookie",
    "Popcorn",
    "Sculpture",
    "Ujikintoki",
]

DIAG_ROOT = Path("outputs/stage6_ablation_weightmap_y_only_stronger_cbcr")
OUT_PATH = Path("report/3DRR_report_template/figs/fig5/proxy_route_scatter.png")


def load_summary(scene: str) -> Dict[str, float]:
    diag_path = DIAG_ROOT / f"{scene}_new" / "brightness_signal_diagnostics.txt"
    if not diag_path.exists():
        raise FileNotFoundError(f"Missing diagnostics file: {diag_path}")

    summary = None
    for line in diag_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("record_type") == "summary":
            summary = payload

    if summary is None:
        raise RuntimeError(f"No summary record found in {diag_path}")
    return summary


def bbox_intersection_area(a, b) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def place_labels(ax, fig, xs: np.ndarray, ys: np.ndarray, names: List[str]) -> float:
    # Highly curated label placement for the 8 specific scenes for maximum paper quality
    
    # Offsets in points (dx, dy)
    offsets = {
        "Cupcake": (2, 16),
        "MilkCookie": (18, -14),
        "GearWorks": (-16, 15),
        "Sculpture": (0, -16),
        "Ujikintoki": (16, 14),
        "Chocolate": (-20, 12),
        "Laboratory": (-20, 12),
        "Popcorn": (-22, -2),
    }

    # Haligin/Valign adjustments dynamically based on offsets
    for i, name in enumerate(names):
        x, y = xs[i], ys[i]
        dx, dy = offsets.get(name, (20, 20))
        
        ha = "center"
        if dx < -8: ha = "right"
        elif dx > 8: ha = "left"
        
        va = "center"
        if dy < -8: va = "top"
        elif dy > 8: va = "bottom"

        ax.annotate(
            name,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.8, alpha=0.7),
            zorder=5,
        )

    # Return a dummy baseline_y that's slightly below the lowest point so ylim adjusts properly
    return float(np.min(ys)) - 0.005


def main() -> None:
    records = []
    for scene in SCENES:
        s = load_summary(scene)
        mean_gain = float(s["mean_proxy_gain"])
        raw_target = float(s["mean_proxy_target_mean_raw"])
        effective_target = float(s["mean_proxy_effective_y_target_mean"])
        compression = max(0.0, raw_target - effective_target)
        global_ratio = float(s["mean_proxy_post_global_ratio"])
        records.append((scene, mean_gain, compression, global_ratio))

    names = [r[0] for r in records]
    x = np.array([r[1] for r in records], dtype=np.float64)
    y = np.array([r[2] for r in records], dtype=np.float64)
    c = np.array([r[3] for r in records], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    scatter = ax.scatter(
        x,
        y,
        c=c,
        cmap="RdYlGn",
        s=160,
        edgecolors="black",
        linewidths=0.85,
        alpha=0.95,
        zorder=3,
    )

    baseline_y = place_labels(ax, fig, x, y, names)

    ax.set_title("Stage3 adaptive target compression and route behavior", fontsize=18, pad=8)
    ax.set_xlabel("Mean proxy gain", fontsize=14)
    ax.set_ylabel("Raw target - effective target", fontsize=14)
    ax.grid(True, alpha=0.25)

    x_pad = 0.8
    y_pad = 0.006
    ax.set_xlim(float(np.min(x) - x_pad), float(np.max(x) + x_pad))
    ax.set_ylim(float(min(-0.002, np.min(y) - y_pad, baseline_y - 0.01)), float(np.max(y) + y_pad))

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Global-route ratio (0=local, 1=global)", fontsize=14)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=220)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

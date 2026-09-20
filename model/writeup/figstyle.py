"""
Shared figure style for the write-up.

Palette is a muted categorical set built in OKLCh at a uniform chroma of
0.125 -- roughly a quarter less saturated than a standard chart palette --
with per-slot lightness chosen so that the accessibility gates still clear
comfortably. Hue order is unchanged, so a colour means the same thing it did
before.

Measured on the light surface (#fcfcfb), adjacent slot pairs -- the ones that
matter for stacked, bar and line forms:

    lightness band        all 8 inside OKLCh L 0.43-0.77
    chroma floor          all 8 at 0.125, above the 0.10 gray threshold
    CVD separation        worst adjacent pair dE 9.7 (deutan), tritan 5.5
    normal-vision floor   worst adjacent pair dE 17.6
    contrast vs surface   all 8 at or above 3:1

Every slot now clears 3:1 against the surface, so the earlier "relief rule"
-- three slots that needed direct labels to be legible -- no longer applies.
Labels and legends are kept for readability, not for compliance.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# categorical slots, fixed order, never cycled
C = ["#4b81c9", "#a9522f", "#2aa172", "#a97202",
     "#d37497", "#4c8e46", "#7874c8", "#c05f59"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = C

# sequential blue ramp (100 -> 700), single hue at the categorical blue
SEQ = ["#d9e9ff", "#c8dcf6", "#b7ceed", "#a7c1e4", "#96b3db",
       "#86a6d1", "#7699c8", "#668cbf", "#577fb5", "#4772ac",
       "#3765a2", "#275999", "#154c8f"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
STATUS_GOOD, STATUS_WARN, STATUS_CRIT = "#4a8c4e", "#cd9b3c", "#b64f4b"


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 9,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK2,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def no_grid_x(ax):
    ax.grid(axis="x", visible=False)


def no_grid_y(ax):
    ax.grid(axis="y", visible=False)

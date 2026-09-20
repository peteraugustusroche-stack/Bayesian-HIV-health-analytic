"""Figure 7: the calibration anchor as the dominant source of uncertainty.

This is the figure the dissertation's central claim needs and the existing set
does not contain. Three panels, one argument:

  (a) the two posteriors do not overlap
  (b) no single parameter, swept across its own posterior, moves the estimate
      as far as changing the anchor does
  (c) the price arm B pays for its retention fit is paid in p1 and p2

Run after both arms exist:
    python3 writeup/fig_arms.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import figlabels as L
from figstyle import (apply_style, SURFACE, INK, INK2, MUTED, BLUE, ORANGE,
                      RED, STATUS_GOOD, despine, no_grid_x, no_grid_y)

apply_style()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{HERE}/data/results"
OUT = f"{HERE}/writeup"

A_LAB, B_LAB = "UNAIDS-Anchored", "Retention-Anchored"


def draws(arm, metric="DALYs_per_acquisition"):
    p = np.load(f"{RES}/{arm}/posterior_analysis.npz", allow_pickle=True)
    return np.asarray(p["payload"].item()["probabilistic"]["metrics"][metric])


def s2(arm):
    d = np.load(f"{RES}/{arm}/system2.npz", allow_pickle=True)
    return d["extra"].item()


dA, dB = draws("arm_unaids"), draws("arm_retention")
eA, eB = s2("arm_unaids"), s2("arm_retention")

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3),
                         gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})

# ---------------- (a) posterior densities --------------------------------
ax = axes[0]
grid = np.linspace(min(dA.min(), dB.min()) - 0.6, max(dA.max(), dB.max()) + 0.6, 400)
for d, col, lab in ((dA, BLUE, A_LAB), (dB, ORANGE, B_LAB)):
    k = gaussian_kde(d)(grid)
    ax.fill_between(grid, k, color=col, alpha=0.20, zorder=3)
    ax.plot(grid, k, color=col, lw=2.0, zorder=4, label=lab)
    lo, hi = np.quantile(d, [0.05, 0.95])
    ax.plot([lo, hi], [-0.012, -0.012], color=col, lw=3.2, zorder=5,
            solid_capstyle="butt")
    ax.plot([d.mean()], [-0.012], marker="|", color="white", ms=9, mew=1.8, zorder=6)

aq95, bq05 = np.quantile(dA, 0.95), np.quantile(dB, 0.05)
ax.axvspan(aq95, bq05, color=MUTED, alpha=0.10, zorder=1)
ax.set_xlabel("DALYs per acquisition, discounted at 3%")
ax.set_ylabel("Posterior density")
ax.set_ylim(-0.045, None)
ax.legend(loc="upper left", fontsize=8)
despine(ax); no_grid_x(ax)
ax.set_title("(a)", loc="left")

# ---------------- (b) parameter swings vs the anchor ---------------------
ax = axes[1]
sc = np.load(f"{RES}/arm_unaids/scenario_uncertainty.npz",
             allow_pickle=True)["payload"].item()
rows = sorted(sc["rows"], key=lambda r: r["DALYs_per_acquisition_range"])
base = rows[0]["DALYs_per_acquisition_baseline"]
y = np.arange(len(rows))
for i, r in enumerate(rows):
    lo = r["DALYs_per_acquisition_at_q03"]; hi = r["DALYs_per_acquisition_at_q97"]
    ax.plot([min(lo, hi), max(lo, hi)], [i, i], color=BLUE, lw=7,
            solid_capstyle="butt", alpha=0.85, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([L.param(r["param"]) for r in rows], fontsize=7.6)
ax.axvline(base, color=MUTED, lw=1.0, ls=":", zorder=2)

ax.axvline(dA.mean(), color=BLUE, lw=1.8, zorder=5)
ax.axvline(dB.mean(), color=ORANGE, lw=1.8, zorder=5)
top = len(rows) - 0.4
ax.annotate("", xy=(dB.mean(), top), xytext=(dA.mean(), top),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
ax.set_ylim(-0.7, len(rows) + 0.5)
ax.set_xlabel("DALYs per acquisition")
no_grid_y(ax); despine(ax)
ax.set_title("(b)", loc="left")

# ---------------- (c) what arm B pays for the fit ------------------------
ax = axes[2]
keys = [k for k in L.TARGET_LABEL
        if k in eA["targets"] or k in eB["targets"]]


def z(extra, k):
    t = extra["targets"].get(k)
    im = extra["model_implied_at_mean"].get(k)
    if t is None or im is None or not t[1]:
        return None
    return (im - t[0]) / t[1]


yy = np.arange(len(keys))[::-1]
ax.axvspan(-2, 2, color=STATUS_GOOD, alpha=0.07, zorder=0)
ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
for yi, k in zip(yy, keys):
    for extra, col, off in ((eA, BLUE, 0.16), (eB, ORANGE, -0.16)):
        v = z(extra, k)
        if v is None:
            continue
        ax.scatter([v], [yi + off], color=col, s=46, zorder=4,
                   edgecolor=RED if abs(v) > 2 else "white",
                   linewidth=1.8 if abs(v) > 2 else 0.8)
ax.set_yticks(yy); ax.set_yticklabels([L.target(k) for k in keys], fontsize=7.4)
ax.set_xlabel("Standardised residual  (model − target) / SD")
ax.set_xlim(-4.4, 4.4)
no_grid_y(ax); despine(ax)
ax.set_title("(c)", loc="left")
ax.legend(handles=[Patch(facecolor=BLUE, label=A_LAB),
                   Patch(facecolor=ORANGE, label=B_LAB),
                   Line2D([], [], marker="o", ls="none", markerfacecolor=MUTED,
                 markeredgecolor=RED, markeredgewidth=1.8, markersize=7,
                 label="Misses By More Than 2 SD")],
          loc="lower right", fontsize=7.2)

fig.tight_layout()
L.save(fig, f"{OUT}/fig7_arms.png")

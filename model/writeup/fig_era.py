"""Figure 7: burden under two cascade eras.

The argument in one picture: calibrate the same model to a better cascade and
the burden a person carries falls, while the burden the cascade averts rises.
The no-care counterfactual is fixed by natural history and does not move.

    python3 writeup/fig_era.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from figstyle import (apply_style, INK, INK2, MUTED, BLUE, ORANGE, AQUA,
                      despine, no_grid_x, no_grid_y)
import figlabels as L
from run_posterior_analysis import load_posterior_result

apply_style()
OUT = os.path.dirname(os.path.abspath(__file__))

A = load_posterior_result("2019")["probabilistic"]["metrics"]
B = load_posterior_result("2024")["probabilistic"]["metrics"]

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1),
                         gridspec_kw={"width_ratios": [1.15, 1]})

# ---------------- (a) posterior densities ----------------
ax = axes[0]
grid = np.linspace(2.0, 11.0, 400)
for d, col, lab in ((A, BLUE, "2019 Cascade"), (B, ORANGE, "2024 Cascade")):
    v = np.asarray(d["DALYs_per_acquisition"])
    k = gaussian_kde(v)(grid)
    ax.plot(grid, k, color=col, lw=2.0, zorder=3, label=lab)
    ax.fill_between(grid, k, color=col, alpha=0.13, zorder=2)
    ax.axvline(v.mean(), color=col, lw=1.0, ls=":", zorder=4)
ax.set_xlabel("DALYs per acquisition, discounted at 3%")
ax.set_ylabel("Posterior density")
ax.set_xlim(2.0, 11.0)
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", fontsize=8)
no_grid_x(ax); despine(ax)

# ---------------- (b) accrued against averted ----------------
ax = axes[1]
nh = np.asarray(A["DALYs_natural_history"]).mean()
acc = [np.asarray(A["DALYs_per_acquisition"]).mean(),
       np.asarray(B["DALYs_per_acquisition"]).mean()]
avt = [np.asarray(A["DALYs_avoided_through_care"]).mean(),
       np.asarray(B["DALYs_avoided_through_care"]).mean()]

x = np.arange(2)
w = 0.54
ax.bar(x, acc, w, color=BLUE, zorder=3, label="Accrued Under The Cascade")
ax.bar(x, avt, w, bottom=acc, color=AQUA, zorder=3, label="Averted By The Cascade")

for xi, (a, b) in enumerate(zip(acc, avt)):
    ax.text(xi, a / 2, f"{a:.2f}", ha="center", va="center", fontsize=9.5,
            color="white", fontweight="bold")
    ax.text(xi, a + b / 2, f"{b:.2f}", ha="center", va="center", fontsize=9.5,
            color="white", fontweight="bold")
    ax.text(xi, a + b + 0.30, f"{b / (a + b) * 100:.1f}% Averted", ha="center",
            fontsize=8.4, color=INK2)

ax.set_xticks(x); ax.set_xticklabels(["2019 Cascade", "2024 Cascade"], fontsize=9)
ax.set_ylabel("DALYs per acquisition, discounted at 3%")
ax.set_ylim(0, 21.4)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=7.8, ncol=2,
          handlelength=1.2, columnspacing=1.2)
no_grid_x(ax); despine(ax)

fig.tight_layout()
L.save(fig, f"{OUT}/fig7_era.png")

d = np.asarray(A["DALYs_per_acquisition"]) - np.asarray(B["DALYs_per_acquisition"])
print(f"  accrued {acc[0]:.2f} -> {acc[1]:.2f} ({acc[1]-acc[0]:+.2f}); "
      f"averted {avt[0]:.2f} -> {avt[1]:.2f} ({avt[1]-avt[0]:+.2f}); "
      f"share {avt[0]/nh*100:.1f}% -> {avt[1]/nh*100:.1f}%")

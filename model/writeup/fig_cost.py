"""Figure 9: the health-economic lens.

  (a) cost-effectiveness plane -- every posterior draw as a point, cost of the
      cascade against the DALYs it averts, both eras on a common 2024 price
      base so the comparison is real rather than inflationary

  (b) what the money buys -- lifetime discounted cost decomposed by component,
      both eras, again on a common 2024 base

    python3 writeup/fig_cost.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from figstyle import (apply_style, INK, INK2, MUTED, BLUE, ORANGE, AQUA, YELLOW,
                      VIOLET, despine, no_grid_x, no_grid_y)
import figlabels as L
import health_econ_params as hep
import simulate_cascade as sc
from params import load_parameters, N_CD4_BANDS
from calibration_common import load_result
from run_posterior_analysis import load_posterior_result
from states import full_index, CASCADE_STATUSES

apply_style()
OUT = os.path.dirname(os.path.abspath(__file__))
PRICE_YEAR = 2024

# both arms on a common 2024 price base
A = load_posterior_result("2019", PRICE_YEAR)["probabilistic"]["metrics"]
B = load_posterior_result("2024")["probabilistic"]["metrics"]

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3),
                         gridspec_kw={"width_ratios": [1.25, 1]})

# ---------------- (a) cost-effectiveness plane ----------------
ax = axes[0]
for M, col, lab in ((A, BLUE, "2019 Cascade"), (B, ORANGE, "2024 Cascade")):
    x = np.asarray(M["DALYs_avoided_through_care"])
    y = np.asarray(M["cost_per_acquisition"])
    ax.scatter(x, y, s=5, color=col, alpha=0.16, linewidths=0, zorder=3)
    ax.scatter([x.mean()], [y.mean()], s=58, color=col, edgecolor="white",
               linewidth=1.4, zorder=6, label=lab)

XLIM, YLIM = (7.5, 15.0), (1200, 3500)
xs = np.array(XLIM)
for thr, ls in ((150, ":"), (250, "--")):
    ax.plot(xs, thr * xs, color=MUTED, lw=1.0, ls=ls, zorder=2, clip_on=True)
    # label where the ray leaves the axes, kept inside the frame
    x_at_top = YLIM[1] / thr
    lx = min(x_at_top, XLIM[1]) - 0.12
    ax.text(lx, thr * lx, f"${thr}/DALY ", fontsize=7.4, color=MUTED,
            ha="right", va="bottom", rotation=0)

ax.set_xlabel("DALYs averted per acquisition")
ax.set_ylabel(f"Lifetime cost per acquisition, {PRICE_YEAR} US$")
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"${v:,.0f}"))
ax.legend(loc="upper left", fontsize=8, scatterpoints=1)
despine(ax)

# ---------------- (b) cost composition ----------------
hep.set_price_year(PRICE_YEAR)
p = load_parameters()
s1 = load_result("system1")
m1 = {r["param"]: r["mean"] for r in s1["summary"]}
kp, km = m1["κ_prog"], m1["κ_mort"]

COMPONENTS = ["Routine Care, CD4 Below 200", "Routine Care, CD4 200 Or Above",
              "Second-Line Drugs", "First-Line Drugs", "Terminal Care"]
COLS = [BLUE, AQUA, VIOLET, YELLOW, ORANGE]
BAND_BELOW_200 = 4          # first band at or below CD4 200


def decompose(result_name):
    from states import FULL_HIV_DEATH, FULL_BG_DEATH
    c = {r["param"]: r["mean"] for r in load_result(result_name)["summary"]}
    stack = sc.build_full_matrix_stack(c, p.fixed["recov_rate"], horizon_years=71,
                                       kappa_prog=kp, kappa_mort=km)
    traj = sc.propagate(stack, sc.undiagnosed_init(), 852)
    out = dict.fromkeys(COMPONENTS, 0.0)
    dt = 1 / 12
    for t in range(852):
        disc = 1.03 ** (-(t * dt))
        M = stack[min(int(t * dt), stack.shape[0] - 1)]
        for st in CASCADE_STATUSES:
            base = st.split("__")[0]
            for b in range(N_CD4_BANDS):
                idx = full_index(st, b)
                m = traj[t, idx]
                if m <= 0:
                    continue
                if base in ("ART1_Ramp", "ART1_Suppressed", "ART1_Failing"):
                    out["First-Line Drugs"] += m * hep.COST_1L_ART_PER_YEAR * dt * disc
                elif base == "ART2_Suppressed":
                    out["Second-Line Drugs"] += m * hep.COST_2L_ART_PER_YEAR * dt * disc
                if base in hep.ROUTINE_CARE_STATUSES:
                    key = ("Routine Care, CD4 Below 200" if b >= BAND_BELOW_200
                           else "Routine Care, CD4 200 Or Above")
                    out[key] += m * hep.ROUTINE_CARE_BY_BAND[b] * dt * disc
                out["Terminal Care"] += m * (M[idx, FULL_HIV_DEATH]
                                             + M[idx, FULL_BG_DEATH]) * hep.COST_DEATH * disc
    return out


d19 = decompose("system2")
d24 = decompose("system2_sens2024")

ax = axes[1]
x = np.arange(2)
bottom = np.zeros(2)
for comp, col in zip(COMPONENTS, COLS):
    vals = np.array([d19[comp], d24[comp]])
    ax.bar(x, vals, 0.54, bottom=bottom, color=col, zorder=3, label=comp)
    for xi, (v, b0) in enumerate(zip(vals, bottom)):
        if v > 150:
            ax.text(xi, b0 + v / 2, f"${v:,.0f}", ha="center", va="center",
                    fontsize=8.2, color="white", fontweight="bold")
    bottom += vals
for xi, tot in enumerate(bottom):
    ax.text(xi, tot + 40, f"${tot:,.0f}", ha="center", fontsize=9, color=INK,
            fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(["2019 Cascade", "2024 Cascade"], fontsize=9)
ax.set_ylabel(f"Lifetime cost per acquisition, {PRICE_YEAR} US$")
ax.set_ylim(0, 2600)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"${v:,.0f}"))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=7.2, ncol=2,
          handlelength=1.1, columnspacing=1.0)
no_grid_x(ax); despine(ax)

fig.tight_layout()
L.save(fig, f"{OUT}/fig9_cost.png")

for nm, M in (("2019", A), ("2024", B)):
    c = np.asarray(M["cost_per_acquisition"]); a = np.asarray(M["DALYs_avoided_through_care"])
    print(f"  {nm} @ {PRICE_YEAR} US$: cost ${c.mean():,.0f}, averted {a.mean():.2f}, "
          f"${(c/a).mean():.0f}/DALY averted")

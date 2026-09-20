"""Figure 8: immune reconstitution as a mitigation of HIV burden.

Two panels, one argument.

  (a) people climb CD4 strata on treatment -- a quarter start ART at CD4 >=500
      but two thirds of on-ART person-time is spent there

  (b) what that climb is worth -- DALYs under the full model against a
      counterfactual in which ART suppresses virus but produces no CD4
      recovery, so everyone stays in the band they started in

Panel (b)'s counterfactual is not a strawman: holding people in their
CD4-at-initiation category is how Spectrum/AIM represents adults on ART.

    python3 writeup/fig_reconstitution.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from figstyle import (apply_style, SURFACE, INK, INK2, MUTED, BLUE, ORANGE,
                      AQUA, despine, no_grid_x, no_grid_y)
import figlabels as L
import simulate_cascade as sc
from params import load_parameters, N_CD4_BANDS, CD4_BAND_NAMES
from states import full_index, with_base
from calibration_common import load_result
from health_economics import evaluate_health_econ

apply_style()
OUT = os.path.dirname(os.path.abspath(__file__))

p = load_parameters()
s1 = load_result("system1"); s2 = load_result("system2")
m1 = {r["param"]: r["mean"] for r in s1["summary"]}
c = {r["param"]: r["mean"] for r in s2["summary"]}
kp, km = m1["κ_prog"], m1["κ_mort"]

# ---- (a) where people start, and where they spend their time --------------
stack = sc.build_full_matrix_stack(c, p.fixed["recov_rate"], horizon_years=71,
                                   kappa_prog=kp, kappa_mort=km)
traj = sc.propagate(stack, sc.undiagnosed_init(), 852)
init = np.asarray(sc.cd4_distribution_at_art_initiation(stack, horizon_years=71))
init = init / init.sum()
sup = with_base("ART1_Suppressed", "ART2_Suppressed")
pyr = np.array([sum(traj[:, full_index(s, b)].sum() / 12 for s in sup)
                for b in range(N_CD4_BANDS)])
pyr = pyr / pyr.sum()

# ---- (b) the counterfactual ----------------------------------------------
class _Params:
    """params with a substituted reconstitution rate; everything else shared."""
    def __init__(self, base, recov):
        self.fixed = dict(base.fixed); self.fixed["recov_rate"] = recov
        self.targets = base.targets

full = evaluate_health_econ(c, kp, km, p)
norec = evaluate_health_econ(c, kp, km, _Params(p, (0.0, 0.0)))

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3),
                         gridspec_kw={"width_ratios": [1.35, 1]})

# ---------------- (a) ----------------
ax = axes[0]
y = np.arange(N_CD4_BANDS)
h = 0.38
ax.barh(y - h/2, init * 100, h, color=ORANGE, zorder=3, label="At ART Initiation")
ax.barh(y + h/2, pyr * 100, h, color=AQUA, zorder=3, label="On-ART Person-Time")
for yi, v in zip(y - h/2, init * 100):
    ax.text(v + 0.9, yi, f"{v:.1f}", va="center", fontsize=7, color=INK2)
for yi, v in zip(y + h/2, pyr * 100):
    ax.text(v + 0.9, yi, f"{v:.1f}", va="center", fontsize=7, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(CD4_BAND_NAMES, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("% of cohort")
ax.set_xlim(0, 72)
ax.legend(loc="lower right", fontsize=7.6)
no_grid_y(ax); despine(ax)
ax.set_title("(a)", loc="left")

# ---------------- (b) ----------------
ax = axes[1]
labs = ["No Care", "ART Without\nCD4 Recovery", "ART With\nCD4 Recovery"]
yll = [full["YLL_natural_history"], norec["YLL_cascade"], full["YLL_cascade"]]
yld = [full["YLD_natural_history"], norec["YLD_cascade"], full["YLD_cascade"]]
x = np.arange(3)
ax.bar(x, yll, 0.52, color=BLUE, zorder=3, label="Years Of Life Lost")
ax.bar(x, yld, 0.52, bottom=np.array(yll) + 0.07, color=AQUA, zorder=3,
       label="Years Lived With Disability")
for xi, (a, b) in enumerate(zip(yll, yld)):
    ax.text(xi, a / 2, f"{a:.2f}", ha="center", va="center", fontsize=8,
            color="white", fontweight="bold")
    ax.text(xi, a + b / 2 + 0.07, f"{b:.2f}", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")
    ax.text(xi, a + b + 0.45, f"{a+b:.2f}", ha="center", fontsize=9,
            color=INK, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
ax.set_ylabel("DALYs per acquisition, discounted at 3%")
ax.set_ylim(0, 19.5)
ax.legend(loc="upper right", fontsize=7.6)
no_grid_x(ax); despine(ax)
ax.set_title("(b)", loc="left")

fig.tight_layout()
L.save(fig, f"{OUT}/fig8_reconstitution.png")

d = norec["DALYs_per_acquisition"] - full["DALYs_per_acquisition"]
print(f"  reconstitution worth {d:.2f} DALYs "
      f"({d / full['DALYs_avoided_through_care'] * 100:.0f}% of DALYs averted); "
      f"YLL {norec['YLL_cascade'] - full['YLL_cascade']:.2f}, "
      f"YLD {norec['YLD_cascade'] - full['YLD_cascade']:.2f}")

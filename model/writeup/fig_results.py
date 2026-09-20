"""Figures 3-6: calibration fit, cohort trajectory, DALY decomposition, tornado."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from figstyle import (apply_style, SURFACE, INK, INK2, MUTED, GRID, BLUE, ORANGE,
                      AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED, STATUS_GOOD,
                      despine, no_grid_x, no_grid_y)
from params import load_parameters, N_CD4_BANDS
import simulate_cascade as sc
from calibration_common import load_result
import figlabels as L
from run_posterior_analysis import load_posterior_result
from scenario_analysis import load_uncertainty_result
from states import (full_index, FULL_HIV_DEATH, FULL_BG_DEATH,
                    CASCADE_STATUSES, with_base)

apply_style()
p = load_parameters(); s1 = load_result("system1"); s2 = load_result("system2")
m1 = {r["param"]: r["mean"] for r in s1["summary"]}
c = {r["param"]: r["mean"] for r in s2["summary"]}
OUT = os.path.dirname(os.path.abspath(__file__))

# ===================== Figure 3: calibration fit ==========================
# Panel (b) used to plot p1*p2*p3 against 0.870*0.828*0.903 and call it a
# derived validation. It was neither derived from anything new nor a
# validation: the model value is the arithmetic product of three quantities
# already fitted and already plotted here, and the "target" is the product of
# their three targets, so the residual is just the sum of the three relative
# residuals. Worse, it can pass by cancellation -- +2 SD on p1 against -2 SD on
# p2 lands on the target while missing both. Removed; the panel that remains
# runs full width and separates what was held back from what was fitted.

FITTED = ["p1", "p2", "p3", "ART_1m", "CD4_median_ART", "CD4_lt200_ART"]

def _z(store, k):
    ce, sd = store["targets"][k]
    return (store["model_implied_at_mean"][k] - ce) / sd

held = [(L.target(k), _z(s1, k)) for k in L.HELD_BACK if k in s1["targets"]]
_missing = [k for k in FITTED if k not in s2["targets"]]
fitted = [(L.target(k), _z(s2, k)) for k in FITTED if k in s2["targets"]]
if _missing:
    print("  not fitted, omitted: " + ", ".join(_missing))

fig, ax = plt.subplots(figsize=(9.8, 4.4))
GAP = 1.0
rows = [(lab, z, "held") for lab, z in held] + [(lab, z, "fit") for lab, z in fitted]
ypos = []
y = len(rows) + GAP
for i, (_l, _z_, kind) in enumerate(rows):
    if i == len(held):
        y -= GAP
    y -= 1
    ypos.append(y)

ax.axvspan(-2, 2, color=STATUS_GOOD, alpha=0.07, zorder=0)
ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
for yi, (lab, zv, kind) in zip(ypos, rows):
    col = AQUA if kind == "held" else BLUE
    ax.plot([0, zv], [yi, yi], color=MUTED, lw=1.0, zorder=2, alpha=0.45)
    ax.scatter([zv], [yi], color=col, s=58, zorder=4,
               edgecolor=RED if abs(zv) > 2 else SURFACE,
               linewidth=1.8 if abs(zv) > 2 else 0.8)
    ax.text(zv + (0.16 if zv >= 0 else -0.16), yi, f"{zv:+.2f}", va="center",
            ha="left" if zv >= 0 else "right", fontsize=7.4, color=INK2)

sep = ypos[len(held) - 1] - GAP / 2
ax.axhline(sep, color=GRID, lw=1.0, zorder=1)

ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
ax.set_xlabel("Standardised residual  (model − target) / SD")
ax.set_xlim(-3.4, 3.4); ax.set_ylim(min(ypos) - 1.0, ypos[0] + 1.5)
no_grid_y(ax); despine(ax)
ax.legend(handles=[Patch(facecolor=AQUA, label="Held Back"),
                   Patch(facecolor=BLUE, label="Fitted"),
                   Line2D([], [], marker="o", ls="none", markerfacecolor=MUTED,
                          markeredgecolor=RED, markeredgewidth=1.8, markersize=7,
                          label="Misses By More Than 2 SD")],
          loc="lower right", fontsize=7.4)
fig.tight_layout()
L.save(fig, f"{OUT}/fig3_calibration.png")

# ===================== Figure 4: cohort trajectory ========================
# System 1 is no longer two fitted hazard parameters -- untreated progression
# and mortality are Glaubius tables scaled by kappa_prog / kappa_mort.
stack = sc.build_full_matrix_stack(c, p.fixed["recov_rate"], horizon_years=45,
                                   kappa_prog=m1["κ_prog"], kappa_mort=m1["κ_mort"])
traj = sc.propagate(stack, sc.undiagnosed_init(), 540)
t = np.arange(541) / 12

# Statuses carry a reconstitution-ceiling tag since the two-class refactor
# ("ART1_Suppressed__Full", "ART1_Suppressed__Capped"), so a bare base name is
# no longer a member of CASCADE_STATUSES -- full_index("ART1_Suppressed", b)
# raised ValueError. Sum over every tagged variant of each base instead.
def mass(bases):
    return sum(traj[:, full_index(st, b)]
               for st in with_base(*bases) for b in range(N_CD4_BANDS))

# ART1_Ramp did not exist when this figure was written, so its person-time was
# silently missing and the stack did not sum to the alive curve. It is folded
# in with ART1_Failing: together they are exactly the on-treatment states that
# carry the untreated mortality curve, which is the distinction that matters
# here. Keeping seven slots also preserves the validated adjacent-pair
# ordering of the categorical palette.
ORDER = [("ART1_Suppressed",), ("ART2_Suppressed",),
         ("ART1_Ramp", "ART1_Failing"), ("Diagnosed_PreART",),
         ("Undiagnosed",), ("LTFU_Recent",), ("LTFU_LongTerm",)]
NICE = {("ART1_Suppressed",): "ART First Line, Suppressed",
        ("ART2_Suppressed",): "ART Second Line",
        ("ART1_Ramp", "ART1_Failing"): "On ART, Not Suppressed",
        ("Diagnosed_PreART",): "Diagnosed, Pre-ART",
        ("Undiagnosed",): "Undiagnosed",
        ("LTFU_Recent",): "Disengaged, Recent",
        ("LTFU_LongTerm",): "Disengaged, Long-Term"}
COLS = [AQUA, GREEN, ORANGE, YELLOW, BLUE, MAGENTA, VIOLET]

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0),
                         gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
series = [mass(s) for s in ORDER]
ax.stackplot(t, *series, colors=COLS, labels=[NICE[s] for s in ORDER],
             edgecolor=SURFACE, linewidth=0.7, zorder=3)
alive = 1 - traj[:, FULL_HIV_DEATH] - traj[:, FULL_BG_DEATH]
ax.plot(t, alive, color=INK, lw=1.6, zorder=5)
ax.set_xlim(0, 45); ax.set_ylim(0, 1.02)
ax.set_xlabel("Years since acquisition"); ax.set_ylabel("Proportion of cohort")
handles, labs = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labs[::-1], loc="lower left", fontsize=6.9, ncol=2,
          bbox_to_anchor=(0.0, -0.015), framealpha=0.92, frameon=True,
          facecolor="white", edgecolor="none")
despine(ax); no_grid_x(ax)
ax.set_title("(a)", loc="left")

# (b) person-years by status
ax = axes[1]
pyrs = np.array([s.sum() / 12 for s in series])
yy = np.arange(len(ORDER))[::-1]
ax.barh(yy, pyrs, color=COLS, height=0.66, zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([NICE[s] for s in ORDER], fontsize=7.6)
for yi, v in zip(yy, pyrs):
    ax.text(v + 0.25, yi, f"{v:.1f}", va="center", fontsize=7.2, color=INK2)
ax.set_xlabel("Person-years over 45 years"); ax.set_xlim(0, max(pyrs) * 1.22)
no_grid_y(ax); despine(ax)
ax.set_title("(b)", loc="left")

fig.tight_layout()
L.save(fig, f"{OUT}/fig4_trajectory.png")

# ===================== Figure 5: DALY decomposition =======================
post = load_posterior_result()
byn = {r["metric"]: r for r in post["probabilistic"]["summary"]}
fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9),
                         gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
arms = ["No Care\n(Counterfactual)", "With Cascade"]
yll = [byn["YLL_natural_history"]["mean"], byn["YLL_cascade"]["mean"]]
yld = [byn["YLD_natural_history"]["mean"], byn["YLD_cascade"]["mean"]]
xa = np.arange(2)
ax.bar(xa, yll, 0.5, color=BLUE, label="Years Of Life Lost", zorder=3)
ax.bar(xa, yld, 0.5, bottom=np.array(yll) + 0.06, color=AQUA,
       label="Years Lived With Disability", zorder=3)
for xi, (a, b) in enumerate(zip(yll, yld)):
    ax.text(xi, a/2, f"{a:.1f}", ha="center", va="center", fontsize=8,
            color="white", fontweight="bold")
    ax.text(xi, a + b/2 + 0.06, f"{b:.1f}", ha="center", va="center", fontsize=8,
            color="white", fontweight="bold")
    ax.text(xi, a + b + 0.5, f"{a+b:.2f}", ha="center", fontsize=8.6,
            color=INK, fontweight="bold")
ax.set_xticks(xa); ax.set_xticklabels(arms, fontsize=8)
ax.set_ylabel("DALYs per acquisition, discounted at 3%")
ax.set_ylim(0, 19.5); ax.legend(loc="upper right", fontsize=7.6)
no_grid_x(ax); despine(ax)
ax.set_title("(a)", loc="left")

ax = axes[1]
METS = ["DALYs_per_acquisition", "DALYs_avoided_through_care", "DALYs_natural_history"]
NM = ["DALYs Per Acquisition\n(With Cascade)", "DALYs Averted\nBy The Cascade",
      "DALYs Per Acquisition\n(No Care)"]
yv = np.arange(3)[::-1]
for yi, k in zip(yv, METS):
    r = byn[k]
    ax.plot([r["q05"], r["q95"]], [yi, yi], color=BLUE, lw=2.4, zorder=3,
            solid_capstyle="round")
    ax.scatter([r["mean"]], [yi], color=BLUE, s=64, zorder=5,
               edgecolor="white", linewidth=1.6)
    ax.text(r["q95"] + 0.35, yi, f"{r['mean']:.2f}  [{r['q05']:.2f}, {r['q95']:.2f}]",
            va="center", fontsize=7.6, color=INK2)
ax.set_yticks(yv); ax.set_yticklabels(NM, fontsize=8)
ax.set_xlim(0, 25); ax.set_xlabel("DALYs, mean and 90% credible interval")
no_grid_y(ax); despine(ax)
ax.set_title("(b)", loc="left")

fig.tight_layout()
L.save(fig, f"{OUT}/fig5_dalys.png")

# ===================== Figure 6: tornado ==================================
unc = load_uncertainty_result()
rws = sorted(unc["rows"], key=lambda r: r["DALYs_per_acquisition_range"])
base = unc["baseline"]["DALYs_per_acquisition"]
fig, ax = plt.subplots(figsize=(8.4, 4.2))
yy = np.arange(len(rws))
for i, r in enumerate(rws):
    lo = r["DALYs_per_acquisition_at_q03"]; hi = r["DALYs_per_acquisition_at_q97"]
    left, right = min(lo, hi), max(lo, hi)
    ax.barh(i, right - left, left=left, height=0.62, color=BLUE, zorder=3)
    ax.text(right + 0.09, i, f"{r['DALYs_per_acquisition_range']:.2f}",
            va="center", fontsize=7.2, color=INK2)
ax.axvline(base, color=ORANGE, lw=1.8, zorder=4)
ax.set_yticks(yy)
ax.set_yticklabels([L.param(r["param"]) for r in rws], fontsize=8.4)
ax.set_xlabel("DALYs per acquisition")
no_grid_y(ax); despine(ax)
ax.set_xlim(4.2, 8.1)
fig.tight_layout()
L.save(fig, f"{OUT}/fig6_tornado.png")

"""Figure 1: model structure schematic, 107 states.

Two versions of the same diagram, annotated with the UNAIDS-anchored cohort at
one year and at twenty years:

    fig1_structure_1yr.png
    fig1_structure_20yr.png

Each box carries the share of the cohort in that state at the time point; each
arrow carries its per-cycle (monthly) transition probability, cohort-weighted
over the CD4 band mix of the people actually in the origin state.

Note what does and does not move between the two. The cascade hazards are
time-invariant, so the arrow probabilities are near-identical; what changes is
where the cohort SITS, plus diagnosis (the undiagnosed who remain are sicker,
and the hazard rises with severity) and mortality (band mix plus ageing).
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from figstyle import (apply_style, SURFACE, INK, INK2, MUTED, AXIS,
                      BLUE, ORANGE, AQUA, VIOLET, RED)
import simulate_cascade as sc
from params import load_parameters, N_CD4_BANDS
from states import full_index, with_base, FULL_HIV_DEATH, FULL_BG_DEATH
from calibration_common import load_result

apply_style()

# --------------------------------------------------------------------------
# Cohort state, base case (arm A posterior mean)
# --------------------------------------------------------------------------
p = load_parameters()
_s1 = load_result("system1"); _s2 = load_result("system2")
m1 = {r["param"]: r["mean"] for r in _s1["summary"]}
c = {r["param"]: r["mean"] for r in _s2["summary"]}
STACK = sc.build_full_matrix_stack(c, p.fixed["recov_rate"], horizon_years=45,
                                   kappa_prog=m1["κ_prog"], kappa_mort=m1["κ_mort"])
TRAJ = sc.propagate(STACK, sc.undiagnosed_init(), 540)


def _rows(base):
    return [full_index(st, k) for st in with_base(base) for k in range(N_CD4_BANDS)]


def occupancy(base, t):
    return float(TRAJ[t, _rows(base)].sum())


def flow_prob(origin, dest, t):
    """Per-cycle probability of origin -> dest, weighted by the band mix of the
    people in `origin` at time t. Returns None when origin is unoccupied."""
    M = STACK[min(int(t / 12), STACK.shape[0] - 1)]
    i = _rows(origin)
    w = TRAJ[t, i]
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    if dest == "death":
        return float(w @ (M[i, FULL_HIV_DEATH] + M[i, FULL_BG_DEATH]))
    return float((w @ M[np.ix_(i, _rows(dest))]).sum())


def pct(v):
    if v is None:
        return "--"
    if 0 < v < 0.001:
        return "<0.1%"
    return f"{v*100:.1f}%"


def prob(v):
    """Per-cycle probability -> probability of making the transition within a
    year, as a percentage. Reported rather than the raw hazard because the
    hazards for the fast transitions (initiation, suppression) exceed 100%/yr
    and read as nonsense on a diagram."""
    if v is None:
        return "--"
    annual = 1.0 - (1.0 - v) ** 12
    if 0 < annual < 0.001:
        return "<0.1%/yr"
    if annual > 0.999:
        return ">99.9%/yr"          # avoid the false precision of "100.0%"
    return f"{annual*100:.1f}%/yr"


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
W, H = 15.5, 12.0
TOP, MID, BOT = 40.0, 22.0, 4.0
XS = [1.0, 22.1, 43.2, 64.3, 85.4]

BOX = {   # key: x, y, label, mortality sublabel, colour, model base status
    "undx":  (XS[0], TOP, "Undiagnosed",              "Untreated Mortality", BLUE,   "Undiagnosed"),
    "dx":    (XS[1], TOP, "Diagnosed\nPre-ART",       "Untreated Mortality", BLUE,   "Diagnosed_PreART"),
    "ramp":  (XS[2], TOP, "On ART\nNot Suppressed",   "Untreated Mortality", ORANGE, "ART1_Ramp"),
    "art1s": (XS[3], TOP, "ART 1st Line\nSuppressed", "Background × SMR",    AQUA,   "ART1_Suppressed"),
    "art2s": (XS[4], TOP, "ART 2nd Line\nSuppressed", "Background × SMR",    AQUA,   "ART2_Suppressed"),
    "art1f": (XS[3], MID, "ART 1st Line\nFailing",    "Untreated Mortality", ORANGE, "ART1_Failing"),
    "ltfur": (XS[1], BOT, "Disengaged\nRecent",       "φ × Untreated",       VIOLET, "LTFU_Recent"),
    "ltful": (XS[2], BOT, "Disengaged\nLong-Term",    "φ × Untreated",       VIOLET, "LTFU_LongTerm"),
}

# origin key, dest key, plain-English label, model bases for the probability
ARROWS = [
    ("undx",  "dx",    "Diagnosis",              "Undiagnosed",      "Diagnosed_PreART"),
    ("undx",  "ramp",  "Same-Day Initiation",    "Undiagnosed",      "ART1_Ramp"),
    ("dx",    "ramp",  "ART Initiation",         "Diagnosed_PreART", "ART1_Ramp"),
    ("ramp",  "art1s", "Viral Suppression",      "ART1_Ramp",        "ART1_Suppressed"),
    ("art1s", "art1f", "Treatment Failure",      "ART1_Suppressed",  "ART1_Failing"),
    ("art1f", "art2s", "Switch To 2nd Line",     "ART1_Failing",     "ART2_Suppressed"),
    ("dx",    "ltfur", "Disengagement",          "Diagnosed_PreART", "LTFU_Recent"),
    ("art1s", "ltfur", "Disengagement",          "ART1_Suppressed",  "LTFU_Recent"),
    ("ltfur", "ltful", "Becomes Long-Term",      "LTFU_Recent",      "LTFU_LongTerm"),
    ("ltfur", "dx",    "Re-Engagement",          "LTFU_Recent",      "Diagnosed_PreART"),
    ("ltful", "dx",    "Late Re-Engagement",     "LTFU_LongTerm",    "Diagnosed_PreART"),
]


def port(key, side, f=0.5):
    x, y = BOX[key][0], BOX[key][1]
    if side == "L": return (x, y + f * H)
    if side == "R": return (x + W, y + f * H)
    if side == "T": return (x + f * W, y + H)
    if side == "B": return (x + f * W, y)


def render(t_months: int, tag: str, when: str):
    fig, ax = plt.subplots(figsize=(13.4, 7.4))
    ax.set_xlim(-1, 102); ax.set_ylim(0, 58); ax.axis("off")

    for x, y, lab, sub, col, base in BOX.values():
        ax.add_patch(FancyBboxPatch((x, y), W, H,
                                    boxstyle="round,pad=0.3,rounding_size=1.2",
                                    linewidth=1.7, edgecolor=col,
                                    facecolor="white", zorder=3))
        ax.text(x + W/2, y + H - 3.0, lab, ha="center", va="center", fontsize=8.4,
                color=INK, fontweight="bold", zorder=4, linespacing=1.3)
        ax.text(x + W/2, y + 4.6, pct(occupancy(base, t_months)), ha="center",
                va="center", fontsize=11.5, color=col, fontweight="bold", zorder=4)
        ax.text(x + W/2, y + 1.9, sub, ha="center", va="center", fontsize=6.3,
                color=MUTED, style="italic", zorder=4)

    def arrow(pa, pb, label, value, rad=0.0, lx=0.0, ly=0.0, t=0.5):
        """`t` slides the label along the chord (0 = origin end, 1 = target
        end). Long arrows put their label near the origin, where the diagram
        is emptier, instead of at the midpoint where the traffic crosses. The
        opaque bbox then keeps any residual overlap reading as occlusion
        rather than as two things printed on top of each other."""
        ax.add_patch(FancyArrowPatch(pa, pb, connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.3, color=AXIS, zorder=2,
                                     shrinkA=2, shrinkB=3))
        ax.text(pa[0] + t * (pb[0] - pa[0]) + lx,
                pa[1] + t * (pb[1] - pa[1]) + ly,
                f"{label}\n{value}", ha="center", va="center", fontsize=6.6,
                color=INK2, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"),
                zorder=6)

    P = {(a, b): prob(flow_prob(ob, db, t_months))
         for a, b, _l, ob, db in ARROWS}

    arrow(port("undx", "R"), port("dx", "L"), "Diagnosis", P[("undx", "dx")], ly=2.6)
    arrow(port("undx", "T", 0.55), port("ramp", "T", 0.45), "Same-Day Initiation",
          P[("undx", "ramp")], rad=-0.26, ly=3.4)
    arrow(port("dx", "R"), port("ramp", "L"), "ART Initiation", P[("dx", "ramp")], ly=2.6)
    arrow(port("ramp", "R"), port("art1s", "L"), "Viral Suppression",
          P[("ramp", "art1s")], ly=2.6)
    arrow(port("art1s", "B", 0.5), port("art1f", "T", 0.5), "Treatment Failure",
          P[("art1s", "art1f")], lx=7.0)
    arrow(port("art1f", "R", 0.78), port("art2s", "B", 0.42), "Switch To 2nd Line",
          P[("art1f", "art2s")], rad=-0.30, lx=7.4, ly=-1.4)
    arrow(port("dx", "B", 0.3), port("ltfur", "T", 0.4), "Disengagement",
          P[("dx", "ltfur")], lx=-6.4, t=0.30)
    arrow(port("art1s", "B", 0.14), port("ltfur", "R", 0.86), "Disengagement",
          P[("art1s", "ltfur")], rad=0.20, lx=-1.0, ly=-1.0, t=0.16)
    arrow(port("ltfur", "R", 0.45), port("ltful", "L", 0.45), "Becomes Long-Term",
          P[("ltfur", "ltful")], ly=2.5)
    arrow(port("ltfur", "T", 0.74), port("dx", "B", 0.72), "Re-Engagement",
          P[("ltfur", "dx")], lx=6.8, t=0.62)
    arrow(port("ltful", "T", 0.55), port("dx", "B", 0.94), "Late Re-Engagement",
          P[("ltful", "dx")], rad=0.30, lx=10.4, ly=-2.0, t=0.30)

    # absorbing states
    for x, y, lab, col, val in ((XS[4], 15.0, "HIV Death", RED,
                                 pct(float(TRAJ[t_months, FULL_HIV_DEATH]))),
                                (XS[4], 4.0, "Background Death", MUTED,
                                 pct(float(TRAJ[t_months, FULL_BG_DEATH])))):
        ax.add_patch(FancyBboxPatch((x, y), W, 8.4,
                                    boxstyle="round,pad=0.28,rounding_size=1.0",
                                    linewidth=1.7, edgecolor=col,
                                    facecolor="white", zorder=3))
        ax.text(x + W/2, y + 5.8, lab, ha="center", va="center", fontsize=8.0,
                color=INK, fontweight="bold", zorder=4)
        ax.text(x + W/2, y + 2.4, val, ha="center", va="center", fontsize=11.0,
                color=col, fontweight="bold", zorder=4)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"fig1_structure_{tag}.png")
    # transparent=True drops the figure and axes patches; the state boxes keep
    # their explicit white fill, and so do the label pills, so the diagram
    # stays legible on any background it is dropped onto.
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"{out} written")


if __name__ == "__main__":
    render(12, "1yr", "1 year")
    render(240, "20yr", "20 years")

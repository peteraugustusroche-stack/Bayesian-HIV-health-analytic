"""Figure 10: what the per-acquisition burden implies for long-acting PrEP.

  (a) the targeting frontier -- incremental cost per DALY averted against how many
      person-years of PrEP it takes to avert one acquisition, at four all-in annual
      prices, with the cost-saving region shaded and two trial cohorts marked

  (b) the denominator -- the same scenario valued with per-acquisition burdens taken
      from the literature instead of from a calibrated cascade

  (c) the offset -- the same scenario as the analysis credits more of the treatment
      cost a prevented acquisition avoids

    python3 writeup/fig_prevention.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from figstyle import (apply_style, INK, INK2, MUTED, AXIS, BLUE, ORANGE, AQUA,
                      YELLOW, VIOLET, STATUS_GOOD, despine, no_grid_x)
import figlabels as L
import prevention_analysis as PA

apply_style()
OUT = os.path.dirname(os.path.abspath(__file__))

dalys, cost, nocare = PA.model_quantities()
D, C = dalys.mean(), cost.mean()
NOCARE = nocare.mean()

REF_NNT, REF_PRICE = 50, 60          # the scenario panels (b) and (c) hold fixed
dollars = FuncFormatter(lambda v, _p: f"${v:,.0f}")

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3),
                         gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})

# ---------------- (a) the targeting frontier ----------------
ax = axes[0]
nnt = np.linspace(10, 160, 600)
COLS = {25: AQUA, 40: BLUE, 60: VIOLET, 100: ORANGE}

ax.axhspan(-400, 0, color=STATUS_GOOD, alpha=0.09, zorder=1)
ax.axhline(0, color=AXIS, lw=1.0, zorder=2)
ax.text(13, -378, "Cost-Saving", ha="left", va="bottom", fontsize=8.4,
        color=STATUS_GOOD, fontweight="bold")

for price, col in COLS.items():
    y = (nnt * price - C) / D
    ax.plot(nnt, y, color=col, lw=2.0, zorder=4, label=f"${price} per year")

for inc, lab, _src in PA.ANCHORS:
    n = 100.0 / inc
    if 10 <= n <= 160:
        ax.axvline(n, color=MUTED, lw=0.9, ls=":", zorder=3)
        ax.text(n + 2.2, 930, lab.split(",")[0], rotation=90, fontsize=7.2,
                color=INK2, va="top", ha="left")

ax.set_xlabel("Person-years of PrEP per acquisition averted")
ax.set_ylabel("Incremental cost per DALY averted")
ax.set_xlim(10, 160)
ax.set_ylim(-400, 950)
ax.yaxis.set_major_formatter(dollars)
ax.legend(loc="lower right", fontsize=8, title="All-in annual cost",
          title_fontsize=8, frameon=False)
despine(ax)

# ---------------- (b) the denominator ----------------
ax = axes[1]
net = REF_NNT * REF_PRICE - cost
rows = [(D, f"This model,\n{PA.CASCADE_YEAR} cascade", BLUE)]
rows += [(v, lab.replace("A commonly used ", "").replace(
              "Representative sub-Saharan African value in current use",
              "Representative\nSSA value in use").replace(
              "round figure", "Round figure\nin common use"), MUTED)
         for v, lab in PA.LITERATURE_DALYS]
rows += [(NOCARE, "This model,\nno-care counterfactual", MUTED)]

y = np.arange(len(rows))[::-1]
vals = [float((net / d).mean()) for d, _, _ in rows]
cols = [c for _, _, c in rows]
ax.barh(y, vals, 0.62, color=cols, zorder=3)
for yi, v, (d, _, _) in zip(y, vals, rows):
    ax.text(v + 8, yi, f"${v:,.0f}", va="center", fontsize=8.6, color=INK)
    ax.text(4, yi + 0.34, f"{d:.1f} DALYs", va="bottom", fontsize=7.4, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels([lab for _, lab, _ in rows], fontsize=7.8)
ax.set_xlabel("Incremental cost per DALY averted")
ax.set_xlim(0, max(vals) * 1.30)
ax.xaxis.set_major_formatter(dollars)
ax.set_title(f"Same scenario, different denominator", fontsize=9, color=INK, pad=8)
no_grid_x(ax); despine(ax, keep=("bottom",))
ax.tick_params(axis="y", length=0)

# ---------------- (c) the offset ----------------
ax = axes[2]
share = np.linspace(0, 1, 200)
y = (REF_NNT * REF_PRICE - share * C) / D
ax.plot(share * 100, y, color=BLUE, lw=2.2, zorder=4)
ax.scatter([0, 100], [(REF_NNT * REF_PRICE) / D, (REF_NNT * REF_PRICE - C) / D],
           s=46, color=BLUE, edgecolor="white", linewidth=1.3, zorder=6)
ax.annotate("Treatment cost\nignored", xy=(0, (REF_NNT * REF_PRICE) / D),
            xytext=(14, (REF_NNT * REF_PRICE) / D - 20), fontsize=7.6, color=INK2,
            va="top")
ax.annotate("Fully credited", xy=(100, (REF_NNT * REF_PRICE - C) / D),
            xytext=(96, (REF_NNT * REF_PRICE - C) / D + 45), fontsize=7.6,
            color=INK2, ha="right")
ax.set_xlabel("Share of avoided treatment cost credited, %")
ax.set_ylabel("Incremental cost per DALY averted")
ax.set_xlim(0, 100)
ax.set_ylim(0, (REF_NNT * REF_PRICE) / D * 1.15)
ax.yaxis.set_major_formatter(dollars)
ax.set_title("Same scenario, with and without the offset", fontsize=9,
             color=INK, pad=8)
despine(ax)

fig.tight_layout()
L.save(fig, f"{OUT}/fig10_prevention.png")

print(f"  reference scenario: NNT {REF_NNT}, ${REF_PRICE}/yr all-in")
print(f"  this model     {D:5.2f} DALYs -> ${float((net/dalys).mean()):,.0f}/DALY")
for v, lab in PA.LITERATURE_DALYS:
    print(f"  literature     {v:5.2f} DALYs -> ${float((net/v).mean()):,.0f}/DALY")
print(f"  no-care        {NOCARE:5.2f} DALYs -> ${float((net/nocare).mean()):,.0f}/DALY")
print(f"  offset ignored -> ${REF_NNT*REF_PRICE/D:,.0f}/DALY; "
      f"fully credited -> ${(REF_NNT*REF_PRICE-C)/D:,.0f}/DALY")

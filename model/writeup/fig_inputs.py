"""Figure 2: sourced model inputs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
from figstyle import (apply_style, SURFACE, INK, INK2, MUTED, BLUE, ORANGE,
                      AQUA, RED, despine, no_grid_x, no_grid_y)
import acquisition_cd4 as acq, life_table as lt
from params import CD4_BAND_NAMES, ANCHOR_AGE
from hazards import on_art_excess_curve, cap_on_art_at_untreated
from natural_history_params import mortality_curve_at_age
from params import load_parameters
import figlabels as L

apply_style()

from matplotlib.ticker import FuncFormatter, LogLocator

# Hazards are plotted as annual mortality PERCENT: easier to read than a bare
# rate, and the log axis still carries the three orders of magnitude between
# background mortality and untreated HIV at CD4 <50.
_PCT = FuncFormatter(lambda v, _p: (f"{v:g}%" if v >= 1 else f"{v:.3g}%"))

p = load_parameters()
mc = mortality_curve_at_age()          # Glaubius 2021, cohort's age group
d = acq.initial_band_distribution()

fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))

# ---- (a) acquisition CD4 -------------------------------------------------
ax = axes[0]
y = np.arange(len(CD4_BAND_NAMES))
ax.barh(y, d * 100, color=BLUE, height=0.62, zorder=3)
ax.set_yticks(y); ax.set_yticklabels(CD4_BAND_NAMES)
ax.invert_yaxis(); ax.set_xlabel("% of cohort at seroconversion")
ax.set_xlim(0, 68); no_grid_y(ax); despine(ax)
for i, v in enumerate(d * 100):
    ax.text(v + 1.4, i, f"{v:.1f}", va="center", fontsize=7.4, color=INK2)
ax.set_title("(a)", loc="left")

# ---- (b) mortality by CD4 band -------------------------------------------
ax = axes[1]
bg29 = lt.background_hazard(ANCHOR_AGE)
# Both curves contain EXACT ZEROS -- Glaubius reports no untreated HIV
# mortality in the top two CD4 bands, and since the SMR floor was removed the
# on-ART excess at CD4>=500 is exactly zero (SMR 1.00, Rodger et al.). A zero
# is at -infinity on a log axis: matplotlib clips zeros out of a line quietly,
# but an ARTIST anchored to one (the annotation below used to sit at
# mc[1] * 0.35 = 0.0) makes bbox_inches="tight" expand without bound, which is
# what produced "Image size 3035x297884 pixels is too large". Mask them.
# ALL THREE SERIES ARE TOTAL ALL-CAUSE MORTALITY, not the excesses the code
# stores. Internally the model keeps HIV-attributable hazards on the hiv_death
# channel (the Glaubius curve when untreated, (SMR - 1) x bg when suppressed)
# and background separately on bg_death, so the arrays hold excesses. Plotting
# an excess for one series and a total for another -- as an earlier version of
# this panel did -- makes untreated HIV at CD4 250-349 look LOWER than on-ART
# mortality, which is an artefact of the mismatch. Totals throughout.
#
# THE ON-ART SERIES IS THE CAPPED CURVE, i.e. what the model actually applies.
# simulate_cascade builds the on-ART hazard as
#     cap_on_art_at_untreated(on_art_excess_curve(bg), untreated)
# and this panel has to do the same. Plotting the raw SMR ladder instead drew
# on-ART above untreated at CD4 350-499 (0.380% against 0.211%) and 250-349
# (0.486% against 0.411%) -- the two bands where the cap binds -- which
# contradicted both the input table in the write-up and the model's own
# guarantee that treatment is never represented as harmful at a given CD4.
#
# Whether the raw crossing is epidemiologically absurd is a separate question,
# and it is not: Rodger's SMR is measured among TREATED patients whose achieved
# CD4 is in that band, who reached it by reconstituting from a lower nadir,
# whereas the Glaubius curve is for someone DECLINING through it who has never
# been immunosuppressed. Those are different populations. The cap is a
# STRUCTURAL choice, not an epidemiological correction: within the model both
# hazards attach to the same state, so an uncapped ladder would make treatment
# actively harmful at CD4 350-499. It is worth 0.33 DALYs per acquisition at
# posterior means (6.00 capped against 6.33 uncapped) because that band holds
# 30.4% of on-ART person-time, so it belongs in the structural sensitivity
# analysis rather than being passed over silently.
#
# No masking needed once these are totals: adding background lifts the two
# zero-hazard Glaubius bands off the log axis floor.
_untr_total = (mc + bg29) * 100.0
_art_total = (cap_on_art_at_untreated(on_art_excess_curve(bg29), mc) + bg29) * 100.0
ax.plot(y, _untr_total, "o-", color=ORANGE,
        label=f"Untreated At Age {ANCHOR_AGE:.0f}", zorder=4)
ax.plot(y, _art_total, "s--", color=AQUA,
        label=f"On ART, Suppressed, At Age {ANCHOR_AGE:.0f}", zorder=4)
ax.plot(y, np.full_like(mc, bg29) * 100.0, "^:", color=MUTED,
        label=f"Background At Age {ANCHOR_AGE:.0f}", zorder=4)
ax.set_yscale("log"); ax.set_xticks(y)
ax.set_xticklabels(CD4_BAND_NAMES, rotation=45, ha="right", fontsize=7)
ax.set_xlabel("CD4 band, cells/µL")
ax.set_ylabel("Annual mortality, % (log scale)")
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_formatter(_PCT)
ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ""))
ax.legend(loc="upper left", fontsize=7.2)
despine(ax); no_grid_x(ax)
ax.set_title("(b)", loc="left")

# ---- (c) life table ------------------------------------------------------
ax = axes[2]
ages = np.arange(29, 91)
haz = np.array([lt.background_hazard(a) for a in ages])
ax.plot(ages, haz * 100.0, color=BLUE, zorder=4, label="Cause-Deleted Hazard")
ax.set_yscale("log"); ax.set_xlabel("Age, years")
ax.set_ylabel("Annual non-HIV mortality, % (log scale)")
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_formatter(_PCT)
ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ""))
despine(ax); no_grid_x(ax)
ax.set_title("(c)", loc="left")

fig.tight_layout()
L.save(fig, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig2_inputs.png"))

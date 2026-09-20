"""Shared display labels and save helper for the write-up figures.

Figures show what a quantity IS, not what the code calls it. Every symbol that
appears in a figure gets its plain-English name here, so the naming is
consistent across panels and there is one place to change it.

Conventions applied throughout:
  * Title Case for panel titles, legends, category labels and annotations
  * sentence case for axis labels
  * no "Figure N" titles -- the caption lives in the document, not the image
  * transparent canvas; boxes and label pills keep an explicit white fill so
    they stay legible on whatever background they are placed over
"""
from __future__ import annotations

# Calibrated cascade parameters -------------------------------------------
PARAM_LABEL = {
    "δ_bg":         "Background Testing Rate",
    "δ_symp":       "Symptomatic Diagnosis Rate",
    "γ_diag":       "Diagnosis CD4 Gradient",
    "λ_init":       "ART Initiation Rate",
    "μ_vf1":        "First-Line Failure Rate",
    "μ_ltfu":       "Disengagement Rate",
    "μ_re-engage":  "Re-Engagement Rate",
    "μ_switch":     "Switch To Second Line Rate",
    "φ_ltfu":       "Disengaged Mortality Multiplier",
    "κ_prog":       "CD4 Progression Scale",
    "κ_mort":       "Untreated Mortality Scale",
}

# Calibration targets ------------------------------------------------------
TARGET_LABEL = {
    "p1":             "Diagnosed, Of All Living With HIV",
    "p2":             "On Treatment, Of Those Diagnosed",
    "p3":             "Virally Suppressed, Of Those Treated",
    "ART_1m":         "Started ART Within One Month",
    "CD4_median_ART": "Median CD4 At ART Start",
    "CD4_lt200_ART":  "CD4 Below 200 At ART Start",
    "Supp_2L":        "Resuppression On Second Line",
    "T_<350":         "Years To CD4 Below 350",
    "T_<200":         "Years To CD4 Below 200",
    "T_survival":     "Years Of Untreated Survival",
    "Ret_12m":        "Retention At 12 Months",
    "Ret_24m":        "Retention At 24 Months",
    "Reeng_1y":       "Re-Engagement Within 1 Year",
    "Reeng_2y":       "Re-Engagement Within 2 Years",
    "Reeng_3y":       "Re-Engagement Within 3 Years",
    "HR_LTFU":        "Mortality Hazard Ratio, Disengaged",
}

# Health-economic outputs --------------------------------------------------
METRIC_LABEL = {
    "DALYs_per_acquisition":      "DALYs Per Acquisition",
    "YLL_cascade":                "Years Of Life Lost",
    "YLD_cascade":                "Years Lived With Disability",
    "DALYs_natural_history":      "DALYs Without Care",
    "YLL_natural_history":        "Years Of Life Lost, No Care",
    "YLD_natural_history":        "Years Lived With Disability, No Care",
    "DALYs_avoided_through_care": "DALYs Averted By Care",
    "cost_per_acquisition":       "Lifetime Cost Per Acquisition",
    "cost_recurring":             "Recurring Cost",
    "cost_death":                 "Terminal Care Cost",
}

# Natural-history targets are held back: System 1 is drawn from its priors
# rather than fitted, so these are genuine out-of-sample checks. Everything
# else in the target list was in the likelihood.
HELD_BACK = ("T_<350", "T_<200", "T_survival")


def param(sym: str) -> str:
    return PARAM_LABEL.get(sym, sym)


def target(sym: str) -> str:
    return TARGET_LABEL.get(sym, sym)


def metric(sym: str) -> str:
    return METRIC_LABEL.get(sym, sym)


def save(fig, path: str, dpi: int = 300) -> None:
    """Transparent canvas, tight crop, no figure-level title."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    print(f"{path} written")

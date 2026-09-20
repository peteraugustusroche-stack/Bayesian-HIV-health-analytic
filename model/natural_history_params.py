"""System 1 natural history: SOURCED tables, not calibrated quantities.

Progression and mortality are adopted from Glaubius et al. 2021 Table 3.
Uncertainty is carried as two multiplicative factors on their level. MANGAL_* is
a cross-check for sensitivity, not the base case. See PARAMETERS.md.
"""

from __future__ import annotations

import numpy as np

from params import N_CD4_BANDS, ANCHOR_AGE

# Glaubius Table 3 age groups. The oldest is open-ended (45+ applies at every
# age above 45, as in Spectrum). The cohort is age-synchronised, so the age
# group in model year y is known exactly with no distributional assumption.
AGE_GROUP_LOWER = (15.0, 25.0, 35.0, 45.0)
AGE_GROUP_NAMES = ("15-24", "25-34", "35-44", "45+")

# --- Glaubius 2021 Table 3 -------------------------------------------------
# Published as events (or deaths) per 100 person-years; stored here as annual
# rates per person-year. Rows are age groups, columns are CD4 bands in the
# model's band order.

# Progression to the next lower CD4 category off ART. The <50 column is zero
# (lowest category, no onward progression).
#
# NB the non-monotonicity at 200-249 -> 100-199: 200-249 is a 50-cell band and
# 100-199 a 100-cell band, so the per-band exit rate drops even though CD4
# decline does not slow. A hazard monotone in band index structurally cannot
# reproduce this.
GLAUBIUS_PROGRESSION = np.array([
    [15.3, 25.1, 34.8,  65.1, 29.4, 50.0, 0.0],   # 15-24
    [18.3, 28.6, 39.5,  74.0, 33.5, 56.9, 0.0],   # 25-34
    [22.5, 33.3, 46.1,  86.2, 39.0, 66.4, 0.0],   # 35-44
    [28.6, 40.2, 55.6, 104.0, 47.1, 80.0, 0.0],   # 45+
]) / 100.0

# HIV/AIDS mortality off ART. Exactly zero in the top two bands as published:
# Glaubius attributes no AIDS mortality above CD4 350, so the model books no
# HIV deaths (and no YLL) at CD4 counts where the evidence records none.
GLAUBIUS_MORTALITY = np.array([
    [0.0, 0.0, 0.2, 0.7, 4.8, 22.7,  71.2],       # 15-24
    [0.0, 0.0, 0.2, 1.0, 6.4, 30.5,  95.4],       # 25-34
    [0.0, 0.0, 0.3, 1.2, 8.0, 38.2, 119.7],       # 35-44
    [0.0, 0.0, 0.3, 1.5, 9.7, 45.9, 143.9],       # 45+
]) / 100.0

# CD4 at seroconversion, per cent. NOT ADOPTED -- the model uses
# acquisition_cd4.initial_band_distribution() (Pantazis 2012), which is
# SSA-specific where this is not. Retained for the annex comparison, where the
# two agree closely -- an independent check on the acquisition distribution.
GLAUBIUS_INITIAL_CD4 = np.array([
    [60.5, 20.8, 10.8, 3.6, 3.7, 0.5, 0.1],       # 15-24
    [57.4, 21.9, 11.8, 4.1, 4.2, 0.6, 0.1],       # 25-34
    [53.9, 22.9, 13.0, 4.6, 4.8, 0.7, 0.1],       # 35-44
    [50.3, 23.8, 14.3, 5.2, 5.5, 0.8, 0.1],       # 45+
]) / 100.0

# --- Mangal 2017 Tables 3 and 4, Africa (SENSITIVITY ONLY) -----------------
# Annual rates at the baseline age group (15-24). Mortality age effects are
# published as hazard ratios, not a full age x band table, so the age dimension
# is applied multiplicatively. Progression is sex-averaged (sex effect is small).
MANGAL_PROGRESSION = np.array([0.191, 0.317, 0.324, 0.490, 0.552, 0.835, 0.0])
MANGAL_MORTALITY = np.array([0.003, 0.009, 0.0075, 0.013, 0.022, 0.038, 0.862])
MANGAL_MORT_AGE_HR = (1.0, 1.25, 1.55, 2.22)   # <25 ref; 25-34; 35-44; >45

# --- System 1 uncertainty --------------------------------------------------
# Two multiplicative factors on the tables above, each lognormal with median 1.
# Puts uncertainty on their LEVEL (what transports least well) while preserving
# their published SHAPE (what the evidence constrains well), and keeps System 1's
# draw two-dimensional so the downstream cut machinery is unchanged.

# Glaubius's own CrIs imply ~0.03; rounded up because one common factor is
# standing in for seven separately-estimated cells.
KAPPA_PROG_SIGMA = 0.05

# Glaubius's own CrIs imply ~0.09. DELIBERATELY WIDENED: the honest uncertainty
# is between-source disagreement on where untreated mortality sits (Mangal vs
# Glaubius), which exceeds either source's internal precision.
KAPPA_MORT_SIGMA = 0.15

SYSTEM1_SYMBOLS = ["κ_prog", "κ_mort"]


def age_group_index(age: float) -> int:
    """Glaubius age-group row for an absolute age. Ages above 45 all use the
    open-ended 45+ row, as Spectrum does."""
    idx = 0
    for i, lower in enumerate(AGE_GROUP_LOWER):
        if age >= lower:
            idx = i
    return idx


def progression_schedule(kappa_prog: float = 1.0, n_years: int = 71,
                          anchor_age: float = ANCHOR_AGE,
                          table: np.ndarray | None = None) -> np.ndarray:
    """(n_years, N_CD4_BANDS) annual CD4-progression hazards, one row per year of
    model time, stepping the cohort through Glaubius age groups as it ages.

    Age is resolved from model time because the cohort is age-synchronised.
    Progression accelerates with age as well as with CD4 depletion: holding the
    cohort at its entry age overstates survival by about a year (12.42 vs 11.42)."""
    tab = GLAUBIUS_PROGRESSION if table is None else table
    rows = [tab[age_group_index(anchor_age + y)] for y in range(int(n_years))]
    return np.asarray(rows, dtype=float) * float(kappa_prog)


def untreated_mortality_schedule(kappa_mort: float = 1.0, n_years: int = 71,
                                  anchor_age: float = ANCHOR_AGE,
                                  table: np.ndarray | None = None) -> np.ndarray:
    """(n_years, N_CD4_BANDS) annual untreated HIV mortality hazards, one row
    per year of model time. Same age-stepping as progression_schedule."""
    tab = GLAUBIUS_MORTALITY if table is None else table
    rows = [tab[age_group_index(anchor_age + y)] for y in range(int(n_years))]
    return np.asarray(rows, dtype=float) * float(kappa_mort)


def mangal_tables() -> tuple[np.ndarray, np.ndarray]:
    """Mangal 2017 Africa rates expanded to the Glaubius (4, N_CD4_BANDS) shape,
    for sensitivity analysis. Progression is published without an age breakdown
    so is held constant across groups; mortality is scaled by the published HRs."""
    prog = np.tile(MANGAL_PROGRESSION, (4, 1))
    mort = np.outer(np.asarray(MANGAL_MORT_AGE_HR), MANGAL_MORTALITY)
    return prog, mort


def mortality_curve_at_age(age: float = ANCHOR_AGE, kappa_mort: float = 1.0,
                            table: np.ndarray | None = None) -> np.ndarray:
    """Single-age untreated mortality vector, for short-horizon micro targets and
    diagnostics. Anything running over model time wants
    untreated_mortality_schedule instead."""
    tab = GLAUBIUS_MORTALITY if table is None else table
    return tab[age_group_index(age)] * float(kappa_mort)


assert GLAUBIUS_PROGRESSION.shape == (4, N_CD4_BANDS)
assert GLAUBIUS_MORTALITY.shape == (4, N_CD4_BANDS)
assert np.all(GLAUBIUS_PROGRESSION[:, -1] == 0.0), "no progression out of <50"

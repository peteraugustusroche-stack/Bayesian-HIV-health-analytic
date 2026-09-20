"""Cost and disability-weight inputs for the health-economic outputs.

Costs are adopted wholesale from Hyle et al. 2025 Table S6 and restated in the
analysis price year by US CPI-U. Disability weights are a continuous function of
CD4, RECONSTRUCTED against GBD benchmarks since the original formula was not
specified. Sourcing and caveats: PARAMETERS.md.
"""

from __future__ import annotations

import math

from params import N_CD4_BANDS

# ---------------------------------------------------------------------------
# Costs (2024 USD)
# ---------------------------------------------------------------------------

# --- price-year adjustment -------------------------------------------------
# Every unit cost is recorded at its SOURCE's price year and converted to the
# analysis year by US all-items CPI-U (BLS, 1982-84 = 100), per CHEERS 2022
# item 15. A US index rather than local-currency deflation: the sources do not
# report the original local-currency values.
CPI_U = {2008: 215.303, 2009: 214.537, 2010: 218.056, 2011: 224.939,
         2012: 229.594, 2013: 232.957, 2014: 236.736, 2015: 237.017,
         2016: 240.007, 2017: 245.120, 2018: 251.107, 2019: 255.657,
         2020: 258.811, 2021: 270.970, 2022: 292.655, 2023: 304.702,
         2024: 313.689}

# Analysis price year. Set per analysis by set_price_year(); the base case
# matches the price year to the cascade year, so the 2019-anchored analysis
# reports 2019 USD and the 2024-anchored analysis reports 2024 USD.
PRICE_YEAR = 2024


def inflate(value: float, from_year: int, to_year: int) -> float:
    """Convert a nominal USD value between price years using US CPI-U."""
    return value * CPI_U[to_year] / CPI_U[from_year]


# --- source values, at their own price years -------------------------------
# ALL unit costs are adopted from ONE source (Hyle et al. 2025 Table S6), whose
# CD4-stratified routine care is BUILT from resource utilisation by CD4 rather
# than asserted -- an auditable chain the previous mixed set did not have.
# CAVEAT: a Malawi cost set, at the low end of the ESA range, so conservative on
# cost and favourable on cost-effectiveness. See PARAMETERS.md.

# First-line ART drugs: TDF/3TC + DTG, $3.60/month.
SRC_COST_1L_ART_PER_YEAR = (3.60 * 12, 2023)

# Second-line ART drugs: AZT/3TC + LPV/r, $18.57/month.
SRC_COST_2L_ART_PER_YEAR = (18.57 * 12, 2023)

# Routine HIV care by CD4 band, monthly. Hyle's six strata map onto this model's
# seven bands exactly (their 200-349 spans this model's 250-349 and 200-249).
#
# REPLACES both previous non-drug terms: Hyle's routine care already contains
# the service-delivery component (the outpatient visit) and the advanced-disease
# component (inpatient days), so keeping the old flat service-delivery cost
# alongside it would double-count the clinic visit.
_HYLE_ROUTINE_MONTHLY = [2.86, 3.96, 4.90, 4.90, 10.77, 18.51, 26.24]
SRC_ROUTINE_CARE_BY_BAND = ([v * 12 for v in _HYLE_ROUTINE_MONTHLY], 2023)
assert len(SRC_ROUTINE_CARE_BY_BAND[0]) == N_CD4_BANDS

# One-off terminal-care cost at death, any cause; not carried by the routine
# schedule.
SRC_COST_DEATH = (104.26, 2023)

# EVERY living state, including Undiagnosed -- this model's one deliberate
# departure from Hyle's accounting, who charge routine care to people in care.
# The underlying utilisation is driven by immunosuppression, and people are
# treated for advanced HIV before anyone records them as having it. Excluding
# the undiagnosed would understate the cost of late diagnosis, a quantity this
# model exists to estimate. Charging in-care states only is reported as a scenario.
ROUTINE_CARE_STATUSES = {"Undiagnosed", "Diagnosed_PreART", "ART1_Ramp",
                         "ART1_Suppressed", "ART1_Failing", "ART2_Suppressed",
                         "LTFU_Recent", "LTFU_LongTerm"}

# RETIRED, kept as names so imports don't break. Service delivery is subsumed
# into the routine-care schedule; Hyle carry no ART initiation cost, so the
# previous unsourced $50 is gone rather than sitting in a fully sourced set.
HIV_CARE_COST_STATUSES = ROUTINE_CARE_STATUSES        # backwards-compatible alias
SERVICE_DELIVERY_STATUSES: set = set()

# NOT ADOPTED (no state to attach them to): Hyle's per-incident opportunistic
# infection costs, diagnostic test costs, and TB/cryptococcal modules, all of
# which they charge ON TOP of routine care. Omitting them makes this model's
# cost per acquisition a LOWER BOUND -- state in the limitations.

# --- effective (price-year adjusted) values --------------------------------
# What the rest of the model reads, recomputed by set_price_year().
# health_economics imports this MODULE, not these names, so updates propagate.
COST_1L_ART_PER_YEAR = 0.0
COST_2L_ART_PER_YEAR = 0.0
COST_SERVICE_DELIVERY_PER_YEAR = 0.0        # retired, always 0
COST_ART_STARTUP = 0.0                       # retired, always 0
COST_DEATH = 0.0
ROUTINE_CARE_BY_BAND = []
HIV_CARE_COST_BY_BAND = []                   # backwards-compatible alias


def set_price_year(year: int) -> None:
    """Restate every unit cost in `year` US dollars."""
    global PRICE_YEAR, COST_1L_ART_PER_YEAR, COST_2L_ART_PER_YEAR
    global COST_SERVICE_DELIVERY_PER_YEAR, COST_ART_STARTUP, COST_DEATH
    global ROUTINE_CARE_BY_BAND, HIV_CARE_COST_BY_BAND
    if year not in CPI_U:
        raise ValueError(f"no CPI-U entry for {year}; have {min(CPI_U)}-{max(CPI_U)}")
    PRICE_YEAR = year
    v, y = SRC_COST_1L_ART_PER_YEAR;   COST_1L_ART_PER_YEAR = inflate(v, y, year)
    v, y = SRC_COST_2L_ART_PER_YEAR;   COST_2L_ART_PER_YEAR = inflate(v, y, year)
    v, y = SRC_COST_DEATH;             COST_DEATH = inflate(v, y, year)
    vs, y = SRC_ROUTINE_CARE_BY_BAND
    ROUTINE_CARE_BY_BAND = [inflate(v, y, year) for v in vs]
    HIV_CARE_COST_BY_BAND = ROUTINE_CARE_BY_BAND
    COST_SERVICE_DELIVERY_PER_YEAR = 0.0
    COST_ART_STARTUP = 0.0


set_price_year(PRICE_YEAR)

# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------

DISCOUNT_RATE = 0.03               # WHO-CHOICE reference case, discrete compounding

# NO LONGER USED FOR YLL, which now comes from the cause-deleted ESA life table
# (life_table.remaining_life_expectancy) so the YLL reference and background
# mortality share a source. Kept as a comparator: the old `72 - age_at_death`
# agrees at the anchor age but diverges sharply after (at 65 it scored 7 years
# lost against a true e_x of 14.0).
REFERENCE_LIFE_EXPECTANCY_LEGACY = 72.0

# ---------------------------------------------------------------------------
# Disability weights
# ---------------------------------------------------------------------------
#
# Specified in the source table as CONTINUOUS functions of CD4, but the exact
# formula and the units of "Slope" were not given. RECONSTRUCTED here as linear
# in CD4 deficit from a 500 cells/uL anchor, in units of 8 cells per slope-step:
#     DW(CD4) = clip(Min + Slope * (500 - CD4) / 8, Min, Max)
#
# The divisor of 8 is a FIT to the table's own cited GBD AIDS benchmark (~0.55)
# at low CD4, NOT the original formula. If the real one surfaces, swap it in --
# everything downstream reads through disability_weight().

DW_ONART_MIN, DW_ONART_MAX, DW_ONART_SLOPE = 0.04, 0.19, 0.010
DW_OFFART_MIN, DW_OFFART_MAX, DW_OFFART_SLOPE = 0.08, 0.57, 0.008
CD4_SLOPE_UNIT_CELLS = 8.0
CD4_DW_ANCHOR = 500.0

# Representative CD4 per band for the continuous DW function. Band 0 uses 575
# to match the acquisition-CD4 median used elsewhere; the rest are midpoints.
CD4_BAND_REPRESENTATIVE = [575.0, 425.0, 300.0, 225.0, 150.0, 75.0, 25.0]
assert len(CD4_BAND_REPRESENTATIVE) == N_CD4_BANDS


def disability_weight(band: int, on_art: bool) -> float:
    cd4 = CD4_BAND_REPRESENTATIVE[band]
    deficit_steps = (CD4_DW_ANCHOR - cd4) / CD4_SLOPE_UNIT_CELLS
    if on_art:
        dw = DW_ONART_MIN + DW_ONART_SLOPE * deficit_steps
        return min(max(dw, DW_ONART_MIN), DW_ONART_MAX)
    else:
        dw = DW_OFFART_MIN + DW_OFFART_SLOPE * deficit_steps
        return min(max(dw, DW_OFFART_MIN), DW_OFFART_MAX)


if __name__ == "__main__":
    from params import CD4_BAND_NAMES
    print("Reconstructed disability weights by band:")
    print(f"{'band':<10}{'on-ART':>10}{'off-ART':>10}")
    for b, name in enumerate(CD4_BAND_NAMES):
        print(f"{name:<10}{disability_weight(b, True):>10.3f}{disability_weight(b, False):>10.3f}")

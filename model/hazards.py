"""Hazard functions shared by the System 1 and System 2 transition engines,
plus the competing-risks machinery turning simultaneous annual hazards into
monthly transition probabilities.

COMPETING RISKS. For a state with simultaneous exit hazards h_1..h_k, total
hazard Lambda = sum(h_i). Over dt years the probability of exiting at all is
1 - exp(-Lambda*dt), and conditional on exiting, the probability it was via
reason i is h_i / Lambda. Exact under the assumption that all h_i are constant
over the step, which is reasonable at dt = 1/12. Resolving sequentially or
hierarchically instead would be wrong.
"""

from __future__ import annotations

import math

import numpy as np

from params import (
    N_CD4_BANDS,
    CD4_BAND_WIDTH,
    ANCHOR_AGE,
    EXCESS_HAZARD_ON_ART,
    ART_SMR, ART_SMR_FLOOR,
    LTFU_MORTALITY_MULTIPLIER,
)
import life_table


def competing_risks_step(hazards: dict, dt: float) -> dict:
    """
    hazards: {label: annual_hazard} for every possible exit from a state
             (excluding "stay").
    dt: step length in years.

    Returns {label: probability} for every exit, plus "stay" for the
    probability of remaining in the state, summing to exactly 1.
    """
    total = sum(hazards.values())
    if total <= 0:
        probs = {label: 0.0 for label in hazards}
        probs["stay"] = 1.0
        return probs

    p_exit = 1.0 - math.exp(-total * dt)
    probs = {label: p_exit * (h / total) for label, h in hazards.items()}
    probs["stay"] = 1.0 - p_exit
    return probs


# Untreated mortality and CD4 decline now live in natural_history_params, as
# sourced age x band tables rather than the interpolated curve and parametric
# hazard that used to be here.


def reconstitution_hazard(band: int, recov_rate_cells_per_yr: float) -> float:
    """Annual hazard of climbing from CD4 band `band` to `band-1`, converting
    a cells/µL/yr recovery rate into a band-crossing hazard using the width
    (in cells) of the band being left. No upward hazard out of band 0."""
    if band == 0:
        return 0.0
    width = CD4_BAND_WIDTH[band]
    return recov_rate_cells_per_yr / width


def choose_recon_rate(band: int, recov_rate_fast: float, recov_rate_slow: float) -> float:
    """Faster reconstitution below CD4 200 (band >= 4), slower above.

    A simplification of the biphasic recovery curve (Lawn et al. 2006); refining
    it needs time-since-suppression, which this state space does not track."""
    return recov_rate_fast if band >= 4 else recov_rate_slow


def diagnosis_hazard(band: int, delta_bg: float, delta_symp: float,
                      gamma_shape: float = 1.0) -> float:
    """Symptomatic-presentation hazard ramping smoothly across all 7 bands.

    severity(band) = (band / (N_CD4_BANDS-1)) ** gamma_shape runs 0 at band 0
    (delta_symp contributes nothing) to 1 at band 6 (full delta_symp).
    gamma_shape = 1 is linear; < 1 front-loads it; > 1 concentrates it in the
    lowest bands, approaching a step function while staying continuous.

    gamma_shape exists because a hard step at band >= 4 forced one delta_symp to
    both match the CD4-at-ART-initiation distribution and leave delta_bg able to
    hit diagnosis coverage; it could not do both (p1 implied 0.70 against a 0.93
    target). gamma_shape is a third lever over WHERE in the CD4 range the
    symptomatic bump sits, not just its magnitude."""
    severity = (band / (N_CD4_BANDS - 1)) ** gamma_shape
    return delta_bg + delta_symp * severity


def background_mortality(age: float | None = None) -> float:
    """Annual non-HIV (cause-deleted) death hazard at `age`, from the ESA life
    table.

    `age` is ABSOLUTE, not time since cohort entry -- callers holding model time
    t should pass ANCHOR_AGE + t, or use background_mortality_at_model_time().
    Defaults to the anchor age so fixed-age callers get a sensible value rather
    than a flat placeholder.
    """
    return life_table.background_hazard(ANCHOR_AGE if age is None else age)


def background_mortality_at_model_time(t_years: float) -> float:
    """Background hazard at `t_years` since cohort entry. Valid only because the
    cohort is age-synchronised."""
    return life_table.background_hazard(ANCHOR_AGE + t_years)


_ART_SMR_EXCESS = np.array([max(v, ART_SMR_FLOOR) - 1.0 for v in ART_SMR])
_ART_SMR_EXCESS_CENTRAL = _ART_SMR_EXCESS.copy()


def smr_excess_for_multiplier(m: float) -> np.ndarray:
    """Excess array for an SMR ladder scaled by `m`.

    The multiplier scales the SMR, NOT the excess. Scaling the excess would pin
    the >=500 band at zero for every draw (its SMR is exactly 1.0, so its excess
    is 0) -- and that is precisely the band Rodger's 1.00 (0.69-1.40) measures.
    Scaling the SMR gives (m x SMR - 1) x background, so the top band spans the
    published interval. m < 1 yields a small NEGATIVE excess, which is what the
    lower half of that interval says; total hazard stays positive across the
    plausible range of m, so nothing is clamped.
    """
    return np.array([max(v, ART_SMR_FLOOR) * float(m) - 1.0 for v in ART_SMR])


def set_smr_multiplier(m: float | None) -> None:
    """Scale the module-level SMR ladder used by on_art_excess_curve.

    Stateful by necessity: the excess array is read at matrix-build time deep
    inside simulate_cascade. Callers MUST restore with set_smr_multiplier(None)
    in a finally block, or every later evaluation inherits the last draw.
    """
    global _ART_SMR_EXCESS
    _ART_SMR_EXCESS = (_ART_SMR_EXCESS_CENTRAL.copy() if m is None
                       else smr_excess_for_multiplier(m))


def cap_on_art_at_untreated(onart_excess, untreated_excess):
    """Elementwise cap: on-ART HIV mortality never exceeds untreated HIV
    mortality at the same CD4 band.

    Both arrays are HIV-attributable excesses over the same background channel,
    so capping the excess caps total mortality. Without it the model asserts
    that a suppressed person at CD4 350-499 dies faster than an untreated person
    at the same CD4 -- Rodger's SMR 1.77 against a Glaubius schedule reporting
    zero HIV mortality above 350 -- and no mechanism supports that. The observed
    elevation in treated cohorts at moderate CD4 is confounding by prior
    immunosuppression: a treated person at CD4 400 recovered there from a nadir,
    an untreated person at CD4 400 is early and undamaged (Engsig et al. 2010,
    doi:10.1186/1471-2334-10-318, locate it in >1 year of prior suppression).

    A cohort study may report a confounded marginal association; a
    state-transition model may not, because the hazard on a state is a claim
    about whoever occupies it. This is a causal constraint, not a convenience.

    COST: overrides Rodger's 1.77 at 350-499 (~27% of on-ART person-time),
    setting it to 1.00. Bites only at 350-499 and 250-349, worth ~0.012 deaths
    per acquisition. Conditioning on the reconstitution class instead -- the
    confounder the state space already tracks -- is noted as further work.
    """
    return np.minimum(onart_excess, untreated_excess)


def on_art_excess_curve(bg_hazard: float) -> np.ndarray:
    """(SMR(band) - 1) x background, for all CD4 bands at once.

    Keyed on ACHIEVED (current) CD4, so someone suppressed but not yet
    reconstituted carries the mortality of their actual CD4 rather than the
    mortality of a reconstituted patient. See params.ART_SMR.
    """
    return _ART_SMR_EXCESS * float(bg_hazard)


def on_art_excess_hazard(band: int | None = None,
                          bg_hazard: float | None = None) -> float:
    """HIV-attributable EXCESS mortality hazard while virally suppressed.

    Returns the excess only -- callers apply it to the `hiv_death` channel
    alongside a separate `bg_death` of background_mortality(age), so the two
    sum to SMR x background with no double counting.

    With (band, bg_hazard) returns the CD4-specific excess; with no arguments
    the flat EXCESS_HAZARD_ON_ART, which is the >=500 ANCHOR ONLY, not a curve.
    """
    if band is None or bg_hazard is None:
        return EXCESS_HAZARD_ON_ART
    return float(_ART_SMR_EXCESS[band] * bg_hazard)


def ltfu_mortality(m_untreated_band: float, phi: float | None = None) -> float:
    """HIV mortality hazard while lost to follow-up: the untreated
    CD4-specific hazard scaled by phi (params.LTFU_MORTALITY_MULTIPLIER when
    not given explicitly, preserving the previous fixed-input behaviour).

    phi = 1.0 means disengaged patients face exactly never-treated mortality,
    which sits at one extreme of the evidence. LTFU dominates HIV deaths here,
    so this is a high-leverage input, not a detail.
    """
    return (LTFU_MORTALITY_MULTIPLIER if phi is None else phi) * m_untreated_band

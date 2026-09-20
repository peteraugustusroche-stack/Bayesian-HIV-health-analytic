"""CD4 distribution at HIV acquisition, and the ESA natural-history targets
derived from the same source (Pantazis et al. 2012).

Sourcing, derivations and caveats: PARAMETERS.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from params import (
    N_CD4_BANDS, CD4_BAND_LOWER, CD4_BAND_UPPER, CD4_BAND_NAMES,
)

# Pantazis 2012 Table 3, adjusted model: (point, ci_low, ci_high).
PANTAZIS = {
    "intercept_ref":  (23.53, 23.08, 23.97),
    "intercept_ssa":  (-0.76, -1.28, -0.25),
    "intercept_female": (1.11, 0.65, 1.57),
    "slope_ref":      (-1.49, -1.69, -1.30),
    "slope_ssa":      (0.34, 0.15, 0.54),
    # Validation check only; not used to build the model.
    "intercept_euro_african": (-2.97, -3.69, -2.25),
    "slope_euro_african": (0.51, 0.21, 0.81),
}

# ESA epidemic composition (UNAIDS). Sex enters the intercept only.
PCT_FEMALE = 0.63

# Lower truncation on the acquisition distribution, cells/uL. A modelling
# judgement, not a sourced value; vary it in sensitivity.
TRUNCATION_FLOOR_CELLS = 100.0

# Upper bound for integrating the open-ended top band.
_TRUNCATION_CEILING_CELLS = 2000.0

# Spread of the acquisition distribution on the sqrt scale. FIXED, not
# calibrated -- nothing in the target set identifies it cleanly.
ACQUISITION_SIGMA0 = 5.0

# Lodi et al. 2011 Table 2: proportion below each CD4 threshold at 1, 2 and 5
# years. Identifies the SHAPE of the acquisition distribution only.
LODI_PROPORTION_TARGETS = {
    # (threshold cells/uL, years) -> proportion below
    (200, 1): 0.088, (200, 2): 0.122, (200, 5): 0.323,
    (350, 1): 0.261, (350, 2): 0.332, (350, 5): 0.550,
    (500, 1): 0.480, (500, 2): 0.559, (500, 5): 0.727,
}
# Widens Lodi's published CIs, which are far too tight to use directly.
LODI_TARGET_SD_FRAC = 0.25


def sqrt_cd4_mean(pct_female: float = PCT_FEMALE) -> float:
    """Mean sqrt(CD4) at acquisition for an ESA cohort of the given sex mix."""
    return (PANTAZIS["intercept_ref"][0]
            + PANTAZIS["intercept_ssa"][0]
            + pct_female * PANTAZIS["intercept_female"][0])


def sqrt_cd4_slope() -> float:
    """Annual decline in sqrt(CD4) for an ESA cohort (negative)."""
    return PANTAZIS["slope_ref"][0] + PANTAZIS["slope_ssa"][0]


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def initial_band_distribution() -> np.ndarray:
    """The acquisition CD4 distribution at the model's fixed settings, cached."""
    global _CACHED_INITIAL
    if _CACHED_INITIAL is None:
        _CACHED_INITIAL = band_distribution(ACQUISITION_SIGMA0)
    return _CACHED_INITIAL


_CACHED_INITIAL = None


def band_distribution(sigma0: float, pct_female: float = PCT_FEMALE,
                       floor_cells: float = TRUNCATION_FLOOR_CELLS) -> np.ndarray:
    """Probability mass over the 7 CD4 bands at acquisition, healthiest first.

    sqrt(CD4) ~ Normal(sqrt_cd4_mean(), sigma0), truncated and renormalised.
    """
    if sigma0 <= 0:
        raise ValueError(f"sigma0 must be positive, got {sigma0}")

    mu = sqrt_cd4_mean(pct_female)
    lo_t, hi_t = math.sqrt(floor_cells), math.sqrt(_TRUNCATION_CEILING_CELLS)
    denom = _normal_cdf(hi_t, mu, sigma0) - _normal_cdf(lo_t, mu, sigma0)
    if denom <= 0:
        raise ValueError("Truncation bounds exclude all probability mass")

    probs = np.zeros(N_CD4_BANDS)
    for b in range(N_CD4_BANDS):
        lo = max(math.sqrt(CD4_BAND_LOWER[b]), lo_t)
        hi = min(math.sqrt(min(CD4_BAND_UPPER[b], _TRUNCATION_CEILING_CELLS)), hi_t)
        if hi <= lo:
            continue
        probs[b] = (_normal_cdf(hi, mu, sigma0) - _normal_cdf(lo, mu, sigma0)) / denom

    total = probs.sum()
    return probs / total if total > 0 else probs


def mean_cd4_at_acquisition(sigma0: float, pct_female: float = PCT_FEMALE,
                            floor_cells: float = TRUNCATION_FLOOR_CELLS) -> float:
    """Band-weighted mean CD4 (cells/uL) from band midpoints. Diagnostic only."""
    probs = band_distribution(sigma0, pct_female, floor_cells)
    mids = [(CD4_BAND_LOWER[b] + min(CD4_BAND_UPPER[b], 1200.0)) / 2.0
            for b in range(N_CD4_BANDS)]
    return float(sum(p * m for p, m in zip(probs, mids)))


def time_to_threshold(threshold_cells: float, pct_female: float = PCT_FEMALE) -> float:
    """Years for the average sqrt(CD4) trajectory to reach `threshold_cells`."""
    mu, slope = sqrt_cd4_mean(pct_female), sqrt_cd4_slope()
    return (mu - math.sqrt(threshold_cells)) / (-slope)


def derive_target_sd(threshold_cells: float, pct_female: float = PCT_FEMALE,
                      n_draws: int = 200_000, seed: int = 0) -> dict:
    """Propagate Pantazis's 95% CIs into an SD for the time-to-threshold target.

    Coefficients are treated as INDEPENDENT (no published covariance), which
    probably overstates the SD -- see PARAMETERS.md. Use time_to_threshold()
    for the central value and this only for the SD.
    """
    rng = np.random.default_rng(seed)

    def draw(key):
        pt, lo, hi = PANTAZIS[key]
        return rng.normal(pt, (hi - lo) / 3.92, n_draws)

    intercept = draw("intercept_ref") + draw("intercept_ssa") + pct_female * draw("intercept_female")
    slope = -(draw("slope_ref") + draw("slope_ssa"))
    t = (intercept - math.sqrt(threshold_cells)) / slope
    return {
        "point": time_to_threshold(threshold_cells, pct_female),
        "mc_mean": float(t.mean()),
        "sd": float(t.std()),
        "q025": float(np.quantile(t, 0.025)),
        "q975": float(np.quantile(t, 0.975)),
    }


ESA_TARGET_THRESHOLDS = {"T_<500": 500.0, "T_<350": 350.0, "T_<200": 200.0}


def esa_natural_history_targets(pct_female: float = PCT_FEMALE) -> dict:
    """symbol -> (central, sd) ESA natural-history targets, shaped like
    params.Parameters.targets."""
    out = {}
    for sym, th in ESA_TARGET_THRESHOLDS.items():
        d = derive_target_sd(th, pct_female)
        out[sym] = (d["point"], d["sd"])
    return out


if __name__ == "__main__":
    print("VALIDATION against quantities Pantazis reports independently")
    print("-" * 62)
    I, S, F = PANTAZIS["intercept_ref"][0], PANTAZIS["slope_ref"][0], PANTAZIS["intercept_female"][0]
    groups = {
        "non-African European women": (I + F, S),
        "Europe-African origin women": (I + PANTAZIS["intercept_euro_african"][0] + F,
                                         S + PANTAZIS["slope_euro_african"][0]),
        "SSA women": (I + PANTAZIS["intercept_ssa"][0] + F, S + PANTAZIS["slope_ssa"][0]),
    }
    print(f"{'group':<30}{'CD4 at SC':>11}{'4yr loss':>10}{'T_<350':>9}")
    for g, (i, s) in groups.items():
        print(f"{g:<30}{i*i:>11.0f}{i*i - (i + 4*s)**2:>10.0f}{(i - math.sqrt(350)) / -s:>9.2f}")
    print("paper reports:                     607/469/570   259/155/199")
    delay = (groups["SSA women"][0] - math.sqrt(350)) / -groups["SSA women"][1] \
            - (groups["non-African European women"][0] - math.sqrt(350)) / -groups["non-African European women"][1]
    print(f"SSA delay to CD4<350: {delay*12:.1f} months   (paper: 'almost 6 months later')")

    print(f"\nESA COHORT ({PCT_FEMALE:.0%} female, truncated at {TRUNCATION_FLOOR_CELLS:.0f} cells/uL)")
    print("-" * 62)
    print(f"  mean sqrt(CD4) at acquisition : {sqrt_cd4_mean():.3f}  ({sqrt_cd4_mean()**2:.0f} cells/uL)")
    print(f"  decline                       : {sqrt_cd4_slope():.2f} sqrt-cells/yr")

    print("\nDERIVED TARGETS (central from point estimate, SD from CI propagation)")
    for sym, th in ESA_TARGET_THRESHOLDS.items():
        d = derive_target_sd(th)
        print(f"  {sym:<10} {d['point']:.2f} +/- {d['sd']:.2f} yr"
              f"   [MC mean {d['mc_mean']:.2f}, 95% {d['q025']:.2f}-{d['q975']:.2f}]")
    print("  (previous Lodi-derived targets: T_<350 4.19 +/- 0.4, T_<200 7.93 +/- 0.5)")

    print("\nACQUISITION CD4 BAND DISTRIBUTION by sigma0")
    sigmas = (3.0, 5.0, 7.0, 9.0)
    print(f"{'band':<10}" + "".join(f"{'s=%.0f' % s:>9}" for s in sigmas))
    dists = [band_distribution(s) for s in sigmas]
    for b, name in enumerate(CD4_BAND_NAMES):
        print(f"{name:<10}" + "".join(f"{d[b]*100:>8.1f}%" for d in dists))
    print(f"{'mean CD4':<10}" + "".join(f"{mean_cd4_at_acquisition(s):>9.0f}" for s in sigmas))
    print(f"\n(previous model behaviour: 100% in band 0, i.e. the sigma0 -> 0 limit)")

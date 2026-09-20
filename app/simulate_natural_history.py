"""
System 1: Natural History engine.

Pure untreated progression (no diagnosis/ART at all) -- deliberately kept
separate from the full cascade model so it can be calibrated in isolation,
per the sequential design (System 1 posterior locked before System 2 sees
any health-system data).

ACQUISITION CD4: the cohort no longer starts as a point mass in band 0. It
enters spread across CD4 bands per acquisition_cd4.initial_band_distribution(),
built from the Pantazis 2012 sub-Saharan African seroconverter estimates. About
59% start at CD4 >=500, 17% below 350 and 3% below 200.

AGE-VARYING RATES: three of the four hazards in this engine now vary with the
cohort's absolute age, which the model knows exactly because the cohort is
age-synchronised -- everyone enters at params.ANCHOR_AGE, so age = ANCHOR_AGE
+ t.

- Background (non-HIV) mortality is read from the cause-deleted ESA life
  table by age.
- CD4 progression and untreated HIV mortality are read from the Glaubius 2021
  age x CD4-band tables (natural_history_params.py), stepping the cohort
  through age groups as it ages.

So the engine works on a STACK of matrices, one per year of model time. One
year is the resolution of the underlying life table and of the Glaubius age
groups, so per-month matrices would be false precision (and ~12x slower for
no information gain).

PARAMETERS: this engine no longer takes lambda_base and alpha_shape. Those
defined a parametric CD4-decline hazard that was fitted here; progression is
now a sourced table. What remains are two multiplicative factors on the
published schedules, kappa_prog and kappa_mort, which carry System 1's
uncertainty without re-estimating its shape. See natural_history_params.
"""

from __future__ import annotations

import math

import numpy as np

from params import N_CD4_BANDS, CD4_BAND_NAMES, ANCHOR_AGE
from states import N_NH_STATES, NH_HIV_DEATH, NH_BG_DEATH
from acquisition_cd4 import initial_band_distribution
from hazards import competing_risks_step, background_mortality
from natural_history_params import (
    progression_schedule, untreated_mortality_schedule,
)


def build_nh_transition_matrix(progression_row, mortality_row,
                                bg_mortality, dt=1 / 12) -> np.ndarray:
    """9x9 monthly transition probability matrix for the natural-history-only
    chain: 7 CD4 bands + HIV Death + Background Death.

    `progression_row` and `mortality_row` are the CD4-band hazard vectors for
    ONE age group, and `bg_mortality` the background hazard for ONE age. Use
    build_nh_matrix_stack() for the full age-varying schedule.
    """
    M = np.zeros((N_NH_STATES, N_NH_STATES))

    for band in range(N_CD4_BANDS):
        hazards = {
            "hiv_death": float(mortality_row[band]),
            "bg_death": bg_mortality,
        }
        decline_h = float(progression_row[band])
        if decline_h > 0:
            hazards["decline"] = decline_h

        probs = competing_risks_step(hazards, dt)

        M[band, band] = probs["stay"]
        M[band, NH_HIV_DEATH] = probs["hiv_death"]
        M[band, NH_BG_DEATH] = probs["bg_death"]
        if "decline" in probs:
            M[band, band + 1] = probs["decline"]

    M[NH_HIV_DEATH, NH_HIV_DEATH] = 1.0
    M[NH_BG_DEATH, NH_BG_DEATH] = 1.0

    assert np.allclose(M.sum(axis=1), 1.0), "Transition matrix rows must sum to 1"
    return M


def build_nh_matrix_stack(kappa_prog=1.0, kappa_mort=1.0,
                           dt=1 / 12, horizon_years=45,
                           anchor_age=ANCHOR_AGE,
                           prog_table=None, mort_table=None) -> np.ndarray:
    """Stack of shape (n_years, 9, 9): one transition matrix per year of
    model time, each using the CD4 progression and untreated-mortality rows
    for the cohort's Glaubius age group that year, and the background hazard
    for its exact age. Index with year = int(t_years).

    prog_table / mort_table override the Glaubius tables -- used by the
    sensitivity analysis to substitute the Mangal 2017 Africa rates."""
    n_years = int(math.ceil(horizon_years))
    prog = progression_schedule(kappa_prog, n_years, anchor_age, prog_table)
    mort = untreated_mortality_schedule(kappa_mort, n_years, anchor_age, mort_table)
    return np.stack([
        build_nh_transition_matrix(prog[y], mort[y],
                                    background_mortality(anchor_age + y), dt)
        for y in range(n_years)
    ])


def propagate(M, init_vec: np.ndarray, n_steps: int, dt=1 / 12) -> np.ndarray:
    """Return trajectory of shape (n_steps+1, n_states).

    `M` is either a single (n, n) matrix applied at every step, or a stack
    of shape (n_years, n, n) indexed by year of model time -- the latter is
    what the age-varying background produces. Years beyond the end of the
    stack reuse its final matrix.
    """
    M = np.asarray(M)
    time_varying = M.ndim == 3
    n_states = M.shape[-1]

    traj = np.zeros((n_steps + 1, n_states))
    traj[0] = init_vec
    for t in range(n_steps):
        step_M = M[min(int(t * dt), M.shape[0] - 1)] if time_varying else M
        traj[t + 1] = traj[t] @ step_M
    return traj


def first_passage_median_years(threshold_band: int, kappa_prog=1.0,
                                kappa_mort=1.0, dt=1 / 12,
                                horizon_years=40,
                                anchor_age=ANCHOR_AGE,
                                prog_table=None, mort_table=None) -> float | None:
    """Median time (years) for the cohort to first reach `threshold_band`,
    as a cumulative-incidence function that properly accounts for the
    competing risk of dying before reaching it.

    All three hazards vary with age across the passage window (ages ~29-40
    for the CD4<350 / CD4<200 targets), so this builds a per-year matrix
    stack rather than a single matrix.
    """
    n = threshold_band + 3
    REACHED, HIV_D, BG_D = threshold_band, threshold_band + 1, threshold_band + 2
    n_years = int(math.ceil(horizon_years))
    prog = progression_schedule(kappa_prog, n_years, anchor_age, prog_table)
    mort = untreated_mortality_schedule(kappa_mort, n_years, anchor_age, mort_table)

    def _matrix_for(progression_row, mortality_row, bg_mortality):
        M = np.zeros((n, n))
        for band in range(threshold_band):
            hazards = {"hiv_death": float(mortality_row[band]),
                       "bg_death": bg_mortality}
            decline_h = float(progression_row[band])
            if decline_h > 0:
                hazards["decline"] = decline_h
            probs = competing_risks_step(hazards, dt)

            M[band, band] = probs["stay"]
            M[band, HIV_D] = probs["hiv_death"]
            M[band, BG_D] = probs["bg_death"]
            if "decline" in probs:
                target = band + 1 if band + 1 < threshold_band else REACHED
                M[band, target] = probs["decline"]

        M[REACHED, REACHED] = 1.0
        M[HIV_D, HIV_D] = 1.0
        M[BG_D, BG_D] = 1.0
        assert np.allclose(M.sum(axis=1), 1.0)
        return M

    stack = np.stack([_matrix_for(prog[y], mort[y],
                                   background_mortality(anchor_age + y))
                      for y in range(n_years)])

    n_steps = int(horizon_years / dt)
    # Cohort enters spread across CD4 bands (acquisition_cd4). Anyone acquiring
    # HIV already at or below the threshold band has, by definition, reached the
    # threshold at time zero, so their mass starts in REACHED.
    init = np.zeros(n)
    acq = initial_band_distribution()
    init[:threshold_band] = acq[:threshold_band]
    init[REACHED] = acq[threshold_band:].sum()
    traj = propagate(stack, init, n_steps, dt)
    cum_reached = traj[:, REACHED]

    idx = np.argmax(cum_reached >= 0.5)
    if cum_reached[idx] < 0.5:
        return None  # never crosses 50% within horizon
    return idx * dt


def overall_survival_median_years(kappa_prog=1.0, kappa_mort=1.0,
                                   dt=1 / 12, horizon_years=40,
                                   anchor_age=ANCHOR_AGE,
                                   prog_table=None, mort_table=None) -> float | None:
    stack = build_nh_matrix_stack(kappa_prog, kappa_mort, dt, horizon_years,
                                   anchor_age, prog_table, mort_table)
    n_steps = int(horizon_years / dt)
    init = np.zeros(N_NH_STATES)
    init[:N_CD4_BANDS] = initial_band_distribution()
    traj = propagate(stack, init, n_steps, dt)
    alive = 1.0 - traj[:, NH_HIV_DEATH] - traj[:, NH_BG_DEATH]

    idx = np.argmax(alive <= 0.5)
    if alive[idx] > 0.5:
        return None
    return idx * dt


if __name__ == "__main__":
    from natural_history_params import (
        AGE_GROUP_NAMES, GLAUBIUS_MORTALITY, GLAUBIUS_PROGRESSION,
    )

    print("Adopted Glaubius 2021 schedules (annual hazards):")
    for label, tab in (("progression", GLAUBIUS_PROGRESSION),
                       ("untreated mortality", GLAUBIUS_MORTALITY)):
        print(f"\n  {label}")
        print("    " + "band".ljust(10) + "".join(g.rjust(10) for g in AGE_GROUP_NAMES))
        for i, name in enumerate(CD4_BAND_NAMES):
            print("    " + name.ljust(10) + "".join(f"{tab[g, i]:10.4f}" for g in range(4)))

    t350 = first_passage_median_years(2)
    t200 = first_passage_median_years(4)
    surv = overall_survival_median_years()

    print(f"\nValidation (no fitted parameters):")
    print(f"  median time to CD4<350: {t350:.2f} yr  (target 4.14 ± 0.64)")
    print(f"  median time to CD4<200: {t200:.2f} yr  (target 8.11 ± 1.11)")
    print(f"  median untreated survival: {surv:.2f} yr  (target 11.50 ± 0.50)")

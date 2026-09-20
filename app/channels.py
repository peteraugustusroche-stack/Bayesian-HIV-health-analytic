"""Collapse a Markov run onto the channels the repricing engine multiplies.

Shared by the offline exporter (export_for_app.py, over the whole posterior)
and the app's live cascade re-run (one parameter set at a time), so both
produce channels the engine treats identically.
"""
from __future__ import annotations

import numpy as np

import life_table
import simulate_cascade as casc
import simulate_natural_history as nh
from acquisition_cd4 import initial_band_distribution
from health_economics import HORIZON_YEARS
from params import ANCHOR_AGE, N_CD4_BANDS
from states import (CASCADE_STATUSES, FULL_BG_DEATH, FULL_HIV_DEATH,
                    N_FULL_STATES, N_NH_STATES, NH_HIV_DEATH, base_of,
                    full_index)

DT = 1 / 12
N_STEPS = int(HORIZON_YEARS / DT)
N_YEARS = int(HORIZON_YEARS)
N_MOMENTS = 3

#: The eight living cascade bases. The engine charges cost against a subset of
#: these, so person-time is kept at base-status resolution.
BASE_STATUSES = ["Undiagnosed", "Diagnosed_PreART", "ART1_Ramp",
                 "ART1_Suppressed", "ART1_Failing", "ART2_Suppressed",
                 "LTFU_Recent", "LTFU_LongTerm"]
#: Bases taking the on-ART disability weight: on treatment AND suppressed.
ON_ART_BASES = ["ART1_Suppressed", "ART2_Suppressed"]

#: Within-year offset of each month, u = k/12, and its powers.
_U = (np.arange(N_STEPS) % 12) / 12.0
_UPOW = np.stack([_U ** j for j in range(N_MOMENTS)], axis=-1)


def moments(monthly: np.ndarray) -> np.ndarray:
    """(n_steps, ...) monthly series -> (n_years, ..., 3) within-year moments."""
    flat = monthly.reshape(N_STEPS, -1)
    m = (flat[:, :, None] * _UPOW[:, None, :]).reshape(
        N_YEARS, 12, -1, N_MOMENTS).sum(axis=1)
    return m.reshape(N_YEARS, *monthly.shape[1:], N_MOMENTS)


def _base_selector():
    """(n_bases, n_full_states): collapse onto base status, ignoring CD4. Used
    to attribute a death to the state the person occupied when they died."""
    sel = np.zeros((len(BASE_STATUSES), N_FULL_STATES))
    pos = {b: i for i, b in enumerate(BASE_STATUSES)}
    for status in CASCADE_STATUSES:
        i = pos[base_of(status)]
        for band in range(N_CD4_BANDS):
            sel[i, full_index(status, band)] = 1.0
    return sel


def _selectors():
    sel = np.zeros((len(BASE_STATUSES), N_CD4_BANDS, N_FULL_STATES))
    pos = {b: i for i, b in enumerate(BASE_STATUSES)}
    for status in CASCADE_STATUSES:
        i = pos[base_of(status)]
        for band in range(N_CD4_BANDS):
            sel[i, band, full_index(status, band)] = 1.0
    return sel.reshape(len(BASE_STATUSES) * N_CD4_BANDS, N_FULL_STATES)


_SEL = _selectors()
_SEL_BASE = _base_selector()


def cascade_monthly(cascade_vals, kappa_prog, kappa_mort, params):
    """Monthly person-time by base status x band, and the death flows."""
    stack = casc.build_full_matrix_stack(cascade_vals, params.fixed["recov_rate"],
                                         DT, HORIZON_YEARS,
                                         kappa_prog=kappa_prog, kappa_mort=kappa_mort)
    traj = casc.propagate(stack, casc.undiagnosed_init(), N_STEPS, DT)[:N_STEPS]
    py = (traj @ _SEL.T) * DT

    year_of = np.minimum((np.arange(N_STEPS) * DT).astype(int), stack.shape[0] - 1)
    living = traj[:, :N_FULL_STATES - 2]
    hiv_state = living * stack[year_of][:, :N_FULL_STATES - 2, FULL_HIV_DEATH]
    hiv_flow = hiv_state.sum(axis=1)
    bg_flow = (living * stack[year_of][:, :N_FULL_STATES - 2, FULL_BG_DEATH]).sum(axis=1)
    # A draw whose SMR multiplier puts the top-band SMR below 1 gives a NEGATIVE
    # on-ART HIV-death hazard. health_economics guards YLL accumulation with
    # `if death_flow > 0` but charges terminal cost against the unguarded flow,
    # so the clamped and raw channels are both carried.
    #
    # The by-state channel clamps each state SEPARATELY, because a negative
    # segment in a decomposition is meaningless. Its total therefore exceeds
    # the total-clamped channel above by about 3%, the offsetting negatives
    # having been removed; engine.daly_by_state rescales the shares back to
    # the published YLL and returns the factor so the figure can report it.
    hiv_by_base = np.maximum(hiv_state[:, :N_FULL_STATES - 2], 0.0) @ _SEL_BASE[:, :N_FULL_STATES - 2].T
    return (py.reshape(N_STEPS, len(BASE_STATUSES), N_CD4_BANDS),
            hiv_flow, bg_flow, np.maximum(hiv_flow, 0.0), hiv_by_base)


def nh_monthly(kappa_prog, kappa_mort):
    stack = nh.build_nh_matrix_stack(kappa_prog, kappa_mort, DT, HORIZON_YEARS)
    init = np.zeros(N_NH_STATES)
    init[:N_CD4_BANDS] = initial_band_distribution()
    traj = nh.propagate(stack, init, N_STEPS, DT)[:N_STEPS]
    year_of = np.minimum((np.arange(N_STEPS) * DT).astype(int), stack.shape[0] - 1)
    out = stack[year_of][:, :N_CD4_BANDS, NH_HIV_DEATH]
    death = (traj[:, :N_CD4_BANDS] * out).sum(axis=1)
    return traj[:, :N_CD4_BANDS] * DT, np.maximum(death, 0.0)


def life_expectancy_grid():
    """Cause-deleted remaining life expectancy at u = 0, 0.5, 1 within each
    year, so the engine can fit PV(L, r) across the year rather than assume the
    mid-year value."""
    return np.array([[life_table.remaining_life_expectancy_interp(ANCHOR_AGE + y + u)
                      for u in (0.0, 0.5, 1.0)] for y in range(N_YEARS)])

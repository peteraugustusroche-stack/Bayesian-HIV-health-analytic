"""DALYs, YLL, YLD and cost per acquisition, from a cohort trajectory.

Every output is a linear functional of the trajectory, which is what lets the
app reprice without re-running the engine.
"""

from __future__ import annotations

import math

import numpy as np

from params import N_CD4_BANDS, ANCHOR_AGE
import life_table
import simulate_natural_history as nh
import simulate_cascade as casc
from states import (
    N_NH_STATES, NH_HIV_DEATH, N_FULL_STATES, FULL_HIV_DEATH, FULL_BG_DEATH,
    CASCADE_STATUSES, full_index,
    with_base, tagged, base_of, CEILING_CLASSES,
)
import health_econ_params as hep
from health_econ_params import (
    disability_weight, DISCOUNT_RATE,
    ROUTINE_CARE_STATUSES,
)

# Accounting horizon runs the cohort to extinction. NOT the calibration horizon,
# which integrates over 25 years for the stationary cascade correction.
from params import HORIZON_YEARS   # single source of truth; see params.py


def _pv_of_years(L: float, r: float) -> float:
    if L <= 0:
        return 0.0
    if r <= 0:
        return L
    return (1 - math.exp(-r * L)) / r


def _discount_factor(t_years: float, r: float) -> float:
    return (1.0 + r) ** (-t_years) if r > 0 else 1.0


def cost_per_year(status: str, band: int) -> float:
    """Annual cost of occupying `status` at `band`, in health_econ_params.PRICE_YEAR
    dollars. Unit costs are read through the module, not bound at import, so that
    set_price_year() takes effect here.

    Two components only, following Hyle et al. Table S6: the antiretroviral
    regimen for whoever is on one, and routine HIV care priced by current CD4
    count. The flat ART service-delivery term the model used to carry is gone --
    it is inside the routine-care schedule, whose outpatient-visit component IS
    the cost of a clinic visit."""
    cost = 0.0
    base = base_of(status)
    if base in ("ART1_Ramp", "ART1_Suppressed", "ART1_Failing"):
        cost += hep.COST_1L_ART_PER_YEAR
    elif base == "ART2_Suppressed":
        cost += hep.COST_2L_ART_PER_YEAR
    if base in ROUTINE_CARE_STATUSES:
        cost += hep.ROUTINE_CARE_BY_BAND[band]
    return cost


def _natural_history_yll_yld(kappa_prog, kappa_mort,
                              dt=1 / 12, horizon_years=HORIZON_YEARS, r=DISCOUNT_RATE):
    stack = nh.build_nh_matrix_stack(kappa_prog, kappa_mort, dt, horizon_years)
    n_steps = int(horizon_years / dt)
    from acquisition_cd4 import initial_band_distribution
    init = np.zeros(N_NH_STATES)
    init[:N_CD4_BANDS] = initial_band_distribution()
    traj = nh.propagate(stack, init, n_steps, dt)

    yll, yld = 0.0, 0.0
    for t in range(n_steps):
        t_years = t * dt
        disc = _discount_factor(t_years, r)
        Mt = stack[min(int(t_years), stack.shape[0] - 1)]

        yld_month = sum(traj[t, band] * disability_weight(band, on_art=False)
                         for band in range(N_CD4_BANDS)) * dt
        yld += yld_month * disc

        death_flow = sum(traj[t, band] * Mt[band, NH_HIV_DEATH] for band in range(N_CD4_BANDS))
        if death_flow > 0:
            age_at_death = ANCHOR_AGE + t_years
            L = life_table.remaining_life_expectancy_interp(age_at_death)
            yll += death_flow * _pv_of_years(L, r) * disc

    return yll, yld


def _cascade_yll_yld_cost(cascade_vals: dict, kappa_prog, kappa_mort, params,
                           dt=1 / 12, horizon_years=HORIZON_YEARS, r=DISCOUNT_RATE):
    central = dict(cascade_vals)
    stack = casc.build_full_matrix_stack(central, params.fixed["recov_rate"],
                                          dt, horizon_years,
                                          kappa_prog=kappa_prog,
                                          kappa_mort=kappa_mort)

    n_steps = int(horizon_years / dt)
    init = casc.undiagnosed_init()
    traj = casc.propagate(stack, init, n_steps, dt)

    # NB `on_art_idx` really means SUPPRESSED -- ART1_Failing is already
    # excluded, and ART1_Ramp is excluded for the same reason: both are on
    # treatment but viraemic, so both take the unsuppressed disability weight.
    art1_idx = [full_index(st, b) for st in with_base("ART1_Suppressed")
                for b in range(N_CD4_BANDS)]
    art2_idx = [full_index(st, b) for st in with_base("ART2_Suppressed")
                for b in range(N_CD4_BANDS)]
    on_art_idx = set(art1_idx) | set(art2_idx)

    yll, yld, cost_recurring, cost_death = 0.0, 0.0, 0.0, 0.0
    for t in range(n_steps):
        t_years = t * dt
        disc = _discount_factor(t_years, r)
        Mt = stack[min(int(t_years), stack.shape[0] - 1)]

        yld_month = 0.0
        cost_month = 0.0
        for status_idx, status in enumerate(CASCADE_STATUSES):
            for band in range(N_CD4_BANDS):
                idx = full_index(status, band)
                mass = traj[t, idx]
                if mass <= 0:
                    continue
                on_art = idx in on_art_idx
                yld_month += mass * disability_weight(band, on_art=on_art)
                cost_month += mass * cost_per_year(status, band)
        yld += yld_month * dt * disc
        cost_recurring += cost_month * dt * disc

        death_flow = sum(traj[t, idx] * Mt[idx, FULL_HIV_DEATH]
                          for idx in range(N_FULL_STATES - 2))
        if death_flow > 0:
            age_at_death = ANCHOR_AGE + t_years
            L = life_table.remaining_life_expectancy_interp(age_at_death)
            yll += death_flow * _pv_of_years(L, r) * disc

        # One-off cost at death, any cause (Hyle et al. Table S6). Both death
        # channels, because their entry is "death cost, any cause of death" --
        # terminal care does not depend on what is written on the certificate.
        # This replaces the previous ART startup cost, which had no source and
        # no counterpart in the adopted cost set.
        bg_death_flow = sum(traj[t, idx] * Mt[idx, FULL_BG_DEATH]
                            for idx in range(N_FULL_STATES - 2))
        cost_death += (death_flow + bg_death_flow) * hep.COST_DEATH * disc

    return yll, yld, cost_recurring, cost_death


def evaluate_health_econ(cascade_vals: dict, kappa_prog: float, kappa_mort: float,
                          params, r: float = DISCOUNT_RATE) -> dict:
    """Full health-econ evaluation for one parameter draw. Returns per-1-
    acquisition (cohort normalized to 1) discounted outputs for both
    scenarios plus the derived comparison metrics."""
    yll_nh, yld_nh = _natural_history_yll_yld(kappa_prog, kappa_mort, r=r)
    dalys_nh = yll_nh + yld_nh

    yll_c, yld_c, cost_recurring, cost_death = _cascade_yll_yld_cost(
        cascade_vals, kappa_prog, kappa_mort, params, r=r)
    dalys_c = yll_c + yld_c
    cost_total = cost_recurring + cost_death

    return {
        "YLL_natural_history": yll_nh,
        "YLD_natural_history": yld_nh,
        "DALYs_natural_history": dalys_nh,
        "YLL_cascade": yll_c,
        "YLD_cascade": yld_c,
        "DALYs_per_acquisition": dalys_c,
        "DALYs_avoided_through_care": dalys_nh - dalys_c,
        "cost_recurring": cost_recurring,
        "cost_death": cost_death,
        "cost_per_acquisition": cost_total,
    }


if __name__ == "__main__":
    from params import load_parameters

    import likelihoods as _lk

    p = load_parameters()
    central = _lk.all_central_estimates(p)
    cascade_vals = {k: central[k] for k in _lk.SYSTEM2_SYMBOLS}

    result = evaluate_health_econ(cascade_vals, central["κ_prog"], central["κ_mort"], p)
    for k, v in result.items():
        print(f"  {k}: {v:.3f}")

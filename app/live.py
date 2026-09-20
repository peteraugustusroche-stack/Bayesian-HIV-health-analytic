"""Live cascade re-run: evaluate the Markov engine at a modified parameter set
and reprice it through the same path the posterior takes.

The rest of the app reads a precomputed occupancy tensor, so cost, disability
weights and the discount rate are instant. Changing a cascade hazard changes
the trajectory itself, which the tensor cannot represent, so this module runs
the engine -- about a quarter of a second for one parameter set -- and wraps
the result as a one-draw occupancy the engine reprices identically.

At a multiplier of one this lands within about 1% of
run_posterior_analysis.run_posterior_deterministic, not on it. The residual is
the kappa source: the reference takes the System 1 summary means, this takes
the mean of the per-draw kappas in the pairing System 2 was calibrated under.
Neither is the posterior mean of the outputs, which is what the other tabs
report.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

import channels
import engine as E

MODEL_AVAILABLE = True
try:
    from params import load_parameters
except Exception:                                       # pragma: no cover
    MODEL_AVAILABLE = False


class _OneDraw:
    """Duck-types engine.Occupancy for a single parameter set."""

    def __init__(self, py, hiv, hiv_yll, bg, nh_py, nh_death, life_expectancy):
        self.py = py[None, ...]
        self.hiv_death = hiv[None, ...]
        self.hiv_death_yll = hiv_yll[None, ...]
        self.bg_death = bg[None, ...]
        self.nh_py = nh_py[None, ...]
        self.nh_death = nh_death[None, ...]
        self.life_expectancy = life_expectancy
        self.bases = list(channels.BASE_STATUSES)
        self.on_art_bases = list(channels.ON_ART_BASES)
        self.n_draws, self.n_years = 1, channels.N_YEARS


@lru_cache(maxsize=1)
def _params():
    return load_parameters()


@lru_cache(maxsize=1)
def _life_expectancy():
    return channels.life_expectancy_grid()


@lru_cache(maxsize=64)
def _run(param_items: tuple, kappa_prog: float, kappa_mort: float) -> _OneDraw:
    vals = dict(param_items)
    m_py, m_hiv, m_bg, m_hiv_yll, _m_hiv_state = channels.cascade_monthly(
        vals, kappa_prog, kappa_mort, _params())
    mn_py, mn_death = channels.nh_monthly(kappa_prog, kappa_mort)
    return _OneDraw(channels.moments(m_py), channels.moments(m_hiv),
                    channels.moments(m_hiv_yll), channels.moments(m_bg),
                    channels.moments(mn_py), channels.moments(mn_death),
                    _life_expectancy())


def posterior_mean_parameters(cascade_year: str) -> tuple[dict, float, float]:
    """The calibrated point the cascade sliders start from."""
    occ = E.load_occupancy(cascade_year)
    means = occ.cascade_draws.mean(axis=0)
    vals = dict(zip(occ.cascade_param_names, (float(v) for v in means)))
    kp, km = (float(v) for v in occ.kappas.mean(axis=0))
    return vals, kp, km


def evaluate_point(a: E.Assumptions, multipliers: dict | None = None) -> dict:
    """Point evaluation at the posterior mean, each cascade hazard scaled by
    `multipliers`. Returns scalars, not draws."""
    vals, kp, km = posterior_mean_parameters(a.cascade_year)
    if multipliers:
        vals = {k: v * float(multipliers.get(k, 1.0)) for k, v in vals.items()}
    occ = _run(tuple(sorted(vals.items())), kp, km)
    out = E.evaluate_occupancy(occ, a)
    return {k: float(v[0]) for k, v in out.items()}

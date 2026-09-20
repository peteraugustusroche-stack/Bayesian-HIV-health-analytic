"""Repricing engine for the interactive app: cost, DALYs and ICERs from the
exported occupancy tensor, for any cost schedule, disability weight set and
discount rate.

No Shiny import, so it can be tested headless. `selftest.py` checks that at
default inputs it reproduces run_posterior_analysis draw-for-draw.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

import data_source

DATA_DIR = Path(__file__).parent / "data"

N_BANDS = 7
BAND_NAMES = ["≥500", "350-499", "250-349", "200-249", "100-199", "50-99", "<50"]

#: Hyle EP, et al. Lancet Glob Health 2025;13(8):e1436-e1447,
#: doi:10.1016/S2214-109X(25)00190-1, supplementary Table S6. Malawi, 2023 USD.
#: Monthly figures in the source; annualised here.
SOURCE_PRICE_YEAR = 2023
DEFAULT_COST_1L = 3.60 * 12
DEFAULT_COST_2L = 18.57 * 12
DEFAULT_ROUTINE = [v * 12 for v in (2.86, 3.96, 4.90, 4.90, 10.77, 18.51, 26.24)]
DEFAULT_COST_DEATH = 104.26

#: US CPI-U, all-items annual average (BLS, 1982-84 = 100). The sources report
#: USD already converted, so a US index moves them between price years.
CPI_U = {2008: 215.303, 2009: 214.537, 2010: 218.056, 2011: 224.939,
         2012: 229.594, 2013: 232.957, 2014: 236.736, 2015: 237.017,
         2016: 240.007, 2017: 245.120, 2018: 251.107, 2019: 255.657,
         2020: 258.811, 2021: 270.970, 2022: 292.655, 2023: 304.702,
         2024: 313.689}

#: Reconstructed continuous disability weights, evaluated at the representative
#: CD4 of each band. DW(CD4) = clip(Min + Slope * (500 - CD4) / 8, Min, Max).
CD4_REPRESENTATIVE = [575.0, 425.0, 300.0, 225.0, 150.0, 75.0, 25.0]
DW_ONART = (0.04, 0.19, 0.010)
DW_OFFART = (0.08, 0.57, 0.008)


def reconstructed_dw(on_art: bool) -> list[float]:
    lo, hi, slope = DW_ONART if on_art else DW_OFFART
    return [float(min(max(lo + slope * (500.0 - cd4) / 8.0, lo), hi))
            for cd4 in CD4_REPRESENTATIVE]


#: GBD-faithful alternative: the published weights for the nearest GBD state
#: rather than the reconstructed continuous function. Offered as a preset
#: because it moves DALYs per acquisition by about half a DALY.
GBD_DW_ONART = [0.053, 0.053, 0.053, 0.053, 0.078, 0.221, 0.547]
GBD_DW_OFFART = [0.078, 0.078, 0.221, 0.221, 0.274, 0.547, 0.582]


@dataclass(frozen=True)
class CostSchedule:
    """Unit costs at SOURCE_PRICE_YEAR. `price_year` restates outputs."""
    art_1l: float = DEFAULT_COST_1L
    art_2l: float = DEFAULT_COST_2L
    routine: tuple = tuple(DEFAULT_ROUTINE)
    death: float = DEFAULT_COST_DEATH
    price_year: int = 2024
    #: Which living cascade statuses incur routine care. The base case charges
    #: every living state including Undiagnosed, because the resource use is
    #: driven by immunosuppression, which precedes diagnosis.
    routine_statuses: tuple = ("Undiagnosed", "Diagnosed_PreART", "ART1_Ramp",
                               "ART1_Suppressed", "ART1_Failing",
                               "ART2_Suppressed", "LTFU_Recent", "LTFU_LongTerm")

    @property
    def inflation(self) -> float:
        return CPI_U[self.price_year] / CPI_U[SOURCE_PRICE_YEAR]


@dataclass(frozen=True)
class Weights:
    on_art: tuple = field(default_factory=lambda: tuple(reconstructed_dw(True)))
    off_art: tuple = field(default_factory=lambda: tuple(reconstructed_dw(False)))


@dataclass(frozen=True)
class Assumptions:
    cascade_year: str = "2019"
    discount_rate: float = 0.03
    costs: CostSchedule = field(default_factory=CostSchedule)
    weights: Weights = field(default_factory=Weights)


# Occupancy tensor

class Occupancy:
    """One cascade arm's exported trajectory moments."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.py = z["py"]                       # (draw, year, base, band, moment)
        self.hiv_death = z["hiv_death"]         # (draw, year, moment), as charged
        # YLL sees the HIV death flow clamped at zero. A draw whose SMR
        # multiplier puts the top-band SMR below 1 gives a negative on-ART
        # HIV-death hazard; health_economics skips those months when
        # accumulating YLL but still charges terminal care against them, so the
        # two channels differ. Both are carried to reproduce the model as run.
        self.hiv_death_yll = z["hiv_death_yll"]
        # HIV deaths disaggregated by the state occupied at death, each state
        # clamped at zero. Absent from exports written before the DALY
        # decomposition was added.
        self.hiv_death_by_state = (z["hiv_death_by_state"]
                                   if "hiv_death_by_state" in z.files else None)
        self.bg_death = z["bg_death"]
        self.nh_py = z["nh_py"]                 # (draw, year, band, moment)
        self.nh_death = z["nh_death"]
        self.life_expectancy = z["life_expectancy"]     # (year, 3) at u=0,.5,1
        self.bases = [str(s) for s in z["base_statuses"]]
        self.on_art_bases = [str(s) for s in z["on_art_bases"]]
        self.cascade_draws = z["cascade_draws"]
        self.cascade_param_names = [str(s) for s in z["cascade_param_names"]]
        self.kappas = z["kappas"]
        self.smr_multiplier = z["smr_multiplier"]
        self.n_draws, self.n_years = self.py.shape[0], self.py.shape[1]


@lru_cache(maxsize=2)
def load_occupancy(cascade_year: str) -> Occupancy:
    # data_source resolves a local build first and only downloads from the
    # release if the file is absent, so a checkout with data in place never
    # touches the network. See app/data_source.py.
    return Occupancy(data_source.ensure_occupancy(cascade_year))


# Within-year quadrature

def _quad(f0: float, f_half: float, f1: float) -> np.ndarray:
    """Coefficients contracting the three stored moments against a within-year
    weight sampled at u = 0, 0.5, 1."""
    c = 2.0 * (f1 - 2.0 * f_half + f0)
    return np.array([f0, f1 - f0 - c, c])


def _pv_years(L, r: float):
    """Present value of L future life-years, continuous time (Fox-Rushby &
    Hanson 2001)."""
    L = np.asarray(L, dtype=float)
    return L if r <= 0 else (1.0 - np.exp(-r * L)) / r


def _discount_vectors(occ: Occupancy, r: float):
    lam = np.log1p(r) if r > 0 else 0.0
    dy = (1.0 + r) ** -np.arange(occ.n_years) if r > 0 else np.ones(occ.n_years)
    w_flow = _quad(*[np.exp(-lam * u) for u in (0.0, 0.5, 1.0)])
    # YLL carries the life expectancy at the age of death, which moves within
    # the year, so its within-year weight is fitted rather than assumed flat.
    w_yll = np.stack([_quad(*[np.exp(-lam * u) * _pv_years(occ.life_expectancy[y, j], r)
                              for j, u in enumerate((0.0, 0.5, 1.0))])
                      for y in range(occ.n_years)])
    return dy, w_flow, w_yll


# Evaluation

def evaluate(a: Assumptions) -> dict[str, np.ndarray]:
    """Per-acquisition outcomes for every posterior draw, discounted, in
    `a.costs.price_year` US dollars."""
    return evaluate_occupancy(load_occupancy(a.cascade_year), a)


def evaluate_occupancy(occ, a: Assumptions) -> dict[str, np.ndarray]:
    """The same evaluation against any occupancy tensor, so a live single-draw
    re-run and the stored posterior go through one code path."""
    r = a.discount_rate
    dy, w_flow, w_yll = _discount_vectors(occ, r)

    # Discounted person-years by base status and CD4 band.
    pv_py = np.einsum("dybcm,m,y->dbc", occ.py, w_flow, dy)
    pv_nh_py = np.einsum("dycm,m,y->dc", occ.nh_py, w_flow, dy)

    on_art = np.array([b in occ.on_art_bases for b in occ.bases])
    dw = np.where(on_art[:, None], np.asarray(a.weights.on_art)[None, :],
                  np.asarray(a.weights.off_art)[None, :])
    yld = np.einsum("dbc,bc->d", pv_py, dw)
    yld_nh = pv_nh_py @ np.asarray(a.weights.off_art)

    yll = np.einsum("dym,ym,y->d", occ.hiv_death_yll, w_yll, dy)
    yll_nh = np.einsum("dym,ym,y->d", occ.nh_death, w_yll, dy)

    # Cost. ART drugs follow the regimen; routine care follows the CD4 band of
    # whichever statuses the user charges it to.
    infl = a.costs.inflation
    art1 = [i for i, b in enumerate(occ.bases) if b.startswith("ART1")]
    art2 = [i for i, b in enumerate(occ.bases) if b.startswith("ART2")]
    routine = [i for i, b in enumerate(occ.bases) if b in a.costs.routine_statuses]

    cost_recurring = (pv_py[:, art1, :].sum(axis=(1, 2)) * a.costs.art_1l * infl
                      + pv_py[:, art2, :].sum(axis=(1, 2)) * a.costs.art_2l * infl
                      + pv_py[:, routine, :].sum(axis=1) @ (np.asarray(a.costs.routine) * infl))

    deaths = np.einsum("dym,m,y->d", occ.hiv_death + occ.bg_death, w_flow, dy)
    cost_death = deaths * a.costs.death * infl

    dalys = yll + yld
    dalys_nh = yll_nh + yld_nh
    return {
        "YLL_cascade": yll, "YLD_cascade": yld,
        "DALYs_per_acquisition": dalys,
        "YLL_natural_history": yll_nh, "YLD_natural_history": yld_nh,
        "DALYs_natural_history": dalys_nh,
        "DALYs_avoided_through_care": dalys_nh - dalys,
        "cost_recurring": cost_recurring, "cost_death": cost_death,
        "cost_per_acquisition": cost_recurring + cost_death,
    }


def daly_by_state(a: Assumptions) -> dict:
    """DALYs per acquisition attributed to cascade state.

    Years lived with disability go to the state the person occupied while
    living them. Years of life lost go to the state occupied at death, which is
    the attribution that matters: it is what shows the burden concentrating in
    states the cascade is supposed to move people out of.

    The per-state death channel clamps each state at zero, so it slightly
    exceeds the total the published YLL uses (which clamps only the sum). The
    YLL shares are therefore rescaled to the published total, and the
    adjustment is returned as `yll_rescale` so the figure can report it.
    """
    occ = load_occupancy(a.cascade_year)
    if occ.hiv_death_by_state is None:
        raise SystemExit(f"occupancy_{a.cascade_year}.npz predates the by-state "
                         "death channel; re-run export_for_app.py")
    dy, w_flow, w_yll = _discount_vectors(occ, a.discount_rate)

    pv_py = np.einsum("dybcm,m,y->dbc", occ.py, w_flow, dy)
    on_art = np.array([b in occ.on_art_bases for b in occ.bases])
    dw = np.where(on_art[:, None], np.asarray(a.weights.on_art)[None, :],
                  np.asarray(a.weights.off_art)[None, :])
    yld = pv_py * dw[None, :, :]                       # (draw, base, band)
    yld_by_base = yld.sum(axis=2)

    yll_by_base = np.einsum("dybm,ym,y->db", occ.hiv_death_by_state, w_yll, dy)
    yll_total = np.einsum("dym,ym,y->d", occ.hiv_death_yll, w_yll, dy)
    scale = yll_total / np.maximum(yll_by_base.sum(axis=1), 1e-12)
    yll_by_base = yll_by_base * scale[:, None]

    return {
        "states": list(occ.bases),
        "yld": {b: yld_by_base[:, i] for i, b in enumerate(occ.bases)},
        "yll": {b: yll_by_base[:, i] for i, b in enumerate(occ.bases)},
        "yll_rescale": float(np.mean(scale)),
    }


def cost_components(a: Assumptions) -> dict[str, np.ndarray]:
    """Cost split the way the write-up reports it: routine care above and below
    CD4 200, drugs by line, and terminal care."""
    occ = load_occupancy(a.cascade_year)
    return cost_components_of(occ, a)


def cost_components_of(occ, a: Assumptions) -> dict[str, np.ndarray]:
    dy, w_flow, _ = _discount_vectors(occ, a.discount_rate)
    pv_py = np.einsum("dybcm,m,y->dbc", occ.py, w_flow, dy)
    infl = a.costs.inflation

    art1 = [i for i, b in enumerate(occ.bases) if b.startswith("ART1")]
    art2 = [i for i, b in enumerate(occ.bases) if b.startswith("ART2")]
    routine = [i for i, b in enumerate(occ.bases) if b in a.costs.routine_statuses]
    rc = pv_py[:, routine, :].sum(axis=1) * (np.asarray(a.costs.routine) * infl)
    deaths = np.einsum("dym,m,y->d", occ.hiv_death + occ.bg_death, w_flow, dy)

    return {
        "Routine Care, CD4 ≥200": rc[:, :4].sum(axis=1),
        "Routine Care, CD4 <200": rc[:, 4:].sum(axis=1),
        "First-Line ART Drugs": pv_py[:, art1, :].sum(axis=(1, 2)) * a.costs.art_1l * infl,
        "Second-Line ART Drugs": pv_py[:, art2, :].sum(axis=(1, 2)) * a.costs.art_2l * infl,
        "Terminal Care": deaths * a.costs.death * infl,
    }


def person_years_by_status(a: Assumptions) -> dict[str, np.ndarray]:
    """Discounted person-years by cascade status, for the occupancy panel."""
    occ = load_occupancy(a.cascade_year)
    dy, w_flow, _ = _discount_vectors(occ, a.discount_rate)
    pv_py = np.einsum("dybcm,m,y->dbc", occ.py, w_flow, dy)
    return {b: pv_py[:, i, :].sum(axis=1) for i, b in enumerate(occ.bases)}


# Cost-effectiveness

def treatment_icer(out: dict) -> np.ndarray:
    """Cost per DALY averted by the cascade, against no care. The denominator
    is DALYs avoided through care, so this values the whole cascade rather than
    any one component of it."""
    return out["cost_per_acquisition"] / out["DALYs_avoided_through_care"]


def prevention_icer(nnt: float, price: float, dalys, offset, offset_share: float = 1.0):
    """Incremental cost per DALY averted by prophylaxis. `offset_share` is the
    fraction of avoided lifetime treatment cost credited: 1.0 is the correct
    treatment, 0.0 an analysis that ignores what prevention saves."""
    net = nnt * price - offset_share * np.asarray(offset)
    return net / np.asarray(dalys)


def break_even_price(nnt: float, offset, offset_share: float = 1.0) -> float:
    """Annual cost per person-year at which prophylaxis is cost-neutral."""
    return float(offset_share * np.mean(offset) / nnt)


def summarise(x, lo=5, hi=95) -> tuple[float, float, float]:
    x = np.asarray(x)
    return float(x.mean()), float(np.percentile(x, lo)), float(np.percentile(x, hi))


def with_cost(a: Assumptions, **kw) -> Assumptions:
    return replace(a, costs=replace(a.costs, **kw))

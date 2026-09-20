"""Export the posterior as an occupancy tensor the interactive app can reprice.

Cost, YLD and YLL are all linear functionals of the cohort trajectory, so the
app does not need the Markov engine to answer "what if costs, disability
weights or the discount rate were different". It needs the trajectory,
collapsed onto the channels those inputs multiply:

    py[draw, year, status, band, moment]   person-years, by cascade status and CD4
    hiv_death[draw, year, moment]          deaths on the HIV channel, as charged
    hiv_death_yll[draw, year, moment]      the same, clamped at zero, as YLL sees it
    bg_death[draw, year, moment]           deaths on the background channel
    nh_py[draw, year, band, moment]        the no-care counterfactual
    nh_death[draw, year, moment]

Storing the trajectory monthly for 1,000 draws is ~200 MB, which is more than
an app should load. Instead each year holds three moments of the within-year
mass distribution, sum(mass * u^j) for u = month/12 and j = 0,1,2. Any smooth
within-year weight -- the discount factor, or the discount factor times the
present value of remaining life expectancy -- is then recovered by fitting it
at u = 0, 0.5, 1 and contracting against the three moments. `--check` reports
the residual against exact monthly evaluation; it is ~1e-6 relative.

    python3 export_for_app.py [cascade_year] [--draws N] [--check]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

import hazards
from calibrate_system2 import result_name_for_year
from calibration_common import load_result
from params import ART_SMR_LOG_SD, N_CD4_BANDS, load_parameters
import health_econ_params as hep
import life_table
from engine import _quad as quad_coeffs
from params import ANCHOR_AGE
from channels import (BASE_STATUSES, DT, N_MOMENTS, N_STEPS, N_YEARS,
                      ON_ART_BASES, cascade_monthly, life_expectancy_grid,
                      moments, nh_monthly)

OUT_DIR = Path(__file__).parent / "app" / "data"


def export(cascade_year: str = "2019", n_draws: int = 1000, seed: int = 1,
           check: bool = False, progress_every: int = 100) -> Path:
    params = load_parameters()
    s2 = load_result(result_name_for_year(cascade_year))
    chains, param_names = s2["chains"], list(s2["param_names"])
    nh_draws_by_chain = s2["nh_draws_by_chain"]
    n_chains, n_per_chain, _ = chains.shape

    # Reproduces run_posterior_analysis.run_posterior_probabilistic exactly:
    # same seed, same draw order, same SMR multiplier sequence.
    rng = np.random.default_rng(seed)
    flat = rng.integers(0, n_chains * n_per_chain, size=n_draws)
    smr_mult = rng.lognormal(0.0, ART_SMR_LOG_SD, size=n_draws)

    nb = len(BASE_STATUSES)
    py = np.zeros((n_draws, N_YEARS, nb, N_CD4_BANDS, N_MOMENTS), dtype=np.float32)
    hiv_death = np.zeros((n_draws, N_YEARS, N_MOMENTS), dtype=np.float32)
    hiv_death_yll = np.zeros((n_draws, N_YEARS, N_MOMENTS), dtype=np.float32)
    hiv_death_by_state = np.zeros((n_draws, N_YEARS, nb, N_MOMENTS), dtype=np.float32)
    bg_death = np.zeros((n_draws, N_YEARS, N_MOMENTS), dtype=np.float32)
    nh_py = np.zeros((n_draws, N_YEARS, N_CD4_BANDS, N_MOMENTS), dtype=np.float32)
    nh_death = np.zeros((n_draws, N_YEARS, N_MOMENTS), dtype=np.float32)
    cascade_draws = np.zeros((n_draws, len(param_names)))
    kappas = np.zeros((n_draws, 2))

    errors, t0 = [], time.time()
    try:
        for i in range(n_draws):
            c, d = int(flat[i]) // n_per_chain, int(flat[i]) % n_per_chain
            vals = {name: float(chains[c, d, j]) for j, name in enumerate(param_names)}
            kp, km = (float(x) for x in nh_draws_by_chain[c])
            hazards.set_smr_multiplier(float(smr_mult[i]))

            m_py, m_hiv, m_bg, m_hiv_yll, m_hiv_state = cascade_monthly(
                vals, kp, km, params)
            mn_py, mn_death = nh_monthly(kp, km)

            py[i] = moments(m_py)
            hiv_death[i], bg_death[i] = moments(m_hiv), moments(m_bg)
            hiv_death_yll[i] = moments(m_hiv_yll)
            hiv_death_by_state[i] = moments(m_hiv_state)
            nh_py[i], nh_death[i] = moments(mn_py), moments(mn_death)
            cascade_draws[i] = [vals[n] for n in param_names]
            kappas[i] = (kp, km)

            if check and i < 10:
                errors.append(_moment_error(py[i], hiv_death_yll[i], m_py, m_hiv_yll))
            if progress_every and (i + 1) % progress_every == 0:
                el = time.time() - t0
                print(f"  draw {i+1}/{n_draws}  ({el:.0f}s, "
                      f"{el / (i + 1) * (n_draws - i - 1):.0f}s left)", flush=True)
    finally:
        hazards.set_smr_multiplier(None)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"occupancy_{cascade_year}.npz"
    np.savez_compressed(
        path,
        py=py, hiv_death=hiv_death, hiv_death_yll=hiv_death_yll,
        hiv_death_by_state=hiv_death_by_state, bg_death=bg_death,
        nh_py=nh_py, nh_death=nh_death,
        life_expectancy=life_expectancy_grid(), smr_multiplier=smr_mult,
        cascade_draws=cascade_draws, cascade_param_names=np.array(param_names),
        kappas=kappas, base_statuses=np.array(BASE_STATUSES),
        on_art_bases=np.array(ON_ART_BASES), cascade_year=np.array(cascade_year),
    )
    print(f"\nwrote {path}  ({path.stat().st_size / 1e6:.1f} MB, "
          f"{n_draws} draws, {time.time() - t0:.0f}s)")
    if errors:
        e = np.abs(np.array(errors))
        print(f"moment-expansion error over {len(e)} draws (max relative): "
              f"cost {e[:, 0].max():.2e}, YLD {e[:, 1].max():.2e}, YLL {e[:, 2].max():.2e}")
    return path


def _moment_error(mom_py, mom_hiv, m_py, m_hiv, r=hep.DISCOUNT_RATE):
    """Residual of the three-moment expansion against exact monthly evaluation,
    for a cost-like, a YLD-like and a YLL-like functional."""
    lam = np.log1p(r)
    Ly = life_expectancy_grid()
    dy = (1 + r) ** -np.arange(N_YEARS)
    dm = (1 + r) ** -(np.arange(N_STEPS) * DT)
    pv = lambda L: (1 - np.exp(-r * L)) / r

    w_disc = quad_coeffs(*[np.exp(-lam * u) for u in (0.0, 0.5, 1.0)])
    approx = ((mom_py.sum(axis=(1, 2)) @ w_disc) * dy).sum()
    exact = (m_py.sum(axis=(1, 2)) * dm).sum()
    e_cost = approx / exact - 1

    w = np.arange(N_CD4_BANDS) + 1.0                      # any band weighting
    approx_y = (((mom_py.sum(axis=1) * w[None, :, None]).sum(axis=1) @ w_disc) * dy).sum()
    exact_y = ((m_py.sum(axis=1) @ w) * dm).sum()
    e_yld = approx_y / exact_y - 1

    w_yll = np.array([quad_coeffs(*[np.exp(-lam * u) * pv(Ly[y, j])
                                    for j, u in enumerate((0.0, 0.5, 1.0))])
                      for y in range(N_YEARS)])
    approx_l = ((mom_hiv * w_yll).sum(axis=1) * dy).sum()
    L_month = np.array([life_table.remaining_life_expectancy_interp(ANCHOR_AGE + t * DT)
                        for t in range(N_STEPS)])
    exact_l = (m_hiv * pv(L_month) * dm).sum()
    e_yll = approx_l / exact_l - 1
    return e_cost, e_yld, e_yll


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    year = args[0] if args else "2019"
    n = 1000
    if "--draws" in sys.argv:
        n = int(sys.argv[sys.argv.index("--draws") + 1])
    export(year, n_draws=n, check="--check" in sys.argv)

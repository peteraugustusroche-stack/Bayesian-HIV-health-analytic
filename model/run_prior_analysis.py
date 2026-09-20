"""
Pre-calibration health-econ run: a single deterministic evaluation at every
free parameter's central (prior) estimate, plus an optional probabilistic
run (Monte Carlo over the PRIOR distributions -- not the posterior) giving
uncertainty intervals on DALYs/YLL/YLD/cost before any calibration data has
touched the model at all.

Run directly:
    python3 run_prior_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from params import load_parameters
from likelihoods import (
    build_specs, all_central_estimates,
    SYSTEM1_SYMBOLS, SYSTEM1_KINDS, SYSTEM2_SYMBOLS, SYSTEM2_KINDS,
)
from health_economics import evaluate_health_econ

RESULTS_DIR = Path(__file__).parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = RESULTS_DIR / "prior_analysis.npz"

METRIC_NAMES = [
    "DALYs_per_acquisition", "DALYs_avoided_through_care", "DALYs_natural_history",
    "YLL_natural_history", "YLD_natural_history", "YLL_cascade", "YLD_cascade",
    "cost_per_acquisition", "cost_recurring", "cost_death",
]


def _all_specs(params):
    return build_specs(params, SYSTEM1_SYMBOLS, SYSTEM1_KINDS) + \
           build_specs(params, SYSTEM2_SYMBOLS, SYSTEM2_KINDS)


def _evaluate(theta: dict, params) -> dict:
    cascade_vals = {k: theta[k] for k in SYSTEM2_SYMBOLS}
    return evaluate_health_econ(cascade_vals, theta["κ_prog"], theta["κ_mort"], params)


def run_prior_deterministic(params=None) -> dict:
    params = params or load_parameters()
    # all_central_estimates, NOT params.central_estimates(): the latter reads
    # only data/parameters.xlsx and silently omits every parameter defined in
    # code (phi_ltfu, and now kappa_prog/kappa_mort), so _evaluate below fails
    # with a bare KeyError on the first missing symbol.
    central = all_central_estimates(params)
    result = _evaluate(central, params)

    specs = _all_specs(params)
    param_table = [{"symbol": s.symbol, "kind": s.kind, "central": s.central, "sd": s.sd}
                   for s in specs]
    return {"metrics": result, "param_table": param_table}


def run_prior_probabilistic(params=None, n_draws=1000, seed=1, progress_every=None) -> dict:
    params = params or load_parameters()
    specs = _all_specs(params)
    rng = np.random.default_rng(seed)

    metrics = {name: np.zeros(n_draws) for name in METRIC_NAMES}
    for i in range(n_draws):
        theta = {}
        for spec in specs:
            val = spec.sample_prior(rng)
            tries = 0
            while spec.kind == "lognormal" and val <= 0 and tries < 20:
                val = spec.sample_prior(rng)
                tries += 1
            theta[spec.symbol] = val
        out = _evaluate(theta, params)
        for name in METRIC_NAMES:
            metrics[name][i] = out[name]
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  draw {i+1}/{n_draws}")

    summary = []
    for name in METRIC_NAMES:
        vals = metrics[name]
        summary.append({
            "metric": name, "mean": float(vals.mean()), "sd": float(vals.std()),
            "q05": float(np.quantile(vals, 0.05)), "median": float(np.quantile(vals, 0.5)),
            "q95": float(np.quantile(vals, 0.95)),
        })

    return {"n_draws": n_draws, "metrics": metrics, "summary": summary}


def save_prior_result(deterministic: dict, probabilistic: dict | None) -> Path:
    payload = {"deterministic": deterministic}
    if probabilistic is not None:
        payload["probabilistic"] = {
            "n_draws": probabilistic["n_draws"],
            "metrics": probabilistic["metrics"],
            "summary": probabilistic["summary"],
        }
    np.savez(RESULT_PATH, payload=np.array(payload, dtype=object))
    return RESULT_PATH


def load_prior_result() -> dict | None:
    if not RESULT_PATH.exists():
        return None
    data = np.load(RESULT_PATH, allow_pickle=True)
    return data["payload"].item()


if __name__ == "__main__":
    params = load_parameters()

    print("Deterministic run (central/prior estimates)...")
    det = run_prior_deterministic(params)
    for k, v in det["metrics"].items():
        print(f"  {k}: {v:.3f}")

    print("\nProbabilistic run (1000 draws from priors)...")
    prob = run_prior_probabilistic(params, n_draws=1000, progress_every=250)
    print(f"\n{'metric':<32}{'mean':>10}{'q05':>10}{'median':>10}{'q95':>10}")
    for row in prob["summary"]:
        print(f"{row['metric']:<32}{row['mean']:>10.2f}{row['q05']:>10.2f}"
              f"{row['median']:>10.2f}{row['q95']:>10.2f}")

    path = save_prior_result(det, prob)
    print(f"\nSaved to {path}")

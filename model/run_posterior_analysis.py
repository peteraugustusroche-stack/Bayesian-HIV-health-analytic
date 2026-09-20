"""
Post-calibration health-econ run: the same DALYs/YLL/YLD/cost metrics as
run_prior_analysis.py, but evaluated over PAIRED posterior draws (System 1's
kappa_prog/kappa_mort together with System 2's cascade parameters, using
the pairing System 2's calibration actually ran under -- see
calibrate_system2.py's nh_draws_by_chain). Requires both calibrations to
have been run first.

Run directly:
    python3 run_posterior_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import hazards
from params import load_parameters, ART_SMR_LOG_SD
from health_economics import evaluate_health_econ
from run_prior_analysis import METRIC_NAMES

RESULTS_DIR = Path(__file__).parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_NAME = "posterior_analysis"

# Health-econ outputs are specific to the cascade target set the System 2
# posterior was calibrated against, so they need the same per-year filenames
# the calibration results use. Before this, every year wrote to a single
# posterior_analysis.npz and running the summary under a sensitivity target
# set silently destroyed the base-case DALY results.
#
# Mirrors calibrate_system2.result_name_for_year: '2019' keeps the original
# bare name so existing files and any code reading it keep working; anything
# else gets a suffix.


def posterior_result_path(cascade_year: str = "2019", price_year=None) -> Path:
    stem = RESULT_NAME if cascade_year == "2019" else f"{RESULT_NAME}_sens{cascade_year}"
    # The price year is normally matched to the cascade year, and that case keeps
    # the plain filename so existing readers are unaffected. A DIFFERENT price
    # year -- used to put two arms on a common base for comparison -- is a
    # distinct analysis and gets its own file, so it cannot overwrite the base
    # case. Without this, `run_posterior_analysis.py 2019 2024` destroyed the
    # 2019-priced base-case results in place.
    if price_year is not None and str(price_year) != str(cascade_year):
        stem = f"{stem}_p{price_year}"
    return RESULTS_DIR / f"{stem}.npz"


# Retained so existing imports do not break; equals the 2019 path.
RESULT_PATH = posterior_result_path("2019")


def run_posterior_deterministic(system1_result, system2_result, params=None) -> dict:
    """Point evaluation at the posterior MEAN of every parameter (quick
    summary, not a substitute for the full probabilistic spread below)."""
    params = params or load_parameters()
    lb_mean = {row["param"]: row["mean"] for row in system1_result["summary"]}
    cascade_mean = {row["param"]: row["mean"] for row in system2_result["summary"]}
    result = evaluate_health_econ(cascade_mean, lb_mean["κ_prog"], lb_mean["κ_mort"], params)
    return {"metrics": result}


def run_posterior_probabilistic(system2_result, params=None, n_draws=1000, seed=1,
                                 progress_every=None) -> dict:
    params = params or load_parameters()
    chains = system2_result["chains"]                    # (n_chains, n_draws, 7)
    param_names = system2_result["param_names"]
    nh_draws_by_chain = system2_result["nh_draws_by_chain"]  # (n_chains, 2)

    n_chains, n_draws_per_chain, _ = chains.shape
    total_available = n_chains * n_draws_per_chain

    rng = np.random.default_rng(seed)
    flat_idx = rng.integers(0, total_available, size=n_draws)
    chain_idx = flat_idx // n_draws_per_chain
    draw_idx = flat_idx % n_draws_per_chain

    # On-ART mortality uncertainty: one multiplicative draw on the whole SMR
    # ladder per posterior draw. Not calibrated -- see params.ART_SMR_LOG_SD.
    smr_mult = rng.lognormal(0.0, ART_SMR_LOG_SD, size=n_draws)

    metrics = {name: np.zeros(n_draws) for name in METRIC_NAMES}
    try:
        for i in range(n_draws):
            c, d = int(chain_idx[i]), int(draw_idx[i])
            cascade_vals = {name: float(chains[c, d, j]) for j, name in enumerate(param_names)}
            kappa_prog, kappa_mort = (float(x) for x in nh_draws_by_chain[c])
            hazards.set_smr_multiplier(float(smr_mult[i]))
            out = evaluate_health_econ(cascade_vals, kappa_prog, kappa_mort, params)
            for name in METRIC_NAMES:
                metrics[name][i] = out[name]
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  draw {i+1}/{n_draws}")
    finally:
        # Never leave the scaled ladder in place: later callers (the
        # deterministic run, scenario_analysis) would silently inherit it.
        hazards.set_smr_multiplier(None)

    summary = []
    for name in METRIC_NAMES:
        vals = metrics[name]
        summary.append({
            "metric": name, "mean": float(vals.mean()), "sd": float(vals.std()),
            "q05": float(np.quantile(vals, 0.05)), "median": float(np.quantile(vals, 0.5)),
            "q95": float(np.quantile(vals, 0.95)),
        })

    return {"n_draws": n_draws, "metrics": metrics, "summary": summary,
            "smr_multiplier": smr_mult}


def save_posterior_result(deterministic: dict, probabilistic: dict | None,
                           cascade_year: str = "2019", price_year=None) -> Path:
    payload = {"deterministic": deterministic, "cascade_year": cascade_year,
               "price_year": price_year}
    if probabilistic is not None:
        payload["probabilistic"] = {
            "n_draws": probabilistic["n_draws"],
            "metrics": probabilistic["metrics"],
            "summary": probabilistic["summary"],
            # The per-draw on-ART SMR multiplier, so its contribution to the
            # spread can be recovered without regenerating the seed sequence.
            "smr_multiplier": probabilistic.get("smr_multiplier"),
        }
    path = posterior_result_path(cascade_year, price_year)
    np.savez(path, payload=np.array(payload, dtype=object))
    return path


def load_posterior_result(cascade_year: str = "2019", price_year=None) -> dict | None:
    path = posterior_result_path(cascade_year, price_year)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return data["payload"].item()


if __name__ == "__main__":
    import sys

    from calibration_common import load_result
    from calibrate_system2 import result_name_for_year

    # Cascade year, matching calibrate_system2.py's argument. '2019' is the
    # base case and reads/writes the unsuffixed files; any other year reads
    # system2_sens<year>.npz and writes posterior_analysis_sens<year>.npz.
    #
    # Without this the script loaded "system2" and saved to the unsuffixed
    # posterior_analysis.npz whatever year you thought you were running, so a
    # sensitivity year silently produced base-case numbers AND overwrote the
    # base-case health economics.
    cascade_year = sys.argv[1] if len(sys.argv) > 1 else "2019"
    s2_name = result_name_for_year(cascade_year)

    s1 = load_result("system1")
    s2 = load_result(s2_name)
    if s1 is None or s2 is None:
        raise SystemExit(
            f"Missing input: need system1 and {s2_name}. "
            f"Run calibrate_system1.py and calibrate_system2.py {cascade_year} first.")
    # Price year follows the cascade year: the 2019-anchored analysis reports
    # 2019 USD and the 2024-anchored analysis 2024 USD, so each arm is
    # internally consistent in both epidemiology and prices. Override with a
    # second argument if a common price base is wanted for a direct comparison.
    import health_econ_params as hep
    price_year = int(sys.argv[2]) if len(sys.argv) > 2 else (
        int(cascade_year) if cascade_year.isdigit() else 2024)
    hep.set_price_year(price_year)

    print(f"cascade_year={cascade_year}  (System 2 result: {s2_name})")
    print(f"price_year={price_year}  (costs in {price_year} USD; "
          f"1L ${hep.COST_1L_ART_PER_YEAR:.2f}/yr, 2L ${hep.COST_2L_ART_PER_YEAR:.2f}/yr, "
          f"routine care ${hep.ROUTINE_CARE_BY_BAND[0]:.2f}-{hep.ROUTINE_CARE_BY_BAND[-1]:.2f}/yr, "
          f"death ${hep.COST_DEATH:.2f})")

    params = load_parameters()

    print("Deterministic run (posterior means)...")
    det = run_posterior_deterministic(s1, s2, params)
    for k, v in det["metrics"].items():
        print(f"  {k}: {v:.3f}")

    print("\nProbabilistic run (1000 paired posterior draws)...")
    prob = run_posterior_probabilistic(s2, params, n_draws=1000, progress_every=250)
    print(f"\n{'metric':<32}{'mean':>10}{'q05':>10}{'median':>10}{'q95':>10}")
    for row in prob["summary"]:
        print(f"{row['metric']:<32}{row['mean']:>10.2f}{row['q05']:>10.2f}"
              f"{row['median']:>10.2f}{row['q95']:>10.2f}")

    path = save_posterior_result(det, prob, cascade_year, price_year)
    print(f"\nSaved to {path}")

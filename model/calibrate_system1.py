"""
System 1: draw the natural-history uncertainty. NOT a calibration any more.

The CD4-progression and untreated-mortality schedules are adopted from
Glaubius et al. 2021 rather than estimated here (see natural_history_params
for why, and for the evidence that the adopted tables reproduce this model's
three natural-history targets better than the previous fitted version did).
What remains is two multiplicative scale factors, kappa_prog and kappa_mort,
which carry uncertainty about the LEVEL of those published schedules. They are
drawn from their priors.

The output has exactly the same structure as the old calibration result --
chains, param_names, summary, model_implied_at_mean, targets -- so everything
downstream (calibrate_system2's outer resampling, the app's result loading,
run_posterior_analysis) is unchanged. Convergence diagnostics are reported and
are trivially satisfied: the draws are independent, so R-hat is 1 by
construction and ESS equals the number of draws. That is not a modelling
achievement, it is a statement that there is nothing left here to converge.

If USE_SYSTEM1_LIKELIHOOD is turned on in likelihoods.py, this reverts to
running an actual sampler against the three targets. Left off by default: the
targets are worth more as out-of-sample validation than as a second helping of
information the priors already carry.

Run directly to produce and cache a result the Shiny app will pick up:
    python3 calibrate_system1.py
"""

from __future__ import annotations

import numpy as np

from params import load_parameters
import likelihoods as lk
from likelihoods import (
    build_specs, SYSTEM1_SYMBOLS, SYSTEM1_KINDS, system1_targets,
    system1_log_posterior, system1_log_likelihood, system1_model_implied,
)
from calibration_common import run_calibration, save_result, load_result
from mcmc_numpy import summarize

RESULT_NAME = "system1"


def _prior_draws(specs, draws: int, chains: int, seed: int) -> np.ndarray:
    """(chains, draws, n_params) independent prior draws, shaped like a
    sampler's output so the same summarize() and the same downstream
    resampling apply unchanged."""
    rng = np.random.default_rng(seed)
    return np.stack([
        np.stack([np.array([spec.sample_prior(rng) for spec in specs])
                  for _ in range(draws)])
        for _ in range(chains)
    ])


def run_system1_calibration(draws=2000, tune=1500, chains=4, seed=1,
                             backend="auto", progress_every=None) -> dict:
    params = load_parameters()
    specs = build_specs(params, SYSTEM1_SYMBOLS, SYSTEM1_KINDS)

    if lk.USE_SYSTEM1_LIKELIHOOD:
        result = run_calibration(
            log_posterior_fn=system1_log_posterior,
            log_likelihood_fn=system1_log_likelihood,
            specs=specs,
            extra_args=(params,),
            draws=draws, tune=tune, chains=chains, seed=seed,
            backend=backend, progress_every=progress_every,
        )
    else:
        param_names = [s.symbol for s in specs]
        chains_arr = _prior_draws(specs, draws, chains, seed)
        result = {
            "backend": "prior (System 1 not fitted)",
            "param_names": param_names,
            "chains": chains_arr,
            "summary": summarize({"chains": chains_arr,
                                   "param_names": param_names}),
        }

    means = {row["param"]: row["mean"] for row in result["summary"]}
    result["model_implied_at_mean"] = system1_model_implied(
        means["κ_prog"], means["κ_mort"], params)
    result["targets"] = system1_targets(params)
    # Flag for the display layer: these are validation checks, not fitted
    # targets, unless the likelihood was switched back on.
    result["targets_are_fitted"] = bool(lk.USE_SYSTEM1_LIKELIHOOD)
    return result


if __name__ == "__main__":
    fitted = lk.USE_SYSTEM1_LIKELIHOOD
    print("Running System 1 calibration..." if fitted else
          "Drawing System 1 natural-history uncertainty (not fitted)...")
    result = run_system1_calibration(progress_every=500)
    print(f"\nBackend used: {result['backend']}")
    for row in result["summary"]:
        print(f"  {row['param']}: mean={row['mean']:.4f} sd={row['sd']:.4f} "
              f"r_hat={row['r_hat']:.3f} ess={row['ess']:.0f}")

    label = "fitted targets" if fitted else "VALIDATION (not fitted)"
    print(f"\nModel-implied (at posterior mean) vs {label}:")
    for key, (central, sd) in result["targets"].items():
        implied = result["model_implied_at_mean"][key]
        z = (implied - central) / sd
        print(f"  {key}: implied={implied:.2f}  target={central:.2f} ± {sd}"
              f"   z={z:+.2f}")

    path = save_result(result, RESULT_NAME)
    print(f"\nSaved to {path}")

"""
Shared plumbing between calibrate_system1.py and calibrate_system2.py:
a PyMC-first, numpy-fallback sampling harness, and a unified result format
both the CLI scripts and the Shiny app consume identically regardless of
which backend actually ran.

Result dict format:
  backend: "pymc" or "numpy" (whichever actually ran)
  param_names: list[str]
  chains: ndarray (n_chains, n_draws, n_params) -- post-tuning samples only
  summary: list[dict] from mcmc_numpy.summarize (mean/sd/q03/q97/r_hat/ess)
  accept_rates: list[float] (numpy backend only; None for pymc)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mcmc_numpy import run_mcmc as run_mcmc_numpy, summarize

RESULTS_DIR = Path(__file__).parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _try_import_pymc():
    try:
        import pymc as pm
        import pytensor.tensor as pt
        from pytensor.compile.ops import as_op
        return pm, pt, as_op
    except ImportError:
        return None, None, None


def run_calibration(log_posterior_fn, log_likelihood_fn, specs, extra_args=(),
                     draws=2000, tune=1500, chains=4, seed=None,
                     backend="auto", progress_every=None) -> dict:
    """
    log_posterior_fn(theta, specs, *extra_args) -> float   (used by numpy backend)
    log_likelihood_fn(*param_values, *extra_args) -> float (used by pymc backend,
        called with each free parameter as its own positional scalar argument)
    """
    pm, pt, as_op = _try_import_pymc() if backend in ("auto", "pymc") else (None, None, None)

    if pm is not None and backend in ("auto", "pymc"):
        result = _run_pymc(pm, pt, as_op, log_likelihood_fn, specs, extra_args,
                            draws, tune, chains, seed)
        result["backend"] = "pymc"
    else:
        if backend == "pymc":
            raise ImportError("PyMC was explicitly requested but is not installed "
                               "(pip install pymc). Falling back is disabled when "
                               "backend='pymc' is set explicitly.")
        raw = run_mcmc_numpy(log_posterior_fn, specs, args=extra_args,
                              draws=draws, tune=tune, chains=chains,
                              seed=seed or 0, progress_every=progress_every)
        result = {"backend": "numpy", "param_names": raw["param_names"],
                  "chains": raw["chains"], "accept_rates": raw["accept_rates"]}

    result["summary"] = summarize({"chains": result["chains"],
                                    "param_names": result["param_names"]})
    return result


def _run_pymc(pm, pt, as_op, log_likelihood_fn, specs, extra_args,
              draws, tune, chains, seed):
    n_params = len(specs)
    itypes = [pt.dscalar] * n_params

    @as_op(itypes=itypes, otypes=[pt.dscalar])
    def loglik(*values):
        return np.array(log_likelihood_fn(*values, *extra_args))

    with pm.Model() as model:
        rvs = []
        for spec in specs:
            if spec.kind == "lognormal":
                rv = pm.Lognormal(spec.symbol, mu=spec.log_mu, sigma=spec.log_sigma)
            else:
                rv = pm.Normal(spec.symbol, mu=spec.central, sigma=spec.sd)
            rvs.append(rv)
        pm.Potential("loglik", loglik(*rvs))

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                           step=pm.DEMetropolisZ(), random_seed=seed,
                           progressbar=False, compute_convergence_checks=False)

    param_names = [s.symbol for s in specs]
    chains_arr = np.stack([idata.posterior[name].values for name in param_names], axis=-1)
    return {"param_names": param_names, "chains": chains_arr, "accept_rates": None}


_CORE_KEYS = {"chains", "param_names", "backend", "summary"}


def save_result(result: dict, name: str) -> Path:
    """Persists the full result dict, including any extra keys a caller
    added on top of run_calibration()'s output (e.g. calibrate_system1.py
    adds 'targets' and 'model_implied_at_mean'). Anything beyond the core
    chains/param_names/backend fields is pickled via an object array so
    load_result() round-trips the exact same dict shape callers expect."""
    path = RESULTS_DIR / f"{name}.npz"
    extra = {k: v for k, v in result.items() if k not in _CORE_KEYS}
    np.savez(path, chains=result["chains"],
             param_names=np.array(result["param_names"]),
             backend=result["backend"],
             extra=np.array(extra, dtype=object))
    return path


def load_result(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    chains = data["chains"]
    param_names = list(data["param_names"])
    result = {"backend": str(data["backend"]), "param_names": param_names, "chains": chains}
    if "extra" in data:
        result.update(data["extra"].item())
    result["summary"] = summarize({"chains": chains, "param_names": param_names})
    return result

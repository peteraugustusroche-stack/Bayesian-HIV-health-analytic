"""
A small, dependency-free random-walk Metropolis sampler.

Two jobs:
1. Verification harness -- runs entirely on numpy (already installed), so
   the priors/likelihood in likelihoods.py can be checked end-to-end in any
   environment, including this one, without needing PyMC.
2. Automatic fallback -- calibrate_system1.py / calibrate_system2.py import
   this and use it if `import pymc` fails, so the pipeline still runs even
   on a machine without PyMC installed (just with a less sophisticated
   sampler -- no NUTS-style efficiency, no automatic step-size adaptation
   as robust as PyMC's).

Not a general-purpose MCMC library: per-parameter step sizes are adapted
during tuning to a target acceptance rate, then frozen. Diagnostics
(R-hat, effective sample size) use the standard split-chain Gelman-Rubin
estimator, same definition PyMC/ArviZ use, so results are comparable.
"""

from __future__ import annotations

import numpy as np


def run_mcmc(log_posterior_fn, specs, args=(), draws=2000, tune=1000, chains=4,
             seed=0, target_accept=0.30, progress_every=None):
    """
    log_posterior_fn(theta, specs, *args) -> float

    Returns dict with:
      chains: ndarray (chains, draws, n_params) -- post-tuning samples only
      param_names: list[str]
      accept_rates: list[float], one per chain
    """
    rng_master = np.random.default_rng(seed)
    n_params = len(specs)
    all_chains = np.zeros((chains, draws, n_params))
    accept_rates = []

    for c in range(chains):
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        theta = np.array([specs[i].sample_prior(rng) for i in range(n_params)])
        lp = log_posterior_fn(theta, specs, *args)
        # guard against an unlucky prior draw landing at -inf
        tries = 0
        while not np.isfinite(lp) and tries < 200:
            theta = np.array([specs[i].sample_prior(rng) for i in range(n_params)])
            lp = log_posterior_fn(theta, specs, *args)
            tries += 1

        step_scale = np.full(n_params, 0.3)
        batch_accept = np.zeros(n_params)
        batch_total = np.zeros(n_params)
        n_accept = 0
        n_total = 0
        samples = np.zeros((draws, n_params))

        total_iters = tune + draws
        for it in range(total_iters):
            for j in range(n_params):
                proposal = theta.copy()
                proposal[j] = specs[j].propose(theta[j], rng, step_scale[j])
                lp_new = log_posterior_fn(proposal, specs, *args)
                accepted = np.log(rng.uniform()) < (lp_new - lp)
                if accepted:
                    theta, lp = proposal, lp_new
                    if it >= tune:
                        n_accept += 1
                if it >= tune:
                    n_total += 1
                if it < tune:
                    batch_accept[j] += float(accepted)
                    batch_total[j] += 1

            if it < tune and (it + 1) % 50 == 0:
                # per-parameter step adaptation toward target_accept
                rate = batch_accept / np.maximum(batch_total, 1)
                step_scale *= np.exp(0.5 * (rate - target_accept))
                step_scale = np.clip(step_scale, 0.02, 5.0)
                batch_accept[:] = 0
                batch_total[:] = 0

            if it >= tune:
                samples[it - tune] = theta

            if progress_every and (it + 1) % progress_every == 0:
                print(f"  chain {c}: iter {it+1}/{total_iters}")

        all_chains[c] = samples
        accept_rates.append(n_accept / max(n_total, 1))

    return {
        "chains": all_chains,
        "param_names": [s.symbol for s in specs],
        "accept_rates": accept_rates,
    }


def split_rhat(chain_samples: np.ndarray) -> float:
    """Standard split-R-hat (Gelman-Rubin) for one parameter.
    chain_samples: shape (n_chains, n_draws)."""
    n_chains, n_draws = chain_samples.shape
    half = n_draws // 2
    split = chain_samples[:, :2 * half].reshape(n_chains * 2, half)
    means = split.mean(axis=1)
    variances = split.var(axis=1, ddof=1)
    W = variances.mean()
    B = half * means.var(ddof=1)
    var_hat = (1 - 1 / half) * W + B / half
    if W <= 0:
        return float("nan")
    return float(np.sqrt(var_hat / W))


def effective_sample_size(chain_samples: np.ndarray) -> float:
    """Rough ESS via the standard autocorrelation-sum estimator, pooling
    chains after centering each on its own mean."""
    n_chains, n_draws = chain_samples.shape
    centered = chain_samples - chain_samples.mean(axis=1, keepdims=True)
    acov = np.zeros(n_draws)
    for lag in range(n_draws):
        prod = (centered[:, :n_draws - lag] * centered[:, lag:]).mean()
        acov[lag] = prod
        if lag > 0 and acov[lag] < 0 and lag % 2 == 0:
            break
    var0 = acov[0]
    if var0 <= 0:
        return float("nan")
    rho = acov[1:len(acov)] / var0
    tau = 1 + 2 * rho.sum()
    n_total = n_chains * n_draws
    return float(n_total / max(tau, 1e-6))


def summarize(result: dict) -> "list[dict]":
    """Per-parameter summary: mean, sd, 3%/97% HDI-ish quantiles, R-hat, ESS."""
    chains = result["chains"]
    names = result["param_names"]
    rows = []
    for i, name in enumerate(names):
        samples_by_chain = chains[:, :, i]
        pooled = samples_by_chain.reshape(-1)
        rows.append({
            "param": name,
            "mean": float(pooled.mean()),
            "sd": float(pooled.std()),
            "q03": float(np.quantile(pooled, 0.03)),
            "q97": float(np.quantile(pooled, 0.97)),
            "r_hat": split_rhat(samples_by_chain),
            "ess": effective_sample_size(samples_by_chain),
        })
    return rows

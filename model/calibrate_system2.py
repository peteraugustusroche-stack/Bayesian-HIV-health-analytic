"""System 2 calibration: fit the 9 cascade hazards against the macro (p1/p2/p3)
and micro targets, with System 1 LOCKED.

Uncertainty propagates as a "cut": each of N draws from System 1 conditions its
own independent System 2 chain, and the chains are pooled. Cascade data therefore
cannot feed back into the natural-history parameters.

Run after calibrate_system1.py:  python3 calibrate_system2.py [cascade_year]
"""

from __future__ import annotations

import numpy as np

from params import load_parameters
from likelihoods import (
    build_specs, SYSTEM2_SYMBOLS, SYSTEM2_KINDS, CASCADE_YEAR_KEYS,
    REENGAGEMENT_TARGETS, HR_LTFU_TARGET, ART_INITIATION_TARGET,
    system2_log_likelihood, system2_model_implied, cascade_target,
)
from calibration_common import run_calibration, save_result, load_result


def lk_uses_retention() -> bool:
    """Cohort retention is a target only when actually fitted -- see
    likelihoods.USE_RETENTION_LIKELIHOOD."""
    import likelihoods
    return likelihoods.USE_RETENTION_LIKELIHOOD


def lk_uses_reeng() -> bool:
    """Mali re-engagement figures are targets only when actually fitted --
    see likelihoods.USE_REENGAGEMENT_LIKELIHOOD."""
    import likelihoods
    return likelihoods.USE_REENGAGEMENT_LIKELIHOOD


def lk_uses_supp2l() -> bool:
    """Second-line resuppression is a target only when actually fitted --
    see likelihoods.USE_SUPP2L_LIKELIHOOD."""
    import likelihoods
    return likelihoods.USE_SUPP2L_LIKELIHOOD


def lk_uses_hr() -> bool:
    """HR_LTFU is only a target when the likelihood actually uses it -- see
    likelihoods.USE_HR_LTFU_LIKELIHOOD. Otherwise it is reported as a
    diagnostic in model_implied, and listing it under targets would wrongly
    imply the calibration was asked to fit it."""
    import likelihoods
    return likelihoods.USE_HR_LTFU_LIKELIHOOD

RESULT_NAME = "system2"


def result_name_for_year(cascade_year: str) -> str:
    """Result-file base name for a given cascade_year -- '2019' (the base
    case) keeps the original 'system2' name so existing downstream code
    (health_economics posterior runs, scenario_analysis, app.py's default
    load) keeps working unchanged; any other year gets a distinguishing
    suffix so it never overwrites the base case."""
    return RESULT_NAME if cascade_year == "2019" else f"{RESULT_NAME}_sens{cascade_year}"


def _make_log_posterior(nh_draw, params, symbols, cascade_year):
    def log_post(theta, specs):
        lp = sum(spec.log_prior(theta[i]) for i, spec in enumerate(specs))
        if not np.isfinite(lp):
            return -np.inf
        cascade_vals = dict(zip(symbols, theta))
        ll = system2_log_likelihood(cascade_vals, nh_draw[0], nh_draw[1], params,
                                     cascade_year=cascade_year)
        return lp + ll
    return log_post


def _make_log_likelihood(nh_draw, params, symbols, cascade_year):
    def loglik(*values):
        cascade_vals = dict(zip(symbols, values))
        return system2_log_likelihood(cascade_vals, nh_draw[0], nh_draw[1], params,
                                       cascade_year=cascade_year)
    return loglik


# Per-outer-draw sampling budget, sized for DEMetropolisZ, which needs a long
# enough tune phase to populate its proposal archive. The previous 300/300/2
# (tuned for the single-component fallback) did not converge: pooled R-hat 1.5-2.4.


def _block_convergence(chain_pieces, param_names, nh_draw_pieces):
    """Convergence diagnostics that respect the cut's block structure.

    R-hat on the POOLED chains is not a convergence diagnostic here. Each
    outer block is conditioned on its own (kappa_prog, kappa_mort), so blocks
    target DIFFERENT distributions and between-block spread is a legitimate
    feature of the posterior rather than a failure to mix. Pooling the two
    together can make a badly-mixed sampler look acceptable.

    What is reported instead:
      within_block   R-hat among the chains inside one block, which DO share a
                     target. This is the real convergence diagnostic.
      between_share  the fraction of each parameter's total variance that is
                     between-block, i.e. how much natural-history level
                     uncertainty actually propagates into the cascade
                     posterior. This is a RESULT, not a diagnostic.
    """
    from mcmc_numpy import summarize
    import numpy as np

    within = {p: {"r_hat": [], "ess": []} for p in param_names}
    block_means = {p: [] for p in param_names}
    for blk in chain_pieces:
        for row in summarize({"chains": blk, "param_names": param_names}):
            within[row["param"]]["r_hat"].append(float(row["r_hat"]))
            within[row["param"]]["ess"].append(float(row["ess"]))
        for j, p in enumerate(param_names):
            block_means[p].append(float(blk[..., j].mean()))

    pooled = np.concatenate(chain_pieces, axis=0)
    out = {}
    for j, p in enumerate(param_names):
        r = np.array(within[p]["r_hat"])
        e = np.array(within[p]["ess"])
        total_var = float(pooled[..., j].var())
        between = float(np.var(block_means[p])) if len(block_means[p]) > 1 else 0.0
        out[p] = {
            "worst_within_r_hat": float(r.max()),
            "median_within_r_hat": float(np.median(r)),
            "total_ess": float(e.sum()),
            "between_block_variance_share": (between / total_var) if total_var else float("nan"),
        }
    return out


def convergence_report(conv: dict, threshold: float = 1.05) -> str:
    """Human-readable block diagnostics, with an explicit pass/fail gate."""
    lines = [f"{'parameter':<16}{'within r_hat':>14}{'(median)':>11}"
             f"{'total ESS':>12}{'kappa share':>13}"]
    lines.append("-" * 66)
    failed = []
    for p, d in conv.items():
        flag = ""
        if d["worst_within_r_hat"] > threshold:
            flag = "  <--"
            failed.append(p)
        lines.append(f"{p:<16}{d['worst_within_r_hat']:>14.3f}"
                     f"{d['median_within_r_hat']:>11.3f}{d['total_ess']:>12.0f}"
                     f"{d['between_block_variance_share']:>12.1%}{flag}")
    lines.append("")
    if failed:
        lines.append(f"NOT CONVERGED: {len(failed)} of {len(conv)} parameters have "
                     f"within-block R-hat > {threshold}")
        lines.append(f"  {', '.join(failed)}")
        lines.append("  Chains sharing an identical target have not found the same "
                     "region.")
        lines.append("  Raise draws_per_outer / tune_per_outer before quoting any "
                     "result from this run.")
    else:
        lines.append(f"Converged: every within-block R-hat is at or below {threshold}.")
    lines.append("")
    lines.append("'kappa share' is the fraction of each parameter's posterior variance")
    lines.append("that is between-block, i.e. attributable to natural-history level")
    lines.append("uncertainty. It is a result to report, not a diagnostic to pass.")
    return "\n".join(lines)


def run_system2_calibration(system1_result: dict, n_outer_draws=5,
                             draws_per_outer=6000, tune_per_outer=3000,
                             chains_per_outer=4, seed=1, backend="auto",
                             progress_every=None, cascade_year: str = "2019") -> dict:
    """cascade_year selects which p1/p2/p3 target symbols to calibrate
    against (see likelihoods.CASCADE_YEAR_KEYS): '2019' (UNAIDS Data 2020,
    p.40 -- default base case, closer in time to the 2016-2019 de Waal et
    al. CD4-at-ART-initiation data) or '2024' (UNAIDS 2025 Global AIDS
    Update), or 'WC2023' (Western Cape cyclical cascade, Euvrard et al. 2024
    -- individually-linked provincial records rather than programme-derived
    counts; see likelihoods.CASCADE_TARGET_OVERRIDES for the derivation).
    2024 and WC2023 carry the SAME p1 and differ only in p2/p3, so running
    them as a pair isolates measurement method from calendar time. Every
    other target (retention, CD4-at-init, Supp_2L, ART_1m) is unaffected by
    this choice."""
    if cascade_year not in CASCADE_YEAR_KEYS:
        raise ValueError(f"cascade_year must be one of {list(CASCADE_YEAR_KEYS)}, got {cascade_year!r}")
    params = load_parameters()
    specs = build_specs(params, SYSTEM2_SYMBOLS, SYSTEM2_KINDS)

    s1_names = system1_result["param_names"]
    s1_chains = system1_result["chains"]  # (chains, draws, 2)
    s1_flat = s1_chains.reshape(-1, s1_chains.shape[-1])
    lb_idx = s1_names.index("κ_prog")
    al_idx = s1_names.index("κ_mort")

    rng = np.random.default_rng(seed)
    outer_row_idx = rng.integers(0, len(s1_flat), size=n_outer_draws)

    import time as _time
    _t_block = _time.perf_counter()
    chain_pieces = []
    nh_draw_pieces = []  # the (kappa_prog, kappa_mort) each chain-block was conditioned on
    backend_used = None
    for k, row in enumerate(outer_row_idx):
        nh_draw = (float(s1_flat[row, lb_idx]), float(s1_flat[row, al_idx]))
        piece = run_calibration(
            log_posterior_fn=_make_log_posterior(nh_draw, params, SYSTEM2_SYMBOLS, cascade_year),
            log_likelihood_fn=_make_log_likelihood(nh_draw, params, SYSTEM2_SYMBOLS, cascade_year),
            specs=specs, extra_args=(),
            draws=draws_per_outer, tune=tune_per_outer, chains=chains_per_outer,
            seed=(seed * 1000 + k), backend=backend,
        )
        chain_pieces.append(piece["chains"])
        n_chains_this_piece = piece["chains"].shape[0]
        nh_draw_pieces.append(np.tile(np.array(nh_draw), (n_chains_this_piece, 1)))
        backend_used = piece["backend"]
        _now = _time.perf_counter()
        _elapsed = _now - _t_block
        _ms = _elapsed / ((draws_per_outer + tune_per_outer) * chains_per_outer) * 1000
        print(f"  outer draw {k+1}/{n_outer_draws} done in {_elapsed:.0f}s "
              f"({_ms:.0f} ms/eval)", flush=True)
        _t_block = _now

    pooled_chains = np.concatenate(chain_pieces, axis=0)  # (n_outer*chains_per_outer, draws, 7)
    # nh_draws_by_chain[c] = the (kappa_prog, kappa_mort) pair chain c was run
    # under -- every draw within chain c shares this pair, since it was fixed
    # for that whole chain. Needed to reconstruct valid *joint* posterior draws
    # (nh params + cascade params) downstream in the health-econ posterior run
    # -- pairing a cascade draw with an unrelated nh draw would be invalid.
    nh_draws_by_chain = np.concatenate(nh_draw_pieces, axis=0)  # (n_chains_total, 2)

    from mcmc_numpy import summarize
    param_names = [s.symbol for s in specs]
    summary = summarize({"chains": pooled_chains, "param_names": param_names})

    result = {
        "backend": backend_used,
        "param_names": param_names,
        "chains": pooled_chains,
        "summary": summary,
        "n_outer_draws": n_outer_draws,
        "nh_draws_by_chain": nh_draws_by_chain,
        "nh_param_names": ["κ_prog", "κ_mort"],
        "cascade_year": cascade_year,
        "chains_per_outer": chains_per_outer,
        "draws_per_outer": draws_per_outer,
        # Block-aware diagnostics -- see _block_convergence for why pooled
        # R-hat is not the right quantity in this architecture.
        "convergence": _block_convergence(chain_pieces, param_names, nh_draw_pieces),
    }

    means = {row["param"]: row["mean"] for row in summary}
    lb_mean = float(s1_flat[:, lb_idx].mean())
    al_mean = float(s1_flat[:, al_idx].mean())
    result["model_implied_at_mean"] = system2_model_implied(means, lb_mean, al_mean, params)
    # "targets" is keyed generically (p1/p2/p3, not p1_2019/p1 etc.) so it
    # lines up 1:1 with model_implied_at_mean's keys regardless of which
    # cascade_year was actually used -- the *values* are pulled from the
    # year-specific spreadsheet symbols selected by cascade_year.
    year_keys = CASCADE_YEAR_KEYS[cascade_year]
    # Only quantities the likelihood ACTUALLY fits belong in targets; anything
    # else is a diagnostic and listing it here would imply the calibration was
    # asked to match it.
    non_cascade_keys = ["CD4_median_ART", "CD4_lt200_ART"]
    if lk_uses_supp2l():
        non_cascade_keys += ["Supp_2L"]
    if lk_uses_retention():
        non_cascade_keys += ["Ret_12m", "Ret_24m"]
    result["targets"] = {generic: cascade_target(params, actual)
                          for generic, actual in year_keys.items()}
    # Provenance for the DISPLAY layer. The generic p1/p2/p3 keys above collide
    # with the 2024 spreadsheet symbols of the same name, so anything labelling
    # them straight from the spreadsheet reports the wrong year. Recording the
    # actual symbols used lets the UI label them correctly.
    result["target_symbols"] = dict(year_keys)
    result["targets"].update({k: params.targets[k] for k in non_cascade_keys
                               if k in params.targets})
    # Targets defined in code rather than the spreadsheet (LTFU dynamics).
    if lk_uses_reeng():
        result["targets"].update({k: (c, sd) for k, (c, sd, _m) in REENGAGEMENT_TARGETS.items()})
    result["targets"][ART_INITIATION_TARGET[0]] = (ART_INITIATION_TARGET[1],
                                                    ART_INITIATION_TARGET[2])
    if lk_uses_hr():
        result["targets"][HR_LTFU_TARGET[0]] = (HR_LTFU_TARGET[1], HR_LTFU_TARGET[2])
    return result


if __name__ == "__main__":
    import sys

    cascade_year = sys.argv[1] if len(sys.argv) > 1 else "2019"

    s1 = load_result("system1")
    if s1 is None:
        raise SystemExit("No System 1 result found -- run calibrate_system1.py first.")

    print(f"Running System 2 calibration (cascade_year={cascade_year}, "
          f"resampling System 1's locked posterior)...")
    result = run_system2_calibration(s1, progress_every=5, cascade_year=cascade_year)
    print(f"\nBackend used: {result['backend']}  ({result['n_outer_draws']} outer draws)")
    for row in result["summary"]:
        print(f"  {row['param']}: mean={row['mean']:.4f} sd={row['sd']:.4f} "
              f"r_hat={row['r_hat']:.3f} ess={row['ess']:.0f}")

    print("\nBlock-aware convergence (pooled R-hat above is NOT a convergence "
          "diagnostic\nhere -- blocks target different distributions):")
    print(convergence_report(result["convergence"]))

    print("\nModel-implied (posterior mean) vs targets:")
    for key, (central, sd) in result["targets"].items():
        implied = result["model_implied_at_mean"][key]
        print(f"  {key}: implied={implied:.3f}  target={central} ± {sd}")

    path = save_result(result, result_name_for_year(cascade_year))
    print(f"\nSaved to {path}")

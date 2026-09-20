"""
Scenario / tornado analysis: how much do DALYs and cost move if each
cascade parameter individually improves or worsens by a fixed relative
amount, holding everything else fixed? This calls evaluate_health_econ()
with perturbed parameter dicts -- no changes to the underlying engine.

Three variants, all read-only compositions of existing functions -- none of
this touches simulate_cascade.py / health_economics.py / calibrate_*.py:
- POINT: perturbs the posterior MEAN of each cascade parameter by a FIXED
  +-20%, one at a time (14 evaluations total, <1s). This is a "policy
  scenario" framing -- what's the DALY payoff of a program moving this
  cascade behavior by a fixed relative amount? -- NOT a classic uncertainty
  DSA, since every parameter gets the same +-20% regardless of how
  certain/uncertain we actually are about its value.
- PROBABILISTIC: for each of ~250 posterior draws, perturbs that SAME
  draw's parameters by +-20% and compares to that same draw's own
  baseline (paired/common-random-numbers design -- cancels out noise from
  the other 8 parameters varying draw to draw, isolating the effect of the
  one parameter actually changed). Produces a distribution of DALY impact
  per parameter that properly reflects posterior uncertainty in everything
  else, rather than treating the posterior mean as ground truth. Still a
  fixed +-20% shift, same policy-scenario framing as POINT.
- UNCERTAINTY: the classic DSA convention -- each cascade parameter is
  varied at its OWN posterior uncertainty bound (q03/q97, reusing the same
  ~94% credible interval already shown elsewhere in the app), holding the
  other parameters at their means. Answers "how robust is our DALY estimate
  to what we don't know about this parameter?" -- a different question from
  POINT/PROBABILISTIC's "what's the payoff of a fixed policy shift?".

Both POINT and UNCERTAINTY also report the resulting shift in the macro/
micro cascade TARGETS (p1/p2/p3/T_ART/retention/Supp_2L) alongside the DALY
impact, via likelihoods.system2_model_implied() -- since the perturbed
quantities are the underlying hazard *components*, not the targets
themselves, and that translation is nonlinear (competing risks, saturation),
it's worth seeing explicitly rather than just the downstream DALY number.

Elasticity = (%change in metric) / (%change in parameter), using the full
+20%/-20% swing (symmetric arc elasticity) -- lets you compare parameters
that live on very different scales (e.g. lambda_init ~8/yr vs mu_vf1 ~0.01/yr).

Run directly (after both calibrations have been run):
    python3 scenario_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from params import load_parameters
from health_economics import evaluate_health_econ
from likelihoods import (
    SYSTEM1_SYMBOLS, SYSTEM1_KINDS, SYSTEM2_SYMBOLS, SYSTEM2_KINDS,
    build_specs, system2_model_implied,
)

RESULTS_DIR = Path(__file__).parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
POINT_RESULT_PATH = RESULTS_DIR / "scenario_point.npz"
PROB_RESULT_PATH = RESULTS_DIR / "scenario_prob.npz"
UNCERTAINTY_RESULT_PATH = RESULTS_DIR / "scenario_uncertainty.npz"
PRIOR_POINT_RESULT_PATH = RESULTS_DIR / "scenario_prior_point.npz"

PERTURBATION = 0.20
TARGET_METRICS = ["DALYs_per_acquisition", "DALYs_avoided_through_care", "cost_per_acquisition"]
TARGET_KEYS = ["p1", "p2", "p3", "T_ART", "Ret_12m", "Ret_24m",
               "CD4_median_ART", "CD4_lt200_ART", "Supp_2L"]

# Which direction counts as a programmatic "improvement" for each cascade
# hazard -- for readability only, doesn't affect the computed numbers.
IMPROVEMENT_DIRECTION = {
    "δ_bg": "increase", "δ_symp": "increase", "γ_diag": "n/a (shape param)",
    "λ_init": "increase",
    "μ_vf1": "decrease", "μ_ltfu": "decrease", "μ_re-engage": "increase",
    "μ_switch": "increase",
    # Added by the LTFU split; it is an epidemiological scaling of mortality
    # while disengaged, not a programme lever, so it has no policy direction.
    "φ_ltfu": "n/a (mortality scaling)",
}


def _evaluate(cascade_vals, kappa_prog, kappa_mort, params):
    return evaluate_health_econ(cascade_vals, kappa_prog, kappa_mort, params)


def run_point_tornado(system1_result, system2_result, params=None,
                       perturbation=PERTURBATION) -> dict:
    params = params or load_parameters()
    lb_mean = {row["param"]: row["mean"] for row in system1_result["summary"]}
    cascade_mean = {row["param"]: row["mean"] for row in system2_result["summary"]}

    baseline = _evaluate(cascade_mean, lb_mean["κ_prog"], lb_mean["κ_mort"], params)
    baseline_targets = system2_model_implied(cascade_mean, lb_mean["κ_prog"], lb_mean["κ_mort"], params)

    rows = []
    for sym in SYSTEM2_SYMBOLS:
        up_vals = dict(cascade_mean); up_vals[sym] = cascade_mean[sym] * (1 + perturbation)
        down_vals = dict(cascade_mean); down_vals[sym] = cascade_mean[sym] * (1 - perturbation)
        up = _evaluate(up_vals, lb_mean["κ_prog"], lb_mean["κ_mort"], params)
        down = _evaluate(down_vals, lb_mean["κ_prog"], lb_mean["κ_mort"], params)
        up_targets = system2_model_implied(up_vals, lb_mean["κ_prog"], lb_mean["κ_mort"], params)
        down_targets = system2_model_implied(down_vals, lb_mean["κ_prog"], lb_mean["κ_mort"], params)

        row = {"param": sym, "direction_of_improvement": IMPROVEMENT_DIRECTION[sym]}
        for metric in TARGET_METRICS:
            row[f"{metric}_baseline"] = baseline[metric]
            row[f"{metric}_up"] = up[metric]
            row[f"{metric}_down"] = down[metric]
            pct_change_metric = ((up[metric] - down[metric]) / baseline[metric]
                                  if baseline[metric] else float("nan"))
            row[f"{metric}_elasticity"] = pct_change_metric / (2 * perturbation)
        for tk in TARGET_KEYS:
            row[f"target_{tk}_baseline"] = baseline_targets[tk]
            row[f"target_{tk}_up"] = up_targets[tk]
            row[f"target_{tk}_down"] = down_targets[tk]
        rows.append(row)

    return {"baseline": baseline, "baseline_targets": baseline_targets,
            "rows": rows, "perturbation": perturbation}


def run_uncertainty_tornado(system1_result, system2_result, params=None) -> dict:
    """Classic DSA convention: vary each cascade parameter at its OWN
    posterior uncertainty bound (q03/q97 -- the same ~94% credible interval
    already shown in the System 2 tab's summary table), holding the other
    parameters at their posterior means. Distinct question from the point/
    probabilistic tornadoes: this asks how robust the DALY estimate is to
    what we don't know about each parameter, not what a policy-driven shift
    would achieve."""
    params = params or load_parameters()
    lb_mean = {row["param"]: row["mean"] for row in system1_result["summary"]}
    cascade_mean = {row["param"]: row["mean"] for row in system2_result["summary"]}
    cascade_bounds = {row["param"]: (row["q03"], row["q97"]) for row in system2_result["summary"]}

    baseline = _evaluate(cascade_mean, lb_mean["κ_prog"], lb_mean["κ_mort"], params)
    baseline_targets = system2_model_implied(cascade_mean, lb_mean["κ_prog"], lb_mean["κ_mort"], params)

    rows = []
    for sym in SYSTEM2_SYMBOLS:
        q03, q97 = cascade_bounds[sym]
        low_vals = dict(cascade_mean); low_vals[sym] = q03
        high_vals = dict(cascade_mean); high_vals[sym] = q97
        low = _evaluate(low_vals, lb_mean["κ_prog"], lb_mean["κ_mort"], params)
        high = _evaluate(high_vals, lb_mean["κ_prog"], lb_mean["κ_mort"], params)
        low_targets = system2_model_implied(low_vals, lb_mean["κ_prog"], lb_mean["κ_mort"], params)
        high_targets = system2_model_implied(high_vals, lb_mean["κ_prog"], lb_mean["κ_mort"], params)

        row = {"param": sym, "q03": q03, "q97": q97, "mean": cascade_mean[sym]}
        for metric in TARGET_METRICS:
            row[f"{metric}_baseline"] = baseline[metric]
            row[f"{metric}_at_q03"] = low[metric]
            row[f"{metric}_at_q97"] = high[metric]
            row[f"{metric}_range"] = abs(high[metric] - low[metric])
        for tk in TARGET_KEYS:
            row[f"target_{tk}_at_q03"] = low_targets[tk]
            row[f"target_{tk}_at_q97"] = high_targets[tk]
        rows.append(row)

    return {"baseline": baseline, "baseline_targets": baseline_targets, "rows": rows}


def _perturb_value(spec, direction: int, perturbation: float) -> float:
    """direction is +1 or -1. Lognormal (positive, multiplicative) parameters
    get the standard +/-20% relative shift. Normal parameters can't take a
    relative shift when centred at zero (any percentage of zero is zero), so
    they instead shift by +/- (perturbation * that parameter's own prior SD),
    i.e. the same 20% figure is read as '20% of one prior SD'. Flagged in each
    row's 'perturbation_kind' field so it's visible in the output.

    Every parameter is lognormal now -- the one Normal parameter was
    alpha_shape, which went with the parametric CD4-decline hazard when
    natural history became a sourced table. The branch is kept because
    nothing guarantees the next parameter added will be positive.

    NOTE on reading the tornado: kappa_prog and kappa_mort are scale factors
    on published schedules, both centred at 1, so a +/-20% shift moves the
    LEVEL of every progression or mortality rate at once. That is a blunter
    instrument than perturbing a single cascade hazard, and their bars should
    be read as "how much would it matter if the whole published
    natural-history schedule were 20% off" -- not as a parameter this study
    could have estimated better."""
    if spec.kind == "lognormal":
        return spec.central * (1 + direction * perturbation)
    return spec.central + direction * perturbation * spec.sd


def run_prior_point_tornado(params=None, perturbation=PERTURBATION) -> dict:
    """Same construction as run_point_tornado, but BEFORE calibration --
    uses the spreadsheet's prior central estimates for all 9 free parameters
    (the 2 natural-history parameters plus the 7 cascade parameters), not a
    fitted posterior. Lets you see which parameters the model is most
    sensitive to before any calibration data has touched it, and compare
    that against the same view post-calibration (Scenario Analysis tab) to
    see how the picture changes as you step through the pipeline."""
    params = params or load_parameters()
    specs = (build_specs(params, SYSTEM1_SYMBOLS, SYSTEM1_KINDS)
             + build_specs(params, SYSTEM2_SYMBOLS, SYSTEM2_KINDS))
    spec_by_symbol = {s.symbol: s for s in specs}
    central = {s.symbol: s.central for s in specs}

    def _split(vals):
        return ({k: vals[k] for k in SYSTEM2_SYMBOLS}, vals["κ_prog"], vals["κ_mort"])

    cascade_c, lb_c, al_c = _split(central)
    baseline = _evaluate(cascade_c, lb_c, al_c, params)
    baseline_targets = system2_model_implied(cascade_c, lb_c, al_c, params)

    rows = []
    for sym in central:
        spec = spec_by_symbol[sym]
        up_central = dict(central); up_central[sym] = _perturb_value(spec, +1, perturbation)
        down_central = dict(central); down_central[sym] = _perturb_value(spec, -1, perturbation)

        cascade_up, lb_up, al_up = _split(up_central)
        cascade_down, lb_down, al_down = _split(down_central)
        up = _evaluate(cascade_up, lb_up, al_up, params)
        down = _evaluate(cascade_down, lb_down, al_down, params)
        up_targets = system2_model_implied(cascade_up, lb_up, al_up, params)
        down_targets = system2_model_implied(cascade_down, lb_down, al_down, params)

        row = {
            "param": sym,
            "direction_of_improvement": IMPROVEMENT_DIRECTION.get(sym, "n/a (biological parameter)"),
            "perturbation_kind": ("±20% relative" if spec.kind == "lognormal"
                                   else "±20% of prior SD (relative undefined at central=0)"),
        }
        for metric in TARGET_METRICS:
            row[f"{metric}_baseline"] = baseline[metric]
            row[f"{metric}_up"] = up[metric]
            row[f"{metric}_down"] = down[metric]
            pct_change_metric = ((up[metric] - down[metric]) / baseline[metric]
                                  if baseline[metric] else float("nan"))
            row[f"{metric}_elasticity"] = (pct_change_metric / (2 * perturbation)
                                            if spec.kind == "lognormal" else float("nan"))
        for tk in TARGET_KEYS:
            row[f"target_{tk}_baseline"] = baseline_targets[tk]
            row[f"target_{tk}_up"] = up_targets[tk]
            row[f"target_{tk}_down"] = down_targets[tk]
        rows.append(row)

    return {"baseline": baseline, "baseline_targets": baseline_targets,
            "rows": rows, "perturbation": perturbation}


def save_prior_point_result(result: dict) -> Path:
    np.savez(PRIOR_POINT_RESULT_PATH, payload=np.array(result, dtype=object))
    return PRIOR_POINT_RESULT_PATH


def load_prior_point_result() -> dict | None:
    if not PRIOR_POINT_RESULT_PATH.exists():
        return None
    return np.load(PRIOR_POINT_RESULT_PATH, allow_pickle=True)["payload"].item()


def save_uncertainty_result(result: dict) -> Path:
    np.savez(UNCERTAINTY_RESULT_PATH, payload=np.array(result, dtype=object))
    return UNCERTAINTY_RESULT_PATH


def load_uncertainty_result() -> dict | None:
    if not UNCERTAINTY_RESULT_PATH.exists():
        return None
    return np.load(UNCERTAINTY_RESULT_PATH, allow_pickle=True)["payload"].item()


def run_probabilistic_tornado(system2_result, params=None, n_draws=250,
                               perturbation=PERTURBATION, seed=1,
                               progress_every=None) -> dict:
    params = params or load_parameters()
    chains = system2_result["chains"]
    param_names = system2_result["param_names"]
    nh_draws_by_chain = system2_result["nh_draws_by_chain"]

    n_chains, n_draws_per_chain, _ = chains.shape
    total_available = n_chains * n_draws_per_chain
    rng = np.random.default_rng(seed)
    flat_idx = rng.integers(0, total_available, size=n_draws)
    chain_idx = flat_idx // n_draws_per_chain
    draw_idx = flat_idx % n_draws_per_chain

    impact_up = {sym: {m: np.zeros(n_draws) for m in TARGET_METRICS} for sym in SYSTEM2_SYMBOLS}
    impact_down = {sym: {m: np.zeros(n_draws) for m in TARGET_METRICS} for sym in SYSTEM2_SYMBOLS}
    elasticity = {sym: {m: np.zeros(n_draws) for m in TARGET_METRICS} for sym in SYSTEM2_SYMBOLS}
    baseline_vals = {m: np.zeros(n_draws) for m in TARGET_METRICS}

    for i in range(n_draws):
        c, d = int(chain_idx[i]), int(draw_idx[i])
        cascade_vals = {name: float(chains[c, d, j]) for j, name in enumerate(param_names)}
        kappa_prog, kappa_mort = (float(x) for x in nh_draws_by_chain[c])
        baseline = _evaluate(cascade_vals, kappa_prog, kappa_mort, params)
        for m in TARGET_METRICS:
            baseline_vals[m][i] = baseline[m]

        for sym in SYSTEM2_SYMBOLS:
            up_vals = dict(cascade_vals); up_vals[sym] = cascade_vals[sym] * (1 + perturbation)
            down_vals = dict(cascade_vals); down_vals[sym] = cascade_vals[sym] * (1 - perturbation)
            up = _evaluate(up_vals, kappa_prog, kappa_mort, params)
            down = _evaluate(down_vals, kappa_prog, kappa_mort, params)
            for m in TARGET_METRICS:
                impact_up[sym][m][i] = up[m] - baseline[m]
                impact_down[sym][m][i] = down[m] - baseline[m]
                pct_change_metric = (up[m] - down[m]) / baseline[m] if baseline[m] else float("nan")
                elasticity[sym][m][i] = pct_change_metric / (2 * perturbation)

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  draw {i+1}/{n_draws}")

    baseline_mean = {m: float(baseline_vals[m].mean()) for m in TARGET_METRICS}

    # Absolute outcome VALUES (not deltas) at each bound, re-centered on the
    # pooled baseline mean -- this is what lets the plot draw a proper
    # tornado bar (low-value -> high-value) instead of a delta forest plot,
    # while impact_*/elasticity_* below are kept too since the delta form is
    # still the more direct way to read "how much did this move DALYs".
    summary = []
    for sym in SYSTEM2_SYMBOLS:
        for m in TARGET_METRICS:
            up_v, down_v, el_v = impact_up[sym][m], impact_down[sym][m], elasticity[sym][m]
            bm = baseline_mean[m]
            summary.append({
                "param": sym, "metric": m, "direction_of_improvement": IMPROVEMENT_DIRECTION[sym],
                "impact_up_mean": float(up_v.mean()), "impact_up_q05": float(np.quantile(up_v, 0.05)),
                "impact_up_q95": float(np.quantile(up_v, 0.95)),
                "impact_down_mean": float(down_v.mean()), "impact_down_q05": float(np.quantile(down_v, 0.05)),
                "impact_down_q95": float(np.quantile(down_v, 0.95)),
                "elasticity_mean": float(el_v.mean()), "elasticity_q05": float(np.quantile(el_v, 0.05)),
                "elasticity_q95": float(np.quantile(el_v, 0.95)),
                "value_up_mean": bm + float(up_v.mean()),
                "value_up_q05": bm + float(np.quantile(up_v, 0.05)),
                "value_up_q95": bm + float(np.quantile(up_v, 0.95)),
                "value_down_mean": bm + float(down_v.mean()),
                "value_down_q05": bm + float(np.quantile(down_v, 0.05)),
                "value_down_q95": bm + float(np.quantile(down_v, 0.95)),
            })

    return {"n_draws": n_draws, "perturbation": perturbation, "summary": summary,
            "baseline_mean": baseline_mean}


def save_point_result(result: dict) -> Path:
    np.savez(POINT_RESULT_PATH, payload=np.array(result, dtype=object))
    return POINT_RESULT_PATH


def load_point_result() -> dict | None:
    if not POINT_RESULT_PATH.exists():
        return None
    return np.load(POINT_RESULT_PATH, allow_pickle=True)["payload"].item()


def save_prob_result(result: dict) -> Path:
    np.savez(PROB_RESULT_PATH, payload=np.array(result, dtype=object))
    return PROB_RESULT_PATH


def load_prob_result() -> dict | None:
    if not PROB_RESULT_PATH.exists():
        return None
    return np.load(PROB_RESULT_PATH, allow_pickle=True)["payload"].item()


if __name__ == "__main__":
    from calibration_common import load_result

    s1 = load_result("system1")
    s2 = load_result("system2")
    if s1 is None or s2 is None:
        raise SystemExit("Run calibrate_system1.py and calibrate_system2.py first.")

    params = load_parameters()

    print("Point tornado (posterior means, ±20%)...")
    point = run_point_tornado(s1, s2, params)
    print(f"\nBaseline DALYs_per_acquisition = {point['baseline']['DALYs_per_acquisition']:.3f}"
          f"   Baseline p1/p2/p3 = {point['baseline_targets']['p1']:.3f}/"
          f"{point['baseline_targets']['p2']:.3f}/{point['baseline_targets']['p3']:.3f}\n")
    print(f"{'param':<14}{'DALYs@-20%':>12}{'DALYs@+20%':>12}{'elasticity':>12}"
          f"{'p1@-20%':>10}{'p1@+20%':>10}")
    for row in sorted(point["rows"], key=lambda r: -abs(r["DALYs_per_acquisition_elasticity"])):
        print(f"{row['param']:<14}{row['DALYs_per_acquisition_down']:>12.3f}"
              f"{row['DALYs_per_acquisition_up']:>12.3f}"
              f"{row['DALYs_per_acquisition_elasticity']:>12.3f}"
              f"{row['target_p1_down']:>10.3f}{row['target_p1_up']:>10.3f}")
    save_point_result(point)

    print("\nUncertainty-bound tornado (each parameter at its own posterior q03/q97)...")
    unc = run_uncertainty_tornado(s1, s2, params)
    print(f"{'param':<14}{'q03':>10}{'q97':>10}{'DALYs@q03':>12}{'DALYs@q97':>12}")
    for row in sorted(unc["rows"], key=lambda r: -r["DALYs_per_acquisition_range"]):
        print(f"{row['param']:<14}{row['q03']:>10.4f}{row['q97']:>10.4f}"
              f"{row['DALYs_per_acquisition_at_q03']:>12.3f}"
              f"{row['DALYs_per_acquisition_at_q97']:>12.3f}")
    save_uncertainty_result(unc)

    print("\nProbabilistic tornado (250 paired posterior draws, ±20%)...")
    prob = run_probabilistic_tornado(s2, params, n_draws=250, progress_every=50)
    print(f"\n{'param':<14}{'metric':<28}{'impact +20%':>14}{'[q05,q95]':>22}")
    for row in prob["summary"]:
        if row["metric"] != "DALYs_per_acquisition":
            continue
        print(f"{row['param']:<14}{row['metric']:<28}{row['impact_up_mean']:>14.3f}"
              f"   [{row['impact_up_q05']:.3f}, {row['impact_up_q95']:.3f}]")
    save_prob_result(prob)
    print("\nSaved.")

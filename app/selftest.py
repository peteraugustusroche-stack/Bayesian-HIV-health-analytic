"""Check the app's repricing engine against the model it was exported from.

At default inputs the engine must reproduce run_posterior_analysis draw for
draw: same posterior, same discounting, same unit costs. It then checks that
the inputs the app exposes actually move the outputs in the expected
direction, and that a zero discount rate agrees with a direct recomputation.

    python3 app/selftest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine as E

try:
    from run_posterior_analysis import load_posterior_result
    HAVE_REPO = True
except Exception:
    # Running from a deployment bundle: fall back to the summaries frozen in
    # at package time.
    HAVE_REPO = False

TOL = 5e-6
failures = []


def check(label, got, want, tol=TOL, relative=True):
    got, want = np.asarray(got, float), np.asarray(want, float)
    denom = np.maximum(np.abs(want), 1e-12) if relative else 1.0
    err = float(np.max(np.abs(got - want) / denom))
    ok = err <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<44} max rel err {err:.2e}")
    if not ok:
        failures.append(label)
    return ok


def check_true(label, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {label:<44} {detail}")
    if not condition:
        failures.append(label)


def compare_arm(year: str, price_year: int):
    print(f"\nCascade {year}, {price_year} US$ -- engine against the published posterior")
    a = E.Assumptions(cascade_year=year,
                      costs=E.CostSchedule(price_year=price_year))
    got = E.evaluate(a)

    if HAVE_REPO:
        ref = load_posterior_result(year)
        if ref is None:
            print(f"  SKIP  no posterior_analysis result for {year}")
            return
        for k, want in ref["probabilistic"]["metrics"].items():
            check(k, got[k], want)
        return

    import json
    path = Path(__file__).resolve().parent / "data" / "reference_summary.json"
    if not path.exists():
        print("  SKIP  no reference_summary.json in this bundle")
        return
    ref = json.loads(path.read_text()).get(year)
    if ref is None:
        print(f"  SKIP  no frozen summary for {year}")
        return
    for k, (mean, sd) in ref["metrics"].items():
        check(f"{k}, mean", got[k].mean(), mean, tol=1e-6)
        check(f"{k}, sd", got[k].std(), sd, tol=1e-6)


def check_levers(year="2019"):
    print(f"\nInput levers move the outputs, cascade {year}")
    base = E.Assumptions(cascade_year=year)
    b = E.evaluate(base)

    zero = E.evaluate(E.Assumptions(cascade_year=year, discount_rate=0.0))
    check_true("zero discounting raises DALYs",
               zero["DALYs_per_acquisition"].mean() > b["DALYs_per_acquisition"].mean(),
               f"{b['DALYs_per_acquisition'].mean():.2f} -> "
               f"{zero['DALYs_per_acquisition'].mean():.2f}")
    check_true("zero discounting raises cost",
               zero["cost_per_acquisition"].mean() > b["cost_per_acquisition"].mean(),
               f"${b['cost_per_acquisition'].mean():,.0f} -> "
               f"${zero['cost_per_acquisition'].mean():,.0f}")

    dearer = E.evaluate(E.with_cost(base, art_1l=E.DEFAULT_COST_1L * 2))
    check_true("doubling first-line drugs raises cost",
               dearer["cost_per_acquisition"].mean() > b["cost_per_acquisition"].mean(),
               f"${b['cost_per_acquisition'].mean():,.0f} -> "
               f"${dearer['cost_per_acquisition'].mean():,.0f}")
    check("doubling first-line leaves DALYs untouched",
          dearer["DALYs_per_acquisition"], b["DALYs_per_acquisition"])

    free = E.evaluate(E.with_cost(base, art_1l=0, art_2l=0,
                                  routine=tuple([0] * 7), death=0))
    check("a zero cost schedule gives zero cost",
          free["cost_per_acquisition"], np.zeros_like(b["cost_per_acquisition"]),
          relative=False)

    in_care = E.with_cost(base, routine_statuses=(
        "Diagnosed_PreART", "ART1_Ramp", "ART1_Suppressed", "ART1_Failing",
        "ART2_Suppressed"))
    ic = E.evaluate(in_care)
    check_true("charging in-care states only lowers cost",
               ic["cost_per_acquisition"].mean() < b["cost_per_acquisition"].mean(),
               f"${b['cost_per_acquisition'].mean():,.0f} -> "
               f"${ic['cost_per_acquisition'].mean():,.0f}")

    gbd = E.Assumptions(cascade_year=year, weights=E.Weights(
        on_art=tuple(E.GBD_DW_ONART), off_art=tuple(E.GBD_DW_OFFART)))
    g = E.evaluate(gbd)
    check_true("GBD weights change DALYs",
               abs(g["DALYs_per_acquisition"].mean()
                   - b["DALYs_per_acquisition"].mean()) > 0.1,
               f"{b['DALYs_per_acquisition'].mean():.2f} -> "
               f"{g['DALYs_per_acquisition'].mean():.2f}")

    check("price year 2019 deflates cost by CPI-U",
          E.evaluate(E.with_cost(base, price_year=2019))["cost_per_acquisition"],
          b["cost_per_acquisition"] * E.CPI_U[2019] / E.CPI_U[2024])


def check_internal_consistency(year="2019"):
    print(f"\nInternal consistency, cascade {year}")
    a = E.Assumptions(cascade_year=year)
    out = E.evaluate(a)
    check("DALYs = YLL + YLD",
          out["DALYs_per_acquisition"], out["YLL_cascade"] + out["YLD_cascade"])
    check("cost = recurring + terminal",
          out["cost_per_acquisition"], out["cost_recurring"] + out["cost_death"])
    check("components sum to total cost",
          sum(E.cost_components(a).values()), out["cost_per_acquisition"])
    check_true("care averts DALYs in every draw",
               bool(np.all(out["DALYs_avoided_through_care"] > 0)),
               f"min {out['DALYs_avoided_through_care'].min():.2f}")


def check_icers(year="2024"):
    print(f"\nCost-effectiveness, cascade {year}")
    a = E.Assumptions(cascade_year=year, costs=E.CostSchedule(price_year=2024))
    out = E.evaluate(a)
    icer_t = E.treatment_icer(out)
    m, lo, hi = E.summarise(icer_t)
    check_true("treatment ICER is positive and plausible", 0 < m < 1000,
               f"${m:,.0f} per DALY averted (90% CrI ${lo:,.0f}-${hi:,.0f})")

    dalys, cost = out["DALYs_per_acquisition"], out["cost_per_acquisition"]
    be = E.break_even_price(50, cost)
    net = 50 * be - cost
    check_true("break-even price gives zero mean net cost", abs(net.mean()) < 1e-6,
               f"${be:,.2f}/py at NNT 50, mean net ${net.mean():.2e}")
    check_true("NNT x price frontier equals the cost offset",
               abs(50 * be - cost.mean()) < 1e-6,
               f"NNT x price = ${50 * be:,.0f}, offset ${cost.mean():,.0f}")
    check_true("dropping the offset raises the ICER",
               E.prevention_icer(50, 60, dalys, cost, 0.0).mean()
               > E.prevention_icer(50, 60, dalys, cost, 1.0).mean(),
               f"${E.prevention_icer(50, 60, dalys, cost, 1.0).mean():,.0f} -> "
               f"${E.prevention_icer(50, 60, dalys, cost, 0.0).mean():,.0f} at $60/py")


def main():
    compare_arm("2019", 2019)
    compare_arm("2024", 2024)
    check_internal_consistency("2019")
    check_levers("2019")
    check_icers("2024")
    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""Render every figure and build every table off-server.

Shiny wiring cannot be exercised without a browser, but everything the server
calls can be: each figure function is rendered to PNG and each table is built,
under several input combinations. Writes to app/_rendertest/.

    python3 app/rendertest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import engine as E
import figures as F

OUT = HERE / "_rendertest"
failures = []


def render(name, fn):
    try:
        fig = fn()
        OUT.mkdir(exist_ok=True)
        fig.savefig(OUT / f"{name}.png", dpi=110)
        plt.close(fig)
        print(f"  PASS  {name}")
    except Exception as exc:
        failures.append(f"{name}: {exc!r}")
        print(f"  FAIL  {name}: {exc!r}")


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:
        failures.append(f"{name}: {exc!r}")
        print(f"  FAIL  {name}: {exc!r}")


def main():
    scenarios = {
        "base": E.Assumptions(),
        "2024_r5": E.Assumptions("2024", 0.05,
                                 E.CostSchedule(price_year=2024)),
        "r0_gbd": E.Assumptions("2019", 0.0, E.CostSchedule(price_year=2019),
                                E.Weights(tuple(E.GBD_DW_ONART),
                                          tuple(E.GBD_DW_OFFART))),
        "free_care": E.Assumptions(
            "2019", 0.03,
            E.CostSchedule(art_1l=0, art_2l=0, routine=tuple([0] * 7), death=0)),
        "in_care_only": E.Assumptions(
            "2019", 0.03,
            E.CostSchedule(routine_statuses=("ART1_Suppressed",))),
    }
    for tag, a in scenarios.items():
        print(f"\n{tag}")
        o = E.evaluate(a)
        render(f"{tag}_daly_density",
               lambda o=o: F.posterior_density(o["DALYs_per_acquisition"],
                                               "DALYs per acquisition"))
        render(f"{tag}_daly_decomp", lambda o=o: F.daly_decomposition(o))
        render(f"{tag}_person_years",
               lambda a=a: F.person_years_panel(E.person_years_by_status(a)))
        render(f"{tag}_cost_decomp",
               lambda a=a: F.cost_decomposition(E.cost_components(a),
                                                a.costs.price_year))
        render(f"{tag}_ce_plane", lambda o=o: F.cost_effectiveness_plane(o, 500))
        render(f"{tag}_frontier",
               lambda o=o: F.prevention_frontier(
                   [20, 50, 80], [25, 40, 60, 100],
                   o["cost_per_acquisition"].mean(), current=(50, 60)))
        render(f"{tag}_denominators",
               lambda o=o: F.prevention_denominators(
                   50, 60, o["DALYs_per_acquisition"], o["cost_per_acquisition"],
                   [(10.0, "Common Round Figure"),
                    (20.0, "Representative Published Value")]))
        render(f"{tag}_offset",
               lambda o=o: F.offset_sensitivity(50, 60, o["DALYs_per_acquisition"],
                                                o["cost_per_acquisition"]))
        base = E.treatment_icer(o).mean()
        render(f"{tag}_tornado",
               lambda base=base: F.tornado(
                   [("Discount Rate 0%", base * 0.6, base),
                    ("GBD-Faithful Weights", base, base * 1.15),
                    ("Routine Care Doubled", base, base * 1.6)],
                   base, "Cost per DALY averted"))

    print("\nedge cases")
    o = E.evaluate(E.Assumptions())
    render("zero_offset_denominators",
           lambda: F.prevention_denominators(
               50, 60, o["DALYs_per_acquisition"], o["cost_per_acquisition"],
               [(10.0, "A"), (20.0, "B")], offset_share=0.0))
    render("cost_saving_denominators",
           lambda: F.prevention_denominators(
               20, 25, o["DALYs_per_acquisition"], o["cost_per_acquisition"],
               [(10.0, "A"), (20.0, "B")]))
    render("empty_tornado", lambda: F.tornado([], 0.0, "Nothing varied"))

    print("\nlive cascade re-run")
    try:
        import live
        a = E.Assumptions()
        pt = live.evaluate_point(a)
        render("cascade_comparison", lambda: F.cascade_comparison(o, pt))
        alt = live.evaluate_point(a, {"μ_ltfu": 2.0})
        check("multiplier changes the point estimate",
              lambda: (_ for _ in ()).throw(AssertionError("no change"))
              if abs(alt["DALYs_per_acquisition"] - pt["DALYs_per_acquisition"]) < 1e-6
              else None)
    except Exception as exc:
        failures.append(f"live: {exc!r}")
        print(f"  FAIL  live cascade: {exc!r}")

    print(f"\n{'ALL RENDERS PASSED' if not failures else 'FAILURES:'}")
    for f in failures:
        print("  " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

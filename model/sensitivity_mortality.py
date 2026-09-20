"""
Sensitivity of the health-economic outputs to the two unsourced mortality
inputs introduced alongside the age-varying background:

  EXCESS_HAZARD_ON_ART       -- additive HIV-attributable excess hazard while
                                virally suppressed (params.py)
  LTFU_MORTALITY_MULTIPLIER  -- phi, scaling the untreated CD4-specific
                                mortality curve for disengaged patients

Both are PLACEHOLDERS. This script exists so the choice of value is an
informed, explicit decision rather than an invisible default -- run it, look
at how far the headline moves, then set the values in params.py and record
the source.

Note on relative leverage: LTFU accounts for roughly a quarter of person-time
and the majority of HIV deaths in this model, so phi typically moves the
headline further than the on-ART excess does. That is a statement about this
model's structure, not about the underlying epidemiology.

Run directly (after both calibrations have been run):
    python3 sensitivity_mortality.py
"""

from __future__ import annotations

import importlib

import numpy as np

import params
import hazards
import health_economics as he
from params import load_parameters
from calibration_common import load_result

EXCESS_GRID = [0.000, 0.0025, 0.005, 0.0075, 0.010]
PHI_GRID = [0.4, 0.6, 0.8, 1.0]

METRICS = ["DALYs_per_acquisition", "DALYs_avoided_through_care",
           "cost_per_acquisition"]


def _set(excess: float | None = None, phi: float | None = None) -> None:
    """Both params.py and hazards.py hold module-level bindings (hazards
    imports the names directly), so both must be rebound for a sweep."""
    if excess is not None:
        params.EXCESS_HAZARD_ON_ART = excess
        hazards.EXCESS_HAZARD_ON_ART = excess
    if phi is not None:
        params.LTFU_MORTALITY_MULTIPLIER = phi
        hazards.LTFU_MORTALITY_MULTIPLIER = phi


def _evaluate(cascade_vals, lb, al, p) -> dict:
    return he.evaluate_health_econ(cascade_vals, lb, al, p)


def run(sweep_excess=True, sweep_phi=True, joint=True) -> dict:
    p = load_parameters()
    s1, s2 = load_result("system1"), load_result("system2")
    if s1 is None or s2 is None:
        raise SystemExit("Run calibrate_system1.py and calibrate_system2.py first.")
    lb = {r["param"]: r["mean"] for r in s1["summary"]}
    cascade = {r["param"]: r["mean"] for r in s2["summary"]}

    base_excess = params.EXCESS_HAZARD_ON_ART
    base_phi = params.LTFU_MORTALITY_MULTIPLIER
    out = {}

    if sweep_excess:
        print(f"\nExcess hazard on ART (phi held at {base_phi})")
        print(f"{'excess/yr':>11}" + "".join(f"{m:>30}" for m in METRICS))
        rows = []
        for e in EXCESS_GRID:
            _set(excess=e, phi=base_phi)
            r = _evaluate(cascade, lb["κ_prog"], lb["κ_mort"], p)
            rows.append({"excess": e, **{m: r[m] for m in METRICS}})
            print(f"{e:>11.4f}" + "".join(f"{r[m]:>30.3f}" for m in METRICS))
        out["excess"] = rows

    if sweep_phi:
        print(f"\nLTFU mortality multiplier phi (excess held at {base_excess})")
        print(f"{'phi':>11}" + "".join(f"{m:>30}" for m in METRICS))
        rows = []
        for f in PHI_GRID:
            _set(excess=base_excess, phi=f)
            r = _evaluate(cascade, lb["κ_prog"], lb["κ_mort"], p)
            rows.append({"phi": f, **{m: r[m] for m in METRICS}})
            print(f"{f:>11.2f}" + "".join(f"{r[m]:>30.3f}" for m in METRICS))
        out["phi"] = rows

    if joint:
        print("\nJoint grid -- DALYs_avoided_through_care")
        header = "excess/phi"
        print(f"{header:>11}" + "".join(f"{f:>10.2f}" for f in PHI_GRID))
        grid = np.zeros((len(EXCESS_GRID), len(PHI_GRID)))
        for i, e in enumerate(EXCESS_GRID):
            line = f"{e:>11.4f}"
            for j, f in enumerate(PHI_GRID):
                _set(excess=e, phi=f)
                r = _evaluate(cascade, lb["κ_prog"], lb["κ_mort"], p)
                grid[i, j] = r["DALYs_avoided_through_care"]
                line += f"{grid[i, j]:>10.3f}"
            print(line)
        out["joint"] = {"excess_grid": EXCESS_GRID, "phi_grid": PHI_GRID,
                        "dalys_avoided": grid}

    _set(excess=base_excess, phi=base_phi)   # restore
    return out


if __name__ == "__main__":
    res = run()
    print("\nNOTE: both inputs are placeholders. Set them in params.py and "
          "record the source before quoting any of these numbers.")

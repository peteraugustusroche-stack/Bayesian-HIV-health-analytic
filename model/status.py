"""
Pipeline status: what has been calibrated, when, and what is now stale.

Every cached result in data/results/ depends on a set of source files. If any
dependency has been modified more recently than the result, that result was
produced by code that no longer exists and must be regenerated before it means
anything. This script makes that explicit rather than leaving it to timestamps
you have to eyeball.

Run directly:
    python3 status.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "data" / "results"

# Files every part of the pipeline depends on.
CORE = ["params.py", "hazards.py", "life_table.py", "states.py",
        "bg_life_table_provenance.csv", "data/parameters.xlsx"]
# NOTE: add "acquisition_cd4.py" to NH once it is actually wired into the
# engines. Until then it is present on disk but imported by nothing, so
# listing it here would flag every result STALE for no reason.
NH = CORE + ["simulate_natural_history.py", "likelihoods.py"]
CASCADE = NH + ["simulate_cascade.py"]
HEALTH_ECON = CASCADE + ["health_economics.py", "health_econ_params.py"]

# result name -> (dependency source files, prerequisite result files)
PIPELINE = {
    "system1":            (NH + ["calibrate_system1.py"], []),
    "system2":            (CASCADE + ["calibrate_system2.py"], ["system1"]),
    "system2_sens2024":   (CASCADE + ["calibrate_system2.py"], ["system1"]),
    "prior_analysis":     (HEALTH_ECON + ["run_prior_analysis.py"], []),
    "posterior_analysis": (HEALTH_ECON + ["run_posterior_analysis.py"], ["system1", "system2"]),
    "scenario_point":     (HEALTH_ECON + ["scenario_analysis.py"], ["system1", "system2"]),
    "scenario_uncertainty": (HEALTH_ECON + ["scenario_analysis.py"], ["system1", "system2"]),
    "scenario_prob":      (HEALTH_ECON + ["scenario_analysis.py"], ["system2"]),
    "scenario_prior_point": (HEALTH_ECON + ["scenario_analysis.py"], []),
}

# The order you must regenerate things in.
REGEN_ORDER = ["system1", "system2", "prior_analysis", "posterior_analysis",
               "scenario_point", "scenario_uncertainty", "scenario_prob",
               "scenario_prior_point"]

RUN_CMD = {
    "system1": "python3 calibrate_system1.py",
    "system2": "python3 calibrate_system2.py",
    "system2_sens2024": "python3 calibrate_system2.py 2024",
    "prior_analysis": "python3 run_prior_analysis.py",
    "posterior_analysis": "python3 run_posterior_analysis.py",
    "scenario_point": "python3 scenario_analysis.py",
    "scenario_uncertainty": "python3 scenario_analysis.py",
    "scenario_prob": "python3 scenario_analysis.py",
    "scenario_prior_point": "python3 scenario_analysis.py",
}


def _mtime(p: Path) -> float | None:
    return p.stat().st_mtime if p.exists() else None


def _fmt(ts: float | None) -> str:
    if ts is None:
        return "never"
    d = dt.datetime.fromtimestamp(ts)
    age = (dt.datetime.now() - d).total_seconds()
    if age < 3600:
        rel = f"{age/60:.0f}m ago"
    elif age < 86400:
        rel = f"{age/3600:.1f}h ago"
    else:
        rel = f"{age/86400:.1f}d ago"
    return f"{d:%Y-%m-%d %H:%M} ({rel})"


def evaluate() -> dict:
    out = {}
    for name, (srcs, prereqs) in PIPELINE.items():
        res_mt = _mtime(RESULTS / f"{name}.npz")
        newer = []
        if res_mt is not None:
            for s in srcs:
                mt = _mtime(ROOT / s)
                if mt is not None and mt > res_mt:
                    newer.append((s, mt))
            for pre in prereqs:
                mt = _mtime(RESULTS / f"{pre}.npz")
                if mt is not None and mt > res_mt:
                    newer.append((f"[result] {pre}", mt))
        out[name] = {"mtime": res_mt, "stale_against": newer,
                     "missing": res_mt is None}
    return out


def main() -> None:
    status = evaluate()

    print("=" * 78)
    print("PIPELINE STATUS")
    print("=" * 78)
    print(f"{'result':<24}{'state':<12}{'written'}")
    print("-" * 78)
    for name in REGEN_ORDER + ["system2_sens2024"]:
        s = status[name]
        if s["missing"]:
            state = "MISSING"
        elif s["stale_against"]:
            state = "STALE"
        else:
            state = "ok"
        print(f"{name:<24}{state:<12}{_fmt(s['mtime'])}")

    stale = [n for n in REGEN_ORDER + ["system2_sens2024"]
             if status[n]["stale_against"] or status[n]["missing"]]
    if stale:
        print("\n" + "-" * 78)
        print("WHY")
        print("-" * 78)
        for name in stale:
            s = status[name]
            if s["missing"]:
                print(f"  {name}: never generated")
                continue
            newest = sorted(s["stale_against"], key=lambda x: -x[1])[:4]
            others = len(s["stale_against"]) - len(newest)
            bits = ", ".join(f"{f}" for f, _ in newest)
            print(f"  {name}: modified since it ran -> {bits}"
                  + (f" (+{others} more)" if others > 0 else ""))

        print("\n" + "-" * 78)
        print("REGENERATE IN THIS ORDER")
        print("-" * 78)
        seen = set()
        for name in REGEN_ORDER:
            if name in stale and RUN_CMD[name] not in seen:
                seen.add(RUN_CMD[name])
                print(f"  {RUN_CMD[name]}")
    else:
        print("\nEverything is current.")

    # --- show whatever IS loadable ---
    print("\n" + "=" * 78)
    print("CURRENT CALIBRATION")
    print("=" * 78)
    try:
        from calibration_common import load_result
    except Exception as e:                                    # pragma: no cover
        print(f"  (could not import calibration_common: {e})")
        return

    for name in ("system1", "system2"):
        r = load_result(name)
        if r is None:
            print(f"\n{name}: not present")
            continue
        flag = "  [STALE]" if status[name]["stale_against"] else ""
        print(f"\n{name}{flag}  backend={r['backend']}  chains={r['chains'].shape}")
        worst_rhat = max((row["r_hat"] for row in r["summary"]), default=float("nan"))
        min_ess = min((row["ess"] for row in r["summary"]), default=float("nan"))
        for row in r["summary"]:
            print(f"    {row['param']:<12} mean={row['mean']:>9.4f}  sd={row['sd']:>8.4f}"
                  f"  r_hat={row['r_hat']:.3f}  ess={row['ess']:.0f}")
        print(f"    -> worst r_hat {worst_rhat:.3f}, min ESS {min_ess:.0f}"
              + ("   <-- convergence is weak" if worst_rhat > 1.05 or min_ess < 200 else ""))

        if "targets" in r and "model_implied_at_mean" in r:
            print(f"    {'target':<16}{'implied':>10}{'central':>10}{'sd':>8}{'z':>8}")
            for k, (c, sd) in r["targets"].items():
                im = r["model_implied_at_mean"].get(k)
                if im is None or c is None:
                    continue
                z = (im - c) / sd if sd else float("nan")
                mark = "  <--" if abs(z) > 2 else ""
                print(f"    {k:<16}{im:>10.3f}{c:>10.3f}{sd:>8.3f}{z:>8.2f}{mark}")


if __name__ == "__main__":
    main()

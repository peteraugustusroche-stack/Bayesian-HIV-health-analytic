"""
Side-by-side comparison of the two calibration arms.

    ARM A  unaids     likelihood = p1, p2, p3, Supp_2L, CD4 at initiation
    ARM B  retention  the same, plus Ret_12m/Ret_24m and Reeng_1y/2y/3y

The question the pair answers is not "which model is right" -- it is the same
model both times, with the same priors and the same state space. It is: how
much of the headline DALY figure is a consequence of which cascade data the
analyst chose to anchor on?

likelihoods.py already records the expected answer, from an earlier joint fit:
fitting all three sources simultaneously pushed ~31% of post-acquisition
person-time into LTFU and inflated DALYs per acquisition by roughly 30%. This
script replaces that recollection with two posteriors.

Writes writeup/arm_comparison.md alongside the console output.

    python3 compare_arms.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "data" / "results"
ARMS = [("unaids", "UNAIDS-anchored", RESULTS / "arm_unaids"),
        ("retention", "Retention-anchored", RESULTS / "arm_retention")]

try:
    from run_prior_analysis import METRIC_NAMES
except Exception:                                    # keep this runnable alone
    METRIC_NAMES = ["YLL_natural_history", "YLD_natural_history",
                    "DALYs_natural_history", "YLL_cascade", "YLD_cascade",
                    "DALYs_per_acquisition", "DALYs_avoided_through_care",
                    "cost_recurring", "cost_startup", "cost_per_acquisition"]

HEADLINE = ["DALYs_per_acquisition", "YLL_cascade", "YLD_cascade",
            "DALYs_natural_history", "DALYs_avoided_through_care",
            "cost_per_acquisition"]

PRETTY = {
    "DALYs_per_acquisition":      "DALYs per acquisition",
    "YLL_cascade":                "  years of life lost",
    "YLD_cascade":                "  years lived with disability",
    "DALYs_natural_history":      "DALYs, no care",
    "DALYs_avoided_through_care": "DALYs averted by the cascade",
    "cost_per_acquisition":       "Lifetime cost per acquisition (USD)",
}


def load_posterior(arm_dir: Path):
    path = arm_dir / "posterior_analysis.npz"
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=True)["payload"].item()
    prob = payload.get("probabilistic")
    if not prob:
        return None
    return {row["metric"]: row for row in prob["summary"]}


def load_chains(arm_dir: Path):
    """Convergence and fitted parameters, read straight off the npz so this
    does not depend on calibration_common's RESULTS_DIR."""
    path = arm_dir / "system2.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    out = {"param_names": [str(n) for n in data["param_names"]],
           "chains": data["chains"], "backend": str(data["backend"])}
    if "extra" in data:
        out.update(data["extra"].item())
    try:
        from mcmc_numpy import summarize
        out["summary"] = summarize({"chains": out["chains"],
                                    "param_names": out["param_names"]})
    except Exception:
        out["summary"] = None
    return out


def fmt(row, money=False):
    if row is None:
        return "--"
    if money:
        return f"${row['mean']:,.0f} (${row['q05']:,.0f}-${row['q95']:,.0f})"
    return f"{row['mean']:.2f} ({row['q05']:.2f}-{row['q95']:.2f})"


def main():
    posts, chains = {}, {}
    for key, _label, d in ARMS:
        posts[key] = load_posterior(d)
        chains[key] = load_chains(d)

    missing = [k for k, v in posts.items() if v is None]
    if missing:
        raise SystemExit(
            f"No posterior found for arm(s): {', '.join(missing)}.\n"
            f"Run ./run_all.sh first (or check {RESULTS}).")

    a, b = posts["unaids"], posts["retention"]
    md = []
    w = md.append

    # ---- headline table --------------------------------------------------
    print("\n" + "=" * 88)
    print("DALY OUTCOMES BY CALIBRATION ANCHOR   (mean, 90% credible interval)")
    print("=" * 88)
    head = f"{'':<38}{'UNAIDS-anchored':>24}{'Retention-anchored':>24}{'diff':>9}"
    print(head)
    print("-" * 95)

    w("**Health-economic outcomes by calibration anchor.** Same model, same "
      "priors, same state space; the arms differ only in which cascade data "
      "enter the likelihood.")
    w("")
    w("| Outcome (discounted at 3%) | UNAIDS-anchored | Retention-anchored | Difference |")
    w("|---|---|---|---|")

    for m in HEADLINE:
        money = m.startswith("cost")
        ra, rb = a.get(m), b.get(m)
        if ra is None or rb is None:
            continue
        pct = (rb["mean"] - ra["mean"]) / ra["mean"] * 100 if ra["mean"] else float("nan")
        print(f"{PRETTY.get(m, m):<38}{fmt(ra, money):>24}{fmt(rb, money):>24}"
              f"{pct:>+8.1f}%")
        w(f"| {PRETTY.get(m, m).strip()} | {fmt(ra, money)} | {fmt(rb, money)} | {pct:+.1f}% |")

    # ---- the sentence the dissertation needs -----------------------------
    da, db = a["DALYs_per_acquisition"], b["DALYs_per_acquisition"]
    gap = (db["mean"] - da["mean"]) / da["mean"] * 100
    overlap = not (da["q95"] < db["q05"] or db["q95"] < da["q05"])
    print("\n" + "-" * 95)
    print(f"DALYs per acquisition moves {gap:+.1f}% between anchors "
          f"({da['mean']:.2f} vs {db['mean']:.2f}).")
    print(f"90% credible intervals {'OVERLAP' if overlap else 'DO NOT OVERLAP'}: "
          f"[{da['q05']:.2f}, {da['q95']:.2f}] vs [{db['q05']:.2f}, {db['q95']:.2f}].")
    if not overlap:
        print("The choice of calibration anchor moves the estimate by more than "
              "its own\nparameter uncertainty -- i.e. it is not absorbed by the "
              "credible interval.")
    w("")
    w(f"DALYs per acquisition move {gap:+.1f}% between the two anchors "
      f"({da['mean']:.2f} against {db['mean']:.2f}). The 90% credible intervals "
      f"{'overlap' if overlap else 'do not overlap'} "
      f"([{da['q05']:.2f}, {da['q95']:.2f}] against [{db['q05']:.2f}, {db['q95']:.2f}]).")

    # ---- fit to targets, both arms --------------------------------------
    print("\n" + "=" * 88)
    print("MODEL-IMPLIED vs TARGETS   (z = (implied - central) / sd)")
    print("=" * 88)
    w("")
    w("**Fit to targets under each anchor.** A target with no z under an arm "
      "was a diagnostic, not a likelihood term, in that arm.")
    w("")
    w("| Target | Observed | UNAIDS-anchored | Retention-anchored |")
    w("|---|---|---|---|")

    keys = []
    for k in ("unaids", "retention"):
        for t in (chains[k] or {}).get("targets", {}) or {}:
            if t not in keys:
                keys.append(t)

    print(f"{'target':<20}{'observed':>10}"
          f"{'UNAIDS':>12}{'z':>8}{'retention':>12}{'z':>8}")
    print("-" * 70)
    for t in keys:
        cells = {}
        for k in ("unaids", "retention"):
            ch = chains[k] or {}
            tg = (ch.get("targets") or {}).get(t)
            imp = (ch.get("model_implied_at_mean") or {}).get(t)
            if tg is None or imp is None:
                cells[k] = (None, None, None)
                continue
            central, sd = tg[0], tg[1]
            cells[k] = (imp, central, (imp - central) / sd if sd else float("nan"))
        obs = next((c[1] for c in cells.values() if c[1] is not None), None)
        ia, za = cells["unaids"][0], cells["unaids"][2]
        ib, zb = cells["retention"][0], cells["retention"][2]
        f = lambda v, s: (s.format(v) if v is not None else "--")
        print(f"{t:<20}{f(obs,'{:>10.3f}')}"
              f"{f(ia,'{:>12.3f}')}{f(za,'{:>+8.2f}')}"
              f"{f(ib,'{:>12.3f}')}{f(zb,'{:>+8.2f}')}")
        w(f"| {t} | {f(obs,'{:.3f}')} | {f(ia,'{:.3f}')} (z {f(za,'{:+.2f}')}) "
          f"| {f(ib,'{:.3f}')} (z {f(zb,'{:+.2f}')}) |")

    # ---- parameters and convergence -------------------------------------
    print("\n" + "=" * 88)
    print("POSTERIOR PARAMETERS AND CONVERGENCE")
    print("=" * 88)
    w("")
    w("**Posterior parameters under each anchor.**")
    w("")
    w("| Parameter | UNAIDS-anchored | Retention-anchored |")
    w("|---|---|---|")

    sa = {r["param"]: r for r in (chains["unaids"] or {}).get("summary") or []}
    sb = {r["param"]: r for r in (chains["retention"] or {}).get("summary") or []}
    print(f"{'parameter':<16}{'UNAIDS mean':>14}{'sd':>9}{'r_hat':>8}"
          f"{'reten. mean':>14}{'sd':>9}{'r_hat':>8}{'ratio':>8}")
    print("-" * 86)
    for p in sa:
        ra, rb = sa.get(p), sb.get(p)
        if rb is None:
            continue
        ratio = rb["mean"] / ra["mean"] if ra["mean"] else float("nan")
        print(f"{p:<16}{ra['mean']:>14.4f}{ra['sd']:>9.4f}{ra['r_hat']:>8.3f}"
              f"{rb['mean']:>14.4f}{rb['sd']:>9.4f}{rb['r_hat']:>8.3f}{ratio:>8.2f}")
        w(f"| `{p}` | {ra['mean']:.4f} ({ra['sd']:.4f}) | {rb['mean']:.4f} ({rb['sd']:.4f}) |")

    for k, label, _d in ARMS:
        s = (chains[k] or {}).get("summary")
        if not s:
            continue
        worst = max(r["r_hat"] for r in s)
        ess = min(r["ess"] for r in s)
        flag = "  <-- weak" if (worst > 1.05 or ess < 400) else ""
        print(f"\n{label}: backend {chains[k]['backend']}, "
              f"worst r_hat {worst:.3f}, min ESS {ess:.0f}{flag}")
        w("")
        w(f"*{label}: worst r-hat {worst:.3f}, minimum effective sample size "
          f"{ess:.0f}.*")

    out = HERE / "writeup" / "arm_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(f"\nMarkdown written to {out}")


if __name__ == "__main__":
    main()

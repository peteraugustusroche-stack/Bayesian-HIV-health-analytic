"""
Is CD4 at presentation doing identifying work, or is it decoration?

The question: p1 (proportion of PLHIV diagnosed) is a STOCK. It constrains the
diagnosis process only through the stationary share of the cohort sitting in
Undiagnosed. Two parameters govern that process:

    delta_symp   the symptomatic diagnosis rate scale
    gamma_diag   its CD4-dependence exponent -- how much faster the sick get
                 diagnosed than the well

Along a ridge in (delta_symp, gamma_diag), p1 can be held fixed: raise the
CD4-dependence and lower the scale, and the same fraction ends up diagnosed.
But those two points are NOT the same model. A high gamma_diag world diagnoses
people late and sick; a low gamma_diag world diagnoses them evenly. Same stock,
different untreated person-time at low CD4 -- which is exactly what DALYs
integrate.

This traces that ridge:

  for each gamma_diag on a grid:
     solve for delta_symp such that model p1 == p1 at the posterior mean
     report p2, p3 (are the other UNAIDS terms also preserved?)
     report CD4 median and P(CD4<200) at initiation  (does CD4 discriminate?)
     report DALYs per acquisition                    (does it matter?)

If DALYs move a lot along the ridge while p1/p2/p3 stay put, then UNAIDS data
alone do not identify the headline number, and whatever does pin gamma_diag is
load-bearing. If the CD4 columns move in step, CD4 at presentation is that
thing, and dropping it would hand the most influential dimension of the model
back to the prior.

    python3 ridge_cd4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import likelihoods as lk
import simulate_cascade as casc
from health_economics import evaluate_health_econ
from params import load_parameters, HORIZON_YEARS

RESULTS = Path(__file__).parent / "data" / "results"
ARM = RESULTS / "arm_unaids" / "system2.npz"

N_STEPS = int(HORIZON_YEARS / (1 / 12))


def posterior_mean_params():
    d = np.load(ARM, allow_pickle=True)
    names = [str(n) for n in d["param_names"]]
    chains = d["chains"]                       # (chain, draw, param)
    flat = chains.reshape(-1, chains.shape[-1])
    means = flat.mean(axis=0)
    extra = d["extra"].item()
    nh = np.asarray(extra["nh_draws_by_chain"])     # (chain, 2) or list
    nh = np.asarray([np.asarray(x, dtype=float).ravel()[:2] for x in nh])
    kappa_prog, kappa_mort = nh.mean(axis=0)
    return dict(zip(names, means)), float(kappa_prog), float(kappa_mort)


def summaries(vals, kp, km, params):
    """p1/p2/p3 and the two CD4-at-initiation statistics, one build."""
    stack = casc.build_full_matrix_stack(
        vals, params.fixed["recov_rate"], horizon_years=HORIZON_YEARS,
        kappa_prog=kp, kappa_mort=km)
    traj = casc.propagate(stack, casc.undiagnosed_init(), N_STEPS)
    stat = casc.stationary_cascade_proportions(traj)
    cd4 = casc.cd4_distribution_at_art_initiation(stack, horizon_years=HORIZON_YEARS)
    return {
        "p1": stat["p1_diagnosed_of_alive"],
        "p2": stat["p2_onart_of_diagnosed"],
        "p3": stat["p3_suppressed_of_onart"],
        "CD4_median": casc.cd4_percentile_at_art_initiation(cd4, 0.50),
        "CD4_lt200": casc.cd4_lt200_frac_at_art_initiation(cd4),
    }


def solve_delta_for_p1(gamma, target_p1, base, kp, km, params,
                       lo=0.001, hi=2.0, tol=1e-4, maxit=40):
    """Bisect on delta_symp so that model p1 matches target_p1, at this gamma.

    p1 is monotone decreasing in delta_symp (faster diagnosis -> smaller
    undiagnosed pool -> larger diagnosed share), so bisection is safe.
    """
    def p1_of(delta):
        v = dict(base); v["γ_diag"] = gamma; v["δ_symp"] = delta
        return summaries(v, kp, km, params)["p1"], v

    f_lo, _ = p1_of(lo)
    f_hi, _ = p1_of(hi)
    if (f_lo - target_p1) * (f_hi - target_p1) > 0:
        return None, None                       # target not bracketed
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        f_mid, v = p1_of(mid)
        if abs(f_mid - target_p1) < tol:
            return mid, v
        if (f_lo - target_p1) * (f_mid - target_p1) <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return mid, v


def main():
    params = load_parameters()
    base, kp, km = posterior_mean_params()
    g0, d0 = base["γ_diag"], base["δ_symp"]

    ref = summaries(base, kp, km, params)
    ref_he = evaluate_health_econ(base, kp, km, params)
    print(f"posterior mean   γ_diag {g0:.4f}   δ_symp {d0:.4f}   "
          f"κ_prog {kp:.3f}  κ_mort {km:.3f}")
    print(f"reference        p1 {ref['p1']:.4f}  p2 {ref['p2']:.4f}  "
          f"p3 {ref['p3']:.4f}  CD4med {ref['CD4_median']:.1f}  "
          f"DALYs {ref_he['DALYs_per_acquisition']:.3f}\n")

    # Target SDs, so movement can be read in z units rather than raw units.
    t = params.targets
    sd_cd4med = t["CD4_median_ART"][1]
    sd_cd4lt = t["CD4_lt200_ART"][1]

    grid = g0 * np.array([0.50, 0.65, 0.80, 0.90, 1.0, 1.15, 1.30, 1.50, 1.75])

    print(f"{'γ_diag':>8}{'δ_symp':>9}{'p1':>8}{'p2':>8}{'p3':>8}"
          f"{'CD4med':>9}{'z':>7}{'CD4<200':>9}{'z':>7}{'DALYs':>8}{'Δ%':>8}")
    print("-" * 96)

    rows = []
    for g in grid:
        delta, vals = solve_delta_for_p1(g, ref["p1"], base, kp, km, params)
        if delta is None:
            print(f"{g:>8.4f}   -- p1 not reachable at this γ --")
            continue
        s = summaries(vals, kp, km, params)
        he = evaluate_health_econ(vals, kp, km, params)
        dal = he["DALYs_per_acquisition"]
        zmed = (s["CD4_median"] - t["CD4_median_ART"][0]) / sd_cd4med
        zlt = (s["CD4_lt200"] - t["CD4_lt200_ART"][0]) / sd_cd4lt
        pct = (dal - ref_he["DALYs_per_acquisition"]) / ref_he["DALYs_per_acquisition"] * 100
        rows.append((g, delta, s, dal, zmed, zlt, pct))
        print(f"{g:>8.4f}{delta:>9.4f}{s['p1']:>8.4f}{s['p2']:>8.4f}"
              f"{s['p3']:>8.4f}{s['CD4_median']:>9.1f}{zmed:>+7.2f}"
              f"{s['CD4_lt200']:>9.3f}{zlt:>+7.2f}{dal:>8.3f}{pct:>+7.1f}%")

    if not rows:
        return

    # ---- the two numbers the argument turns on --------------------------
    dals = np.array([r[3] for r in rows])
    p1s = np.array([r[2]["p1"] for r in rows])
    p2s = np.array([r[2]["p2"] for r in rows])
    p3s = np.array([r[2]["p3"] for r in rows])
    zs = np.array([r[4] for r in rows])

    print("\n" + "=" * 96)
    print(f"Along the ridge, the UNAIDS terms barely move: "
          f"p1 {p1s.min():.4f}-{p1s.max():.4f}, "
          f"p2 {p2s.min():.4f}-{p2s.max():.4f}, "
          f"p3 {p3s.min():.4f}-{p3s.max():.4f}.")
    print(f"DALYs per acquisition move {dals.min():.2f}-{dals.max():.2f} "
          f"({(dals.max()-dals.min())/dals.mean()*100:.0f}% of the mean).")
    print(f"CD4 median moves {zs.min():+.2f} to {zs.max():+.2f} in target-SD units, "
          f"i.e. a span of {zs.max()-zs.min():.1f} SD.")
    print("\nRead: if the DALY span is wide and the CD4 z span is wide while "
          "p1/p2/p3\nsit still, CD4 at presentation is the only target "
          "separating these worlds.")

    # Restricted ridge: only the segment CD4 data would tolerate (|z| < 2)
    ok = [r for r in rows if abs(r[4]) < 2 and abs(r[5]) < 2]
    if ok:
        d_ok = np.array([r[3] for r in ok])
        print(f"\nRestricting to the segment CD4 data tolerate (|z| < 2 on both "
              f"CD4 targets):\n  γ_diag {min(r[0] for r in ok):.3f}-"
              f"{max(r[0] for r in ok):.3f}, DALYs {d_ok.min():.2f}-{d_ok.max():.2f}"
              f"  (span cut from {dals.max()-dals.min():.2f} to "
              f"{d_ok.max()-d_ok.min():.2f}).")


if __name__ == "__main__":
    main()

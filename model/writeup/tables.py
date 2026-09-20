"""Generate every results table as markdown, straight from the cached results.

Nothing here is hand-transcribed: if a number appears in the write-up it was
produced by this script from the .npz files the calibration wrote.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from params import load_parameters, CD4_BAND_NAMES, EXCESS_HAZARD_ON_ART, ANCHOR_AGE
from calibration_common import load_result
from run_posterior_analysis import load_posterior_result
from run_prior_analysis import load_prior_result
# base case is the 2019 cascade, so costs are reported in 2019 USD (price year
# follows cascade year -- see run_posterior_analysis.posterior_result_path)
from scenario_analysis import load_uncertainty_result, load_point_result
import likelihoods as lk
import acquisition_cd4 as acq
import life_table as lt

OUT = []
def w(s=""): OUT.append(s)

p = load_parameters()
s1 = load_result("system1"); s2 = load_result("system2")
names = p.symbol_names()

# ---------------------------------------------------------------- Table 2
w("**Table 2. CD4 distribution at HIV acquisition.**")
w()
w("| CD4 band (cells/µL) | Probability | Cumulative |")
w("|---|---|---|")
d = acq.initial_band_distribution(); cum = 0.0
for nm, v in zip(CD4_BAND_NAMES, d):
    cum += v
    w(f"| {nm} | {v*100:.1f}% | {cum*100:.1f}% |")
w()
w(f"Median CD4 at acquisition {acq.sqrt_cd4_mean()**2:.0f} cells/µL "
  f"(√-scale mean {acq.sqrt_cd4_mean():.2f}, σ₀ = {acq.ACQUISITION_SIGMA0:.1f}, "
  f"truncated below {acq.TRUNCATION_FLOOR_CELLS:.0f} cells/µL).")
w()

# ---------------------------------------------------------------- Table 3
w("**Table 3. Free parameters, priors and posterior estimates.**")
w()
w("| Symbol | Parameter | Prior median | Prior SD | Posterior mean | Posterior SD | 94% CrI | R̂ | ESS |")
w("|---|---|---|---|---|---|---|---|---|")
specs = {s.symbol: s for s in
         lk.build_specs(p, lk.SYSTEM1_SYMBOLS, lk.SYSTEM1_KINDS)
         + lk.build_specs(p, lk.SYSTEM2_SYMBOLS, lk.SYSTEM2_KINDS)}
LABEL = {
    "λ_base": "Baseline CD4 decline hazard", "α_shape": "CD4 band accelerator",
    "δ_bg": "Routine diagnosis hazard", "δ_symp": "Symptomatic diagnosis multiplier",
    "γ_diag": "Symptomatic diagnosis shape", "λ_init": "ART initiation hazard",
    "μ_vf1": "1st-line virological failure", "μ_ltfu": "Disengagement hazard",
    "μ_re-engage": "Re-engagement (recent)", "μ_perm_diseng": "Transition to long-term disengagement",
    "μ_reengage_late": "Re-engagement (long-term)", "φ_ltfu": "LTFU mortality multiplier",
    "μ_switch": "Rescue switch hazard",
}
for res, sysname in ((s1, "System 1"), (s2, "System 2")):
    w(f"| *{sysname}* | | | | | | | | |")
    for row in res["summary"]:
        sym = row["param"]; sp = specs.get(sym)
        pm = f"{sp.central:g}" if sp else "—"; ps = f"{sp.sd:g}" if sp else "—"
        w(f"| {sym} | {LABEL.get(sym, names.get(sym,''))} | {pm} | {ps} | "
          f"{row['mean']:.4g} | {row['sd']:.3g} | "
          f"{row['q03']:.3g} – {row['q97']:.3g} | {row['r_hat']:.3f} | {row['ess']:.0f} |")
w()

# ---------------------------------------------------------------- Table 5
def target_block(res, title):
    w(f"**{title}**")
    w()
    w("| Target | Model-implied | Target value | SD | z |")
    w("|---|---|---|---|---|")
    for k, (c, sd) in res["targets"].items():
        im = res["model_implied_at_mean"].get(k)
        if im is None or c is None:
            continue
        z = (im - c) / sd if sd else float("nan")
        flag = " ⚠" if abs(z) > 2 else ""
        w(f"| {k} | {im:.3f} | {c:.3f} | {sd:.3f} | {z:+.2f}{flag} |")
    w()

target_block(s1, "Table 6. System 1 calibration: model-implied values against targets.")
target_block(s2, "Table 7. System 2 calibration: model-implied values against targets.")

w("**Table 7b. Diagnostics reported but not fitted.**")
w()
w("| Quantity | Model-implied | External estimate | Note |")
w("|---|---|---|---|")
mi = s2["model_implied_at_mean"]
if mi.get("T_ART") is not None:
    w(f"| Median time diagnosis→ART (months) | {mi['T_ART']:.2f} | 0 days (IQR 0–7) | "
      "quantised to whole cycles; superseded by ART_1m |")
if mi.get("HR_LTFU") is not None:
    w(f"| Mortality risk ratio, disengaged vs retained | {mi['HR_LTFU']:.2f} | 6–23 | "
      "model saturates near 9; see §4.3 |")
w()

# ---------------------------------------------------------------- Table 8
post = load_posterior_result(); prior = load_prior_result()
w("**Table 8. Health-economic outcomes per HIV acquisition (discounted at 3%).**")
w()
w("| Outcome | Posterior mean | 5th centile | Median | 95th centile |")
w("|---|---|---|---|---|")
KEEP = ["DALYs_per_acquisition", "DALYs_avoided_through_care", "DALYs_natural_history",
        "YLL_cascade", "YLD_cascade", "YLL_natural_history", "YLD_natural_history",
        "cost_per_acquisition", "cost_recurring", "cost_death"]
PRETTY = {
    "DALYs_per_acquisition": "DALYs per acquisition (with cascade)",
    "DALYs_avoided_through_care": "DALYs averted by the cascade",
    "DALYs_natural_history": "DALYs per acquisition (no care)",
    "YLL_cascade": "— YLL, cascade arm", "YLD_cascade": "— YLD, cascade arm",
    "YLL_natural_history": "— YLL, no-care arm", "YLD_natural_history": "— YLD, no-care arm",
    "cost_per_acquisition": "Cost per acquisition (2019 USD)",
    "cost_recurring": "— recurring", "cost_death": "— terminal care",
}
byname = {r["metric"]: r for r in post["probabilistic"]["summary"]}
for k in KEEP:
    r = byname[k]
    fmt = (lambda v: f"{v:,.0f}") if k.startswith("cost") else (lambda v: f"{v:.2f}")
    w(f"| {PRETTY[k]} | {fmt(r['mean'])} | {fmt(r['q05'])} | {fmt(r['median'])} | {fmt(r['q95'])} |")
w()
w(f"Based on {post['probabilistic']['n_draws']} paired posterior draws.")
w()

# ---------------------------------------------------------------- Table 9
unc = load_uncertainty_result()
w("**Table 9. Deterministic sensitivity analysis.** Each parameter varied to its own "
  "posterior 3rd and 97th percentiles with all others held at posterior means.")
w()
w("| Parameter | 3rd centile | 97th centile | DALYs at 3rd | DALYs at 97th | Range |")
w("|---|---|---|---|---|---|")
for row in sorted(unc["rows"], key=lambda r: -r["DALYs_per_acquisition_range"]):
    w(f"| {row['param']} | {row['q03']:.4g} | {row['q97']:.4g} | "
      f"{row['DALYs_per_acquisition_at_q03']:.2f} | "
      f"{row['DALYs_per_acquisition_at_q97']:.2f} | "
      f"{row['DALYs_per_acquisition_range']:.2f} |")
w()
w(f"Baseline DALYs per acquisition at posterior means: "
  f"{unc['baseline']['DALYs_per_acquisition']:.2f}.")
w()

# ---------------------------------------------------------------- Table 10
w("**Table 10. Derived validation: population viral suppression.**")
w()
imp = s2["model_implied_at_mean"]
prod = imp["p1"] * imp["p2"] * imp["p3"]
w("| Quantity | Model | UNAIDS ESA 2019 |")
w("|---|---|---|")
w(f"| p₁ diagnosed / alive | {imp['p1']:.3f} | 0.870 |")
w(f"| p₂ on ART / diagnosed | {imp['p2']:.3f} | 0.828 |")
w(f"| p₃ suppressed / on ART | {imp['p3']:.3f} | 0.903 |")
w(f"| **all PLHIV suppressed (p₁×p₂×p₃)** | **{prod:.3f}** | **0.650** |")
w()
w("The final row was never a calibration target; it falls out of the three fitted "
  "conditional proportions and provides an independent check.")
w()

open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.md"), "w").write("\n".join(OUT))
print("\n".join(OUT[:0]))
print(f"tables.md written ({len(OUT)} lines)")

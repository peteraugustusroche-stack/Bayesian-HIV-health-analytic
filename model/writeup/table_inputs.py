"""Appendix B: complete classification of model inputs (CHEERS 2022 item 22).

Every quantity the model reads, classified as one of four kinds:

    adopted     fixed at a published value, no uncertainty carried
    sampled     drawn from a prior on every run, never fitted
    calibrated  estimated by MCMC against the calibration targets
    derived     a deterministic function of the above, no prior of its own

Values are read from the live modules wherever possible rather than typed in,
so this table cannot drift from the model. Where a value is a table rather
than a scalar the entry names the table and its source; the full arrays are
in natural_history_params.py and health_econ_params.py.

    python3 writeup/table_inputs.py            > writeup/appendix_b.md
    python3 writeup/table_inputs.py --check    (provenance gaps only)
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import params as P
import natural_history_params as NH
import health_econ_params as HE
import acquisition_cd4 as ACQ
import likelihoods as LK
import life_table as LT
from calibration_common import load_result

OUT = []
def w(s=""):
    OUT.append(s)


p = P.load_parameters()
s1 = load_result("system1")
s2 = load_result("system2")
s2b = load_result("system2_sens2024")
post = {r["param"]: r for r in s2["summary"]}
post24 = {r["param"]: r for r in s2b["summary"]}
nhpost = {r["param"]: r for r in s1["summary"]}

specs = {s.symbol: s for s in
         LK.build_specs(p, LK.SYSTEM1_SYMBOLS, LK.SYSTEM1_KINDS)
         + LK.build_specs(p, LK.SYSTEM2_SYMBOLS, LK.SYSTEM2_KINDS)}

# Prior mean/SD on the natural scale, by sampling -- the (central, sd) pair in
# the spreadsheet is a lognormal MEDIAN and log-scale spread, so quoting it as
# if it were a mean and SD understates the prior's true width.
_rng = np.random.default_rng(20240904)
def prior_moments(sym, n=200_000):
    sp = specs.get(sym)
    if sp is None:
        return None
    x = np.array([float(sp.sample_prior(_rng)) for _ in range(n)])
    return x.mean(), x.std()


# ---------------------------------------------------------------------------
# A. Calibrated -- System 2 cascade hazards
# ---------------------------------------------------------------------------
CAL_LABEL = {
    "δ_bg": ("Routine diagnosis hazard", "/yr",
             "Rate of diagnosis independent of disease stage (routine, antenatal, partner and provider-initiated testing)."),
    "δ_symp": ("Symptomatic diagnosis multiplier", "/yr",
               "Additional diagnosis hazard at the lowest CD4 band; scaled across bands by γ_diag."),
    "γ_diag": ("Symptomatic diagnosis shape", "—",
               "Exponent of the severity ramp, severity(band) = (band/6)^γ. Governs where in the CD4 range symptomatic presentation concentrates."),
    "λ_init": ("ART initiation hazard", "/yr",
               "Rate of starting ART from Diagnosed_PreART, for the fraction not initiating on the day of diagnosis."),
    "μ_vf1": ("First-line virological failure", "/yr",
              "Rate of loss of viral suppression on first-line ART."),
    "μ_ltfu": ("Disengagement hazard", "/yr",
               "Rate of disengaging from care, applied from every in-care state."),
    "μ_re-engage": ("Re-engagement, recent", "/yr",
                    "Rate of returning to care from recent disengagement."),
    "φ_ltfu": ("Disengaged mortality multiplier", "—",
               "Multiplier on the untreated CD4-specific mortality curve while disengaged."),
    "μ_switch": ("Rescue switch hazard", "/yr",
                 "Rate at which first-line failure is detected and the patient switched to second line."),
}

w("**Table B1. Calibrated parameters.** Estimated by Markov chain Monte Carlo "
  "against the targets in Table B5. Priors are lognormal; the prior mean and SD "
  "below are on the natural scale, computed from the prior itself rather than "
  "from its median and log-scale spread. The 2019 cascade is the base case.")
w()
w("| Symbol | Quantity | Unit | Prior mean (SD) | Posterior mean (SD) | 94% CrI | R̂ | ESS | Posterior, 2024 cascade |")
w("|---|---|---|---|---|---|---|---|---|")
for sym in LK.SYSTEM2_SYMBOLS:
    lab, unit, _ = CAL_LABEL[sym]
    pm, ps = prior_moments(sym)
    r, r24 = post[sym], post24[sym]
    w(f"| {sym} | {lab} | {unit} | {pm:.3g} ({ps:.3g}) | "
      f"{r['mean']:.3g} ({r['sd']:.3g}) | {r['q03']:.3g}–{r['q97']:.3g} | "
      f"{r['r_hat']:.3f} | {r['ess']:.0f} | {r24['mean']:.3g} ({r24['sd']:.3g}) |")
w()

# ---------------------------------------------------------------------------
# B. Sampled but not calibrated
# ---------------------------------------------------------------------------
w("**Table B2. Sampled parameters.** Uncertain, drawn from their priors on every "
  "model run, and never confronted with the calibration targets. The natural-history "
  "scale factors are the mechanism by which uncertainty about untreated disease "
  "reaches the result without the cascade data being permitted to revise it.")
w()
w("| Symbol | Quantity | Prior | Realised mean (SD) | Basis for the spread |")
w("|---|---|---|---|---|")
for sym, lab, basis in (
    ("κ_prog", "Scale factor on the adopted CD4 progression schedule",
     "The relative half-widths of Glaubius's own credible intervals on the progression cells (3.8–10.5%, mean 5.7% at ages 25–34), rounded up because one factor stands in for seven separately-estimated cells."),
    ("κ_mort", "Scale factor on the adopted untreated mortality schedule",
     "Deliberately wider than Glaubius's internal precision (which implies σ ≈ 0.09), because between-source disagreement on where untreated mortality sits — Glaubius against Mangal 2017 — exceeds either source's own uncertainty."),
):
    sigma = NH.KAPPA_PROG_SIGMA if sym == "κ_prog" else NH.KAPPA_MORT_SIGMA
    r = nhpost[sym]
    w(f"| {sym} | {lab} | Lognormal, median 1, σ = {sigma} | "
      f"{r['mean']:.3g} ({r['sd']:.3g}) | {basis} |")
w(f"| σ_SMR | Multiplicative uncertainty on the on-ART SMR ladder | "
  f"Lognormal, log-scale SD {P.ART_SMR_LOG_SD} | — | "
  f"The two rungs Rodger et al. 2013 actually estimate (SMR 1.00, 0.69–1.40 at CD4 ≥500; "
  f"1.77, 1.17–2.55 at 350–499) imply log-scale SDs of 0.171–0.211. Drawn once per posterior "
  f"draw in the health-economic run rather than by the sampler, because the cascade targets "
  f"barely inform it. Understates the uncertainty on the 7.0 and 10.0 rungs, which are "
  f"extrapolations rather than estimates. |")
w()
w("*Not calibrated:* System 1 is not fitted. `backend` in the saved System 1 result "
  "records this as `\"" + str(s1.get("backend")) + "\"`. The three natural-history "
  "quantities that were formerly calibration targets — median time to CD4 <350, to "
  "CD4 <200, and median untreated survival — are therefore available as out-of-sample "
  "validation checks the model can fail (Table B7).")
w()

# ---------------------------------------------------------------------------
# C. Adopted
# ---------------------------------------------------------------------------
w("**Table B3. Adopted inputs.** Fixed at a published value; no uncertainty is "
  "carried except where a sampled scale factor in Table B2 applies to the whole "
  "schedule.")
w()
w("| Input | Value | Source |")
w("|---|---|---|")

GLAUB = ("Glaubius R, Stover J, Johnson LF, et al. J Int AIDS Soc 2021;24(Suppl 5):e25784. "
         "doi:10.1002/jia2.25784, Table 3. The parameterisation behind the UNAIDS Spectrum "
         "cascade estimates the model calibrates to.")

rows_nh = [
    ("CD4 progression schedule, untreated",
     f"{NH.GLAUBIUS_PROGRESSION.shape[0]}×{NH.GLAUBIUS_PROGRESSION.shape[1]} table, age group × CD4 band; "
     f"{NH.GLAUBIUS_PROGRESSION[NH.GLAUBIUS_PROGRESSION > 0].min():.3g}–"
     f"{NH.GLAUBIUS_PROGRESSION.max():.3g}/yr, zero out of the lowest band. Scaled by κ_prog.",
     GLAUB),
    ("HIV mortality schedule, untreated",
     f"{NH.GLAUBIUS_MORTALITY.shape[0]}×{NH.GLAUBIUS_MORTALITY.shape[1]} table; 0/yr above CD4 350, "
     f"rising to {NH.GLAUBIUS_MORTALITY[3][6]:.3g}/yr at CD4 <50, age 45+. Scaled by κ_mort.",
     GLAUB),
    ("CD4 distribution at acquisition",
     f"7-band distribution; central CD4 {ACQ.sqrt_cd4_mean()**2:.0f} cells/µL "
     f"(square of the √-scale mean, sex-weighted at {ACQ.PCT_FEMALE:.0%} female)",
     "Pantazis N, et al. PLoS ONE 2012;7(3):e32369. doi:10.1371/journal.pone.0032369, Table 3 "
     "(sub-Saharan African seroconverter group). Reconstructed from the published mixed-model "
     "fixed effects and validated against three quantities the paper reports independently."),
    ("Age at acquisition",
     f"{P.ANCHOR_AGE:.0f} years",
     "Mean age at acquisition among adults in eastern and southern Africa (UNAIDS). Indexes both "
     "the background mortality schedule and the YLL reference, so both move with it. **[SOURCE "
     "PLACEHOLDER — needs a citation]**"),
    ("Background (non-HIV) mortality",
     f"Cause-deleted annual hazard by single year of age, {LT.background_hazard(29):.5f}/yr at 29 "
     f"rising to {LT.background_hazard(70):.4f}/yr at 70",
     "UN World Population Prospects 2024, eastern and southern Africa all-cause mortality, with the "
     "GBD 2021 regional HIV share of all-cause mortality deleted. Cause-deleted rather than "
     "all-cause so the background channel represents the same person without HIV; an ESA all-cause "
     "table carries 25–34% HIV mortality in the adult ages and would double-count against the "
     "model's own HIV death channel."),
    ("Reference life expectancy for YLL",
     f"Cause-deleted remaining life expectancy, {LT.remaining_life_expectancy(P.ANCHOR_AGE):.1f} "
     f"years at age {P.ANCHOR_AGE:.0f}",
     "Same table as background mortality, deliberately: an HIV death at age t costs exactly the "
     "non-HIV life the person would otherwise have had, under the identical schedule the surviving "
     "cohort faces."),
    ("CD4 reconstitution on suppressive ART",
     f"{p.fixed['recov_rate'][0]:.0f} cells/µL/yr below CD4 200, {p.fixed['recov_rate'][1]:.0f} above",
     "Biphasic recovery, fast phase at low CD4. Lawn et al. 2006. **[SOURCE INCOMPLETE — the "
     "spreadsheet row carries no link]**"),
    ("On-ART mortality, virally suppressed",
     "SMR by current CD4 band: " + ", ".join(f"{v:g}" for v in P.ART_SMR) +
     f"; applied as (SMR − 1) × background. Universal floor set to {P.ART_SMR_FLOOR:g} (inert).",
     "The ≥500 rung is Rodger AJ, et al. AIDS 2013;27(6):973–979. "
     "doi:10.1097/QAD.0b013e32835cae9c (SMR 1.00, 95% CI 0.69–1.40), and 1.8 is that paper's 1.77 "
     "for CD4 350–499. **The CD4 gradient below 350 is a modelling assumption:** 2.3 and 4.6 are "
     "attributed in the project notes to PISCIS but no publication reporting those values could be "
     "located, and 7.0 and 10.0 are recorded there as extrapolations. Capped at the untreated curve."),
    ("Same-day ART initiation",
     f"{P.SAME_DAY_INITIATION:.2f} of diagnoses",
     "Ross J, Brazier E, Fatti G, et al. Clin Infect Dis 2023;76(1):39–47. doi:10.1093/cid/ciac759. "
     "18,584/29,017 (64.0%) initiated on the same day, median 0 days (IQR 0–7), across 11 "
     "sub-Saharan African countries of which six are in eastern and southern Africa. Measured "
     "relative to enrolment in care rather than to diagnosis, so this is a mild overestimate of the "
     "diagnosis-to-ART same-day fraction."),
    ("Time to viral suppression on ART",
     f"{P.SUPPRESSION_RATE:g}/yr exit hazard from the unsuppressed ramp (median 28 days)",
     "The value most commonly reported for dolutegravir-based first line; consistent with ADVANCE "
     "(Venter et al. NEJM 2019, doi:10.1056/NEJMoa1902824). The exponential is a poor shape here — "
     "suppression is closer to a fixed delay than a memoryless process — and no single rate "
     "reproduces both the median and the tail."),
    ("Eventual return to care after disengagement",
     f"{P.LTFU_RETURN_FRACTION:.3f}",
     "Baldé A, Lièvre L, Maiga AI, et al. PLoS ONE 2020;15(9):e0238687. "
     "doi:10.1371/journal.pone.0238687. n = 3,650 across 16 facilities; cumulative re-engagement "
     "39.0% at one year, 45.0% at two, 47.0% at three. Fixes the plateau of the re-engagement "
     "curve while μ_re-engage sets its level. **Mali is West African**, so the shape of the "
     "re-engagement process is imported from outside the study region."),
    ("Re-engagement from long-term disengagement",
     f"{P.LTFU_REENGAGE_LATE:.4f}/yr",
     "Same source: the ~2-point rise between years two and three among the ~53% still disengaged."),
]
for name, val, src in rows_nh:
    w(f"| {name} | {val} | {src} |")

# --- economics
HE.set_price_year(2024)
c1_24, c2_24, dth_24 = HE.COST_1L_ART_PER_YEAR, HE.COST_2L_ART_PER_YEAR, HE.COST_DEATH
care_24 = list(HE.ROUTINE_CARE_BY_BAND)
HE.set_price_year(2019)
c1_19, c2_19, dth_19 = HE.COST_1L_ART_PER_YEAR, HE.COST_2L_ART_PER_YEAR, HE.COST_DEATH
care_19 = list(HE.ROUTINE_CARE_BY_BAND)
HE.set_price_year(2024)

HYLE = ("Hyle EP, Maphosa T, Rangaraj A, et al. Lancet Glob Health 2025;13(8):e1436–e1447. "
        "doi:10.1016/S2214-109X(25)00190-1, supplementary appendix Table S6 (Malawi, 2023 US$).")

rows_he = [
    ("Discount rate", f"{HE.DISCOUNT_RATE:.0%} per year, discrete compounding, applied to both health and cost",
     "WHO-CHOICE reference case; iDSI Reference Case principle 6."),
    ("Time horizon", f"{P.HORIZON_YEARS} years (lifetime, to age {P.ANCHOR_AGE + P.HORIZON_YEARS:.0f})",
     "Integrated to extinction rather than truncated: truncation drops long-duration survivors, who "
     "are disproportionately diagnosed and on treatment, and biases the diagnosis and treatment "
     "coverage the model is calibrated against downward."),
    ("Disability weight, on ART",
     f"{HE.DW_ONART_MIN}–{HE.DW_ONART_MAX}, linear in CD4 deficit from {HE.CD4_DW_ANCHOR:.0f} cells/µL",
     "Benchmarked to GBD HIV on ART. The functional form is a **reconstruction**, and the benchmarks "
     "used mix GBD rounds. **[NEEDS REFITTING TO A SINGLE GBD ROUND — see the structural sensitivity]**"),
    ("Disability weight, off ART",
     f"{HE.DW_OFFART_MIN}–{HE.DW_OFFART_MAX}, same form",
     "Same reconstruction caveat. Current GBD weights are 0.274 symptomatic pre-AIDS, 0.078 AIDS on "
     "ART, 0.582 AIDS not on ART; asymptomatic HIV carries no weight."),
    ("First-line ART drugs",
     f"${c1_19:,.0f}/yr (2019 US$); ${c1_24:,.0f}/yr (2024 US$)",
     f"TDF/3TC + DTG at $3.60/month. {HYLE}"),
    ("Second-line ART drugs",
     f"${c2_19:,.0f}/yr (2019 US$); ${c2_24:,.0f}/yr (2024 US$)",
     "AZT/3TC + LPV/r at $18.57/month, from the same table. Replaces a DRV/r-based figure of "
     "$274.20/yr built from the 2024 CHAI HIV Market Report; the two agree to within 20%."),
    ("Routine HIV care, by CD4 band",
     "2024 US$/yr: " + ", ".join(f"${v:,.0f}" for v in care_24) +
     "; 2019 US$/yr: " + ", ".join(f"${v:,.0f}" for v in care_19),
     "Monthly $2.86 / $3.96 / $4.90 / $10.77 / $18.51 / $26.24 across their six CD4 strata, which map "
     "exactly onto this model's seven bands. Built by Hyle et al. from resource utilisation by CD4 "
     "(Holmes 2006, Cape Town) priced with Malawi unit costs for an inpatient day and an outpatient "
     "visit (Maheswaran 2017, 2018). Replaces BOTH previous non-drug terms — the flat Tagar service-"
     "delivery cost and the CD4-tiered Menzies cost — because their outpatient-visit component is the "
     "clinic visit and their inpatient component is the advanced-disease care."),
    ("Applied to", "every living state, including Undiagnosed",
     "The one deliberate departure from Hyle's accounting, which charges people in care. The "
     "underlying resource use is driven by immunosuppression, and excluding the undiagnosed would "
     "understate the cost of late diagnosis. The in-care-only variant is reported as a scenario."),
    ("Terminal care, one off at death",
     f"${dth_19:,.0f} (2019 US$); ${dth_24:,.0f} (2024 US$)",
     "Any cause of death, from the same table. Replaces the previous unsourced $50 ART initiation "
     "cost, which has no counterpart in the adopted cost set."),
    ("Not adopted from the source",
     "Per-incident OI costs; diagnostic test costs; TB and cryptococcal modules",
     "Malaria $185.49, serious bacterial infection $142.80, other WHO stage 3/4 disease $86.72 per "
     "incident, and CD4 ($5.80) and viral load ($19.58) tests. This model has no state to attach them "
     "to. In Hyle's model they are charged ON TOP of routine care, so cost per acquisition here is a "
     "**lower bound** relative to a full implementation of their cost set."),
    ("Price-year adjustment",
     "US CPI-U, all items, annual average (BLS, 1982–84 = 100), from a 2023 base",
     "Hyle et al. converted to 2023 US$ using the Malawi inflation index and the 2023 average "
     "exchange rate — deflating in local currency then converting, which is the Global Health Cost "
     "Consortium preference. The CPI-U step here only moves an already-USD figure between price "
     "years, which is the appropriate use of a US index."),
]
for name, val, src in rows_he:
    w(f"| {name} | {val} | {src} |")
w()
w("*Not listed above, because they are inert in the base case:* the Mangal 2017 "
  "progression and mortality tables and the Glaubius acquisition-CD4 distribution, all "
  "retained as comparators for the structural sensitivity analysis; the universal on-ART "
  "SMR floor, set to 1.0 and therefore having no effect; the flat on-ART excess hazard, "
  "superseded by the CD4-dependent SMR ladder but kept for the mortality sensitivity "
  "script; and the legacy fixed reference life expectancy, superseded by the cause-deleted "
  "life table. Every other module-level constant in `params.py`, `health_econ_params.py` "
  "and `natural_history_params.py` appears in this appendix; the check is automated in "
  "`writeup/table_inputs.py`.")
w()

# ---------------------------------------------------------------------------
# D. Derived
# ---------------------------------------------------------------------------
mu_re = post["μ_re-engage"]["mean"]
w("**Table B4. Derived quantities.** Deterministic functions of the inputs above, "
  "with no prior of their own.")
w()
w("| Symbol | Quantity | Definition | Value at posterior mean |")
w("|---|---|---|---|")
w(f"| μ_perm_diseng | Transition from recent to long-term disengagement | "
  f"μ_re-engage × (1 − π)/π, where π = {P.LTFU_RETURN_FRACTION:.3f} is the eventual-return "
  f"fraction | {P.ltfu_perm_disengage(mu_re):.3g}/yr |")
w()
w("Parameterised as a fraction rather than as a fixed companion hazard because the "
  "plateau of the re-engagement curve is r/(r + d): fixing d and freeing r does not "
  "hold the shape the source pins down. Holding the fraction and deriving the "
  "companion keeps the plateau exactly while letting the speed of the process vary "
  "with the calibration.")
w()

# ---------------------------------------------------------------------------
# E. Targets
# ---------------------------------------------------------------------------
TGT_LABEL = {
    "p1": "Diagnosed, % of all people living with HIV",
    "p2": "On ART, % of those diagnosed",
    "p3": "Virally suppressed, % of those on ART",
    "ART_1m": "Started ART within one month of diagnosis",
    "CD4_median_ART": "Median CD4 at first-line ART initiation",
    "CD4_lt200_ART": "CD4 <200 at first-line ART initiation",
    "T_ART": "Median time from diagnosis to ART",
    "Ret_12m": "Retention at 12 months", "Ret_24m": "Retention at 24 months",
    "Supp_2L": "Resuppression on second line", "HR_LTFU": "Mortality hazard ratio, disengaged",
    "Reeng_1y": "Cumulative re-engagement, 1 year",
    "Reeng_2y": "Cumulative re-engagement, 2 years",
    "Reeng_3y": "Cumulative re-engagement, 3 years",
}
TGT_SRC = {
    "p1": "UNAIDS Data 2020, p.40, eastern and southern Africa (2019). The 2024-cascade arm uses the UNAIDS 2025 Global AIDS Update ESA profile instead.",
    "p2": "UNAIDS Data 2020, derived: 72% on treatment of all PLHIV ÷ 87% diagnosed. Cross-checks against the same page's directly reported second-90 figure (0.828 against 0.83).",
    "p3": "UNAIDS Data 2020, derived: 65% suppressed of all PLHIV ÷ 72% on treatment. Cross-checks against the reported third-90 (0.903 against 0.90).",
    "ART_1m": "Ross J, Brazier E, Fatti G, et al. Clin Infect Dis 2023;76(1):39–47. doi:10.1093/cid/ciac759 — at least 85% initiating within a month. Declared in `likelihoods.py`, not in the parameter spreadsheet.",
    "CD4_median_ART": "de Waal R, et al. 2024. Pooled Southern Africa (excl. South Africa) and East Africa, 2016–2019, N = 82,713 with a CD4 result at ART start. SD is the between-region/between-year spread of the eight available estimates, doubled after a full run showed the tighter value forced the fit into a corner.",
    "CD4_lt200_ART": "de Waal R, et al. 2024, same pooling. SD widened to span the ~11-point gap between the prevalence among those with a CD4 result (32.2%) and the lower bound across all ART starters (~21.7%), which is ascertainment uncertainty rather than sampling noise.",
}
NOT_FITTED_SRC = {
    "T_ART": "Ross et al. 2023. Superseded in the likelihood by ART_1m, which uses the same source but a threshold the model can represent.",
    "Ret_12m": "Pooled GLMM estimate for LMICs under universal test-and-treat.",
    "Ret_24m": "Same source. Dropped from the likelihood: p2 and Ret_12m could not be satisfied simultaneously, and retaining both drove the fit to an implausible corner.",
    "Supp_2L": "Sub-Saharan African second-line cohorts. **[SECONDARY SOURCE — the spreadsheet cites a news summary; the primary study is needed]**",
    "HR_LTFU": "Brinkhof et al., pooled mortality hazard ratio for disengaged against retained patients. Excluded from the likelihood on the grounds that the model could not reach it — a sweep recorded in `likelihoods.py` had the implied ratio saturating near 9 at φ = 32. **That is no longer true:** with the adopted Glaubius mortality table replacing the previously interpolated curve, the ratio now runs 11.0 at φ = 1 to 29.4 at φ = 8 and crosses the target at φ ≈ 1.14. The model reproduces this value without being fitted to it, which makes it the strongest out-of-sample check in the cascade module rather than a discarded target.",
    "Reeng_1y": "Baldé et al. 2020, doi:10.1371/journal.pone.0238687. Demoted from a target to a fixed shape: the figures measure clinic re-engagement rather than population coverage, and are West African.",
    "Reeng_2y": "Baldé et al. 2020, as above.",
    "Reeng_3y": "Baldé et al. 2020, as above.",
}

fitted = dict(s2.get("targets", {}))
implied = dict(s2.get("model_implied_at_mean", {}))
alias = dict(s2.get("target_symbols", {}))

w("**Table B5. Calibration targets in the likelihood.** These six quantities, and no "
  "others, enter the objective the sampler maximises. Standard deviations are the weight "
  "each target carries, not sampling error alone; several are widened deliberately, and "
  "the reason is given with the source. Model values are at the posterior mean of the "
  "base-case (2019 cascade) fit.")
w()
w("| Symbol | Quantity | Target (SD) | Model | z | Source |")
w("|---|---|---|---|---|---|")
for sym, (c, sd) in fitted.items():
    lab = TGT_LABEL.get(sym, sym)
    src = TGT_SRC.get(sym, "—")
    mv = implied.get(sym)
    z = (float(mv) - c) / sd if mv is not None else None
    mv_s = f"{float(mv):.4g}" if mv is not None else "—"
    z_s = f"{z:+.2f}" if z is not None else "—"
    note = f" (spreadsheet row `{alias[sym]}`)" if sym in alias else ""
    w(f"| {sym} | {lab}{note} | {c:g} ({sd:g}) | {mv_s} | {z_s} | {src} |")
w()

w("**Table B6. Comparators reported but not fitted.** Each was considered as a target "
  "and is computed at every evaluation, but does not enter the likelihood. Reporting "
  "them is the point: a quantity the model reproduces without having been fitted to it "
  "is evidence, and a quantity it misses is a limitation that would otherwise be hidden "
  "by fitting to it.")
w()
w("| Symbol | Quantity | Reference value | Model | Status and source |")
w("|---|---|---|---|---|")
_REFVALS = {"T_ART": "0.5 months", "Ret_12m": "0.796", "Ret_24m": "0.812",
            "Supp_2L": "0.85", "HR_LTFU": f"{LK.HR_LTFU_TARGET[1]:g} ({LK.HR_LTFU_TARGET[2]:g})",
            "Reeng_1y": "0.390", "Reeng_2y": "0.450", "Reeng_3y": "0.470"}
for sym, mv in implied.items():
    if sym in fitted:
        continue
    lab = TGT_LABEL.get(sym, sym)
    w(f"| {sym} | {lab} | {_REFVALS.get(sym, '—')} | {float(mv):.4g} | "
      f"{NOT_FITTED_SRC.get(sym, '—')} |")
w()
w(f"Switches controlling these blocks, as run: `USE_RETENTION_LIKELIHOOD = "
  f"{LK.USE_RETENTION_LIKELIHOOD}`, `USE_SUPP2L_LIKELIHOOD = {LK.USE_SUPP2L_LIKELIHOOD}`, "
  f"`USE_REENGAGEMENT_LIKELIHOOD = {LK.USE_REENGAGEMENT_LIKELIHOOD}`, "
  f"`USE_HR_LTFU_LIKELIHOOD = {LK.USE_HR_LTFU_LIKELIHOOD}`.")
w()

# ---------------------------------------------------------------------------
# G. Out-of-sample validation of the adopted natural history
# ---------------------------------------------------------------------------
import simulate_natural_history as SNH

w("**Table B7. Out-of-sample validation of the adopted natural history.** System 1 is "
  "not fitted, so these three quantities are predictions rather than fits, and the model "
  "can fail them. Computed at κ_prog = κ_mort = 1.")
w()
w("| Quantity | Model | Reference (SD) | z | Reference source |")
w("|---|---|---|---|---|")
_VALID = [
    ("Median time to CD4 <350",
     SNH.first_passage_median_years(P.BAND_INDEX_CD4_350), "T_<350",
     "Pooled untreated seroconverter cohorts (CASCADE)."),
    ("Median time to CD4 <200",
     SNH.first_passage_median_years(P.BAND_INDEX_CD4_200), "T_<200",
     "Pooled untreated seroconverter cohorts (CASCADE)."),
    ("Median untreated survival",
     SNH.overall_survival_median_years(), "T_survival",
     "Pooled median survival for a cohort aged 25–34 at seroconversion."),
]
for lab, modelled, sym, src in _VALID:
    c, sd = p.targets[sym]
    w(f"| {lab} | {modelled:.2f} yr | {c:g} ({sd:g}) | {(modelled - c)/sd:+.2f} | {src} |")
w()

# ---------------------------------------------------------------------------
# Provenance gaps
# ---------------------------------------------------------------------------
GAPS = [
    "Age at acquisition (29 years) has no citation in the code or the spreadsheet.",
    "CD4 reconstitution rates (306/92 cells/µL/yr) cite 'Lawn et al. 2006' with no link.",
    "The on-ART SMR gradient below CD4 350 (2.3, 4.6, 7.0, 10.0) is a modelling assumption; "
    "the PISCIS attribution in the project notes could not be verified.",
    "The disability-weight functional form is a reconstruction, not the original formula, and its "
    "benchmarks mix GBD 2010 and GBD 2013 values. Costs now come from a single source; the "
    "disability weights are the remaining place where the model mixes provenance.",
    "The spreadsheet still carries rows for λ_base and α_shape, and fixed rows for Acq_CD4 (579 "
    "cells/µL) and m_untreated, none of which the model now uses. It also still carries the "
    "superseded cost rows. Delete them or mark them superseded.",
    "The spreadsheet's natural-history reference values (T_<350 4.19 ± 0.40, T_<200 7.93 ± 0.50) "
    "differ from those quoted in the natural_history_params.py docstring (4.14 ± 0.64, 8.11 ± 1.11). "
    "Table B7 uses the spreadsheet. Reconcile them, since the z-scores move with the choice.",
    "The spreadsheet's T_ART row is described as '17 days (IQR 9–42) for adolescents and 0 days "
    "(64% same-day) for broader adults'. Ross et al. report median 0 days (IQR 0–7) overall and "
    "no such adolescent stratum; the 17-day figure is untraceable to that source.",
    "Ross et al. 2023 measure same-day initiation relative to enrolment in care, not to diagnosis, "
    "and pool 11 countries of which five are in western and central Africa.",
    "The φ_ltfu sweep recorded in likelihoods.py (HR_LTFU saturating near 9 at φ = 32) predates the "
    "adopted Glaubius mortality table and no longer holds — the current sweep gives 11.0 at φ = 1 "
    "and 29.4 at φ = 8. The comment justifying USE_HR_LTFU_LIKELIHOOD = False should be updated, "
    "and re-enabling the term is now worth considering.",
    "COSTS: the adopted cost set is Malawi-specific and Malawi sits at the low end of the eastern "
    "and southern African range, so the estimate is conservative on cost and favourable on "
    "cost-effectiveness. This is now the single largest transportability limitation and needs a "
    "one-way sensitivity spanning Malawi to the CEPAC Cote d'Ivoire routine-care costs.",
    "COSTS: Hyle et al. charge per-incident opportunistic infection costs and diagnostic tests on "
    "top of routine care. This model has no state to attach them to, so cost per acquisition is a "
    "lower bound relative to a full implementation of their cost set.",
]

if "--check" in sys.argv:
    print("Provenance gaps (%d):" % len(GAPS))
    for g in GAPS:
        print("  -", g)
    raise SystemExit

w("**Outstanding provenance gaps.** Listed here rather than left implicit; each is a "
  "row above that an examiner could reasonably ask about.")
w()
for g in GAPS:
    w(f"- {g}")
w()

print("\n".join(OUT))

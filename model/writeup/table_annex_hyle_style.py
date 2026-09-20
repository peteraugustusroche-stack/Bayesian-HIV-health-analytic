"""Model input annex laid out in the style of Hyle et al. 2025 (Lancet Glob Health)
supplementary appendix: Parameter / Value / Reference, grouped by domain, with the
price year stated in the table title and lettered footnotes.

Mirrors their Tables S1 (natural history), S2 (treatment and care continuum),
S6 (costs) and S7 (quality of life). Every value is read from the live modules
or the saved calibration results.

    python3 writeup/table_annex_hyle_style.py > writeup/annex_inputs.md
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

PRICE_YEAR = int(os.environ.get("PRICE_YEAR", "2024"))
HE.set_price_year(PRICE_YEAR)

OUT = []
def w(s=""):
    OUT.append(s)

p = P.load_parameters()
s2 = load_result("system2")
post = {r["param"]: r for r in s2["summary"]}
nh = {r["param"]: r for r in load_result("system1")["summary"]}

BANDS = P.CD4_BAND_NAMES


def row(param, value, ref, indent=0):
    pad = "&nbsp;&nbsp;&nbsp;&nbsp;" * indent
    w(f"| {pad}{param} | {value} | {ref} |")


def head(title):
    w(f"| **{title}** | | |")


def table_open(title, note=None):
    w(f"**{title}**")
    if note:
        w()
        w(note)
    w()
    w("| Parameter | Value | Reference |")
    w("|---|---|---|")


def table_close(abbrev=None, footnotes=()):
    w()
    for marker, text in footnotes:
        w(f"<sup>{marker}</sup> {text}  ")
    if footnotes:
        w()
    if abbrev:
        w(f"*Abbreviations:* {abbrev}")
    w()


# ===========================================================================
# Table A1 -- natural history
# ===========================================================================
table_open("Table A1. Detailed input parameters for HIV natural history.")

head("Cohort at HIV acquisition")
row("Age, years", f"{P.ANCHOR_AGE:.0f}", "UNAIDS <sup>a</sup>")
row("CD4 count distribution, cells/µL",
    ", ".join(f"{b} {v*100:.1f}%" for b, v in zip(BANDS, ACQ.initial_band_distribution())),
    "Pantazis 2012 <sup>b</sup>")
row("Central CD4 count at acquisition, cells/µL", f"{ACQ.sqrt_cd4_mean()**2:.0f}", "Pantazis 2012")

head("CD4 progression when untreated, annual rate")
for i, name in enumerate(NH.AGE_GROUP_NAMES):
    row(f"Age {name}",
        ", ".join(f"{v:.3f}" for v in NH.GLAUBIUS_PROGRESSION[i]),
        "Glaubius 2021" if i == 0 else "")

head("HIV mortality when untreated, annual rate")
for i, name in enumerate(NH.AGE_GROUP_NAMES):
    row(f"Age {name}",
        ", ".join(f"{v:.3f}" for v in NH.GLAUBIUS_MORTALITY[i]),
        "Glaubius 2021" if i == 0 else "")

head("Mortality on suppressive ART")
row("Standardised mortality ratio by current CD4 count",
    ", ".join(f"{v:g}" for v in P.ART_SMR),
    "Rodger 2013 (≥350); assumption <sup>c</sup>")
row("Applied as", "(SMR − 1) × background hazard, capped at the untreated rate", "")
row("Uncertainty on the ladder, log-scale SD", f"{P.ART_SMR_LOG_SD:g}", "Rodger 2013")

head("Mortality when disengaged from care")
row("Multiplier on the untreated CD4-specific rate",
    f"{post['φ_ltfu']['mean']:.2f} (94% CrI {post['φ_ltfu']['q03']:.2f}–{post['φ_ltfu']['q97']:.2f})",
    "Calibrated")

head("Background (non-HIV) mortality")
row("Annual hazard, age 29 / 50 / 70",
    f"{LT.background_hazard(29):.5f} / {LT.background_hazard(50):.4f} / {LT.background_hazard(70):.4f}",
    "UN WPP 2024, HIV cause-deleted using GBD 2021 <sup>d</sup>")
row("Remaining life expectancy at age 29, years",
    f"{LT.remaining_life_expectancy(P.ANCHOR_AGE):.1f}", "Same table")

head("CD4 reconstitution on suppressive ART")
row("Recovery rate, cells/µL/year",
    f"{p.fixed['recov_rate'][0]:.0f} below CD4 200; {p.fixed['recov_rate'][1]:.0f} above", "Lawn 2006 <sup>a</sup>")
row("Ceiling, nadir ≥200 cells/µL", "≥500 cells/µL", "Microsimulation mapping <sup>e</sup>")
row("Ceiling, nadir <200 cells/µL", "350–499 cells/µL", "Microsimulation mapping <sup>e</sup>")

head("Scale factors carrying natural-history uncertainty")
row("κ_prog, progression schedule",
    f"lognormal, median 1, σ = {NH.KAPPA_PROG_SIGMA}", "Glaubius 2021 credible intervals")
row("κ_mort, mortality schedule",
    f"lognormal, median 1, σ = {NH.KAPPA_MORT_SIGMA}", "Glaubius 2021 vs Mangal 2017 <sup>f</sup>")

table_close(
    abbrev="ART, antiretroviral therapy; CrI, credible interval; SD, standard deviation; "
           "SMR, standardised mortality ratio; UN WPP, United Nations World Population Prospects.",
    footnotes=[
        ("a", "Source not yet verified; see the provenance note in the methods."),
        ("b", "Reconstructed from the published mixed-model fixed effects for the sub-Saharan African "
              "seroconverter group and validated against three quantities the paper reports independently."),
        ("c", "The ≥500 rung is Rodger's point estimate (SMR 1.00) and 1.8 is their 1.77 for CD4 350–499. "
              "The gradient below CD4 350 is a modelling assumption."),
        ("d", "Cause-deleted so the background channel represents the same person without HIV; an all-cause "
              "regional table would double-count against the model's own HIV death channel."),
        ("e", "Reconstitution is capped by the nadir. Seven nadir bands collapse to two distinct ceilings, "
              "so the model carries a binary class rather than a nadir dimension."),
        ("f", "Deliberately wider than Glaubius's internal precision because between-source disagreement on "
              "where untreated mortality sits exceeds either source's own uncertainty."),
    ])


# ===========================================================================
# Table A2 -- care continuum
# ===========================================================================
CASC = [
    ("δ_bg", "Diagnosis, stage-independent, annual rate"),
    ("δ_symp", "Diagnosis, additional rate at the lowest CD4 band"),
    ("γ_diag", "Diagnosis, exponent of the CD4 severity ramp"),
    ("λ_init", "ART initiation after diagnosis, annual rate"),
    ("μ_vf1", "First-line virological failure, annual rate"),
    ("μ_switch", "Switch to second line after failure, annual rate"),
    ("μ_ltfu", "Disengagement from care, annual rate"),
    ("μ_re-engage", "Return to care from recent disengagement, annual rate"),
]
table_open("Table A2. Detailed input parameters for HIV treatment and the care continuum.",
           "Calibrated parameters are posterior means with 94% credible intervals, estimated by "
           "Markov chain Monte Carlo against the targets in Table A5. The base case is calibrated "
           "to the 2019 regional cascade.")

head("Calibrated cascade rates")
for sym, label in CASC:
    r = post[sym]
    row(label, f"{r['mean']:.3g} ({r['q03']:.3g}–{r['q97']:.3g})", "Calibrated")

head("Fixed cascade inputs")
row("Same-day ART initiation, proportion of diagnoses", f"{P.SAME_DAY_INITIATION:.2f}",
    "Ross 2023 <sup>a</sup>")
row("Viral suppression after initiation, annual rate",
    f"{P.SUPPRESSION_RATE:g} (median 28 days)", "Venter 2019")
row("Eventual return to care after disengagement, proportion",
    f"{P.LTFU_RETURN_FRACTION:.3f}", "Baldé 2020 <sup>b</sup>")
row("Return from long-term disengagement, annual rate",
    f"{P.LTFU_REENGAGE_LATE:.4f}", "Baldé 2020")
row("Transition to long-term disengagement, annual rate",
    f"{P.ltfu_perm_disengage(post['μ_re-engage']['mean']):.3g}", "Derived <sup>c</sup>")

head("Analytic conventions")
row("Cycle length", "1 month", "")
row("Time horizon", f"{P.HORIZON_YEARS} years (to age {P.ANCHOR_AGE + P.HORIZON_YEARS:.0f})",
    "Lifetime, integrated to extinction <sup>d</sup>")
row("Discount rate, health and cost", f"{HE.DISCOUNT_RATE:.0%} per year",
    "WHO-CHOICE; iDSI Reference Case")
row("Perspective", "Health-system provider", "iDSI Reference Case <sup>e</sup>")

table_close(
    abbrev="ART, antiretroviral therapy; iDSI, international Decision Support Initiative.",
    footnotes=[
        ("a", "Measured relative to enrolment in care rather than to diagnosis, and pooled across 11 "
              "sub-Saharan African countries of which six are in eastern and southern Africa."),
        ("b", "Fixes the plateau of the re-engagement curve while the calibrated rate sets its level. "
              "The source cohort is West African."),
        ("c", "μ_re-engage × (1 − π)/π, where π is the eventual-return proportion above."),
        ("d", "The horizon must reach extinction because the cascade proportions are computed as ratios "
              "of state occupancy integrated over duration since infection."),
        ("e", "Deviates from the iDSI preference for a limited societal perspective; patient out-of-pocket, "
              "transport, time and productivity costs are excluded."),
    ])


# ===========================================================================
# Table A3 -- costs
# ===========================================================================
table_open(f"Table A3. Detailed cost parameters (USD {PRICE_YEAR}).",
           "Adopted in full from Hyle et al. 2025, supplementary appendix Table S6 (Malawi, USD 2023), "
           "and restated in the analysis price year by US CPI-U. Monthly values are given first, as in "
           "the source, with annual equivalents beneath.")

head("Antiretroviral drugs, monthly")
row("TDF/3TC + DTG", f"{HE.COST_1L_ART_PER_YEAR/12:.2f}", "Hyle 2025 <sup>a</sup>")
row("AZT/3TC + LPV/r", f"{HE.COST_2L_ART_PER_YEAR/12:.2f}", "Hyle 2025")

head("Routine HIV care costs by CD4 count, /µL, monthly")
for b, name in enumerate(BANDS):
    row(name, f"{HE.ROUTINE_CARE_BY_BAND[b]/12:.2f}",
        "Hyle 2025 <sup>b</sup>" if b == 0 else "")

head("Other")
row("Death cost, any cause of death", f"{HE.COST_DEATH:.2f}", "Hyle 2025")

head("Annual equivalents, for reference")
row("First-line ART drugs", f"{HE.COST_1L_ART_PER_YEAR:.2f}", "")
row("Second-line ART drugs", f"{HE.COST_2L_ART_PER_YEAR:.2f}", "")
row("Routine care, CD4 ≥500 to <50",
    " – ".join(f"{HE.ROUTINE_CARE_BY_BAND[b]:.0f}" for b in (0, 6)), "")

head("Application")
row("Routine care charged to", "every living state, including undiagnosed", "<sup>c</sup>")
row("Not adopted from the source",
    "per-incident OI costs; CD4 and viral load test costs; TB and cryptococcal modules",
    "<sup>d</sup>")
row("Price-year conversion", "US CPI-U, all items, annual average, from a 2023 base",
    "US BLS <sup>e</sup>")

table_close(
    abbrev="ART, antiretroviral therapy; AZT, zidovudine; CPI-U, consumer price index for all urban "
           "consumers; DTG, dolutegravir; LPV/r, lopinavir/ritonavir; OI, opportunistic infection; "
           "3TC, lamivudine; TDF, tenofovir disoproxil fumarate; USD, United States dollar.",
    footnotes=[
        ("a", "Hyle EP, Maphosa T, Rangaraj A, et al. Lancet Glob Health 2025;13(8):e1436–e1447, "
              "supplementary appendix Table S6. All cost inputs are taken from this single source so "
              "that price year, setting and costing method are internally consistent."),
        ("b", "Their six CD4 strata map exactly onto this model's seven bands; their 200–349 stratum "
              "spans two of the model's bands and every other boundary coincides. Built from resource "
              "utilisation by CD4 (Holmes 2006, Cape Town) priced with Malawi unit costs for an "
              "inpatient day and an outpatient visit (Maheswaran 2017, 2018). This schedule replaces "
              "both non-drug terms the model previously carried, because the outpatient-visit "
              "component is the clinic visit and the inpatient component is the advanced-disease care."),
        ("c", "Hyle et al. charge routine care to people in care. Extending it to the undiagnosed is "
              "this model's one departure from their accounting: the underlying resource use is driven "
              "by immunosuppression, and excluding the undiagnosed would understate the cost of late "
              "diagnosis. The in-care-only variant is reported as a scenario."),
        ("d", "Malaria $185.49, serious bacterial infection $142.80 and other WHO stage 3/4 disease "
              "$86.72 per incident; CD4 test $5.80; viral load test $19.58. This model has no state to "
              "attach them to, and in the source they are charged on top of routine care, so cost per "
              "acquisition here is a lower bound relative to a full implementation of that cost set."),
        ("e", "Hyle et al. converted to 2023 USD using the Malawi inflation index and the 2023 average "
              "exchange rate, deflating in local currency then converting. The CPI-U step here only "
              "moves an already-USD figure between price years."),
    ])


# ===========================================================================
# Table A4 -- disability weights
# ===========================================================================
table_open("Table A4. Disability weight inputs by CD4 count and treatment status.")
head("Disability weight")
for b, name in enumerate(BANDS):
    row(name,
        f"{HE.disability_weight(b, True):.3f} on suppressive ART; "
        f"{HE.disability_weight(b, False):.3f} otherwise",
        "GBD benchmarks <sup>a</sup>" if b == 0 else "")
head("Functional form")
row("On suppressive ART",
    f"clip({HE.DW_ONART_MIN} + {HE.DW_ONART_SLOPE} × (500 − CD4)/{HE.CD4_SLOPE_UNIT_CELLS:.0f}, "
    f"{HE.DW_ONART_MIN}, {HE.DW_ONART_MAX})", "")
row("Not suppressed",
    f"clip({HE.DW_OFFART_MIN} + {HE.DW_OFFART_SLOPE} × (500 − CD4)/{HE.CD4_SLOPE_UNIT_CELLS:.0f}, "
    f"{HE.DW_OFFART_MIN}, {HE.DW_OFFART_MAX})", "")
row("Representative CD4 by band, cells/µL",
    ", ".join(f"{v:.0f}" for v in HE.CD4_BAND_REPRESENTATIVE), "")
table_close(
    abbrev="ART, antiretroviral therapy; GBD, Global Burden of Disease.",
    footnotes=[
        ("a", "The continuous form is a reconstruction fitted to GBD benchmark weights rather than a "
              "published formula, and the benchmarks used mix GBD rounds. Current GBD weights are 0.274 "
              "(symptomatic, pre-AIDS), 0.078 (AIDS on ART) and 0.582 (AIDS not on ART); asymptomatic HIV "
              "carries no weight. See the structural sensitivity analysis."),
    ])


# ===========================================================================
# Table A5 -- calibration targets
# ===========================================================================
TL = {"p1": "Diagnosed, proportion of all people living with HIV",
      "p2": "On ART, proportion of those diagnosed",
      "p3": "Virally suppressed, proportion of those on ART",
      "ART_1m": "Started ART within one month of diagnosis",
      "CD4_median_ART": "Median CD4 count at first-line ART initiation, cells/µL",
      "CD4_lt200_ART": "CD4 <200 cells/µL at first-line ART initiation, proportion"}
TS = {"p1": "UNAIDS 2020", "p2": "UNAIDS 2020 <sup>a</sup>", "p3": "UNAIDS 2020 <sup>a</sup>",
      "ART_1m": "Ross 2023", "CD4_median_ART": "de Waal 2024", "CD4_lt200_ART": "de Waal 2024 <sup>b</sup>"}
table_open("Table A5. Calibration targets and model fit.",
           "These six quantities, and no others, enter the likelihood. Model values are at the "
           "posterior mean of the base-case fit.")
head("Target (SD) and model-projected value")
imp = s2.get("model_implied_at_mean", {})
for sym, (cval, sd) in s2.get("targets", {}).items():
    mv = float(imp[sym]); z = (mv - cval) / sd
    row(TL.get(sym, sym), f"{cval:g} ({sd:g}); model {mv:.4g}; z = {z:+.2f}", TS.get(sym, ""))
table_close(
    abbrev="ART, antiretroviral therapy; SD, standard deviation.",
    footnotes=[
        ("a", "Derived from the reported all-PLHIV proportions rather than taken directly, and "
              "cross-checked against the same source's own conditional figures."),
        ("b", "Standard deviation widened to span the gap between prevalence among those with a CD4 "
              "result and the lower bound across all ART starters, which is ascertainment uncertainty "
              "rather than sampling noise."),
    ])

print("\n".join(OUT))

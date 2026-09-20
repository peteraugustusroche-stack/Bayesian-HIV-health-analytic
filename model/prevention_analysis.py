"""Prevention analysis: what the per-acquisition burden implies for long-acting PrEP.

The model estimates two quantities that a prevention cost-effectiveness analysis
needs and rarely states:

    DALYs accrued per acquisition     the health loss a prevented acquisition avoids
    cost per acquisition             the treatment spending a prevented acquisition avoids

Both are taken from the CURRENT cascade arm, because a person who acquires HIV in
the region today enters today's cascade. Using the no-care counterfactual instead
would value prevention against a world in which nobody is treated, which is not the
world a prevention programme operates in.

The arithmetic is deliberately transparent:

    cost per acquisition averted      = NNT x annual PrEP cost
    net cost                          = that, minus the treatment cost avoided
    incremental cost per DALY averted = net cost / DALYs accrued per acquisition

where NNT is person-years of PrEP per acquisition averted, 1 / (incidence x efficacy).
NNT is the primary axis rather than incidence because it is what a programme can
observe, and because it already embeds efficacy: at the 96% of PURPOSE 2 rather than
the 100% of PURPOSE 1, an observed incidence of 2.0 per 100 person-years gives NNT 52
rather than 50, which does not move any conclusion here.

Three things are varied, and they are the three things that decide the answer:

  1. TARGETING          NNT, i.e. how well the programme finds people at risk
  2. PRICE              the all-in annual cost per person on PrEP, drug plus delivery
  3. THE DENOMINATOR    the DALYs assumed to be averted by preventing one acquisition
  4. THE COST OFFSET    whether the analysis nets off the treatment it prevents

The last two are the point of the paper. An analysis that takes a per-acquisition
burden from the literature without asking which cascade generated it, and that omits
the treatment cost avoided, will reach a different answer from the same trial data.

    python3 prevention_analysis.py                # table to stdout
    python3 prevention_analysis.py --md           # markdown table for the write-up
    python3 writeup/fig_prevention.py             # the figure
"""
from __future__ import annotations

import sys

import numpy as np

from run_posterior_analysis import load_posterior_result

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

#: Cascade arm the prevented acquisition is valued against. A person acquiring
#: HIV today enters today's cascade, so the contemporary arm is the right basis
#: even though the 2019 arm is the base case elsewhere in the analysis.
CASCADE_YEAR = "2024"
PRICE_YEAR = 2024

#: Targeting scenarios, as person-years of PrEP per acquisition averted.
#: Chosen to span what published trials achieved in eastern and southern Africa
#: rather than to be round numbers -- see ANCHORS below.
NNT_SCENARIOS = [
    (20, "Intensive targeting"),
    (50, "Targeted"),
    (80, "Moderate targeting"),
]

#: All-in annual cost per person-year on PrEP: drug plus delivery, injection
#: visits and periodic testing. Generic lenacapavir is expected at about $40 per
#: person-year in licensed low- and lower-middle-income countries from 2027, so
#: $60 is the realistic all-in column at that drug price and $25 is the floor a
#: production-cost analysis implies.
PRICES = [25, 40, 60, 100]

#: Observed incidence in populations recruited for HIV prevention trials in the
#: region, converted to NNT at 100% efficacy. These are what good targeting has
#: actually achieved, and they bracket the middle scenario.
ANCHORS = [
    (2.41, "PURPOSE 1 background cohort",
     "Bekker et al. NEJM 2024, doi:10.1056/NEJMoa2407001; young women, "
     "South Africa and Uganda"),
    (1.79, "HPTN 084, TDF/FTC arm",
     "Delany-Moretlwe et al. Lancet 2022, doi:10.1016/S0140-6736(22)00538-4; "
     "34 infections among 3,223 cisgender women across seven ESA countries"),
    (0.17, "ESA adults 15-49, untargeted",
     "UNAIDS regional factsheet; roughly 490,000-500,000 new infections across "
     "about 280 million adults"),
]

#: Per-acquisition DALY burdens used in the literature, for the denominator
#: comparison. The point is not that these are wrong but that they describe
#: different cascades -- or none.
LITERATURE_DALYS = [
    (10.0, "A commonly used round figure"),
    (20.0, "Representative sub-Saharan African value in current use"),
]

EFFICACY_NOTE = ("NNT embeds efficacy. PURPOSE 1 recorded no infections on "
                 "lenacapavir (efficacy 100%, 95% CI on the incidence 0.00-0.19 per "
                 "100 person-years); PURPOSE 2 gave 96%. At 96% an observed incidence "
                 "of 2.0 per 100 person-years gives NNT 52 rather than 50.")


# ---------------------------------------------------------------------------
# Model quantities
# ---------------------------------------------------------------------------

def model_quantities(cascade_year: str = CASCADE_YEAR):
    """Posterior draws of the two quantities prevention needs from this model."""
    m = load_posterior_result(cascade_year)["probabilistic"]["metrics"]
    dalys = np.asarray(m["DALYs_per_acquisition"])
    cost = np.asarray(m["cost_per_acquisition"])
    nocare = np.asarray(m["DALYs_natural_history"])
    return dalys, cost, nocare


def icer(nnt: float, price: float, dalys, offset, offset_share: float = 1.0):
    """Incremental cost per DALY averted, as an array over posterior draws.

    `offset_share` is the fraction of the avoided treatment cost the analysis
    credits. 1.0 is the correct treatment; 0.0 reproduces an analysis that counts
    only what PrEP costs and ignores what it saves.
    """
    net = nnt * price - offset_share * np.asarray(offset)
    return net / np.asarray(dalys)


def summarise(a, lo=5, hi=95):
    a = np.asarray(a)
    return a.mean(), np.percentile(a, lo), np.percentile(a, hi)


def fmt_icer(vals) -> str:
    mean, _, _ = summarise(vals)
    return "cost-saving" if mean < 0 else f"${mean:,.0f}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def main(as_markdown: bool = False) -> None:
    dalys, cost, nocare = model_quantities()
    d_m, d_lo, d_hi = summarise(dalys)
    c_m, c_lo, c_hi = summarise(cost)

    rows = [(n, lab, 100.0 / n) for n, lab in NNT_SCENARIOS]
    rows += [(1 / (i / 100), lab, i) for i, lab, _ in ANCHORS]
    rows.sort(key=lambda r: r[0])

    out = []
    w = out.append

    if as_markdown:
        w(f"**Model quantities, {CASCADE_YEAR} cascade, {PRICE_YEAR} US$.** "
          f"DALYs averted by preventing one acquisition {d_m:.2f} "
          f"(90% CrI {d_lo:.2f}–{d_hi:.2f}); lifetime treatment cost avoided "
          f"${c_m:,.0f} (${c_lo:,.0f}–${c_hi:,.0f}).")
        w("")
        w("**Table P1. Incremental cost per DALY averted by long-acting PrEP, "
          "by targeting and all-in annual cost.**")
        w("")
        w("| NNT, person-years | Implied incidence /100py | Scenario | "
          + " | ".join(f"${p}/yr" for p in PRICES) + " |")
        w("|---|---|---|" + "---|" * len(PRICES))
        for nnt, lab, inc in rows:
            cells = " | ".join(fmt_icer(icer(nnt, p, dalys, cost)) for p in PRICES)
            w(f"| {nnt:.0f} | {inc:.2f} | {lab} | {cells} |")
        w("")
        w("*Net of the lifetime treatment cost a prevented acquisition avoids. "
          "'Cost-saving' means the PrEP spend is less than the treatment spend "
          "it displaces.*")
        w("")
        w("**Table P2. Sensitivity to the per-acquisition DALY burden assumed.** "
          f"At NNT 50 and $60 per person-year.")
        w("")
        w("| DALYs per acquisition | Basis | Cost per DALY averted |")
        w("|---|---|---|")
        net = 50 * 60 - cost
        w(f"| {d_m:.2f} | This model, {CASCADE_YEAR} cascade | "
          f"{fmt_icer(net / dalys)} |")
        for v, lab in LITERATURE_DALYS:
            w(f"| {v:.2f} | {lab} | {fmt_icer(net / v)} |")
        w(f"| {nocare.mean():.2f} | This model, no-care counterfactual | "
          f"{fmt_icer(net / nocare)} |")
        w("")
        w("**Table P3. Sensitivity to crediting the treatment cost avoided.** "
          "At NNT 50 and $60 per person-year.")
        w("")
        w("| Share of treatment cost credited | Cost per DALY averted |")
        w("|---|---|")
        for share in (0.0, 0.25, 0.5, 0.75, 1.0):
            w(f"| {share:.0%} | {fmt_icer(icer(50, 60, dalys, cost, share))} |")
        w("")
        w(f"*{EFFICACY_NOTE}*")
    else:
        w(f"Model quantities, {CASCADE_YEAR} cascade, {PRICE_YEAR} US$")
        w(f"  DALYs averted per acquisition prevented : {d_m:6.2f} "
          f"(90% CrI {d_lo:.2f}-{d_hi:.2f})")
        w(f"  lifetime treatment cost avoided         : ${c_m:8,.0f} "
          f"(${c_lo:,.0f}-${c_hi:,.0f})")
        w("")
        w(f"{'NNT':>5} {'inc/100py':>10}  {'scenario':<30}"
          + "".join(f"{'$' + str(p) + '/yr':>15}" for p in PRICES))
        w("-" * (5 + 11 + 32 + 15 * len(PRICES)))
        for nnt, lab, inc in rows:
            line = f"{nnt:5.0f} {inc:10.2f}  {lab:<30}"
            for p in PRICES:
                line += f"{fmt_icer(icer(nnt, p, dalys, cost)):>15}"
            w(line)
        w("")
        w("Break-even: PrEP pays for itself when NNT x annual cost < "
          f"${c_m:,.0f}")
        for p in PRICES:
            w(f"   ${p:>3}/yr -> NNT {c_m / p:5.0f} py "
              f"(incidence {100 * p / c_m:.2f}/100py)")
        w("")
        w("Denominator sensitivity, at NNT 50 and $60/yr:")
        net = 50 * 60 - cost
        w(f"   {d_m:5.2f} DALYs  {'this model, ' + CASCADE_YEAR + ' cascade':<45} "
          f"-> {fmt_icer(net / dalys):>12}/DALY")
        for v, lab in LITERATURE_DALYS:
            w(f"   {v:5.2f} DALYs  {lab:<45} -> {fmt_icer(net / v):>12}/DALY")
        w(f"   {nocare.mean():5.2f} DALYs  {'this model, no-care counterfactual':<45} "
          f"-> {fmt_icer(net / nocare):>12}/DALY")
        w("")
        w("Cost-offset sensitivity, at NNT 50 and $60/yr:")
        for share in (0.0, 0.25, 0.5, 0.75, 1.0):
            w(f"   {share:4.0%} of treatment cost credited -> "
              f"{fmt_icer(icer(50, 60, dalys, cost, share))}/DALY")

    print("\n".join(out))


if __name__ == "__main__":
    main(as_markdown="--md" in sys.argv)

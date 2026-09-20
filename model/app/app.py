"""Interactive companion to the ESA HIV cascade model.

Takes the calibrated posterior and lets the user reprice it: a cost schedule,
disability weights, the discount rate, and the two inputs a prophylaxis
decision needs -- number needed to treat and annual cost. Cascade hazards can
also be moved, which re-runs the Markov engine rather than repricing.

    shiny run --reload app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

import engine as E
import figures as F

try:
    import live
    LIVE_OK = True
except Exception as _exc:                                # pragma: no cover
    LIVE_OK, LIVE_ERROR = False, str(_exc)

CASCADE_YEARS = {"2019": "2019 cascade", "2024": "2024 cascade"}
PRICE_YEARS = [str(y) for y in range(2019, 2025)]

COST_ROWS = [
    ("First-Line ART Drugs", "art_1l", None,
     "TDF/3TC + DTG, $3.60 per month"),
    ("Second-Line ART Drugs", "art_2l", None,
     "AZT/3TC + LPV/r, $18.57 per month"),
    ("Routine Care, CD4 ≥500", "routine", 0, "$2.86 per month"),
    ("Routine Care, CD4 350-499", "routine", 1, "$3.96 per month"),
    ("Routine Care, CD4 250-349", "routine", 2, "$4.90 per month"),
    ("Routine Care, CD4 200-249", "routine", 3, "$4.90 per month"),
    ("Routine Care, CD4 100-199", "routine", 4, "$10.77 per month"),
    ("Routine Care, CD4 50-99", "routine", 5, "$18.51 per month"),
    ("Routine Care, CD4 <50", "routine", 6, "$26.24 per month"),
    ("Terminal Care At Death", "death", None,
     "One-off, any cause of death"),
]

STATUS_CHOICES = {
    "Undiagnosed": "Undiagnosed",
    "Diagnosed_PreART": "Diagnosed, Not Yet Treated",
    "ART1_Ramp": "First Line, Not Yet Suppressed",
    "ART1_Suppressed": "First Line, Suppressed",
    "ART1_Failing": "First Line, Failing",
    "ART2_Suppressed": "Second Line, Suppressed",
    "LTFU_Recent": "Recently Disengaged",
    "LTFU_LongTerm": "Long-Term Disengaged",
}

HAZARD_LABEL = {
    "δ_bg": "Background Testing Rate",
    "δ_symp": "Symptomatic Diagnosis Rate",
    "γ_diag": "Diagnosis CD4 Gradient",
    "λ_init": "ART Initiation Rate",
    "μ_vf1": "First-Line Failure Rate",
    "μ_ltfu": "Disengagement Rate",
    "μ_re-engage": "Re-Engagement Rate",
    "φ_ltfu": "Disengaged Mortality Multiplier",
    "μ_switch": "Switch To Second Line Rate",
}
HAZARD_KEYS = list(HAZARD_LABEL)

LITERATURE_DALYS = [(10.0, "Common Round Figure"),
                    (20.0, "Representative Published Value")]


def default_cost_frame() -> pd.DataFrame:
    values = []
    for _label, field, idx, _note in COST_ROWS:
        if field == "routine":
            values.append(E.DEFAULT_ROUTINE[idx])
        elif field == "art_1l":
            values.append(E.DEFAULT_COST_1L)
        elif field == "art_2l":
            values.append(E.DEFAULT_COST_2L)
        else:
            values.append(E.DEFAULT_COST_DEATH)
    return pd.DataFrame({
        "Input": [r[0] for r in COST_ROWS],
        "Annual cost, 2023 US$": [round(v, 2) for v in values],
        "Source note": [r[3] for r in COST_ROWS],
    })


def default_weight_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "CD4 band": E.BAND_NAMES,
        "On ART, suppressed": [round(v, 4) for v in E.reconstructed_dw(True)],
        "Not suppressed": [round(v, 4) for v in E.reconstructed_dw(False)],
    })


def gbd_weight_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "CD4 band": E.BAND_NAMES,
        "On ART, suppressed": E.GBD_DW_ONART,
        "Not suppressed": E.GBD_DW_OFFART,
    })


def _num(x, fallback):
    try:
        v = float(x)
        return v if np.isfinite(v) else fallback
    except (TypeError, ValueError):
        return fallback


def fmt_money(v, dp=0):
    return f"-${abs(v):,.{dp}f}" if v < 0 else f"${v:,.{dp}f}"


def ci(values, unit="", dp=2):
    m, lo, hi = E.summarise(values)
    f = (lambda v: fmt_money(v, dp)) if unit == "$" else (lambda v: f"{v:,.{dp}f}")
    return f"{f(m)}  (90% CrI {f(lo)} to {f(hi)})"


# UI

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.accordion(
            ui.accordion_panel(
                "Analysis basis",
                ui.input_select("cascade_year", "Care cascade",
                                CASCADE_YEARS, selected="2019"),
                ui.input_slider("discount", "Discount rate, %", 0, 8, 3, step=0.5,
                                post="%"),
                ui.input_select("price_year", "Report costs in", PRICE_YEARS,
                                selected="2024"),
                ui.help_text("Unit costs are entered in 2023 US dollars, the "
                             "price year of the source, and restated here by "
                             "US CPI-U."),
            ),
            ui.accordion_panel(
                "Cost schedule",
                ui.output_data_frame("cost_grid"),
                ui.input_action_button("reset_costs", "Reset to source values",
                                       class_="btn-sm btn-outline-secondary mt-2"),
                ui.hr(),
                ui.input_checkbox_group(
                    "routine_statuses", "Routine care is charged to",
                    STATUS_CHOICES, selected=list(STATUS_CHOICES)),
                ui.help_text("The base case charges every living state, "
                             "including undiagnosed, because the resource use "
                             "is driven by immunosuppression rather than by "
                             "being in care."),
            ),
            ui.accordion_panel(
                "Disability weights",
                ui.output_data_frame("weight_grid"),
                ui.div(
                    ui.input_action_button("reset_weights", "Reconstructed",
                                           class_="btn-sm btn-outline-secondary mt-2"),
                    ui.input_action_button("gbd_weights", "GBD-faithful",
                                           class_="btn-sm btn-outline-secondary mt-2 ms-1"),
                ),
                ui.help_text("The reconstructed set is a continuous function of "
                             "CD4; the GBD-faithful set uses published weights "
                             "for the nearest GBD state."),
            ),
            ui.accordion_panel(
                "Prevention",
                ui.input_slider("nnt", "Number needed to treat, person-years",
                                10, 200, 50, step=1),
                ui.input_slider("prep_price", "All-in annual cost per person-year",
                                10, 200, 60, step=5, pre="$"),
                ui.input_slider("offset_share",
                                "Treatment cost avoided that is credited, %",
                                0, 100, 100, step=5, post="%"),
                ui.input_numeric("threshold",
                                 "Cost-effectiveness threshold, $ per DALY",
                                 500, min=0, step=50),
            ),
            ui.accordion_panel(
                "Cascade adjustments",
                ui.help_text("These re-run the Markov engine rather than "
                             "repricing, so they give a point estimate at the "
                             "posterior mean, not a credible interval."),
                *[ui.input_slider(f"mult_{i}", HAZARD_LABEL[k], 0.25, 3.0, 1.0,
                                  step=0.05)
                  for i, k in enumerate(HAZARD_KEYS)],
                ui.input_action_button("reset_cascade", "Reset to calibrated",
                                       class_="btn-sm btn-outline-secondary mt-2"),
            ),
            id="controls", open=["Analysis basis", "Prevention"], multiple=True,
        ),
        width=380, title="Inputs",
    ),
    ui.navset_card_tab(
        ui.nav_panel(
            "Burden",
            ui.layout_columns(
                ui.value_box("DALYs per acquisition", ui.output_text("vb_dalys"),
                             ui.output_text("vb_dalys_ci")),
                ui.value_box("DALYs averted through care",
                             ui.output_text("vb_averted"),
                             ui.output_text("vb_averted_ci")),
                ui.value_box("Lifetime cost per acquisition",
                             ui.output_text("vb_cost"), ui.output_text("vb_cost_ci")),
                fill=False,
            ),
            ui.layout_columns(
                ui.card(ui.card_header("Posterior, DALYs per acquisition"),
                        ui.output_plot("plot_daly_density", height="260px")),
                ui.card(ui.card_header("Burden with and without care"),
                        ui.output_plot("plot_daly_decomp", height="260px")),
            ),
            ui.card(ui.card_header("Where the person-time is spent"),
                    ui.output_plot("plot_person_years", height="320px")),
        ),
        ui.nav_panel(
            "Cost",
            ui.card(ui.card_header("What the lifetime cost buys"),
                    ui.output_plot("plot_cost_decomp", height="300px")),
            ui.layout_columns(
                ui.card(ui.card_header("Posterior, cost per acquisition"),
                        ui.output_plot("plot_cost_density", height="260px")),
                ui.card(ui.card_header("Cost composition"),
                        ui.output_data_frame("cost_summary")),
            ),
        ),
        ui.nav_panel(
            "Treatment value",
            ui.layout_columns(
                ui.value_box("Cost per DALY averted by the cascade",
                             ui.output_text("vb_icer"), ui.output_text("vb_icer_ci")),
                ui.value_box("Probability below the threshold",
                             ui.output_text("vb_prob"),
                             ui.output_text("vb_prob_note")),
                fill=False,
            ),
            ui.card(ui.card_header("Cost-effectiveness plane, one point per draw"),
                    ui.output_plot("plot_ce_plane", height="340px")),
            ui.card(ui.card_header("What the estimate rests on"),
                    ui.output_plot("plot_tornado", height="330px")),
        ),
        ui.nav_panel(
            "Prevention",
            ui.layout_columns(
                ui.value_box("Cost per DALY averted by prophylaxis",
                             ui.output_text("vb_prev"), ui.output_text("vb_prev_ci")),
                ui.value_box("Break-even annual cost",
                             ui.output_text("vb_breakeven"),
                             ui.output_text("vb_breakeven_note")),
                fill=False,
            ),
            ui.card(ui.card_header("Where prophylaxis pays for itself"),
                    ui.output_plot("plot_frontier", height="340px")),
            ui.layout_columns(
                ui.card(ui.card_header("Under other DALY denominators"),
                        ui.output_plot("plot_denominators", height="280px")),
                ui.card(ui.card_header("Crediting the treatment cost avoided"),
                        ui.output_plot("plot_offset", height="280px")),
            ),
            ui.card(ui.card_header("Cost per DALY averted, by targeting and price"),
                    ui.output_data_frame("prevention_table")),
        ),
        ui.nav_panel(
            "Cascade",
            ui.output_ui("cascade_status"),
            ui.card(ui.card_header("Adjusted cascade against the calibrated model"),
                    ui.output_plot("plot_cascade", height="300px")),
            ui.card(ui.card_header("Hazards applied"),
                    ui.output_data_frame("cascade_table")),
        ),
        ui.nav_panel(
            "Inputs and provenance",
            ui.card(ui.card_header("Current configuration"),
                    ui.output_data_frame("config_table")),
            ui.download_button("download_config",
                               "Download configuration and results, CSV",
                               class_="btn-sm btn-primary"),
            ui.card(
                ui.card_header("Sources and caveats"),
                ui.markdown(
                    "**Costs.** Hyle EP, Maphosa T, Rangaraj A, et al. Clinical "
                    "impact and cost-effectiveness of the WHO-recommended "
                    "advanced HIV disease package of care. *Lancet Global "
                    "Health* 2025;13(8):e1436-e1447. "
                    "doi:10.1016/S2214-109X(25)00190-1, supplementary Table S6. "
                    "A Malawi cost set at the low end of the eastern and "
                    "southern African range, so the treatment cost avoided, and "
                    "therefore the case for prevention, is conservative.\n\n"
                    "**Disability weights.** Reconstructed as a continuous "
                    "function of CD4; the benchmark anchors mix GBD rounds "
                    "(2010 and 2013). The GBD-faithful preset is offered "
                    "because the choice is worth about half a DALY per "
                    "acquisition.\n\n"
                    "**Years of life lost.** Cause-deleted remaining life "
                    "expectancy from the eastern and southern African life "
                    "table, discounted in continuous time following Fox-Rushby "
                    "and Hanson (2001).\n\n"
                    "**Scope.** One index acquisition, no onward transmission, "
                    "and a regional average across populations whose cascade "
                    "access differs substantially. It should not be applied to "
                    "a particular subgroup without adjustment."
                ),
            ),
        ),
    ),
    title="HIV cascade: burden, cost and cost-effectiveness per acquisition",
    fillable=False,
)


# Server

def server(input, output, session):

    # -- editable input tables --------------------------------------------
    cost_data = reactive.value(default_cost_frame())
    weight_data = reactive.value(default_weight_frame())

    @render.data_frame
    def cost_grid():
        return render.DataGrid(cost_data(), editable=True, width="100%",
                               height="330px")

    @render.data_frame
    def weight_grid():
        return render.DataGrid(weight_data(), editable=True, width="100%",
                               height="260px")

    @reactive.effect
    @reactive.event(input.reset_costs)
    def _reset_costs():
        cost_data.set(default_cost_frame())

    @reactive.effect
    @reactive.event(input.reset_weights)
    def _reset_weights():
        weight_data.set(default_weight_frame())

    @reactive.effect
    @reactive.event(input.gbd_weights)
    def _gbd_weights():
        weight_data.set(gbd_weight_frame())

    @reactive.effect
    @reactive.event(input.reset_cascade)
    def _reset_cascade():
        for i in range(len(HAZARD_KEYS)):
            ui.update_slider(f"mult_{i}", value=1.0)

    def _edited(grid, fallback):
        """The grid's current contents, falling back to the stored frame before
        the first render or if an edit leaves a cell unparseable."""
        try:
            df = grid.data_view()
            if df is not None and len(df) == len(fallback):
                return df
        except Exception:
            pass
        return fallback

    @reactive.calc
    def costs() -> E.CostSchedule:
        df = _edited(cost_grid, cost_data())
        col = df.columns[1]
        vals = list(df[col])
        defaults = list(default_cost_frame()[col])
        vals = [_num(v, d) for v, d in zip(vals, defaults)]
        routine = [vals[i] for i, r in enumerate(COST_ROWS) if r[1] == "routine"]
        art1 = next(vals[i] for i, r in enumerate(COST_ROWS) if r[1] == "art_1l")
        art2 = next(vals[i] for i, r in enumerate(COST_ROWS) if r[1] == "art_2l")
        death = next(vals[i] for i, r in enumerate(COST_ROWS) if r[1] == "death")
        chosen = tuple(input.routine_statuses() or ())
        return E.CostSchedule(art_1l=art1, art_2l=art2, routine=tuple(routine),
                              death=death, price_year=int(input.price_year()),
                              routine_statuses=chosen)

    @reactive.calc
    def weights() -> E.Weights:
        df = _edited(weight_grid, weight_data())
        d = default_weight_frame()
        on = [_num(v, dv) for v, dv in zip(df[df.columns[1]], d[d.columns[1]])]
        off = [_num(v, dv) for v, dv in zip(df[df.columns[2]], d[d.columns[2]])]
        return E.Weights(on_art=tuple(on), off_art=tuple(off))

    @reactive.calc
    def assumptions() -> E.Assumptions:
        return E.Assumptions(cascade_year=input.cascade_year(),
                             discount_rate=float(input.discount()) / 100.0,
                             costs=costs(), weights=weights())

    @reactive.calc
    def out():
        return E.evaluate(assumptions())

    @reactive.calc
    def multipliers() -> dict:
        return {k: float(input[f"mult_{i}"]()) for i, k in enumerate(HAZARD_KEYS)}

    # -- headline numbers --------------------------------------------------
    @render.text
    def vb_dalys():
        return f"{out()['DALYs_per_acquisition'].mean():.2f}"

    @render.text
    def vb_dalys_ci():
        return ci(out()["DALYs_per_acquisition"])

    @render.text
    def vb_averted():
        return f"{out()['DALYs_avoided_through_care'].mean():.2f}"

    @render.text
    def vb_averted_ci():
        return ci(out()["DALYs_avoided_through_care"])

    @render.text
    def vb_cost():
        return fmt_money(out()["cost_per_acquisition"].mean())

    @render.text
    def vb_cost_ci():
        return f"{input.price_year()} US$ · " + ci(out()["cost_per_acquisition"], "$", 0)

    @render.text
    def vb_icer():
        return fmt_money(E.treatment_icer(out()).mean())

    @render.text
    def vb_icer_ci():
        return "per DALY averted · " + ci(E.treatment_icer(out()), "$", 0)

    @render.text
    def vb_prob():
        thr = _num(input.threshold(), 500.0)
        return f"{(E.treatment_icer(out()) < thr).mean() * 100:.0f}%"

    @render.text
    def vb_prob_note():
        return f"of draws below {fmt_money(_num(input.threshold(), 500.0))} per DALY"

    @reactive.calc
    def prevention_values():
        o = out()
        return E.prevention_icer(float(input.nnt()), float(input.prep_price()),
                                 o["DALYs_per_acquisition"],
                                 o["cost_per_acquisition"],
                                 float(input.offset_share()) / 100.0)

    @render.text
    def vb_prev():
        v = prevention_values().mean()
        return "Cost-saving" if v < 0 else fmt_money(v)

    @render.text
    def vb_prev_ci():
        return (f"at NNT {input.nnt()} and {fmt_money(float(input.prep_price()))}/yr · "
                + ci(prevention_values(), "$", 0))

    @render.text
    def vb_breakeven():
        be = E.break_even_price(float(input.nnt()), out()["cost_per_acquisition"],
                                float(input.offset_share()) / 100.0)
        return fmt_money(be, 2)

    @render.text
    def vb_breakeven_note():
        offset = out()["cost_per_acquisition"].mean() * float(input.offset_share()) / 100.0
        return (f"per person-year at NNT {input.nnt()}; the frontier is "
                f"NNT × price = {fmt_money(offset)}")

    # -- plots -------------------------------------------------------------
    @render.plot
    def plot_daly_density():
        return F.posterior_density(out()["DALYs_per_acquisition"],
                                   "DALYs per acquisition, discounted")

    @render.plot
    def plot_daly_decomp():
        return F.daly_decomposition(out())

    @render.plot
    def plot_person_years():
        return F.person_years_panel(E.person_years_by_status(assumptions()))

    @render.plot
    def plot_cost_decomp():
        return F.cost_decomposition(E.cost_components(assumptions()),
                                    input.price_year())

    @render.plot
    def plot_cost_density():
        return F.posterior_density(out()["cost_per_acquisition"],
                                   f"Lifetime cost per acquisition, "
                                   f"{input.price_year()} US$", unit="$",
                                   colour=F.AQUA)

    @render.plot
    def plot_ce_plane():
        return F.cost_effectiveness_plane(out(), _num(input.threshold(), None))

    @render.plot
    def plot_tornado():
        base = E.treatment_icer(out()).mean()
        rows = []
        a = assumptions()
        for label, alt in _sensitivity_variants(a):
            vals = E.treatment_icer(E.evaluate(alt))
            rows.append((label, base, vals.mean()))
        rows = [(lab, min(b, v), max(b, v)) for lab, b, v in rows]
        return F.tornado(rows, base, "Cost per DALY averted by the cascade")

    @render.plot
    def plot_frontier():
        return F.prevention_frontier(
            [20, 50, 80], [25, 40, 60, 100],
            out()["cost_per_acquisition"].mean() * float(input.offset_share()) / 100.0,
            current=(float(input.nnt()), float(input.prep_price())))

    @render.plot
    def plot_denominators():
        o = out()
        return F.prevention_denominators(
            float(input.nnt()), float(input.prep_price()),
            o["DALYs_per_acquisition"], o["cost_per_acquisition"],
            LITERATURE_DALYS, float(input.offset_share()) / 100.0)

    @render.plot
    def plot_offset():
        o = out()
        return F.offset_sensitivity(float(input.nnt()), float(input.prep_price()),
                                    o["DALYs_per_acquisition"],
                                    o["cost_per_acquisition"])

    @render.plot
    def plot_cascade():
        if not LIVE_OK:
            return F.tornado([], 0, "Live re-run unavailable")
        return F.cascade_comparison(out(),
                                    live.evaluate_point(assumptions(), multipliers()))

    # -- tables ------------------------------------------------------------
    @render.data_frame
    def cost_summary():
        comp = E.cost_components(assumptions())
        total = sum(v.mean() for v in comp.values())
        return render.DataGrid(pd.DataFrame({
            "Component": list(comp),
            "Mean": [fmt_money(v.mean()) for v in comp.values()],
            "Share": [f"{v.mean() / total * 100:.1f}%" for v in comp.values()],
        }), width="100%")

    @render.data_frame
    def prevention_table():
        o = out()
        share = float(input.offset_share()) / 100.0
        prices = [25, 40, 60, 100, float(input.prep_price())]
        prices = sorted(set(round(p) for p in prices))
        rows = []
        for nnt, label in [(20, "Intensive targeting"), (50, "Targeted"),
                           (80, "Moderate targeting"),
                           (float(input.nnt()), "Current selection")]:
            row = {"NNT": f"{nnt:.0f}", "Implied incidence per 100 py": f"{100 / nnt:.2f}",
                   "Scenario": label}
            for p in prices:
                v = E.prevention_icer(nnt, p, o["DALYs_per_acquisition"],
                                      o["cost_per_acquisition"], share).mean()
                row[f"${p:,.0f}/yr"] = "Cost-saving" if v < 0 else fmt_money(v)
            rows.append(row)
        return render.DataGrid(pd.DataFrame(rows), width="100%")

    @render.data_frame
    def cascade_table():
        if not LIVE_OK:
            return render.DataGrid(pd.DataFrame({"Status": ["Model not importable"]}))
        vals, _kp, _km = live.posterior_mean_parameters(input.cascade_year())
        m = multipliers()
        return render.DataGrid(pd.DataFrame({
            "Hazard": [HAZARD_LABEL[k] for k in HAZARD_KEYS],
            "Calibrated": [f"{vals.get(k, float('nan')):.4f}" for k in HAZARD_KEYS],
            "Multiplier": [f"×{m[k]:.2f}" for k in HAZARD_KEYS],
            "Applied": [f"{vals.get(k, float('nan')) * m[k]:.4f}" for k in HAZARD_KEYS],
        }), width="100%")

    @render.ui
    def cascade_status():
        if not LIVE_OK:
            return ui.markdown(
                "The Markov engine could not be imported, so cascade "
                f"adjustments are unavailable in this deployment: `{LIVE_ERROR}`. "
                "Every other tab reads the exported posterior and is unaffected.")
        return ui.markdown(
            "Moving a cascade hazard re-runs the engine at the **posterior "
            "mean** of the calibrated parameters, so the figures here are point "
            "estimates without a credible interval. They will not equal the "
            "posterior means on the other tabs even at a multiplier of one, "
            "because the mean of a non-linear function is not the function of "
            "the mean.")

    @render.data_frame
    def config_table():
        a = assumptions()
        o = out()
        rows = [("Care cascade", CASCADE_YEARS[a.cascade_year]),
                ("Discount rate", f"{a.discount_rate * 100:.1f}%"),
                ("Price year", str(a.costs.price_year)),
                ("Routine care charged to",
                 ", ".join(STATUS_CHOICES[s] for s in a.costs.routine_statuses)
                 or "no states"),
                ("First-line ART drugs, 2023 US$", f"{a.costs.art_1l:,.2f}"),
                ("Second-line ART drugs, 2023 US$", f"{a.costs.art_2l:,.2f}"),
                ("Terminal care, 2023 US$", f"{a.costs.death:,.2f}")]
        rows += [(f"Routine care {b}, 2023 US$", f"{v:,.2f}")
                 for b, v in zip(E.BAND_NAMES, a.costs.routine)]
        rows += [(f"Disability weight {b}, on ART / not suppressed",
                  f"{on:.3f} / {off:.3f}")
                 for b, on, off in zip(E.BAND_NAMES, a.weights.on_art,
                                       a.weights.off_art)]
        rows += [("Number needed to treat", f"{input.nnt()}"),
                 ("Annual cost per person-year", fmt_money(float(input.prep_price()))),
                 ("Treatment cost credited", f"{input.offset_share()}%"),
                 ("DALYs per acquisition", ci(o["DALYs_per_acquisition"])),
                 ("DALYs averted through care", ci(o["DALYs_avoided_through_care"])),
                 ("Lifetime cost per acquisition", ci(o["cost_per_acquisition"], "$", 0)),
                 ("Cost per DALY averted by the cascade",
                  ci(E.treatment_icer(o), "$", 0)),
                 ("Cost per DALY averted by prophylaxis",
                  ci(prevention_values(), "$", 0))]
        return render.DataGrid(pd.DataFrame(rows, columns=["Input or result", "Value"]),
                               width="100%", height="620px")

    @render.download(filename=lambda: f"cascade_app_{input.cascade_year()}.csv")
    def download_config():
        a = assumptions()
        o = out()
        df = pd.DataFrame({k: v for k, v in o.items()})
        df.insert(0, "draw", np.arange(len(df)))
        df["treatment_icer"] = E.treatment_icer(o)
        df["prevention_icer"] = prevention_values()
        header = (f"# ESA HIV cascade model, {CASCADE_YEARS[a.cascade_year]}\n"
                  f"# discount rate {a.discount_rate:.3f}, "
                  f"price year {a.costs.price_year}\n"
                  f"# NNT {input.nnt()}, prophylaxis "
                  f"${float(input.prep_price()):,.0f}/py, "
                  f"{input.offset_share()}% of treatment cost credited\n")
        yield header
        yield df.to_csv(index=False)


def _sensitivity_variants(a: E.Assumptions):
    """One-way variants for the tornado: each analytic choice moved on its own."""
    in_care = tuple(s for s in a.costs.routine_statuses
                    if s not in ("Undiagnosed", "LTFU_Recent", "LTFU_LongTerm"))
    return [
        ("Discount Rate 0%", E.Assumptions(a.cascade_year, 0.0, a.costs, a.weights)),
        ("Discount Rate 5%", E.Assumptions(a.cascade_year, 0.05, a.costs, a.weights)),
        ("GBD-Faithful Weights",
         E.Assumptions(a.cascade_year, a.discount_rate, a.costs,
                       E.Weights(tuple(E.GBD_DW_ONART), tuple(E.GBD_DW_OFFART)))),
        ("Routine Care, In-Care States Only",
         E.with_cost(a, routine_statuses=in_care)),
        ("ART Drugs At Half Price",
         E.with_cost(a, art_1l=a.costs.art_1l / 2, art_2l=a.costs.art_2l / 2)),
        ("Routine Care Doubled",
         E.with_cost(a, routine=tuple(v * 2 for v in a.costs.routine))),
    ]


app = App(app_ui, server)

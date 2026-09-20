"""
Shiny app: run + explore the two-system calibration.

By design there are no sliders or parameter inputs -- this is a "run and
view results" dashboard, not a what-if explorer. Two buttons (one per
system), each running the exact same calibration code as
`python3 calibrate_system1.py` / `calibrate_system2.py`, so there's no way
for a user to get an inconsistent or hand-edited result through the UI.

Run with:
    shiny run --reload app.py
then open the URL it prints (usually http://127.0.0.1:8000).

Requires `pip install -r requirements.txt` first (PyMC + Shiny in
particular aren't installable in the sandbox this app was built in -- see
README.md for what was and wasn't verified there).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from shiny import App, reactive, render, ui

from calibration_common import load_result, save_result
from calibrate_system1 import run_system1_calibration, RESULT_NAME as S1_NAME
from calibrate_system2 import run_system2_calibration, result_name_for_year
from params import load_parameters
from run_prior_analysis import (
    run_prior_deterministic, run_prior_probabilistic,
    save_prior_result, load_prior_result,
)
from run_posterior_analysis import (
    run_posterior_deterministic, run_posterior_probabilistic,
    save_posterior_result, load_posterior_result, posterior_result_path,
)
from scenario_analysis import (
    run_point_tornado, run_probabilistic_tornado, run_uncertainty_tornado,
    run_prior_point_tornado,
    save_point_result, load_point_result, save_prob_result, load_prob_result,
    save_uncertainty_result, load_uncertainty_result,
    save_prior_point_result, load_prior_point_result, TARGET_KEYS,
)

# Loaded once at import time (the spreadsheet doesn't change mid-session) and
# reused both for labeling and as the app's working Parameters object -- see
# params_obj in server(). Every table/plot with a bare symbol (lambda_base,
# delta_bg, p1, ...) runs it through _label() to show "Descriptive Name
# (symbol)" instead, per Peter's request.
_PARAMS = load_parameters()
SYMBOL_LABEL = {sym: f"{name} ({sym})" for sym, name in _PARAMS.symbol_names().items()}


def _label(symbol: str) -> str:
    return SYMBOL_LABEL.get(symbol, symbol)


def _label_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy of df with `column`'s values run through _label()."""
    if df.empty or column not in df.columns:
        return df
    df = df.copy()
    df[column] = df[column].map(_label)
    return df


def _label_columns_header(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with any column NAME that's itself a known symbol
    (e.g. the target-translation tables' p1/p2/p3/... columns) relabeled."""
    if df.empty:
        return df
    return df.rename(columns=SYMBOL_LABEL)


ABOUT_TEXT = """
## HIV Natural History + Care Cascade Model

A 44-state (7 CD4 bands x 6 cascade statuses, + 2 absorbing death states)
continuous-time competing-risks model, calibrated with a sequential
("cut"-style) Bayesian design: System 1 (untreated natural history) is
fit and locked first; System 2 (care cascade) is fit conditional on System
1's posterior, with System 1's uncertainty propagated in by resampling its
posterior rather than plugging in a single point estimate.

**This dashboard only runs and displays results** -- no parameters can be
edited here. To change priors, targets, or fixed inputs, edit
`data/parameters.xlsx` (or the structural assumptions in `simulate_cascade.py`
/ `hazards.py`) and re-run.

**Known data gaps**, flagged with placeholders pending real sourced values:
excess mortality on suppressive ART (SMR), the background mortality curve,
5 of 7 untreated-mortality-by-CD4-band values (interpolated from 2 anchor
points), and the acquisition CD4 distribution (using a single starting band
instead of a spread). See README.md for the full list and rationale for
every other v1 structural assumption (LTFU routing, no ART2_Failing state,
etc).

**Backend:** uses PyMC (DEMetropolisZ sampler) if installed, otherwise
automatically falls back to a small dependency-free numpy Metropolis
sampler. Both target the identical posterior -- see `likelihoods.py`.

**Health-econ metrics** (Prior Run / Posterior Run Summary tabs): DALYs =
YLL + YLD, discounted at 3%/yr (discrete compounding), YLL referenced to a
72-year standard life expectancy, disability weights and unit costs from
`health_econ_params.py`. "DALYs avoided through care" compares the full
cascade against a no-diagnosis/no-ART natural-history-only counterfactual.
Cost unit prices and the discount rate/reference life expectancy are
sourced values Peter supplied; the continuous CD4-based disability-weight
*formula* was reconstructed (not copied) from a Min/Max/Slope spec that
didn't include the underlying units -- see `health_econ_params.py` for the
exact reconstruction and how it was checked against the source's own GBD
benchmark anchors.
"""


app_ui = ui.page_navbar(
    ui.nav_panel(
        "Prior Run",
        ui.card(
            ui.card_header("Free parameters (priors)"),
            ui.output_table("prior_param_table"),
        ),
        ui.card(
            ui.card_header("Deterministic run (central estimates)"),
            ui.output_table("prior_deterministic_table"),
        ),
        ui.card(
            ui.card_header("Deterministic tornado (prior central estimates, ±20%, all 9 free parameters)"),
            ui.output_plot("prior_tornado_plot"),
            ui.output_table("prior_tornado_table"),
        ),
        ui.card(
            ui.card_header("Deterministic tornado: cascade target translation"),
            ui.output_table("prior_tornado_target_table"),
        ),
        ui.card(
            ui.card_header("Probabilistic run (Monte Carlo over priors)"),
            ui.input_action_button("run_prior_prob", "Run Probabilistic Analysis (1000 draws)",
                                    class_="btn-primary"),
            ui.output_text("prior_prob_status"),
        ),
        ui.card(ui.card_header("Probabilistic summary"), ui.output_table("prior_prob_table")),
        ui.card(ui.card_header("DALYs (mean, 90% interval)"), ui.output_plot("prior_daly_plot")),
        ui.card(ui.card_header("Cost (mean, 90% interval)"), ui.output_plot("prior_cost_plot")),
    ),
    ui.nav_panel(
        "System 1: Natural History (sourced, not fitted)",
        ui.card(
            ui.card_header("Run"),
            ui.input_action_button("run_sys1", "Draw System 1 Natural History", class_="btn-primary"),
            ui.output_text("sys1_status"),
        ),
        ui.card(ui.card_header("Posterior summary"), ui.output_table("sys1_summary")),
        ui.card(ui.card_header("Model-implied vs. targets"), ui.output_table("sys1_targets")),
        ui.card(ui.card_header("Target comparison"), ui.output_plot("sys1_compare_plot")),
        ui.card(ui.card_header("Trace plots"), ui.output_plot("sys1_trace")),
    ),
    ui.nav_panel(
        "System 2: Care Cascade",
        ui.card(
            ui.card_header("Cascade base year"),
            ui.input_radio_buttons(
                "cascade_year", None,
                choices={"2019": "2019 (base case)",
                          "2024": "2024 (UNAIDS, sensitivity)",
                          "WC2023": "WC 2023 (population-linked, sensitivity)"},
                selected="2019", inline=True,
            ),
            ui.markdown(
                "**2019** (UNAIDS Data 2020, p.40) is the base case -- it's "
                "closer in time to the de Waal et al. CD4-at-ART-initiation "
                "data (2016-2019) than the 2024 cascade figures are. **2024** "
                "(UNAIDS 2025 Global AIDS Update) and **WC 2023** (Western Cape "
                "cyclical cascade, Euvrard et al. 2024) are sensitivity "
                "comparisons. 2024 and WC 2023 share the same p1 (93%) and "
                "differ only in p2 and p3, so running them as a pair isolates "
                "the difference between programme-derived and "
                "individually-linked measurement of treatment coverage. "
                "Switching this reloads whichever calibration has "
                "already been run for that year; every other tab (Posterior "
                "Run Summary, Scenario Analysis) follows whichever year is "
                "selected here."
            ),
        ),
        ui.card(
            ui.card_header("Run"),
            ui.input_action_button("run_sys2", "Run System 2 Calibration", class_="btn-primary"),
            ui.output_text("sys2_status"),
        ),
        ui.card(ui.card_header("Posterior summary"), ui.output_table("sys2_summary")),
        ui.card(ui.card_header("Model-implied vs. targets"), ui.output_table("sys2_targets")),
        ui.card(ui.card_header("Target comparison"), ui.output_plot("sys2_compare_plot")),
        ui.card(ui.card_header("Trace plots"), ui.output_plot("sys2_trace")),
    ),
    ui.nav_panel(
        "Posterior Run Summary",
        ui.output_text("post_active_year_note"),
        ui.card(
            ui.card_header("Run"),
            ui.input_action_button("run_post_prob", "Run Posterior Health-Econ Summary (1000 draws)",
                                    class_="btn-primary"),
            ui.output_text("post_status"),
        ),
        ui.card(
            ui.card_header("Deterministic run (posterior means)"),
            ui.output_table("post_deterministic_table"),
        ),
        ui.card(ui.card_header("Probabilistic summary"), ui.output_table("post_prob_table")),
        ui.card(ui.card_header("DALYs (mean, 90% interval)"), ui.output_plot("post_daly_plot")),
        ui.card(ui.card_header("Cost (mean, 90% interval)"), ui.output_plot("post_cost_plot")),
        ui.card(
            ui.card_header("Prior vs. posterior: uncertainty narrowing"),
            ui.output_plot("prior_vs_post_plot"),
        ),
    ),
    ui.nav_panel(
        "Scenario Analysis",
        ui.output_text("scenario_active_year_note"),
        ui.markdown(
            "Two different questions, both below. **Policy tornadoes** (point + "
            "probabilistic) apply the same fixed ±20% shift to every parameter -- "
            "'what's the DALY payoff of a program moving this cascade behavior by "
            "a fixed relative amount?'. The **uncertainty tornado** instead varies "
            "each parameter at its own posterior range -- 'how robust is our DALY "
            "estimate to what we don't know about this parameter?'. These can give "
            "very different answers for the same parameter. All perturb the "
            "underlying hazard *components*, not the p1/p2/p3 targets directly -- "
            "the target-translation tables show how that converts (nonlinearly, "
            "due to competing risks) into actual movement in %diagnosed/%on-ART/"
            "%suppressed/retention."
        ),
        ui.card(
            ui.card_header("Point tornado (posterior means, ±20% per parameter)"),
            ui.output_plot("point_tornado_plot"),
            ui.output_table("point_tornado_table"),
        ),
        ui.card(
            ui.card_header("Point tornado: cascade target translation"),
            ui.output_table("point_target_table"),
        ),
        ui.card(
            ui.card_header("Uncertainty-bound tornado (each parameter at its own posterior range)"),
            ui.output_plot("uncertainty_tornado_plot"),
            ui.output_table("uncertainty_tornado_table"),
        ),
        ui.card(
            ui.card_header("Uncertainty tornado: cascade target translation"),
            ui.output_table("uncertainty_target_table"),
        ),
        ui.card(
            ui.card_header("Probabilistic tornado (paired posterior draws, ±20%)"),
            ui.input_action_button("run_prob_tornado", "Run Probabilistic Tornado (250 draws)",
                                    class_="btn-primary"),
            ui.output_text("prob_tornado_status"),
            ui.output_plot("prob_tornado_plot"),
            ui.output_table("prob_tornado_table"),
        ),
    ),
    ui.nav_panel("About", ui.markdown(ABOUT_TEXT)),
    title="HIV Natural History + Care Cascade Model",
    id="nav",
)


def _trace_plot(result):
    if result is None:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Run the calibration to see trace plots.", ha="center", va="center")
        ax.axis("off")
        return fig
    chains = result["chains"]
    names = result["param_names"]
    n_params = len(names)
    fig, axes = plt.subplots(n_params, 1, figsize=(7, 1.8 * n_params))
    if n_params == 1:
        axes = [axes]
    for i, name in enumerate(names):
        for c in range(chains.shape[0]):
            axes[i].plot(chains[c, :, i], alpha=0.6, linewidth=0.6)
        axes[i].set_ylabel(_label(name), fontsize=8)
    axes[-1].set_xlabel("draw")
    fig.suptitle(f"Trace plots ({chains.shape[0]} chains)")
    fig.tight_layout()
    return fig


def _target_unit_group(symbol: str) -> str:
    """Classify a target symbol into a unit family so the comparison plot can
    give each family its own x-axis. Targets span very different scales
    (proportions ~0-1, T_ART in months, the CD4-at-ART-init targets in
    cells/uL up to ~1400) -- on one shared linear axis the proportion targets
    collapse to an invisible sliver pinned at x=0, which is the "orange
    diamonds on zero, no visible blue bar" symptom."""
    if symbol.startswith("CD4_") and symbol != "CD4_lt200_ART":
        return "CD4 count (cells/µL)"
    if symbol.startswith("T_"):
        return "time (years or months)"
    return "proportion"


def _compare_plot(result, targets_key="targets", implied_key="model_implied_at_mean"):
    if result is None:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Run the calibration to see the comparison.", ha="center", va="center")
        ax.axis("off")
        return fig
    targets = result[targets_key]
    implied = result[implied_key]
    names = list(targets.keys())

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(_target_unit_group(n), []).append(n)
    group_order = [g for g in ("proportion", "time (years or months)", "CD4 count (cells/µL)")
                   if g in groups]

    fig, axes = plt.subplots(
        len(group_order), 1, squeeze=False,
        figsize=(7, sum(0.55 * len(groups[g]) + 1.0 for g in group_order)),
    )
    for row, gname in enumerate(group_order):
        ax = axes[row][0]
        gnames = groups[gname]
        target_vals = [targets[k][0] for k in gnames]
        target_sds = [targets[k][1] for k in gnames]
        implied_vals = [implied[k] if implied[k] is not None else float("nan") for k in gnames]
        y = range(len(gnames))
        ax.errorbar(target_vals, y, xerr=target_sds, fmt="o", color="tab:blue",
                    label="target (± SD)", capsize=3)
        ax.scatter(implied_vals, y, marker="D", color="tab:orange", zorder=5,
                   label="model-implied (posterior mean)")
        ax.set_yticks(list(y))
        ax.set_yticklabels([_label(n) for n in gnames], fontsize=8)
        ax.set_xlabel(gname, fontsize=8)
        if row == 0:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle("Model-implied vs. target", y=0.995)
    fig.tight_layout()
    return fig


DALY_METRICS = ["DALYs_per_acquisition", "DALYs_avoided_through_care", "DALYs_natural_history",
                "YLL_natural_history", "YLD_natural_history", "YLL_cascade", "YLD_cascade"]
COST_METRICS = ["cost_per_acquisition", "cost_recurring", "cost_startup"]


def _uncertainty_bar_plot(summary, metric_names, title, empty_msg="Run the analysis to see this plot."):
    if summary is None:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, empty_msg, ha="center", va="center")
        ax.axis("off")
        return fig
    rows = [r for r in summary if r["metric"] in metric_names]
    names = [r["metric"] for r in rows]
    means = [r["mean"] for r in rows]
    lo = [max(0.0, r["mean"] - r["q05"]) for r in rows]
    hi = [max(0.0, r["q95"] - r["mean"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(rows) + 1))
    y = range(len(rows))
    ax.errorbar(means, y, xerr=[lo, hi], fmt="o", color="tab:green", capsize=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _prior_vs_post_plot(prior_summary, post_summary, metric="DALYs_avoided_through_care"):
    if prior_summary is None or post_summary is None:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Run both the prior and posterior probabilistic "
                           "analyses to see this comparison.", ha="center", va="center")
        ax.axis("off")
        return fig
    prior_row = next(r for r in prior_summary if r["metric"] == metric)
    post_row = next(r for r in post_summary if r["metric"] == metric)

    fig, ax = plt.subplots(figsize=(7, 2.5))
    for i, (label, row) in enumerate([("Prior", prior_row), ("Posterior", post_row)]):
        lo = max(0.0, row["mean"] - row["q05"])
        hi = max(0.0, row["q95"] - row["mean"])
        ax.errorbar([row["mean"]], [i], xerr=[[lo], [hi]], fmt="o",
                    color="tab:blue" if label == "Prior" else "tab:orange", capsize=4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Prior", "Posterior"])
    ax.set_title(f"{metric}: prior vs. posterior 90% interval")
    fig.tight_layout()
    return fig


def _point_tornado_plot(point_result, metric="DALYs_per_acquisition"):
    if point_result is None:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Run System 1 and System 2 first.", ha="center", va="center")
        ax.axis("off")
        return fig
    rows = sorted(point_result["rows"], key=lambda r: -abs(r[f"{metric}_elasticity"]))
    baseline = point_result["baseline"][metric]

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1))
    for i, row in enumerate(rows):
        lo, hi = row[f"{metric}_down"], row[f"{metric}_up"]
        left, width = min(lo, hi), abs(hi - lo)
        ax.barh(i, width, left=left, color="tab:purple", alpha=0.6)
    ax.axvline(baseline, color="black", linewidth=1, linestyle="--", label="baseline")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_label(r["param"]) for r in rows], fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title(f"±20% tornado: {metric}", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _uncertainty_tornado_plot(unc_result, metric="DALYs_per_acquisition"):
    if unc_result is None:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Run System 1 and System 2 first.", ha="center", va="center")
        ax.axis("off")
        return fig
    rows = sorted(unc_result["rows"], key=lambda r: -r[f"{metric}_range"])
    baseline = unc_result["baseline"][metric]

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1))
    for i, row in enumerate(rows):
        lo, hi = row[f"{metric}_at_q03"], row[f"{metric}_at_q97"]
        left, width = min(lo, hi), abs(hi - lo)
        ax.barh(i, width, left=left, color="tab:red", alpha=0.6)
    ax.axvline(baseline, color="black", linewidth=1, linestyle="--", label="baseline")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_label(r["param"]) for r in rows], fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title(f"Uncertainty-bound tornado: {metric}", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _target_translation_table(result, bound_suffixes):
    """Long-format table: one row per (parameter, bound) showing how much
    the underlying hazard perturbation moved each cascade TARGET (p1/p2/p3/
    T_ART/retention/etc), not just the downstream DALY number."""
    if result is None:
        return pd.DataFrame()
    baseline_row = {"param": "(baseline)", "bound": "baseline"}
    baseline_row.update({k: result["baseline_targets"][k] for k in TARGET_KEYS})
    rows = [baseline_row]
    for row in result["rows"]:
        for suffix, label in bound_suffixes:
            r = {"param": _label(row["param"]), "bound": label}
            r.update({k: row[f"target_{k}_{suffix}"] for k in TARGET_KEYS})
            rows.append(r)
    df = pd.DataFrame(rows).round(4)
    return _label_columns_header(df)


def _prob_tornado_plot(prob_result, metric="DALYs_per_acquisition"):
    """Same visual grammar as the point tornado (one floating bar per
    parameter, spanning the outcome value at -20% to +20%, sorted so the
    plot tapers into the classic tornado funnel) -- plus thin whisker caps
    at each bar END showing that bound's own 90% credible interval. This
    is the standard way tornado diagrams get extended to carry probabilistic
    information (e.g. TreeAge's probabilistic tornado), as opposed to a
    forest-plot layout, which answers a different question and looks
    nothing like a tornado."""
    if prob_result is None:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Run the probabilistic tornado to see this plot.", ha="center", va="center")
        ax.axis("off")
        return fig
    rows = [r for r in prob_result["summary"] if r["metric"] == metric]
    rows = sorted(rows, key=lambda r: -abs(r["value_up_mean"] - r["value_down_mean"]))
    baseline = prob_result["baseline_mean"][metric]

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1))
    for i, row in enumerate(rows):
        lo, hi = row["value_down_mean"], row["value_up_mean"]
        left, width = min(lo, hi), abs(hi - lo)
        ax.barh(i, width, left=left, color="tab:purple", alpha=0.4)
        # whisker caps: each bound's own 90% CrI, drawn at that bound's y position
        for value_key, q05_key, q95_key in [
            ("value_down_mean", "value_down_q05", "value_down_q95"),
            ("value_up_mean", "value_up_q05", "value_up_q95"),
        ]:
            mean, q05, q95 = row[value_key], row[q05_key], row[q95_key]
            ax.errorbar([mean], [i], xerr=[[mean - q05], [q95 - mean]], fmt="none",
                        color="black", capsize=4, elinewidth=1)
    ax.axvline(baseline, color="black", linewidth=1, linestyle="--", label="baseline")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_label(r["param"]) for r in rows], fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title(f"Probabilistic tornado: {metric}\n"
                 f"(bar = mean outcome at -20%/+20%; whiskers = 90% credible interval at each bound)",
                 fontsize=9)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def server(input, output, session):
    # Load cached results as plain variables first -- reactive.Value.get()
    # can only be called from inside a reactive context (a @render function,
    # @reactive.effect, or @reactive.calc), not during this plain setup code.
    s1_initial = load_result(S1_NAME)

    sys1_result = reactive.Value(s1_initial)

    def _status_msg(r, extra=""):
        if r is None:
            return "Not yet run."
        return f"Loaded cached result. Backend: {r['backend']}.{extra}"

    sys1_status_msg = reactive.Value(_status_msg(s1_initial))

    # ---------------- System 2 cascade-year cache ----------------
    # Both base-case (2019) and sensitivity (2024) results are loaded up
    # front (whichever have already been calibrated -- either may be None).
    # sys2_result always holds whichever year is currently selected via the
    # "cascade_year" radio buttons on the System 2 tab; switching that
    # control just swaps in the corresponding entry from sys2_cache, no
    # recalculation needed. Running a fresh calibration updates BOTH
    # sys2_result and that year's slot in sys2_cache (and the file on disk),
    # leaving the other year's cached result untouched.
    _CASCADE_YEARS = ("2019", "2024", "WC2023")
    s2_cache_initial = {yr: load_result(result_name_for_year(yr)) for yr in _CASCADE_YEARS}
    sys2_cache = reactive.Value(s2_cache_initial)

    # Cached health-econ results are per cascade_year, same as the System 2
    # calibrations they are derived from -- switching the radio swaps these in
    # rather than showing one year's DALYs against another year's fit.
    def _post_prob_for(year):
        c = load_posterior_result(year)
        return c["probabilistic"] if c and "probabilistic" in c else None


    def _s2_status_msg(r, year):
        if r is None:
            return f"Not yet run for {year}."
        return (f"Loaded cached {year} result. Backend: {r['backend']} "
                f"({r.get('n_outer_draws', '?')} outer draws).")

    sys2_result = reactive.Value(s2_cache_initial["2019"])
    sys2_status_msg = reactive.Value(_s2_status_msg(s2_cache_initial["2019"], "2019"))

    @reactive.effect
    @reactive.event(input.cascade_year)
    def _switch_cascade_year():
        year = input.cascade_year()
        cached = sys2_cache.get().get(year)
        sys2_result.set(cached)
        sys2_status_msg.set(_s2_status_msg(cached, year))
        # Swap the health-econ cache to match, so the DALY tab never shows one
        # year's outputs alongside another year's calibration.
        pp = _post_prob_for(year)
        post_prob_result.set(pp)
        post_status_msg.set(f"Loaded cached {year} result." if pp is not None
                             else f"Not yet run for {year}.")

    def _active_cascade_year_text():
        s2 = sys2_result.get()
        year = s2.get("cascade_year", "?") if s2 else input.cascade_year()
        return (f"Using System 2 cascade base year: {year} "
                f"(set on the “System 2: Care Cascade” tab).")

    # ---------------- System 1 ----------------

    @reactive.effect
    @reactive.event(input.run_sys1)
    def _run_sys1():
        sys1_status_msg.set("Running... this can take roughly 30-60 seconds.")
        with ui.Progress(min=0, max=1, session=session) as p:
            p.set(message="Drawing System 1 natural-history uncertainty...")
            result = run_system1_calibration()
        sys1_result.set(result)
        save_result(result, S1_NAME)
        sys1_status_msg.set(f"Done. Backend: {result['backend']}.")

    @output
    @render.text
    def sys1_status():
        return sys1_status_msg.get()

    @output
    @render.table
    def sys1_summary():
        result = sys1_result.get()
        if result is None:
            return pd.DataFrame()
        df = pd.DataFrame(result["summary"])[["param", "mean", "sd", "q03", "q97", "r_hat", "ess"]].round(4)
        return _label_column(df, "param")

    @output
    @render.table
    def sys1_targets():
        result = sys1_result.get()
        if result is None:
            return pd.DataFrame()
        rows = []
        for key, (central, sd) in result["targets"].items():
            implied = result["model_implied_at_mean"][key]
            rows.append({"target": key, "model_implied": round(implied, 2) if implied is not None else None,
                         "target_central": central, "target_sd": sd})
        return _label_column(pd.DataFrame(rows), "target")

    @output
    @render.plot
    def sys1_trace():
        return _trace_plot(sys1_result.get())

    @output
    @render.plot
    def sys1_compare_plot():
        return _compare_plot(sys1_result.get())

    # ---------------- System 2 ----------------

    @reactive.effect
    @reactive.event(input.run_sys2)
    def _run_sys2():
        s1 = sys1_result.get()
        if s1 is None:
            sys2_status_msg.set("Run System 1 first -- its locked draws are a required input.")
            return
        year = input.cascade_year()
        sys2_status_msg.set(f"Running for cascade_year={year}... this takes roughly 4 minutes "
                             "(20 outer draws from System 1, each "
                             "running its own short cascade chain).")
        with ui.Progress(min=0, max=1, session=session) as p:
            p.set(message=f"Running System 2 calibration (cascade_year={year})...")
            result = run_system2_calibration(s1, cascade_year=year)
        sys2_result.set(result)
        save_result(result, result_name_for_year(year))
        cache = dict(sys2_cache.get())
        cache[year] = result
        sys2_cache.set(cache)
        sys2_status_msg.set(f"Done ({year}). Backend: {result['backend']} "
                             f"({result['n_outer_draws']} outer draws).")

    @output
    @render.text
    def sys2_status():
        return sys2_status_msg.get()

    @output
    @render.table
    def sys2_summary():
        result = sys2_result.get()
        if result is None:
            return pd.DataFrame()
        df = pd.DataFrame(result["summary"])[["param", "mean", "sd", "q03", "q97", "r_hat", "ess"]].round(4)
        return _label_column(df, "param")

    @output
    @render.table
    def sys2_targets():
        result = sys2_result.get()
        if result is None:
            return pd.DataFrame()
        rows = []
        for key, (central, sd) in result["targets"].items():
            implied = result["model_implied_at_mean"][key]
            rows.append({"target": key, "model_implied": round(implied, 3) if implied is not None else None,
                         "target_central": central, "target_sd": sd})
        return _label_column(pd.DataFrame(rows), "target")

    @output
    @render.plot
    def sys2_trace():
        return _trace_plot(sys2_result.get())

    @output
    @render.plot
    def sys2_compare_plot():
        return _compare_plot(sys2_result.get())

    @output
    @render.text
    def post_active_year_note():
        return _active_cascade_year_text()

    @output
    @render.text
    def scenario_active_year_note():
        return _active_cascade_year_text()

    # ---------------- Prior Run ----------------

    params_obj = load_parameters()

    prior_cached = load_prior_result()
    prior_prob_initial = (prior_cached["probabilistic"]
                           if prior_cached and "probabilistic" in prior_cached else None)
    prior_prob_result = reactive.Value(prior_prob_initial)
    prior_prob_status_msg = reactive.Value(
        "Loaded cached result." if prior_prob_initial is not None else "Not yet run."
    )

    @reactive.calc
    def prior_det():
        return run_prior_deterministic(params_obj)

    @output
    @render.table
    def prior_param_table():
        df = pd.DataFrame(prior_det()["param_table"]).round(4)
        return _label_column(df, "symbol")

    @output
    @render.table
    def prior_deterministic_table():
        m = prior_det()["metrics"]
        return pd.DataFrame([{"metric": k, "value": round(v, 3)} for k, v in m.items()])

    @reactive.calc
    def prior_tornado():
        result = run_prior_point_tornado(params_obj)
        save_prior_point_result(result)
        return result

    @output
    @render.plot
    def prior_tornado_plot():
        return _point_tornado_plot(prior_tornado())

    @output
    @render.table
    def prior_tornado_table():
        result = prior_tornado()
        cols = ["param", "perturbation_kind",
                "DALYs_per_acquisition_down", "DALYs_per_acquisition_up",
                "DALYs_per_acquisition_elasticity"]
        df = pd.DataFrame(result["rows"])[cols]
        # NaN elasticity (alpha_shape, relative % undefined at central=0) sorts last
        df = df.reindex(df["DALYs_per_acquisition_elasticity"].abs().sort_values(
            ascending=False, na_position="last").index).round(4)
        return _label_column(df, "param")

    @output
    @render.table
    def prior_tornado_target_table():
        return _target_translation_table(prior_tornado(), [("down", "-20%"), ("up", "+20%")])

    @reactive.effect
    @reactive.event(input.run_prior_prob)
    def _run_prior_prob():
        prior_prob_status_msg.set("Running... roughly 30 seconds (1000 draws from the priors).")
        with ui.Progress(min=0, max=1, session=session) as p:
            p.set(message="Running probabilistic prior analysis...")
            result = run_prior_probabilistic(params_obj, n_draws=1000)
        prior_prob_result.set(result)
        save_prior_result(prior_det(), result)
        prior_prob_status_msg.set(f"Done. {result['n_draws']} draws.")

    @output
    @render.text
    def prior_prob_status():
        return prior_prob_status_msg.get()

    @output
    @render.table
    def prior_prob_table():
        result = prior_prob_result.get()
        if result is None:
            return pd.DataFrame()
        return pd.DataFrame(result["summary"]).round(3)

    @output
    @render.plot
    def prior_daly_plot():
        result = prior_prob_result.get()
        return _uncertainty_bar_plot(result["summary"] if result else None,
                                      DALY_METRICS, "Prior probabilistic: DALYs")

    @output
    @render.plot
    def prior_cost_plot():
        result = prior_prob_result.get()
        return _uncertainty_bar_plot(result["summary"] if result else None,
                                      COST_METRICS, "Prior probabilistic: cost (USD)")

    # ---------------- Posterior Run Summary ----------------

    post_prob_initial = _post_prob_for("2019")
    post_prob_result = reactive.Value(post_prob_initial)
    post_status_msg = reactive.Value(
        "Loaded cached 2019 result." if post_prob_initial is not None else "Not yet run."
    )

    @reactive.effect
    @reactive.event(input.run_post_prob)
    def _run_post_prob():
        s1 = sys1_result.get()
        s2 = sys2_result.get()
        if s1 is None or s2 is None:
            post_status_msg.set("Run System 1 and System 2 first -- "
                                 "both are required inputs.")
            return
        post_status_msg.set("Running... roughly 30 seconds (1000 paired posterior draws).")
        with ui.Progress(min=0, max=1, session=session) as p:
            p.set(message="Running posterior health-econ summary...")
            det = run_posterior_deterministic(s1, s2, params_obj)
            prob = run_posterior_probabilistic(s2, params_obj, n_draws=1000)
        year = input.cascade_year()
        post_prob_result.set(prob)
        save_posterior_result(det, prob, cascade_year=year)
        post_status_msg.set(f"Done ({year}). {prob['n_draws']} draws.")

    @output
    @render.text
    def post_status():
        return post_status_msg.get()

    @output
    @render.table
    def post_deterministic_table():
        s1 = sys1_result.get()
        s2 = sys2_result.get()
        if s1 is None or s2 is None:
            return pd.DataFrame()
        det = run_posterior_deterministic(s1, s2, params_obj)
        return pd.DataFrame([{"metric": k, "value": round(v, 3)} for k, v in det["metrics"].items()])

    @output
    @render.table
    def post_prob_table():
        result = post_prob_result.get()
        if result is None:
            return pd.DataFrame()
        return pd.DataFrame(result["summary"]).round(3)

    @output
    @render.plot
    def post_daly_plot():
        result = post_prob_result.get()
        return _uncertainty_bar_plot(result["summary"] if result else None,
                                      DALY_METRICS, "Posterior probabilistic: DALYs")

    @output
    @render.plot
    def post_cost_plot():
        result = post_prob_result.get()
        return _uncertainty_bar_plot(result["summary"] if result else None,
                                      COST_METRICS, "Posterior probabilistic: cost (USD)")

    @output
    @render.plot
    def prior_vs_post_plot():
        prior_result = prior_prob_result.get()
        post_result = post_prob_result.get()
        return _prior_vs_post_plot(
            prior_result["summary"] if prior_result else None,
            post_result["summary"] if post_result else None,
        )

    # ---------------- Scenario Analysis ----------------

    prob_tornado_cached = load_prob_result()
    prob_tornado_result = reactive.Value(prob_tornado_cached)
    prob_tornado_status_msg = reactive.Value(
        "Loaded cached result." if prob_tornado_cached is not None else "Not yet run."
    )

    @reactive.calc
    def point_tornado():
        s1 = sys1_result.get()
        s2 = sys2_result.get()
        if s1 is None or s2 is None:
            return None
        result = run_point_tornado(s1, s2, params_obj)
        save_point_result(result)
        return result

    @output
    @render.plot
    def point_tornado_plot():
        return _point_tornado_plot(point_tornado())

    @output
    @render.table
    def point_tornado_table():
        result = point_tornado()
        if result is None:
            return pd.DataFrame()
        cols = ["param", "direction_of_improvement",
                "DALYs_per_acquisition_down", "DALYs_per_acquisition_up",
                "DALYs_per_acquisition_elasticity"]
        df = pd.DataFrame(result["rows"])[cols]
        df = df.sort_values("DALYs_per_acquisition_elasticity", key=abs, ascending=False).round(4)
        return _label_column(df, "param")

    @output
    @render.table
    def point_target_table():
        return _target_translation_table(point_tornado(), [("down", "-20%"), ("up", "+20%")])

    @reactive.calc
    def uncertainty_tornado():
        s1 = sys1_result.get()
        s2 = sys2_result.get()
        if s1 is None or s2 is None:
            return None
        result = run_uncertainty_tornado(s1, s2, params_obj)
        save_uncertainty_result(result)
        return result

    @output
    @render.plot
    def uncertainty_tornado_plot():
        return _uncertainty_tornado_plot(uncertainty_tornado())

    @output
    @render.table
    def uncertainty_tornado_table():
        result = uncertainty_tornado()
        if result is None:
            return pd.DataFrame()
        cols = ["param", "q03", "q97", "DALYs_per_acquisition_at_q03",
                "DALYs_per_acquisition_at_q97", "DALYs_per_acquisition_range"]
        df = pd.DataFrame(result["rows"])[cols]
        df = df.sort_values("DALYs_per_acquisition_range", ascending=False).round(4)
        return _label_column(df, "param")

    @output
    @render.table
    def uncertainty_target_table():
        return _target_translation_table(uncertainty_tornado(), [("at_q03", "q03"), ("at_q97", "q97")])

    @reactive.effect
    @reactive.event(input.run_prob_tornado)
    def _run_prob_tornado():
        s2 = sys2_result.get()
        if s2 is None:
            prob_tornado_status_msg.set("Run System 2 calibration first.")
            return
        prob_tornado_status_msg.set("Running... roughly 1.5-2 minutes "
                                     "(250 posterior draws x 7 parameters x 2 directions).")
        with ui.Progress(min=0, max=1, session=session) as p:
            p.set(message="Running probabilistic tornado...")
            result = run_probabilistic_tornado(s2, params_obj, n_draws=250)
        prob_tornado_result.set(result)
        save_prob_result(result)
        prob_tornado_status_msg.set(f"Done. {result['n_draws']} draws.")

    @output
    @render.text
    def prob_tornado_status():
        return prob_tornado_status_msg.get()

    @output
    @render.plot
    def prob_tornado_plot():
        return _prob_tornado_plot(prob_tornado_result.get())

    @output
    @render.table
    def prob_tornado_table():
        result = prob_tornado_result.get()
        if result is None:
            return pd.DataFrame()
        rows = [r for r in result["summary"] if r["metric"] == "DALYs_per_acquisition"]
        rows = sorted(rows, key=lambda r: -abs(r["value_up_mean"] - r["value_down_mean"]))
        df = pd.DataFrame(rows).round(4)
        return _label_column(df, "param")


app = App(app_ui, server)

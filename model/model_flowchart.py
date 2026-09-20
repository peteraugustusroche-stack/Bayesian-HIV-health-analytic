"""
Standalone visualization script (does NOT touch the calibration/simulation
engine -- it only imports and reads from it, the same way scenario_analysis.py
does). Renders FOUR flowchart PNGs, prior and posterior versions of two
detail levels:

  Macro (cascade_flowchart_prior.png / cascade_flowchart_posterior.png):
    A hybrid view. Undiagnosed and both ART-suppressed statuses (ART1 and
    ART2) are each expanded into their own 7-node CD4-band chain -- showing
    the actual decline-down-the-bands dynamic for Undiagnosed and the
    climb-back-up-the-bands (reconstitution) dynamic for the two suppressed
    statuses, with EXACT per-band transition probabilities and costs (no
    weighting needed, same as the full grid below). Diagnosed_PreART,
    ART1_Failing, and LTFU stay as single aggregate nodes (cost broken out
    per band where it varies), since those three aren't the CD4-dynamics of
    interest here. Edges connecting a band-chain to an aggregate node (e.g.
    Undiagnosed-band -> Diagnosed_PreART, or Diagnosed_PreART -> ART1
    Suppressed-band) are fanned/pooled using each parameter set's own
    model-derived, time-integrated within-status CD4 distribution (same
    technique simulate_cascade.stationary_cascade_proportions uses) --
    e.g. the probability of Diagnosed_PreART -> ART1_Suppressed@CD4-250-349
    is the aggregate initiation probability times the modeled share of
    Diagnosed_PreART person-time spent at that band. HIV Death and
    Background Death are separate nodes with separate edges throughout.

  Full grid (cascade_flowchart_full_prior.png / _full_posterior.png):
    All 42 (CD4 band x cascade status) cells as individual nodes, laid out
    as an exact grid (band = row, status = column), plus the same two
    separate death nodes. Every transition probability (band decline,
    reconstitution, cascade-status transition, HIV death, background death)
    is computed EXACTLY per cell -- no band-weighting/approximation needed,
    since every band is its own node. This is dense (44 nodes, ~180 edges)
    by design -- it's the real 44-state engine, not a simplification.
    NOTE: both of the above turned out too dense to read comfortably -- see
    the schematic design below, which is the recommended one.

  Schematic (cascade_schematic_prior.png / cascade_schematic_posterior.png):
    THE RECOMMENDED DIAGRAM. Two-panel design borrowed from a reference
    schematic Peter supplied (a different model, same idea): panel 1 is the
    6-state cascade skeleton with hazard-labeled arrows (symbolic formula
    for the one hazard that's genuinely CD4-band-dependent -- diagnosis --
    plus the resolved annual probability for every band-independent hazard);
    HIV Death and Background Death sit off to the side, reached by thin
    dotted lines from every cascade box (not literal, individually-labeled
    edges -- same abstraction the reference uses). Panel 2 is a SINGLE
    generic 7-node CD4-band sub-chain shown once, captioned as applying
    inside every relevant cascade box -- decline (solid) inside Undiagnosed
    /Diagnosed_PreART/ART1_Failing/LTFU, recovery/reconstitution (dashed)
    inside ART1_Suppressed/ART2_Suppressed. This works as a simplification
    (not just a style choice) because the CD4 decline/reconstitution hazard
    functions in the real engine depend only on band and treatment category,
    never on which specific cascade box they're evaluated in -- so one
    generic template is exact, not approximated. No cost annotations on this
    version (matching the clean reference style) -- see the macro/full-grid
    PNGs above, or the dashboard's Prior/Posterior Run tabs, for cost.

Run directly:
    python3 model_flowchart.py
(requires calibrate_system1.py and calibrate_system2.py to have been run at
least once, for the posterior diagrams)
"""

from __future__ import annotations

import graphviz
import numpy as np

from params import load_parameters, N_CD4_BANDS, CD4_BAND_NAMES, ANCHOR_AGE
from states import CASCADE_STATUSES, full_index, N_FULL_STATES
from hazards import (
    diagnosis_hazard, suppressed_mortality,
    background_mortality, competing_risks_step,
    reconstitution_hazard, choose_recon_rate,
)
import likelihoods as _lk
from natural_history_params import mortality_curve_at_age
from simulate_cascade import progression_row_at_age
import simulate_cascade as casc
from health_economics import cost_per_year
from health_econ_params import COST_ART_STARTUP
from calibration_common import load_result

HORIZON_YEARS = 30
DT = 1 / 12

NODE_LABELS = {
    "Undiagnosed": "Undiagnosed",
    "Diagnosed_PreART": "Diagnosed,\nPre-ART",
    "ART1_Suppressed": "1st-Line ART,\nSuppressed",
    "ART1_Failing": "1st-Line ART,\nFailing",
    "LTFU": "Disengaged\n(LTFU)",
    "ART2_Suppressed": "2nd-Line ART,\nSuppressed",
}
NODE_COLORS = {
    "Undiagnosed": "#f4cccc",
    "Diagnosed_PreART": "#fce5cd",
    "ART1_Suppressed": "#d9ead3",
    "ART1_Failing": "#fff2cc",
    "LTFU": "#ead1dc",
    "ART2_Suppressed": "#cfe2f3",
}
HIV_DEATH_COLOR = "#e06666"
BG_DEATH_COLOR = "#b7b7b7"
STARTUP_EDGES = {("Diagnosed_PreART", "ART1_Suppressed"), ("ART1_Failing", "ART2_Suppressed")}


# ---------------------------------------------------------------------------
# Shared: parameter loading
# ---------------------------------------------------------------------------

def _get_central(kind: str, params) -> dict:
    if kind == "prior":
        # see run_prior_analysis: the spreadsheet dict omits code-defined
        # parameters, so use the merged view
        return _lk.all_central_estimates(params)
    s1 = load_result("system1")
    s2 = load_result("system2")
    if s1 is None or s2 is None:
        raise SystemExit("Run calibrate_system1.py and calibrate_system2.py before "
                          "generating the posterior flowcharts.")
    lb_mean = {row["param"]: row["mean"] for row in s1["summary"]}
    cascade_mean = {row["param"]: row["mean"] for row in s2["summary"]}
    central = dict(cascade_mean)
    central["κ_prog"] = lb_mean["κ_prog"]
    central["κ_mort"] = lb_mean["κ_mort"]
    return central


def _title(kind: str) -> str:
    return ("Prior model (pre-calibration central estimates)" if kind == "prior"
            else "Posterior model (Bayesian-calibrated, posterior mean)")


# ---------------------------------------------------------------------------
# Exact per-(status, band) hazards -- mirrors
# simulate_cascade.build_full_transition_matrix's per-cell hazard sets
# exactly, just exposed per cell instead of assembled into a matrix.
# ---------------------------------------------------------------------------

def _cell_hazards(status, band, central, mortality_curve, recov_fast, recov_slow, bg,
                   progression_curve=None):
    if progression_curve is None:
        progression_curve = progression_row_at_age(ANCHOR_AGE,
                                                    central.get("κ_prog", 1.0))
    h = {}
    if status == "Undiagnosed":
        h["HIV Death"] = mortality_curve[band]
        h["Background Death"] = bg
        h["Diagnosed_PreART"] = diagnosis_hazard(band, central["δ_bg"], central["δ_symp"],
                                                  central.get("γ_diag", 1.0))
        decline = float(progression_curve[band]) if band < N_CD4_BANDS - 1 else 0.0
        if decline > 0:
            h["decline"] = decline
    elif status == "Diagnosed_PreART":
        h["HIV Death"] = mortality_curve[band]
        h["Background Death"] = bg
        h["ART1_Suppressed"] = central["λ_init"]
        h["LTFU"] = central["μ_ltfu"]
        decline = float(progression_curve[band]) if band < N_CD4_BANDS - 1 else 0.0
        if decline > 0:
            h["decline"] = decline
    elif status == "ART1_Suppressed":
        h["HIV Death"] = suppressed_mortality(mortality_curve[band])
        h["Background Death"] = bg
        h["ART1_Failing"] = central["μ_vf1"]
        h["LTFU"] = central["μ_ltfu"]
        recov_h = reconstitution_hazard(band, choose_recon_rate(band, recov_fast, recov_slow))
        if recov_h > 0:
            h["recover"] = recov_h
    elif status == "ART1_Failing":
        h["HIV Death"] = mortality_curve[band]
        h["Background Death"] = bg
        h["ART2_Suppressed"] = central["μ_switch"]
        h["LTFU"] = central["μ_ltfu"]
        decline = float(progression_curve[band]) if band < N_CD4_BANDS - 1 else 0.0
        if decline > 0:
            h["decline"] = decline
    elif status == "LTFU":
        h["HIV Death"] = mortality_curve[band]
        h["Background Death"] = bg
        h["Diagnosed_PreART"] = central["μ_re-engage"]
        decline = float(progression_curve[band]) if band < N_CD4_BANDS - 1 else 0.0
        if decline > 0:
            h["decline"] = decline
    elif status == "ART2_Suppressed":
        h["HIV Death"] = suppressed_mortality(mortality_curve[band])
        h["Background Death"] = bg
        h["LTFU"] = central["μ_ltfu"]
        recov_h = reconstitution_hazard(band, choose_recon_rate(band, recov_fast, recov_slow))
        if recov_h > 0:
            h["recover"] = recov_h
    return h


# ---------------------------------------------------------------------------
# Full 44-state grid diagram
# ---------------------------------------------------------------------------

def build_full_grid_diagram(kind: str, out_path: str) -> str:
    params = load_parameters()
    central = _get_central(kind, params)
    mortality_curve = mortality_curve_at_age(ANCHOR_AGE, central.get("κ_mort", 1.0))
    recov_fast, recov_slow = params.fixed["recov_rate"]
    bg = background_mortality()

    g = graphviz.Digraph("cascade_full", format="png", engine="neato")
    g.attr(label=_title(kind) + " -- full 44-state grid (CD4 band x cascade status)",
           labelloc="t", fontsize="20", fontname="Helvetica",
           overlap="false", splines="true", dpi="170")
    g.attr("node", shape="box", style="filled,rounded", fontname="Helvetica",
           fontsize="9", margin="0.05,0.03", width="0", height="0")
    g.attr("edge", fontname="Helvetica", fontsize="7", arrowsize="0.5")

    dx, dy = 3.3, 2.0
    col_of = {s: i for i, s in enumerate(CASCADE_STATUSES)}

    def nid(status, band):
        return f"{status}_{band}"

    for band in range(N_CD4_BANDS):
        for status in CASCADE_STATUSES:
            cost = cost_per_year(status, band)
            label = f"{NODE_LABELS[status]}\nCD4 {CD4_BAND_NAMES[band]}\n${cost:,.0f}/yr"
            x, y = col_of[status] * dx, -band * dy
            g.node(nid(status, band), label=label, pos=f"{x},{y}!",
                   fillcolor=NODE_COLORS[status])

    right_x = len(CASCADE_STATUSES) * dx + 2.2
    g.node("HIV_DEATH", label="HIV\nDeath", pos=f"{right_x},{-1.5*dy}!",
           fillcolor=HIV_DEATH_COLOR, fontsize="12", width="1.1", height="0.7")
    g.node("BG_DEATH", label="Background\nDeath", pos=f"{right_x},{-4.5*dy}!",
           fillcolor=BG_DEATH_COLOR, fontsize="12", width="1.1", height="0.7")

    for band in range(N_CD4_BANDS):
        for status in CASCADE_STATUSES:
            h = _cell_hazards(status, band, central, mortality_curve, recov_fast, recov_slow, bg)
            probs = competing_risks_step(h, dt=1.0)
            src = nid(status, band)
            for key, p in probs.items():
                if key == "stay" or p <= 0:
                    continue
                lbl = f"{p*100:.1f}%"
                if key == "decline":
                    g.edge(src, nid(status, band + 1), label=lbl, color="#666666", fontcolor="#666666")
                elif key == "recover":
                    g.edge(src, nid(status, band - 1), label=lbl, color="#666666", fontcolor="#666666")
                elif key == "HIV Death":
                    g.edge(src, "HIV_DEATH", label=lbl, color=HIV_DEATH_COLOR,
                           fontcolor=HIV_DEATH_COLOR, penwidth="0.6", arrowsize="0.35")
                elif key == "Background Death":
                    g.edge(src, "BG_DEATH", label=lbl, color=BG_DEATH_COLOR,
                           fontcolor="#888888", penwidth="0.6", arrowsize="0.35")
                else:  # cascade-status transition, same band
                    ecolor = "#1c4587"
                    if (status, key) in STARTUP_EDGES:
                        lbl += f"\n+${COST_ART_STARTUP:.0f}"
                    g.edge(src, nid(key, band), label=lbl, color=ecolor, fontcolor=ecolor)

    g.render(out_path, cleanup=True)
    return f"{out_path}.png"


# ---------------------------------------------------------------------------
# Macro 6-status diagram, with per-CD4-band cost breakdown per node
# ---------------------------------------------------------------------------

def _band_weights_within_status(M) -> dict:
    n_steps = int(HORIZON_YEARS / DT)
    init = np.zeros(N_FULL_STATES)
    init[full_index("Undiagnosed", 0)] = 1.0
    traj = casc.propagate(M, init, n_steps)
    time_in_band = traj.sum(axis=0) * DT

    weights = {}
    for status in CASCADE_STATUSES:
        band_mass = np.array([time_in_band[full_index(status, b)] for b in range(N_CD4_BANDS)])
        total = band_mass.sum()
        weights[status] = band_mass / total if total > 0 else np.ones(N_CD4_BANDS) / N_CD4_BANDS
    return weights


def _cost_label(status: str) -> str:
    costs = [cost_per_year(status, b) for b in range(N_CD4_BANDS)]
    if max(costs) - min(costs) < 0.01:
        return f"${costs[0]:,.0f}/yr"
    lines = [f"{CD4_BAND_NAMES[b]:>7}: ${costs[b]:,.0f}/yr" for b in range(N_CD4_BANDS)]
    return "\n".join(lines)


def build_macro_diagram(kind: str, out_path: str) -> str:
    """Hybrid macro diagram: Undiagnosed and both ART-suppressed statuses are
    expanded into 7-node CD4-band chains (exact per-band hazards, via
    _cell_hazards -- same function the full grid uses). Diagnosed_PreART,
    ART1_Failing, and LTFU stay as single aggregate nodes (band-weighted
    average hazards, same technique as the original macro diagram). Edges
    that cross a band-chain <-> aggregate-node boundary are fanned/pooled
    using the aggregate node's own model-derived within-status CD4-band
    distribution (`weights`, from _band_weights_within_status)."""
    params = load_parameters()
    central = _get_central(kind, params)
    mortality_curve = mortality_curve_at_age(ANCHOR_AGE, central.get("κ_mort", 1.0))
    recov_fast, recov_slow = params.fixed["recov_rate"]
    bg = background_mortality()
    M = casc.build_full_transition_matrix(central, mortality_curve, params.fixed["recov_rate"])
    weights = _band_weights_within_status(M)

    AGG = ("Diagnosed_PreART", "ART1_Failing", "LTFU")
    BAND_CHAIN = ("Undiagnosed", "ART1_Suppressed", "ART2_Suppressed")

    def wavg(status, fn):
        return sum(weights[status][b] * fn(b) for b in range(N_CD4_BANDS))

    avg_hiv_mort = {s: wavg(s, lambda b: mortality_curve[b]) for s in AGG}

    agg_hazards = {
        "Diagnosed_PreART": {"ART1_Suppressed": central["λ_init"], "LTFU": central["μ_ltfu"],
                              "HIV Death": avg_hiv_mort["Diagnosed_PreART"], "Background Death": bg},
        "ART1_Failing": {"ART2_Suppressed": central["μ_switch"], "LTFU": central["μ_ltfu"],
                          "HIV Death": avg_hiv_mort["ART1_Failing"], "Background Death": bg},
        "LTFU": {"Diagnosed_PreART": central["μ_re-engage"],
                 "HIV Death": avg_hiv_mort["LTFU"], "Background Death": bg},
    }
    agg_probs = {s: competing_risks_step(h, dt=1.0) for s, h in agg_hazards.items()}

    g = graphviz.Digraph("cascade_macro", format="png", engine="neato")
    g.attr(label=_title(kind) + " -- macro view (CD4-band chains for Undiagnosed & ART-Suppressed)",
           labelloc="t", fontsize="15", fontname="Helvetica",
           overlap="false", splines="true", dpi="150")
    g.attr("node", shape="box", style="filled,rounded", fontname="Helvetica",
           fontsize="9", margin="0.06,0.04", width="0", height="0")
    g.attr("edge", fontname="Helvetica", fontsize="7", arrowsize="0.5")

    dx, dy = 3.1, 1.15
    col_x = {"Undiagnosed": 0, "Diagnosed_PreART": 1, "ART1_Suppressed": 2,
             "ART1_Failing": 3, "LTFU": 4, "ART2_Suppressed": 5}
    mid_y = -3 * dy  # vertical center of the 7-row (band 0..6) chains

    def band_id(status, band):
        return f"{status}_{band}"

    for status in BAND_CHAIN:
        for band in range(N_CD4_BANDS):
            cost = cost_per_year(status, band)
            label = f"{NODE_LABELS[status]}\nCD4 {CD4_BAND_NAMES[band]}\n${cost:,.0f}/yr"
            x, y = col_x[status] * dx, -band * dy
            g.node(band_id(status, band), label=label, pos=f"{x},{y}!", fillcolor=NODE_COLORS[status])

    for status in AGG:
        label = f"{NODE_LABELS[status]}\n{_cost_label(status)}"
        x = col_x[status] * dx
        g.node(status, label=label, pos=f"{x},{mid_y}!", fillcolor=NODE_COLORS[status], fontname="Courier")

    right_x = col_x["ART2_Suppressed"] * dx + 2.1
    g.node("HIV_DEATH", label="HIV Death", pos=f"{right_x},{mid_y+1.6}!",
           fillcolor=HIV_DEATH_COLOR, fontsize="11", fontname="Helvetica", width="1.0", height="0.6")
    g.node("BG_DEATH", label="Background\nDeath", pos=f"{right_x},{mid_y-1.6}!",
           fillcolor=BG_DEATH_COLOR, fontsize="11", fontname="Helvetica", width="1.0", height="0.6")

    def _death_or_agg_edge(src, key, p):
        lbl = f"{p*100:.1f}%/yr"
        if key == "HIV Death":
            g.edge(src, "HIV_DEATH", label=lbl, color=HIV_DEATH_COLOR,
                   fontcolor=HIV_DEATH_COLOR, penwidth="0.7", arrowsize="0.4")
        elif key == "Background Death":
            g.edge(src, "BG_DEATH", label=lbl, color=BG_DEATH_COLOR,
                   fontcolor="#888888", penwidth="0.7", arrowsize="0.4")
        else:
            g.edge(src, key, label=lbl, color="#1c4587", fontcolor="#1c4587")

    # ---- Undiagnosed band-chain: decline down, diagnose out, death out ----
    for band in range(N_CD4_BANDS):
        h = _cell_hazards("Undiagnosed", band, central, mortality_curve, recov_fast, recov_slow, bg)
        probs = competing_risks_step(h, dt=1.0)
        src = band_id("Undiagnosed", band)
        for key, p in probs.items():
            if key == "stay" or p <= 0:
                continue
            if key == "decline":
                g.edge(src, band_id("Undiagnosed", band + 1), label=f"{p*100:.1f}%/yr",
                       color="#666666", fontcolor="#666666")
            else:
                _death_or_agg_edge(src, key, p)
        g.node(src, xlabel=f"stays {probs['stay']*100:.0f}%")

    # ---- ART1_Suppressed / ART2_Suppressed band-chains: climb up, exit out ----
    for status in ("ART1_Suppressed", "ART2_Suppressed"):
        for band in range(N_CD4_BANDS):
            h = _cell_hazards(status, band, central, mortality_curve, recov_fast, recov_slow, bg)
            probs = competing_risks_step(h, dt=1.0)
            src = band_id(status, band)
            for key, p in probs.items():
                if key == "stay" or p <= 0:
                    continue
                if key == "recover":
                    g.edge(src, band_id(status, band - 1), label=f"{p*100:.1f}%/yr",
                           color="#1155cc", fontcolor="#1155cc")
                else:
                    _death_or_agg_edge(src, key, p)
            g.node(src, xlabel=f"stays {probs['stay']*100:.0f}%")

    # ---- aggregate nodes: Diagnosed_PreART, ART1_Failing, LTFU ----
    fanout_target = {"ART1_Suppressed": "Diagnosed_PreART", "ART2_Suppressed": "ART1_Failing"}
    for status in AGG:
        probs = agg_probs[status]
        for target, p in probs.items():
            if target == "stay" or p <= 0:
                continue
            if target in ("ART1_Suppressed", "ART2_Suppressed"):
                band_weights = weights[fanout_target[target]]
                for band in range(N_CD4_BANDS):
                    share = p * band_weights[band]
                    if share <= 0:
                        continue
                    lbl = f"{share*100:.1f}%/yr"
                    if band == N_CD4_BANDS // 2 and (status, target) in STARTUP_EDGES:
                        lbl += f"\n(+${COST_ART_STARTUP:.0f} startup)"
                    g.edge(status, band_id(target, band), label=lbl)
            elif target == "HIV Death":
                g.edge(status, "HIV_DEATH", label=f"{p*100:.1f}%/yr",
                       color=HIV_DEATH_COLOR, fontcolor=HIV_DEATH_COLOR)
            elif target == "Background Death":
                g.edge(status, "BG_DEATH", label=f"{p*100:.1f}%/yr",
                       color=BG_DEATH_COLOR, fontcolor="#888888")
            else:
                g.edge(status, target, label=f"{p*100:.1f}%/yr")
        g.node(status, xlabel=f"stays {probs['stay']*100:.0f}%/yr")

    g.render(out_path, cleanup=True)
    return f"{out_path}.png"


# ---------------------------------------------------------------------------
# Schematic diagram -- two-panel design borrowed from Peter's reference:
# a clean cascade-status skeleton (panel 1) + one generic CD4 sub-chain
# template (panel 2), no cost clutter, hazard-labeled edges.
# ---------------------------------------------------------------------------

def _fmt_isolated(h: float) -> str:
    """Isolated annual probability 1-exp(-h), formatted; flags near-certain
    (saturated) hazards with the raw rate too, since '100.0%/yr' alone hides
    just how much faster than 'once a year' the underlying rate is."""
    import math
    p = 1 - math.exp(-h)
    if p > 0.995:
        return f">99%/yr (rate {h:.1f}/yr)"
    return f"{p*100:.1f}%/yr"


def _build_cascade_panel(kind, central, p_init, p_vf1, p_ltfu, p_reeng, p_switch):
    g = graphviz.Digraph("panel1", format="png")
    g.attr(rankdir="LR", fontsize="13", fontname="Helvetica-Bold", bgcolor="white",
           label="Care cascade status (6 states)", labelloc="t", labeljust="l",
           splines="true", nodesep="0.6", ranksep="1.0", dpi="160", margin="0.25")
    g.attr("node", fontname="Helvetica", fontsize="10")
    g.attr("edge", fontname="Helvetica", fontsize="8.5")

    g.attr("node", shape="box", style="filled,rounded", margin="0.14,0.09")
    for status in CASCADE_STATUSES:
        g.node(status, label=NODE_LABELS[status], fillcolor=NODE_COLORS[status])

    g.edge("Undiagnosed", "Diagnosed_PreART",
           label="Diagnosis hazard\nδ_bg + δ_symp·(band/6)^γ_diag", style="solid", penwidth="1.4", minlen="2")
    g.edge("Diagnosed_PreART", "ART1_Suppressed",
           label=f"ART Initiation Hazard λ_init\n({p_init})", style="solid", penwidth="1.4")
    g.edge("Diagnosed_PreART", "LTFU",
           label=f"μ_ltfu ({p_ltfu})", style="dashed")
    g.edge("ART1_Suppressed", "ART1_Failing",
           label=f"1st-Line VF Hazard μ_vf1\n({p_vf1})", style="dashed")
    g.edge("ART1_Suppressed", "LTFU",
           label=f"μ_ltfu ({p_ltfu})", style="dashed")
    g.edge("ART1_Failing", "ART2_Suppressed",
           label=f"Rescue Switch Hazard μ_switch\n({p_switch})", style="solid", penwidth="1.4")
    g.edge("ART1_Failing", "LTFU",
           label=f"μ_ltfu ({p_ltfu})", style="dashed")
    g.edge("ART2_Suppressed", "LTFU",
           label=f"μ_ltfu ({p_ltfu})", style="dashed")
    g.edge("LTFU", "Diagnosed_PreART",
           label=f"Re-engagement Hazard μ_re-engage\n({p_reeng})",
           style="solid", penwidth="1.4", constraint="false")

    g.node("HIV_DEATH", label="HIV death\n(absorbing)", shape="ellipse",
           style="filled", fillcolor=HIV_DEATH_COLOR)
    g.node("BG_DEATH", label="Background death\n(absorbing)",
           shape="ellipse", style="filled", fillcolor=BG_DEATH_COLOR)

    for status in CASCADE_STATUSES:
        g.edge(status, "HIV_DEATH", style="dotted", arrowsize="0.5",
               color="#999999", constraint="false")
        g.edge(status, "BG_DEATH", style="dotted", arrowsize="0.5",
               color="#999999", constraint="false")
    return g


def _build_cd4_panel(central, recov_fast, recov_slow):
    g = graphviz.Digraph("panel2", format="png")
    g.attr(rankdir="LR", fontsize="10.5", fontname="Helvetica-Bold", bgcolor="white",
           label=("Within Undiagnosed / Diagnosed_PreART / ART1_Failing / LTFU: CD4 declines (solid).   "
                  "Within ART1_Suppressed / ART2_Suppressed: CD4 recovers toward the band already reached "
                  "(dashed).   Same hazard functions in every case -- shown once."),
           labelloc="t", labeljust="l", splines="true",
           nodesep="0.5", ranksep="1.1", dpi="160", margin="0.25")
    g.attr("edge", fontname="Helvetica", fontsize="8.5")
    g.attr("node", shape="circle", style="filled", fillcolor="#d9ead3",
           fontname="Helvetica", fontsize="10", fixedsize="false", width="0.75")
    for band in range(N_CD4_BANDS):
        g.node(f"cd4_{band}", label=CD4_BAND_NAMES[band])

    for band in range(N_CD4_BANDS - 1):
        h_decline = float(progression_row_at_age(ANCHOR_AGE, central.get("κ_prog", 1.0))[band])
        g.edge(f"cd4_{band}", f"cd4_{band+1}",
               label=f"decline\n({_fmt_isolated(h_decline)})", style="solid")
    for band in range(1, N_CD4_BANDS):
        h_recover = reconstitution_hazard(band, choose_recon_rate(band, recov_fast, recov_slow))
        g.edge(f"cd4_{band}", f"cd4_{band-1}",
               label=f"recovery\n({_fmt_isolated(h_recover)})", style="dashed", constraint="false")
    return g


def build_schematic_diagram(kind: str, out_path: str) -> str:
    """Two-panel schematic, panels rendered as SEPARATE graphviz images and
    stacked with PIL -- more reliable than fighting dot's cluster-ordering
    heuristics to keep panel 1 above panel 2."""
    from PIL import Image, ImageDraw, ImageFont
    import tempfile, os

    params = load_parameters()
    central = _get_central(kind, params)
    recov_fast, recov_slow = params.fixed["recov_rate"]

    p_init = _fmt_isolated(central["λ_init"])
    p_vf1 = _fmt_isolated(central["μ_vf1"])
    p_ltfu = _fmt_isolated(central["μ_ltfu"])
    p_reeng = _fmt_isolated(central["μ_re-engage"])
    p_switch = _fmt_isolated(central["μ_switch"])

    panel1 = _build_cascade_panel(kind, central, p_init, p_vf1, p_ltfu, p_reeng, p_switch)
    panel2 = _build_cd4_panel(central, recov_fast, recov_slow)

    with tempfile.TemporaryDirectory() as tmpdir:
        p1_path = panel1.render(os.path.join(tmpdir, "panel1"), cleanup=True)
        p2_path = panel2.render(os.path.join(tmpdir, "panel2"), cleanup=True)
        def _flatten_to_white(path):
            """graphviz's bgcolor='white' doesn't reliably fill the full
            canvas on every cairo build -- composite onto an opaque white
            background ourselves so transparent margin pixels don't turn
            black when we drop the alpha channel."""
            src = Image.open(path).convert("RGBA")
            bg = Image.new("RGBA", src.size, (255, 255, 255, 255))
            bg.alpha_composite(src)
            return bg.convert("RGB")

        def _autocrop(im, pad=12):
            """Trim uniform-white margin (dot's cluster layout tends to
            leave a lot of empty space below a wide, short diagram)."""
            import PIL.ImageChops as ImageChops
            bg = Image.new("RGB", im.size, "white")
            diff = ImageChops.difference(im, bg)
            bbox = diff.getbbox()
            if bbox is None:
                return im
            l, t, r, b = bbox
            l, t = max(0, l - pad), max(0, t - pad)
            r, b = min(im.width, r + pad), min(im.height, b + pad)
            return im.crop((l, t, r, b))

        im1 = _autocrop(_flatten_to_white(p1_path))
        im2 = _autocrop(_flatten_to_white(p2_path))

        title = _title(kind) + " -- schematic"
        title_h = 60
        pad = 20
        width = max(im1.width, im2.width) + 2 * pad
        # scale panel2 to match panel1's width for a clean stacked look
        if im2.width != im1.width:
            scale = im1.width / im2.width
            im2 = im2.resize((im1.width, int(im2.height * scale)))
        height = title_h + im1.height + pad + im2.height + 2 * pad

        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), title, font=font)
        draw.text(((width - (bbox[2] - bbox[0])) / 2, 15), title, fill="black", font=font)

        canvas.paste(im1, (pad, title_h))
        canvas.paste(im2, (pad, title_h + im1.height + pad))
        canvas.save(f"{out_path}.png")

    return f"{out_path}.png"


if __name__ == "__main__":
    for kind in ("prior", "posterior"):
        p = build_schematic_diagram(kind, f"cascade_schematic_{kind}")
        print("Saved", p)
        p = build_macro_diagram(kind, f"cascade_flowchart_{kind}")
        print("Saved", p)
        p = build_full_grid_diagram(kind, f"cascade_flowchart_full_{kind}")
        print("Saved", p)

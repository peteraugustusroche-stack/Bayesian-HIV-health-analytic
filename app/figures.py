"""Figures for the interactive app, in the write-up's house style.

Every function takes data and returns a matplotlib Figure, so they can be
rendered and inspected without starting a server.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import engine as E
from figstyle import (AQUA, AXIS, BLUE, GRID, INK, INK2, MAGENTA, MUTED,
                      ORANGE, RED, SEQ, SURFACE, VIOLET, YELLOW, apply_style,
                      despine, no_grid_x, no_grid_y)

apply_style()

USD = FuncFormatter(lambda v, _p: f"${v:,.0f}")

STATUS_LABEL = {
    "Undiagnosed": "Undiagnosed",
    "Diagnosed_PreART": "Diagnosed, Not Yet Treated",
    "ART1_Ramp": "First Line, Not Yet Suppressed",
    "ART1_Suppressed": "First Line, Suppressed",
    "ART1_Failing": "First Line, Failing",
    "ART2_Suppressed": "Second Line, Suppressed",
    "LTFU_Recent": "Recently Disengaged",
    "LTFU_LongTerm": "Long-Term Disengaged",
}


def _blank(ax, message):
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED,
            transform=ax.transAxes)
    ax.set_axis_off()


def _band(ax, x, lo, hi, colour, alpha=0.16):
    ax.axvspan(lo, hi, color=colour, alpha=alpha, lw=0, zorder=1)


def posterior_density(values, label, unit="", reference=None,
                      reference_label="Base Case", colour=BLUE):
    """Posterior of one quantity, with its 90% credible interval shaded."""
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    v = np.asarray(values)
    lo, hi = np.percentile(v, [5, 95])
    counts, edges = np.histogram(v, bins=60, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.fill_between(centres, counts, color=colour, alpha=0.30, lw=0, zorder=2)
    ax.plot(centres, counts, color=colour, zorder=3)
    _band(ax, centres, lo, hi, colour)
    ax.axvline(v.mean(), color=INK, lw=1.4, zorder=4)
    ax.annotate(f"Mean {unit}{v.mean():,.2f}" if unit else f"Mean {v.mean():,.2f}",
                xy=(v.mean(), ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", color=INK, fontsize=8.4)
    if reference is not None:
        ax.axvline(reference, color=MUTED, lw=1.2, ls="--", zorder=4)
        ax.annotate(reference_label, xy=(reference, ax.get_ylim()[1] * 0.62),
                    xytext=(6, 0), textcoords="offset points",
                    color=MUTED, fontsize=8)
    ax.set_xlabel(label)
    ax.set_ylabel("Posterior density")
    ax.set_yticks([])
    despine(ax, keep=("bottom",))
    no_grid_y(ax)
    fig.tight_layout()
    return fig


def daly_decomposition(out, out_base=None):
    """Years of life lost against years lived with disability, cascade against
    the no-care counterfactual."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    rows = ["No Care", "With Care"]
    yll = [out["YLL_natural_history"].mean(), out["YLL_cascade"].mean()]
    yld = [out["YLD_natural_history"].mean(), out["YLD_cascade"].mean()]
    y = np.arange(len(rows))
    ax.barh(y, yll, color=ORANGE, height=0.55, label="Years Of Life Lost", zorder=3)
    ax.barh(y, yld, left=yll, color=BLUE, height=0.55,
            label="Years Lived With Disability", zorder=3)
    for i, (a, b) in enumerate(zip(yll, yld)):
        ax.text(a + b + 0.25, i, f"{a + b:.2f}", va="center",
                fontsize=8.4, color=INK2)
        if a > 1.2:
            ax.text(a / 2, i, f"{a:.2f}", va="center", ha="center",
                    fontsize=8, color=SURFACE)
        if b > 1.2:
            ax.text(a + b / 2, i, f"{b:.2f}", va="center", ha="center",
                    fontsize=8, color=SURFACE)
    ax.set_yticks(y)
    ax.set_yticklabels(rows)
    ax.invert_yaxis()
    ax.set_xlabel("DALYs per acquisition, discounted")
    ax.set_xlim(0, max(np.array(yll) + np.array(yld)) * 1.18)
    ax.legend(loc="lower right", ncol=1)
    despine(ax)
    no_grid_y(ax)
    fig.tight_layout()
    return fig


def cost_decomposition(components, price_year):
    """Lifetime cost per acquisition, by what the money buys."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    labels = list(components)
    means = [components[k].mean() for k in labels]
    colours = [SEQ[3], SEQ[9], AQUA, ORANGE, MUTED][:len(labels)]
    left = 0.0
    for lab, v, col in zip(labels, means, colours):
        ax.barh([0], [v], left=[left], color=col, height=0.5, zorder=3, label=lab)
        if v > sum(means) * 0.055:
            ax.text(left + v / 2, 0, f"${v:,.0f}", ha="center", va="center",
                    fontsize=8, color=SURFACE if col != MUTED else INK)
        left += v
    ax.text(left * 1.01, 0, f"${left:,.0f}", va="center", fontsize=9,
            color=INK, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlim(0, max(left * 1.16, 1.0))
    ax.xaxis.set_major_formatter(USD)
    ax.set_xlabel(f"Lifetime cost per acquisition, {price_year} US$, discounted")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=3)
    despine(ax, keep=("bottom",))
    no_grid_y(ax)
    fig.tight_layout()
    return fig


def person_years_panel(py_by_status):
    """Where the discounted person-time is spent."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order = [s for s in STATUS_LABEL if s in py_by_status]
    means = np.array([py_by_status[s].mean() for s in order])
    y = np.arange(len(order))
    ax.barh(y, means, color=BLUE, height=0.62, zorder=3)
    for i, v in enumerate(means):
        ax.text(v + means.max() * 0.015, i, f"{v:.2f}", va="center",
                fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([STATUS_LABEL[s] for s in order])
    ax.invert_yaxis()
    ax.set_xlim(0, max(means.max() * 1.16, 1.0))
    ax.set_xlabel("Discounted person-years per acquisition")
    despine(ax)
    no_grid_y(ax)
    fig.tight_layout()
    return fig


def prevention_frontier(nnts, prices, offset_mean, current=None):
    """Where prophylaxis pays for itself: NNT x annual cost against the
    lifetime treatment cost a prevented acquisition avoids."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    nnt_grid = np.linspace(min(nnts) * 0.5, max(nnts) * 1.35, 240)
    ax.plot(nnt_grid, offset_mean / nnt_grid, color=INK, lw=1.8, zorder=5,
            label="Cost-Neutral Frontier")
    ax.fill_between(nnt_grid, 0, offset_mean / nnt_grid, color=AQUA,
                    alpha=0.16, lw=0, zorder=1)
    ax.text(nnt_grid[len(nnt_grid) // 2], offset_mean / nnt_grid[len(nnt_grid) // 2] * 0.42,
            "Cost-Saving", color=INK2, fontsize=8.6, ha="center")

    for p, col in zip(prices, [SEQ[2], SEQ[6], SEQ[10], ORANGE][:len(prices)]):
        ax.axhline(p, color=col, lw=1.2, ls=":", zorder=3)
        ax.annotate(f"${p:,.0f}/yr", xy=(nnt_grid[-1], p), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=7.6, color=col)
    if current is not None:
        ax.plot([current[0]], [current[1]], "o", color=RED, ms=9, zorder=6,
                label="Current Selection")
    ax.set_xlabel("Number needed to treat, person-years of prophylaxis")
    ax.set_ylabel("All-in annual cost per person-year")
    ax.set_ylim(0, max(max(prices) * 1.45, 120))
    ax.set_xlim(nnt_grid[0], nnt_grid[-1])
    ax.yaxis.set_major_formatter(USD)
    ax.legend(loc="upper right")
    despine(ax)
    fig.tight_layout()
    return fig


def prevention_denominators(nnt, price, dalys_model, offset, literature, offset_share=1.0):
    """The same prophylaxis decision under this model's DALY denominator and
    the round figures the literature uses."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    entries = [("This Model", float(np.mean(dalys_model)), BLUE)]
    entries += [(lab, d, MUTED) for d, lab in literature]
    vals, labels, colours = [], [], []
    for lab, d, col in entries:
        vals.append(float(np.mean(E.prevention_icer(nnt, price, d, offset, offset_share))))
        labels.append(f"{lab}\n{d:.2f} DALYs")
        colours.append(col)
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colours, height=0.58, zorder=3)
    ax.axvline(0, color=AXIS, lw=1.0, zorder=4)
    span = max(abs(min(vals)), abs(max(vals))) or 1.0
    for i, v in enumerate(vals):
        ax.text(v + np.sign(v) * span * 0.03, i,
                "Cost-saving" if v < 0 else f"${v:,.0f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=8.2, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(min(0, min(vals)) - span * 0.30, max(0, max(vals)) + span * 0.30)
    ax.xaxis.set_major_formatter(USD)
    ax.set_xlabel(f"Incremental cost per DALY averted, at NNT {nnt:.0f} and ${price:,.0f}/yr")
    despine(ax)
    no_grid_y(ax)
    fig.tight_layout()
    return fig


def offset_sensitivity(nnt, price, dalys, offset):
    """How much of the answer rests on crediting the treatment cost avoided."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    shares = np.linspace(0, 1, 101)
    means = np.array([E.prevention_icer(nnt, price, dalys, offset, s).mean()
                      for s in shares])
    ax.plot(shares * 100, means, color=BLUE, zorder=4)
    ax.axhline(0, color=AXIS, lw=1.0, zorder=3)
    cross = np.interp(0, means[::-1], shares[::-1] * 100) if means.min() < 0 < means.max() else None
    if cross is not None:
        ax.plot([cross], [0], "o", color=RED, ms=7, zorder=5)
        ax.annotate(f"Cost-Neutral At {cross:.0f}% Credited", xy=(cross, 0),
                    xytext=(8, 10), textcoords="offset points",
                    fontsize=8.2, color=INK2)
    ax.set_xlabel("Share of avoided treatment cost credited, %")
    ax.set_ylabel("Cost per DALY averted")
    ax.yaxis.set_major_formatter(USD)
    ax.set_xlim(0, 100)
    despine(ax)
    fig.tight_layout()
    return fig


def cost_effectiveness_plane(out, threshold=None):
    """Treatment against no care, one point per posterior draw."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = out["DALYs_avoided_through_care"]
    y = out["cost_per_acquisition"]
    ax.scatter(x, y, s=7, color=BLUE, alpha=0.30, lw=0, zorder=3)
    ax.plot([x.mean()], [y.mean()], "o", color=INK, ms=8, zorder=5)
    if threshold:
        xs = np.linspace(0, x.max() * 1.06, 50)
        ax.plot(xs, xs * threshold, color=RED, lw=1.4, ls="--", zorder=4,
                label=f"Threshold, ${threshold:,.0f} per DALY")
        ax.legend(loc="upper left")
    ax.set_xlabel("DALYs averted through care, per acquisition")
    ax.set_ylabel("Lifetime cost per acquisition")
    ax.yaxis.set_major_formatter(USD)
    ax.set_xlim(0, max(x.max() * 1.06, 1.0))
    ax.set_ylim(0, max(y.max() * 1.10, 1.0))
    despine(ax)
    fig.tight_layout()
    return fig


def tornado(rows, base, xlabel):
    """One-way sensitivity: each input swung across its stated range."""
    fig, ax = plt.subplots(figsize=(6.4, max(2.6, 0.42 * len(rows) + 1.2)))
    if not rows:
        _blank(ax, "No inputs varied")
        return fig
    rows = sorted(rows, key=lambda r: abs(r[2] - r[1]))
    y = np.arange(len(rows))
    for i, (label, lo, hi) in enumerate(rows):
        left, right = min(lo, hi), max(lo, hi)
        ax.barh([i], [base - left], left=[left], color=ORANGE, height=0.6, zorder=3)
        ax.barh([i], [right - base], left=[base], color=BLUE, height=0.6, zorder=3)
    ax.axvline(base, color=INK, lw=1.2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(xlabel)
    despine(ax)
    no_grid_y(ax)
    fig.tight_layout()
    return fig


def cascade_comparison(base_out, alt_out, labels=("Calibrated", "Adjusted")):
    """Point estimates under adjusted cascade hazards against the calibrated
    posterior mean."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    keys = [("DALYs_per_acquisition", "DALYs per acquisition", ""),
            ("DALYs_avoided_through_care", "DALYs averted through care", ""),
            ("cost_per_acquisition", "Lifetime cost per acquisition", "$")]
    for ax, (k, lab, unit) in zip(axes, keys):
        vals = [float(np.mean(base_out[k])), float(alt_out[k])]
        ax.bar([0, 1], vals, color=[MUTED, BLUE], width=0.55, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, f"{unit}{v:,.2f}" if not unit else f"{unit}{v:,.0f}",
                    ha="center", fontsize=8.4, color=INK2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(vals) * 1.20)
        ax.set_ylabel(lab)
        despine(ax)
        no_grid_x(ax)
    fig.tight_layout()
    return fig

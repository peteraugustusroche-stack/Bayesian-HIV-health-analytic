# Interactive companion to the ESA HIV cascade model

Takes the calibrated posterior and lets a user reprice it: a cost schedule,
disability weights, the discount rate, and the two inputs a prophylaxis
decision needs. Cascade hazards can also be moved, which re-runs the Markov
engine rather than repricing.

## How it works

Cost, YLD and YLL are all **linear functionals of the cohort trajectory**. The
app therefore does not carry the Markov engine for the common case: it carries
the trajectory, collapsed onto the channels those inputs multiply, for all
1,000 posterior draws.

    py[draw, year, status, band, moment]   person-years, by cascade status and CD4
    hiv_death, hiv_death_yll, bg_death     death flows, as charged and as YLL sees them
    nh_py, nh_death                        the no-care counterfactual

Every slider except the cascade ones is then a contraction over that array —
a few milliseconds, whole posterior, no approximation.

**Within-year resolution.** Storing the trajectory monthly for 1,000 draws is
about 200 MB. Instead each year holds three moments of the within-year mass
distribution, `sum(mass * u^j)` for `u = month/12` and `j = 0,1,2`. Any smooth
within-year weight — the discount factor, or the discount factor times the
present value of remaining life expectancy — is recovered by fitting it at
`u = 0, 0.5, 1` and contracting against the three moments. The residual against
exact monthly evaluation is about 1e-8 relative, reported by
`export_for_app.py --check`.

**Cascade hazards** change the trajectory itself, which the tensor cannot
represent, so `live.py` runs the engine (about a quarter of a second) and wraps
the result as a one-draw occupancy that `engine.py` reprices through exactly
the same code path. Those figures are point estimates at the posterior mean and
carry no credible interval.

## Files

| file | what it does |
|---|---|
| `../export_for_app.py` | writes `data/occupancy_<year>.npz` from the calibration |
| `channels.py` | collapses a Markov run onto the stored channels |
| `engine.py` | reprices an occupancy tensor: cost, DALYs, ICERs |
| `live.py` | live cascade re-run at the posterior mean |
| `figures.py` | every figure, in the write-up's house style |
| `app.py` | the Shiny UI and reactive wiring |
| `selftest.py` | engine against the published posterior, draw for draw |
| `rendertest.py` | every figure rendered off-server, five input scenarios |
| `wiringcheck.py` | static check of the Shiny bindings |
| `package.py` | assembles a self-contained deployment bundle |

## Verifying

```bash
python3 app/selftest.py      # must reproduce run_posterior_analysis exactly
python3 app/rendertest.py    # every figure, including degenerate inputs
python3 app/wiringcheck.py   # no unbound outputs or misspelled inputs
```

`selftest.py` is the guard against the app and the report drifting apart: at
default inputs it checks all ten published metrics on both cascade arms,
draw for draw, to about 1e-8. Run it after any change to the model, the
calibration or the cost schedule.

## One thing to know about the model

A posterior draw whose SMR multiplier puts the top-band on-ART SMR below 1
produces a **negative** HIV-death hazard on treatment. `health_economics.py`
guards its YLL accumulation with `if death_flow > 0`, so those months are
skipped, but it charges terminal care against the unguarded flow. The two
channels are therefore stored separately and the app reproduces the published
behaviour on both. It affects roughly 40% of draws and moves mean YLL by about
0.16%.

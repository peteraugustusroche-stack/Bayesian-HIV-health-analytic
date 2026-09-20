# Lifetime DALY burden of an HIV acquisition under the ESA care cascade

Code archive accompanying the MSc research project. Two things live here: the model
that produced the reported estimates, and the interactive application that lets
someone re-run the economic layer under their own assumptions.

```
model/    the Bayesian-calibrated Markov cohort model, its calibration and
          analysis scripts, and the scripts that generate every figure and
          table in the report
app/      a self-contained deployable copy of the interactive application
          accessible through: https://01a0be68-efe1-30dd-96fe-1a9f95f21343.share.connect.posit.cloud/
```

## model/

A monthly-cycle Markov cohort model of one HIV acquisition in eastern and
southern Africa: 107 states (fifteen tagged cascade statuses across seven CD4
bands, plus two absorbing death states), competing risks resolved exactly.
Calibration is modular — a Bayesian "cut". Natural history (System 1) is
adopted from published schedules and never fitted, which leaves three
quantities free to serve as out-of-sample validation the model can fail. Nine
cascade transition hazards (System 2) are estimated by Markov chain Monte
Carlo against six targets, separately for the 2019 and 2024 cascades.

Run order, from a clean checkout:

```bash
cd model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 calibrate_system1.py          # adopted schedules, sampled from priors
python3 calibrate_system2.py          # the MCMC; roughly 90 minutes
python3 run_posterior_analysis.py     # DALYs, YLL, YLD and cost per acquisition
python3 scenario_analysis.py          # structural and one-way sensitivity
python3 prevention_analysis.py --md   # Tables P1 to P3
```

`run_all.sh` chains the whole sequence, including the two-arm calibration
experiment (UNAIDS-anchored vs retention-anchored); `run_2024.sh` does the same
for the contemporary cascade arm. Both expect the virtual environment at
`model/.venv` and create a `/tmp/hm` symlink, which the `writeup/` scripts use
as their import root.

Figures and tables are generated from the posterior rather than transcribed:
`writeup/fig_*.py` writes each figure, `writeup/table_inputs.py` writes the
input appendix with its provenance-gap list, and
`writeup/table_annex_hyle_style.py` writes the cost annex.

`model/app.py` is a separate, earlier Shiny dashboard that runs and displays
the calibration itself. It deliberately exposes no parameter inputs, so that
no result can reach the screen except by running the same code as the
command-line scripts. The what-if explorer is `app/`, below.

## app/

The application in `app/` answers a different question: what happens to the
estimate under someone else's assumptions. It exposes the cost schedule (the
ten unit costs, which statuses are charged routine care, the price year),
the disability weights, the discount rate, and the two inputs a prophylaxis
decision needs — number needed to treat and annual cost per person-year — and
reports the burden, the cost, and both the treatment and prevention ICERs with
credible intervals over the full posterior.

```bash
cd app
python3 selftest.py                    # must pass before the numbers mean anything
shiny run --reload app.py
```

The occupancy tensors are not in git. `app/data_source.py` uses a local copy if
there is one and otherwise downloads them once from the GitHub Release,
checksum-verified — so a fresh clone works without any manual data step. See
`app/DEPLOY.md` for hosting, including deploying straight from this repository.

The reason it is fast is that cost, YLD and YLL are all linear functionals of
the cohort trajectory. The application therefore does not carry the Markov
engine for the common case; it carries the trajectory for all 1,000 posterior
draws, collapsed onto the channels those inputs multiply, and every change is a
contraction over that array. Cascade hazards do change the trajectory, so those
sliders re-run the engine and return a point estimate at the posterior mean.

`selftest.py` is what keeps the application and the report from drifting apart:
at default inputs it reproduces every published metric on both cascade arms,
draw for draw, to about one part in a hundred million. `DEPLOY.md` covers
hosting; `README.md` inside `model/app/` documents the export format.

## Data availability

The calibrated posteriors and the exported trajectories are **not committed to
git** — they are large binary artefacts. They are attached to the GitHub
Release instead. Download and unpack them before running the analysis scripts
or the application:

| Artefact | Goes in | Needed for |
|---|---|---|
| `results.tar.gz` (~57 MB) | `model/data/results/` | the analysis scripts, without repeating the MCMC |
| `occupancy_2019.npz`, `occupancy_2024.npz` (~47 MB each) | `app/data/` and `model/app/data/` | the interactive application |

Without them, `calibrate_system1.py` and `calibrate_system2.py` regenerate
`model/data/results/` from scratch — roughly 90 minutes for the MCMC.

## Reproducing the exported posterior

The application reads `app/data/occupancy_<year>.npz`. Rebuild it after any
change to the model or the calibration:

```bash
cd model
python3 export_for_app.py 2019 --draws 1000 --check
python3 export_for_app.py 2024 --draws 1000 --check
python3 app/package.py --out ../app
```

`--check` reports the error of the within-year moment expansion against exact
monthly evaluation; it should stay around 1e-8.

## Provenance

Unit costs are adopted wholesale from a single source: Hyle EP, Maphosa T,
Rangaraj A, et al. Clinical impact and cost-effectiveness of the
WHO-recommended advanced HIV disease package of care. *Lancet Global Health*
2025;13(8):e1436-e1447, supplementary Table S6. Natural history follows the
schedules underlying UNAIDS Spectrum, so the model and its calibration targets
share a parameterisation rather than being derived independently. Every
parameter, its source and its price year is listed in Appendix B of the report,
generated by `model/writeup/table_inputs.py`, which also prints the inputs
whose provenance is unresolved.

## Licence

MIT licence, data used is covered by its own respective licence which may attract restrictions.

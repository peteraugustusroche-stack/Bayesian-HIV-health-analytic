"""Verification run: calibrate System 2 against data/parameters_updated.xlsx
(the updated file, NOT YET swapped into data/parameters.xlsx by the user),
to confirm the new CD4-at-ART-init targets are wired correctly end-to-end
and produce a sane fit. Does not overwrite the live data/results/system2.npz.
"""
import sys
sys.path.insert(0, '/sessions/modest-sleepy-newton/mnt/hiv_model')

from pathlib import Path
from params import load_parameters
from calibration_common import load_result
import calibrate_system2
import likelihoods as lk

UPDATED_PATH = Path('/sessions/modest-sleepy-newton/mnt/hiv_model/data/parameters_updated.xlsx')
params = load_parameters(UPDATED_PATH)

print("Targets loaded (n={}):".format(len(params.targets)))
for k, v in params.targets.items():
    print(" ", k, v)

# run_system2_calibration() calls load_parameters() internally with no args
# (i.e. the live default data/parameters.xlsx). Monkeypatch it for this
# verification run only, so it picks up the updated file with the new CD4
# targets, without touching the live default file the user hasn't swapped in.
calibrate_system2.load_parameters = lambda: load_parameters(UPDATED_PATH)

s1 = load_result("system1")
if s1 is None:
    raise SystemExit("No System 1 result cached.")

print("\nRunning System 2 calibration against updated params (reduced draws for speed)...")
result = calibrate_system2.run_system2_calibration(
    s1, n_outer_draws=6, draws_per_outer=200, tune_per_outer=150, progress_every=2)

print(f"\nBackend: {result['backend']}  outer draws: {result['n_outer_draws']}")
print("\nModel-implied (posterior mean) vs targets:")
for key, (central, sd) in result["targets"].items():
    implied = result["model_implied_at_mean"][key]
    print(f"  {key}: implied={implied:.3f}  target={central} +/- {sd}")

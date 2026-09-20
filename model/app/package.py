"""Assemble a self-contained deployment bundle for shinyapps.io / Posit Connect.

rsconnect uploads one directory, so the model modules the live cascade re-run
needs are copied in beside the app rather than imported from the repository.

    python3 app/package.py [--out DIR] [--clean]

Then, with an account configured (see app/DEPLOY.md):

    rsconnect deploy shiny DIR --entrypoint app:app --title "HIV cascade model"
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "app"

#: Modules the live cascade re-run imports, transitively. Verified by importing
#: the app and listing every repository module that ends up in sys.modules.
MODEL_MODULES = [
    "acquisition_cd4.py", "hazards.py", "health_econ_params.py",
    "health_economics.py", "life_table.py", "natural_history_params.py",
    "params.py", "simulate_cascade.py", "simulate_natural_history.py",
    "states.py",
]
APP_FILES = ["app.py", "engine.py", "figures.py", "figstyle.py", "channels.py",
             "live.py", "data_source.py", "selftest.py", "rendertest.py",
             "requirements.txt", "DEPLOY.md"]
DATA_FILES = ["occupancy_2019.npz", "occupancy_2024.npz"]


def write_reference_summary(path: Path) -> None:
    """Freeze the published posterior summaries into the bundle, so selftest.py
    can check the app against the report without the repository present."""
    import json
    sys.path.insert(0, str(REPO))
    from run_posterior_analysis import load_posterior_result
    ref = {}
    for year, price_year in (("2019", 2019), ("2024", 2024)):
        res = load_posterior_result(year)
        if res is None:
            continue
        ref[year] = {"price_year": price_year,
                     "metrics": {k: [float(v.mean()), float(v.std())]
                                 for k, v in res["probabilistic"]["metrics"].items()}}
    path.write_text(json.dumps(ref, indent=1))


def _copy(src: Path, dst: Path) -> None:
    """Overwrite by truncation rather than replacement, so the bundle can be
    rebuilt in a directory where deleting is not permitted."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def build(out: Path, clean: bool = False) -> Path:
    if clean and out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True, exist_ok=True)

    for name in APP_FILES:
        src = APP / name
        if src.exists():
            _copy(src, out / name)
    for name in MODEL_MODULES:
        _copy(REPO / name, out / name)
    for name in DATA_FILES:
        src = APP / "data" / name
        if not src.exists():
            raise SystemExit(f"missing {src}; run export_for_app.py first")
        _copy(src, out / "data" / name)

    # params.load_parameters() reads this workbook at import time.
    _copy(REPO / "data" / "parameters.xlsx", out / "data" / "parameters.xlsx")
    write_reference_summary(out / "data" / "reference_summary.json")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"bundle at {out}  ({size / 1e6:.0f} MB, "
          f"{sum(1 for f in out.rglob('*') if f.is_file())} files)")
    print("\nverify, then deploy:")
    print(f"  python3 {out}/selftest.py")
    print(f"  shiny run {out}/app.py")
    print(f"  rsconnect deploy shiny {out} --entrypoint app:app "
          f"--title 'HIV cascade model'")
    return out


if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else REPO / "dist" / "hiv_cascade_app"
    build(out.resolve(), clean="--clean" in sys.argv)

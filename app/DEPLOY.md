# Running and deploying

## Where the data comes from

The occupancy tensors are ~47 MB each and are not in git. `data_source.py`
resolves them in this order, so you rarely have to think about it:

1. `$HIV_APP_DATA_DIR/occupancy_<year>.npz`, if that variable is set
2. `app/data/occupancy_<year>.npz` — a local build, or a CLI-packaged deploy
3. the cache directory, filled by downloading from the GitHub Release

A checkout with the data already in place never touches the network. A
deploy-from-repo downloads once per container and serves from cache after that.
Downloads are checksum-verified and written atomically.

**Before the first deploy-from-repo**, point it at your release — either set
`HIV_APP_DATA_URL` in the host's environment, or edit `DEFAULT_BASE_URL` in
`data_source.py`:

```
https://github.com/<owner>/<repo>/releases/download/v1.0
```

Re-exporting the tensors changes their digests, so update `EXPECTED_SHA256`
(from `sha256sum`) at the same time or the download will be rejected.

```bash
python3 data_source.py            # resolve both arms, reporting where each came from
```

## Locally

```bash
pip install -r requirements.txt
python3 selftest.py        # engine against the published posterior
python3 rendertest.py      # every figure, off-server
shiny run --reload app.py  # http://127.0.0.1:8000
```

`selftest.py` is the one to run after any change to the cost schedule or the
engine: it checks that at default inputs the app reproduces
`run_posterior_analysis` draw for draw, so the app and the report cannot drift
apart silently.

## Rebuilding the posterior export

The app reads `data/occupancy_<year>.npz`, written by `export_for_app.py` in
the repository root. Re-run it after any change to the model, the calibration
or the natural-history schedules:

```bash
python3 export_for_app.py 2019 --draws 1000 --check
python3 export_for_app.py 2024 --draws 1000 --check
python3 app/package.py
```

`--check` reports the error of the three-moment within-year expansion against
exact monthly evaluation; it should stay around 1e-8.

## Deploying to shinyapps.io

```bash
pip install rsconnect-python
rsconnect add --account <ACCOUNT> --name shinyapps \
              --token <TOKEN> --secret <SECRET>
python3 package.py                      # writes ../dist/hiv_cascade_app
rsconnect deploy shiny ../dist/hiv_cascade_app \
          --entrypoint app:app --title "HIV cascade model"
```

Tokens come from shinyapps.io under Account, Tokens.

**Instance size.** Each cascade arm's occupancy tensor is about 50 MB on disk
and about 200 MB once loaded, and the app caches the arms it has been asked
for. A 1 GB instance holds one arm comfortably and both only just; if the app
restarts when the cascade year is switched, move it to a 2 GB instance
(shinyapps.io: Application, Settings, Instance Size). Posit Connect has no
such default limit.

**Cold start.** Loading an arm takes a few seconds. The first request after a
deploy or an idle timeout will be slow; subsequent ones are immediate, since
every input except the cascade sliders is a contraction over an array already
in memory.

## Deploying to Posit Connect

```bash
rsconnect add --server https://<CONNECT-HOST> --name connect --api-key <KEY>
rsconnect deploy shiny ../dist/hiv_cascade_app --entrypoint app:app
```

## Deploying from the GitHub repository

Posit Connect Cloud and Hugging Face Spaces both deploy straight from a repo
and redeploy on push, which is the tidiest way to keep a public link current.
Neither needs `package.py`: the repo already has the layout, and the tensors
arrive by download rather than being bundled.

- Entry point `app/app.py`, requirements `app/requirements.txt`.
- Set `HIV_APP_DATA_URL` in the host's environment (or edit `DEFAULT_BASE_URL`).
- The repository must be **public** for Connect Cloud to read it. The release
  assets are public once the repo is.

**Memory.** Each arm is ~55 MB of arrays and about 200 MB once working copies
are allocated; the app caches the arms it is asked for. 1 GB holds one arm
comfortably and both only just. Hugging Face's free CPU tier has considerably
more headroom than shinyapps.io's free tier.

**Cold start** is now a download plus a load on the very first request after a
container starts, rather than a load alone. Subsequent requests are immediate.

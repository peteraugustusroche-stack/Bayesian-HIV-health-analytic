# Pushing this to GitHub

The repository is already initialised and committed locally at
`~/Documents/hiv-daly-esa` — one commit on branch `main`, 97 files, 4.1 MB.
Nothing has been sent anywhere. These are the remaining steps, all run by you
so your credentials stay yours.

Delete this file once you've pushed; it isn't part of the archive.

---

## 1. Create the empty private repository

On github.com: **New repository** → name `hiv-daly-esa` (or whatever you
prefer) → **Private** → do **not** add a README, .gitignore or licence, since
the local repo already has them.

## 2. Push

```bash
cd ~/Documents/hiv-daly-esa
git remote add origin git@github.com:<your-username>/hiv-daly-esa.git
git push -u origin main
```

Use the `https://github.com/...` URL instead of `git@github.com:...` if you
aren't set up with SSH keys; it will prompt for a personal access token rather
than a password.

## 3. Attach the large files as a Release

The calibrated posteriors and exported trajectories are deliberately not in
git. They are staged for you at `~/Documents/hiv-daly-esa-release-assets/`:

| File | Size |
|---|---|
| `results.tar.gz` | 18 MB |
| `occupancy_2019.npz` | 47 MB |
| `occupancy_2024.npz` | 47 MB |

On github.com: **Releases** → **Create a new release** → tag `v1.0` → attach
all three files → publish.

Or with the GitHub CLI:

```bash
cd ~/Documents/hiv-daly-esa
gh release create v1.0 \
  ~/Documents/hiv-daly-esa-release-assets/results.tar.gz \
  ~/Documents/hiv-daly-esa-release-assets/occupancy_2019.npz \
  ~/Documents/hiv-daly-esa-release-assets/occupancy_2024.npz \
  --title "MSc submission archive" \
  --notes "Calibrated posteriors and exported trajectories accompanying the model code."
```

## 4. Point the app at your release

`app/data_source.py` downloads the occupancy tensors on first use. Set the
prefix once, replacing the placeholder:

```bash
cd ~/Documents/hiv-daly-esa
sed -i '' 's|OWNER/REPO|<your-username>/hiv-daly-esa|' app/data_source.py model/app/data_source.py
git commit -am "Point data_source at the release"
```

Or leave the file alone and set `HIV_APP_DATA_URL` in the host's environment
instead — useful if you'd rather not bake a URL into the code.

## 5. Check it round-trips

Worth doing once, from a clone rather than the original — this exercises the
download path exactly as a fresh deploy would:

```bash
cd /tmp && git clone git@github.com:<your-username>/hiv-daly-esa.git
cd hiv-daly-esa
tar xzf ~/Documents/hiv-daly-esa-release-assets/results.tar.gz -C model/

cd app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 data_source.py     # downloads both arms, checksum-verified
python3 selftest.py        # should reproduce the published metrics
```

Note there is no longer any need to copy the `.npz` by hand.

## 6. Publish the live app

Deploy straight from the repo (Posit Connect Cloud, or Hugging Face Spaces),
entry point `app/app.py`, requirements `app/requirements.txt`. Both redeploy on
push. `app/DEPLOY.md` covers this, the shinyapps.io CLI route, and the memory
sizing. Then put the link at the top of `README.md`.

Connect Cloud needs the repository to be **public**, which also makes the
dissertation's results public — a separate decision from the repo itself.

---

## Two things to decide before this goes public

**Licence.** There is no `LICENSE` file. Add one before flipping the repo to
public — MIT or BSD-3 for permissive, CC-BY-4.0 if you want attribution on the
model itself, or GPL-3.0 to keep derivatives open.

**Citation.** Consider a `CITATION.cff` so GitHub renders a "Cite this
repository" button, and a Zenodo DOI if you want the archive citable from the
dissertation or the JIAS paper.

## What was left out, and why

| Excluded | Reason |
|---|---|
| `.venv/`, `cyclical/.venv/` | environments, ~300 MB |
| `cyclical/` | the parked cyclical-cascade model — a different project |
| `Draft*.docx`, `HIV_model_writeup*.docx` | unpublished manuscript drafts |
| `data/results_backup_*/` | dated MCMC backups |
| `.*.bak_*`, `*.tar.gz`, `__pycache__`, `.DS_Store` | working scratch |
| `model/data/results/`, `*.npz` | large generated artefacts — see the Release |

`model/app/` is kept despite duplicating most of `app/`, because
`model/export_for_app.py` puts it on `sys.path` at line 33 and imports
`channels`/`engine` from it. Removing it breaks the export step.

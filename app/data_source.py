"""Locate the exported occupancy tensors, downloading them on first use.

The tensors are ~47 MB each and are not committed to git, so a clone or a
deploy-from-repo has the code but not the data. This module resolves that: if
the file is already present it is used as-is, otherwise it is fetched once from
the GitHub Release and cached.

Resolution order for each arm:

1. ``$HIV_APP_DATA_DIR/occupancy_<year>.npz``  (if that variable is set)
2. ``app/data/occupancy_<year>.npz``           (a local build or a CLI deploy)
3. the cache directory, populated by downloading from ``base_url()``

So a local checkout with the data in place never touches the network, and a
deploy-from-repo downloads once per container and serves from cache thereafter.

Downloads are verified against the SHA-256 digests below and written
atomically, so a truncated or interrupted transfer cannot leave a corrupt file
that later loads as garbage.

Standard library only -- adding this must not add a dependency.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

#: Where the release assets live. Override with HIV_APP_DATA_URL, which may be
#: any prefix that ``<prefix>/occupancy_<year>.npz`` resolves under -- a GitHub
#: Release download URL, an S3 bucket, or a plain static host.
DEFAULT_BASE_URL = "https://github.com/peteraugustusroche-stack/Bayesian-HIV-health-analytic/releases/download/v1.0"

#: Expected digests, so a partial download fails loudly rather than silently.
#: Regenerate with `sha256sum` after any re-export.
EXPECTED_SHA256 = {
    "2019": "7b1111b0514eb2b4d35694d0e9380022230c568f514c5a509f7aee8e44fe53be",
    "2024": "1a6682b90bc74e5504b9a1a612c6e7f5d8b1685e9c6da63e0b4d0e29bf86f15f",
}

EXPECTED_BYTES = {"2019": 49380653, "2024": 49386016}


def base_url() -> str:
    return os.environ.get("HIV_APP_DATA_URL", DEFAULT_BASE_URL).rstrip("/")


def cache_dir() -> Path:
    """Where downloads are kept. Ephemeral on most hosts, which is fine: the
    cost of a cold start is one download per container."""
    override = os.environ.get("HIV_APP_CACHE_DIR")
    if override:
        return Path(override)
    root = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(root) / "hiv-daly-esa"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, cascade_year: str) -> None:
    expected = EXPECTED_SHA256.get(cascade_year)
    if expected is None:
        return
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{path.name} failed its checksum (expected {expected[:12]}..., "
            f"got {actual[:12]}...). Delete it and retry; if it persists the "
            "release asset and EXPECTED_SHA256 have diverged."
        )


def _download(cascade_year: str, dest: Path) -> None:
    url = f"{base_url()}/occupancy_{cascade_year}.npz"
    if "OWNER/REPO" in url:
        raise RuntimeError(
            "No download location configured. Set HIV_APP_DATA_URL to the "
            "release asset prefix, or edit DEFAULT_BASE_URL in data_source.py. "
            f"Alternatively place occupancy_{cascade_year}.npz in {DATA_DIR}."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkstemp(dir=dest.parent, suffix=".part")[1])
    try:
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        _verify(tmp, cascade_year)
        tmp.replace(dest)          # atomic, so readers never see a partial file
    finally:
        tmp.unlink(missing_ok=True)


def ensure_occupancy(cascade_year: str) -> Path:
    """Path to the occupancy tensor for ``cascade_year``, downloading it if it
    is not already on disk."""
    name = f"occupancy_{cascade_year}.npz"

    override = os.environ.get("HIV_APP_DATA_DIR")
    if override and (Path(override) / name).exists():
        return Path(override) / name

    local = DATA_DIR / name
    if local.exists():
        return local

    cached = cache_dir() / name
    if not cached.exists():
        _download(cascade_year, cached)
    return cached


if __name__ == "__main__":
    import sys

    years = sys.argv[1:] or ["2019", "2024"]
    for y in years:
        path = ensure_occupancy(y)
        print(f"{y}: {path}  ({path.stat().st_size / 1048576:.1f} MB)")

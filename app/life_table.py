"""Cause-deleted (non-HIV) background mortality for eastern & southern Africa.

Supplies background_hazard(age) and remaining_life_expectancy(age) from the SAME
table, so an HIV death costs exactly the non-HIV life the person would otherwise
have had. Source, method and caveats: PARAMETERS.md.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

LIFE_TABLE_PATH = Path(__file__).parent / "bg_life_table_provenance.csv"

# Age beyond which the table is extrapolated by holding the final hazard
# constant. The table runs to 100; the model horizon ends around 73, so
# this only ever matters for the e_x integration tail.
_MAX_TABLE_AGE = 100


def _load_table(path: Path = LIFE_TABLE_PATH):
    """Returns (ages, qx_non_hiv, qx_all_cause, hiv_share) as parallel arrays."""
    ages, qn, qa, share = [], [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            ages.append(int(row["age"]))
            qn.append(float(row["qx_non_hiv"]))
            qa.append(float(row["qx_all_cause"]))
            share.append(float(row["hiv_share_of_all_cause"]))
    order = np.argsort(ages)
    return (np.array(ages)[order], np.array(qn)[order],
            np.array(qa)[order], np.array(share)[order])


AGES, QX_NON_HIV, QX_ALL_CAUSE, HIV_SHARE = _load_table()

# qx (annual probability) -> annual hazard, assuming a constant force of
# mortality within each single year of age: q = 1 - exp(-m)  =>  m = -ln(1-q).
# This is the correct conversion for feeding a competing-risks step that
# itself works in hazards, and it matters at older ages where qx is large.
HAZARD_NON_HIV = -np.log(1.0 - QX_NON_HIV)

_AGE_MIN, _AGE_MAX = int(AGES[0]), int(AGES[-1])


def background_hazard(age: float) -> float:
    """Annual non-HIV (cause-deleted) death hazard at `age`.

    Piecewise-constant within each single year of age, which is the
    resolution of the underlying data -- there is no information below one
    year, so interpolating to monthly granularity would be false precision.
    Ages outside the table are clamped to its first/last value.
    """
    a = int(math.floor(age))
    if a < _AGE_MIN:
        a = _AGE_MIN
    elif a > _AGE_MAX:
        a = _AGE_MAX
    return float(HAZARD_NON_HIV[a - _AGE_MIN])


@lru_cache(maxsize=256)
def remaining_life_expectancy(age: int) -> float:
    """Cause-deleted remaining life expectancy e_x at integer `age`, i.e. the
    average number of further years a person of this age would live if HIV
    did not exist and they faced only the non-HIV schedule.

    Standard life-table construction with the usual half-year-in-year-of-
    death correction (a_x = 0.5). This is the YLL reference: an HIV death at
    age x costs e_x years, not `72 - x`.

    Why this replaces the previous fixed 72-year anchor: `72 - age_at_death`
    collapses toward zero as age rises (a death at 70 was scored as losing
    2 years), whereas true remaining non-HIV expectancy at 70 is ~11 years.
    Because deaths in the treated arm are concentrated at older ages, the
    old convention systematically understated YLL in the cascade arm and
    therefore OVERSTATED "DALYs avoided through care".
    """
    a = max(_AGE_MIN, min(int(age), _AGE_MAX))
    lx, total = 1.0, 0.0
    for x in range(a, _AGE_MAX + 1):
        q = float(QX_NON_HIV[x - _AGE_MIN])
        total += lx * (1.0 - q / 2.0)
        lx *= (1.0 - q)
    return total


def remaining_life_expectancy_interp(age: float) -> float:
    """Linear interpolation of e_x between integer ages, so YLL varies
    smoothly across a monthly model step rather than jumping once a year."""
    lo = int(math.floor(age))
    frac = age - lo
    if frac <= 0:
        return remaining_life_expectancy(lo)
    return ((1.0 - frac) * remaining_life_expectancy(lo)
            + frac * remaining_life_expectancy(lo + 1))


def survival_between(age_from: float, age_to: float) -> float:
    """Non-HIV survival probability from `age_from` to `age_to`. Diagnostic
    helper -- not used by the engines, but handy for sanity checks and for
    the methods write-up."""
    lo, hi = int(math.floor(age_from)), int(math.floor(age_to))
    s = 1.0
    for x in range(lo, hi):
        s *= (1.0 - float(QX_NON_HIV[max(_AGE_MIN, min(x, _AGE_MAX)) - _AGE_MIN]))
    return s


if __name__ == "__main__":
    print(f"Loaded {len(AGES)} ages ({_AGE_MIN}-{_AGE_MAX}) from {LIFE_TABLE_PATH.name}")
    print(f"HIV share of all-cause mortality peaks at age "
          f"{int(AGES[int(np.argmax(HIV_SHARE))])} = {HIV_SHARE.max():.1%}\n")
    print(f"{'age':>5}{'qx_nonHIV':>12}{'hazard':>10}{'e_x':>9}{'age at death':>14}")
    for a in (28, 35, 45, 55, 65, 72, 80):
        print(f"{a:>5}{float(QX_NON_HIV[a - _AGE_MIN]):>12.5f}"
              f"{background_hazard(a):>10.5f}{remaining_life_expectancy(a):>9.2f}"
              f"{a + remaining_life_expectancy(a):>14.1f}")
    print(f"\nNon-HIV survival age 28 -> 73: {survival_between(28, 73):.4f}"
          f"   (flat 0.006/yr placeholder gave {math.exp(-0.006 * 45):.4f})")

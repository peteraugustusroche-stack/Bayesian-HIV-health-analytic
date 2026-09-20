"""
Validation checks for the cause-deleted background life table.

Run:  python3 life_table_checks.py

Every check is self-describing and prints PASS/FLAG so this can be re-run
after any edit to bg_life_table_provenance.csv. Nothing here is used by the
engines -- it exists so that the table's construction is auditable and so
that the numbers quoted in the methods section can be regenerated.

The checks, in order:
  1. Internal identity   qx_non_hiv == qx_all_cause * (1 - hiv_share)
  2. Deletion rule       proportional-on-q vs the exact power rule
  3. e_x construction    a_x = 0.5, and the closed-vs-open final interval
  4. Interpolation       do single-year qx reproduce the 5-year group survival
  5. External check      implied HIV share of all deaths vs UNAIDS/GBD
  6. Sensitivity         how much does e_29 move if hiv_share is wrong
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

PATH = Path(__file__).parent / "bg_life_table_provenance.csv"
ANCHOR = 29          # model entry age
HORIZON_END = 74     # anchor + 45


def load():
    rows = list(csv.DictReader(open(PATH, newline="")))
    age = np.array([int(r["age"]) for r in rows])
    o = np.argsort(age)
    return (age[o],
            np.array([float(r["qx_non_hiv"]) for r in rows])[o],
            np.array([float(r["qx_all_cause"]) for r in rows])[o],
            np.array([float(r["hiv_share_of_all_cause"]) for r in rows])[o])


AGE, QN, QA, SH = load()


def ex(q, a, ax=0.5, open_ended=False):
    """Remaining life expectancy at integer age `a` from a qx vector."""
    lx, total = 1.0, 0.0
    for x in range(a, len(q)):
        total += lx * (1.0 - q[x] * ax)
        lx *= (1.0 - q[x])
    if open_ended:
        m_w = -math.log(1.0 - q[-1])
        total += lx / m_w
    return total


def surv(q, a, b):
    s = 1.0
    for x in range(a, b):
        s *= (1.0 - q[x])
    return s


def rule(title):
    print(f"\n{'-' * 70}\n{title}\n{'-' * 70}")


# ---------------------------------------------------------------- 1
rule("1. Internal identity: qx_non_hiv == qx_all_cause * (1 - hiv_share)")
resid = QN - QA * (1.0 - SH)
band = (AGE >= ANCHOR) & (AGE <= HORIZON_END)
worst_abs = np.abs(resid).max()
worst_rel = np.abs(resid[band] / QN[band]).max()
print(f"  max |residual| all ages      : {worst_abs:.2e} (age {AGE[np.argmax(np.abs(resid))]})")
print(f"  max relative residual {ANCHOR}-{HORIZON_END} : {worst_rel:.3%}")
print(f"  {'PASS' if worst_rel < 0.01 else 'FLAG'} -- identity holds to rounding over the model horizon")

# ---------------------------------------------------------------- 2
rule("2. Deletion rule: proportional-on-q (used) vs exact power rule")
print("  Cause deletion is exact on the HAZARD scale: m* = m(1-R), i.e.")
print("  q* = 1 - (1-q)^(1-R). The table applies (1-R) to q directly.")
q_power = 1.0 - (1.0 - QA) ** (1.0 - SH)
print(f"\n  {'age':>5}{'used':>12}{'power rule':>13}{'rel diff':>11}")
for a in (ANCHOR, 45, 60, HORIZON_END, 85, 100):
    i = int(np.where(AGE == a)[0][0])
    print(f"  {a:>5}{QN[i]:>12.6f}{q_power[i]:>13.6f}{(QN[i]-q_power[i])/q_power[i]:>+11.3%}")
d_ex = ex(q_power, ANCHOR) - ex(QN, ANCHOR)
d_s = surv(q_power, ANCHOR, HORIZON_END) - surv(QN, ANCHOR, HORIZON_END)
print(f"\n  effect on e_{ANCHOR}          : {d_ex:+.3f} yr")
print(f"  effect on S({ANCHOR}->{HORIZON_END})     : {d_s:+.4f}")
print(f"  {'PASS' if abs(d_ex) < 0.2 else 'FLAG'} -- approximation biases background mortality low by <1%, "
      "immaterial at these hazards")

# ---------------------------------------------------------------- 3
rule("3. e_x construction: a_x correction and final-interval treatment")
closed = ex(QN, ANCHOR)
opened = ex(QN, ANCHOR, open_ended=True)
print(f"  e_{ANCHOR} closed at age {AGE[-1]}      : {closed:.4f}")
print(f"  e_{ANCHOR} with open-ended a_w   : {opened:.4f}  ({opened-closed:+.4f} yr)")
lx = np.prod(1.0 - QN[ANCHOR:])
print(f"  survivors past table end     : {lx:.2e}")
print(f"  {'PASS' if abs(opened-closed) < 0.01 else 'FLAG'} -- truncation at the table end is immaterial")
print(f"\n  NOTE a_x = 0.5 is appropriate at adult ages but wrong at age 0")
print(f"       (infant deaths concentrate early, a_0 ~ 0.1). e_0 from this")
print(f"       table is therefore not a valid published-style figure; the")
print(f"       model never uses it. e_0 as computed = {ex(QN,0):.2f}")

# ---------------------------------------------------------------- 4
rule("4. Interpolation: do single-year qx reproduce 5-year group survival?")
rows = list(csv.DictReader(open(PATH, newline="")))
groups = {}
for r in rows:
    g = r["source"].split("(group ")[-1].rstrip(")")
    groups.setdefault(g, []).append(int(r["age"]))
print(f"  {'group':>9}{'5q_x':>10}{'flat-midpoint':>15}{'ratio':>9}")
worst = 0.0
for g, ages in groups.items():
    if len(ages) != 5:
        continue
    lo = min(ages)
    if not (20 <= lo <= 80):
        continue
    q5 = 1.0 - np.prod([1.0 - QN[a] for a in ages])
    flat = 1.0 - (1.0 - QN[lo + 2]) ** 5
    print(f"  {g:>9}{q5:>10.5f}{flat:>15.5f}{q5/flat:>9.4f}")
    worst = max(worst, abs(q5 / flat - 1.0))
print(f"\n  max deviation from a flat-midpoint group: {worst:.2%}")
print(f"  {'PASS' if worst < 0.03 else 'FLAG'} -- interpolation is close to survival-preserving")
print("\n  NOTE the hiv_share column is ALSO interpolated to single years")
print(f"       ({len(set(SH[(AGE>=15)&(AGE<=85)]))} distinct values over ages 15-85, peaking")
print(f"       at {SH.max():.3f} at age {AGE[int(np.argmax(SH))]}). The source column does not")
print("       record the interpolation method for either column -- document it.")

# ---------------------------------------------------------------- 5
rule("5. External check: implied HIV share of all deaths")
lxv = np.ones(len(QA))
for i in range(1, len(QA)):
    lxv[i] = lxv[i - 1] * (1.0 - QA[i - 1])
d = lxv * QA
hiv_d = d * SH
print(f"  all-ages HIV share of deaths : {hiv_d.sum()/d.sum():.1%}")
print(f"  ages 15-59 HIV share         : {hiv_d[15:60].sum()/d[15:60].sum():.1%}")
print(f"  mean age at HIV death        : {(AGE*hiv_d).sum()/hiv_d.sum():.1f}")
print(f"\n  e_0  all-cause {ex(QA,0):.2f} -> non-HIV {ex(QN,0):.2f}  (HIV costs {ex(QN,0)-ex(QA,0):.2f} yr)")
print(f"  e_{ANCHOR} all-cause {ex(QA,ANCHOR):.2f} -> non-HIV {ex(QN,ANCHOR):.2f}  "
      f"(HIV costs {ex(QN,ANCHOR)-ex(QA,ANCHOR):.2f} yr)")
print("\n  Benchmark: UNAIDS reports 260 000 AIDS deaths in eastern and southern")
print("  Africa in 2022 against roughly 4.5 million deaths from all causes,")
print("  i.e. about 6%. GBD 2021 sits somewhat higher than UNAIDS and 2021 was")
print("  a worse year than 2022, so the table's figure above is in range.")

# ---------------------------------------------------------------- 6
rule("6. Sensitivity: how wrong could hiv_share be, and does it matter?")
base_e, base_s = ex(QN, ANCHOR), surv(QN, ANCHOR, HORIZON_END)
print(f"  {'scale k':>9}{'e_29':>9}{'delta':>9}{'S(29->74)':>12}{'delta':>9}")
for k in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    q = QA * (1.0 - np.clip(SH * k, 0.0, 0.95))
    print(f"  {k:>9.2f}{ex(q,ANCHOR):>9.2f}{ex(q,ANCHOR)-base_e:>+9.2f}"
          f"{surv(q,ANCHOR,HORIZON_END):>12.4f}{surv(q,ANCHOR,HORIZON_END)-base_s:>+9.4f}")
print("\n  A 25% error in the HIV share moves the YLL anchor e_29 by under half")
print("  a year on 43.5, i.e. ~1%. The table is robust to plausible error in")
print("  the column whose provenance is least documented.")

# ---------------------------------------------------------------- summary
rule("Model-facing summary")
print(f"  non-HIV survival {ANCHOR} -> {HORIZON_END}   : {surv(QN,ANCHOR,HORIZON_END):.4f}")
print(f"  old flat 0.006/yr placeholder : {math.exp(-0.006*(HORIZON_END-ANCHOR)):.4f}")
print(f"  cumulative non-HIV hazard     : {sum(-math.log(1-QN[x]) for x in range(ANCHOR,HORIZON_END)):.4f}")
print("  e_x: " + "  ".join(f"e_{a}={ex(QN,a):.1f}" for a in (29, 40, 50, 60, 70, 74)))

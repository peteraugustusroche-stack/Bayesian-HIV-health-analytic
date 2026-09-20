"""Parameter loader. Reads free / fixed / target parameters from
data/parameters.xlsx; constants below are the fixed inputs that are not in it.

Sourcing and derivations: PARAMETERS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl

DATA_PATH = Path(__file__).parent / "data" / "parameters.xlsx"

# CD4 bands, ordered from healthiest (0) to most advanced disease (6).
CD4_BAND_NAMES = ["≥500", "350-499", "250-349", "200-249", "100-199", "50-99", "<50"]
N_CD4_BANDS = len(CD4_BAND_NAMES)

# Band width in cells/µL, used only to convert cells/yr reconstitution into a
# band-crossing hazard.
CD4_BAND_WIDTH = [None, 150, 100, 50, 100, 50, 50]

# CD4<350 is first reached on entry to band index 2 (250-349).
# CD4<200 is first reached on entry to band index 4 (100-199).
BAND_INDEX_CD4_350 = 2
BAND_INDEX_CD4_200 = 4

# Band edges, for interpolating continuous CD4 percentiles from the discrete
# band distribution. The open top is an ASSUMPTION, not sourced data.
CD4_BAND_OPEN_TOP_ASSUMED = 1500.0
CD4_BAND_LOWER = [500.0, 350.0, 250.0, 200.0, 100.0, 50.0, 0.0]
CD4_BAND_UPPER = [CD4_BAND_OPEN_TOP_ASSUMED, 500.0, 350.0, 250.0, 200.0, 100.0, 50.0]


def _parse_number(value):
    """Pull the first float out of strings like '579 cells/µL' or '0.93 (93%)'."""
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    match = re.search(r"-?\d+\.?\d*", str(value))
    return float(match.group()) if match else None


def _parse_range(value):
    """Pull two floats out of strings like '0.002 to 0.712'."""
    nums = re.findall(r"-?\d+\.?\d*", str(value))
    return [float(n) for n in nums]


@dataclass
class FreeParam:
    symbol: str
    name: str
    central: float
    sd: float


@dataclass
class Parameters:
    free: dict = field(default_factory=dict)          # symbol -> FreeParam
    fixed: dict = field(default_factory=dict)          # symbol -> parsed value(s)
    targets: dict = field(default_factory=dict)        # symbol -> (central, sd)
    raw_rows: list = field(default_factory=list)       # full rows for reference

    def central_estimates(self) -> dict:
        """Convenience dict of symbol -> central value for a deterministic run."""
        return {sym: p.central for sym, p in self.free.items()}

    def symbol_names(self) -> dict:
        """symbol -> descriptive spreadsheet name, for every row (free, fixed,
        and target alike). Used by the UI to label plots/tables with a human
        -readable name rather than a bare Greek-letter symbol."""
        return {row["Symbol"]: row["Parameter / Target Name"] for row in self.raw_rows
                if row.get("Symbol")}


def load_parameters(path: Path = DATA_PATH) -> Parameters:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    header, rows = rows[0], rows[1:]

    params = Parameters()
    for row in rows:
        system, category, name, symbol, ptype, central, sd, desc, source = row
        params.raw_rows.append(dict(zip(header, row)))

        if ptype == "Free":
            params.free[symbol] = FreeParam(
                symbol=symbol, name=name,
                central=_parse_number(central), sd=_parse_number(sd),
            )
        elif ptype == "Fixed":
            # Some fixed inputs are single values, some ranges/pairs (mortality
            # by CD4 band, fast/slow reconstitution).
            nums = _parse_range(central)
            params.fixed[symbol] = nums[0] if len(nums) == 1 else nums
        elif ptype == "Target":
            params.targets[symbol] = (_parse_number(central), _parse_number(sd))

    return params


# ---------------------------------------------------------------------------
# Fixed inputs NOT present in the spreadsheet. Sourcing: PARAMETERS.md.
# ---------------------------------------------------------------------------

# Age at cohort entry. The cohort is age-synchronised, so age is exactly
# ANCHOR_AGE + (years since entry) and needs no state dimension.
ANCHOR_AGE = 29.0

# Background mortality is not a constant here -- see hazards.background_mortality.

# Excess HIV mortality on suppressive ART, as an additive hazard over
# background. The >=500 anchor; the CD4 gradient is ART_SMR below. Influential:
# set to 0.0 for a "no excess once suppressed" scenario.
EXCESS_HAZARD_ON_ART = 0.001

# On-ART mortality while suppressed: SMR on CURRENT CD4, applied as excess over
# background. UNVERIFIED gradient -- treat 2.3/4.6 as modelling assumptions.
ART_SMR = [1.0, 1.8, 2.3, 2.3, 4.6, 7.0, 10.0]   # bands >=500 ... <50

# Universal SMR floor, inert at 1.0. Kept as a live mechanism for testing an
# ESA-specific residual-risk assumption in sensitivity.
ART_SMR_FLOOR = 1.0

# Multiplicative uncertainty on the SMR ladder, log-scale SD. Applied to the
# SMR, not the excess. Drawn per posterior draw; not calibrated.
ART_SMR_LOG_SD = 0.19

# Exit hazard from ART1_Ramp, per year: time from starting ART to suppressing.
# In the ramp, CD4 does not recover and untreated mortality applies.
SUPPRESSION_RATE = 9.0

# Same-day ART initiation, SOURCED not fitted. Modelled as a split of the
# diagnosis flow rather than a faster hazard. First diagnosis only.
SAME_DAY_INITIATION = 0.64


# Accounting horizon: one horizon for BOTH calibration and the health-economic
# run. 71 years runs the cohort to extinction, which the stationary cascade
# correction requires.
HORIZON_YEARS = 71

# PLACEHOLDER. Scales untreated mortality for disengaged patients. Dominates the
# mortality model; phi_ltfu is now free and this is the fallback.
LTFU_MORTALITY_MULTIPLIER = 1.0

# Shape of re-engagement, FIXED from Bagayoko et al. 2020 (Mali). Parameterised
# as an eventual-return FRACTION so the plateau holds as mu_re-engage varies.
LTFU_RETURN_FRACTION = 0.437      # Mali: 43.7% of disengaged patients ever return
LTFU_REENGAGE_LATE = 0.0314       # slow trickle out of long-term disengagement


def ltfu_perm_disengage(mu_reengage: float,
                        return_fraction: float = LTFU_RETURN_FRACTION) -> float:
    """Companion hazard implied by holding the eventual-return fraction fixed."""
    return mu_reengage * (1.0 - return_fraction) / return_fraction


if __name__ == "__main__":
    p = load_parameters()
    print("Free parameters:")
    for sym, fp in p.free.items():
        print(f"  {sym}: central={fp.central}, sd={fp.sd}  ({fp.name})")
    print("\nFixed inputs:")
    for sym, val in p.fixed.items():
        print(f"  {sym}: {val}")
    print("\nTargets:")
    for sym, (c, sd) in p.targets.items():
        print(f"  {sym}: central={c}, sd={sd}")

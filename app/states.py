"""State space definitions.

NaturalHistoryStates: 7 CD4 bands + 2 absorbing = 9. CascadeStates: 7 bands x 15
cascade statuses + 2 absorbing = 107. The 15 statuses are Undiagnosed plus 7 base
statuses that each exist once per recovery-ceiling class.

LTFU is split into recent and long-term compartments because observed
re-engagement is a mixture of two exponentials, not one. See PARAMETERS.md.
"""

from __future__ import annotations

from params import CD4_BAND_NAMES, N_CD4_BANDS

# ---------------------------------------------------------------------------
# System 1: Natural history only
# ---------------------------------------------------------------------------

NH_HIV_DEATH = N_CD4_BANDS          # index 7
NH_BG_DEATH = N_CD4_BANDS + 1       # index 8
N_NH_STATES = N_CD4_BANDS + 2       # 9

NH_STATE_NAMES = list(CD4_BAND_NAMES) + ["HIV Death", "Background Death"]


def nh_is_absorbing(state_idx: int) -> bool:
    return state_idx in (NH_HIV_DEATH, NH_BG_DEATH)


# ---------------------------------------------------------------------------
# System 2: Full cascade
# ---------------------------------------------------------------------------

# Recovery-ceiling classes: a coarse nadir. CD4 reconstitution on ART is capped
# by the nadir, but seven nadir bands give only four distinct ceilings and every
# nadir at or above 200 gives the same one -- so this tracks a CLASS, not a
# nadir. A ratchet: FULL -> CAPPED on declining past 200, never back.
CEILING_FULL = "Full"
CEILING_CAPPED = "Capped"
CEILING_CLASSES = (CEILING_FULL, CEILING_CAPPED)

#: Highest band (lowest index) a CAPPED person can reconstitute to.
#
# Band 1, the MODAL ceiling: every person enters Capped at 100-199, which maps
# to a ceiling of 400. Taking the middle value instead costs ~0.4 DALYs.
CEILING_BAND_CAPPED = 1          # 350-499
#: First band at or below 200 cells; entering it triggers the ratchet.
BAND_BELOW_200 = 4               # 100-199

#: Statuses that exist once per ceiling class. Undiagnosed is untagged: before
#: any treatment the nadir IS the current band, so no memory is needed -- the
#: class is assigned on leaving Undiagnosed.
_TAGGED_BASE = (
    "Diagnosed_PreART",
    "ART1_Ramp",
    "ART1_Suppressed",
    "ART1_Failing",
    "ART2_Suppressed",
    "LTFU_Recent",
    "LTFU_LongTerm",
)
_UNTAGGED = ("Undiagnosed",)


def tagged(base: str, cls: str) -> str:
    """Status name for a base status in a ceiling class."""
    return f"{base}__{cls}"


def base_of(status: str) -> str:
    """Base status, ignoring the ceiling class."""
    return status.split("__", 1)[0]


def class_of(status: str) -> str | None:
    """Ceiling class of a status, or None for untagged statuses."""
    return status.split("__", 1)[1] if "__" in status else None


def ceiling_band(cls: str | None) -> int:
    """Highest band (lowest index) reachable on suppressive ART."""
    return 0 if cls != CEILING_CAPPED else CEILING_BAND_CAPPED


def class_for_band(band: int) -> str:
    """Ceiling class implied by a CD4 band, for someone whose nadir is their
    current band -- i.e. anyone who has not yet been treated."""
    return CEILING_CAPPED if band >= BAND_BELOW_200 else CEILING_FULL


CASCADE_STATUSES = list(_UNTAGGED) + [
    tagged(b, c) for b in _TAGGED_BASE for c in CEILING_CLASSES
]

#: Base-status membership helpers, so nothing downstream has to know about tags.
def with_base(*bases: str) -> tuple[str, ...]:
    return tuple(s for s in CASCADE_STATUSES if base_of(s) in bases)


N_CASCADE_STATUSES = len(CASCADE_STATUSES)

# Every status representing disengagement from care. Anything that used to
# test `status == "LTFU"` should test membership here instead.
LTFU_STATUSES = with_base("LTFU_Recent", "LTFU_LongTerm")

# The compartment newly-disengaging patients enter.
_LTFU_ENTRY_BASE = "LTFU_Recent"


def ltfu_entry(cls: str) -> str:
    return tagged(_LTFU_ENTRY_BASE, cls)

N_FULL_STATES = N_CD4_BANDS * N_CASCADE_STATUSES + 2   # 7 x 15 + 2 = 107
FULL_HIV_DEATH = N_CD4_BANDS * N_CASCADE_STATUSES        # index 105
FULL_BG_DEATH = N_CD4_BANDS * N_CASCADE_STATUSES + 1     # index 106


def full_index(cascade_status: str, cd4_band: int) -> int:
    """Map (cascade status name, CD4 band index 0-6) -> flat state index."""
    status_idx = CASCADE_STATUSES.index(cascade_status)
    return status_idx * N_CD4_BANDS + cd4_band


def full_is_absorbing(state_idx: int) -> bool:
    return state_idx in (FULL_HIV_DEATH, FULL_BG_DEATH)


def full_state_name(state_idx: int) -> str:
    if state_idx == FULL_HIV_DEATH:
        return "HIV Death"
    if state_idx == FULL_BG_DEATH:
        return "Background Death"
    status_idx, band = divmod(state_idx, N_CD4_BANDS)
    return f"{CASCADE_STATUSES[status_idx]} | CD4 {CD4_BAND_NAMES[band]}"


FULL_STATE_NAMES = [full_state_name(i) for i in range(N_FULL_STATES)]

assert N_FULL_STATES == 107, f"Expected 107 states, got {N_FULL_STATES}"
assert all(s in CASCADE_STATUSES for s in LTFU_STATUSES)
assert all(ltfu_entry(c) in CASCADE_STATUSES for c in CEILING_CLASSES)

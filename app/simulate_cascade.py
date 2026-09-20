"""System 2: the full 107-state cascade engine.

Reuses the System 1 hazards (CD4 decline, untreated mortality) for every
non-suppressed cascade status and adds cascade transitions on top. System 1 is
sourced rather than fitted, so it enters here through kappa_prog / kappa_mort
draws on the published tables.

STRUCTURAL ASSUMPTIONS, flagged rather than buried:

- CD4 dynamics: Undiagnosed, Diagnosed_PreART and ART1_Failing all use the
  untreated decline hazard and untreated mortality -- failing first line is
  treated as being off treatment. LTFU likewise, with no discount for prior ART
  exposure or partial immune preservation (see params.LTFU_MORTALITY_MULTIPLIER,
  a high-leverage assumption). Suppressed states reconstitute CD4 toward their
  ceiling class and carry background + EXCESS mortality.

- Pre-ART disengagement uses the same mu_ltfu hazard as suppressed and failing
  patients; there is no pre-ART-specific disengagement parameter.

- Re-engagement always re-enters via Diagnosed_PreART regardless of the regimen
  held before disengaging: there is one LTFU pool per ceiling class, not one
  per regimen.

- There is no ART2_Failing state, so a rescue switch reaches ART2_Suppressed
  directly. The Supp_2L target is therefore an emergent property of mu_switch
  rather than a distinct pathway that can be checked against.
"""

from __future__ import annotations

import math

import numpy as np

from params import (N_CD4_BANDS, ANCHOR_AGE, SUPPRESSION_RATE, SAME_DAY_INITIATION,
                    ltfu_perm_disengage, LTFU_REENGAGE_LATE)
from states import (
    N_FULL_STATES, N_CASCADE_STATUSES, CASCADE_STATUSES,
    LTFU_STATUSES,
    full_index, FULL_HIV_DEATH, FULL_BG_DEATH,
    with_base, tagged, base_of, class_of, ceiling_band, class_for_band,
    CEILING_CLASSES, CEILING_CAPPED, BAND_BELOW_200, ltfu_entry)
from hazards import (
    reconstitution_hazard, choose_recon_rate,
    diagnosis_hazard, competing_risks_step,
    background_mortality, on_art_excess_hazard, on_art_excess_curve,
    cap_on_art_at_untreated,
    ltfu_mortality,
)
from natural_history_params import (
    age_group_index, progression_schedule, untreated_mortality_schedule,
    mortality_curve_at_age, GLAUBIUS_PROGRESSION, AGE_GROUP_LOWER,
)


def progression_row_at_age(age: float, kappa_prog: float = 1.0,
                            table=None) -> np.ndarray:
    """CD4-progression hazard vector for the Glaubius age group containing
    `age`. Companion to natural_history_params.mortality_curve_at_age."""
    tab = GLAUBIUS_PROGRESSION if table is None else table
    return tab[age_group_index(age)] * float(kappa_prog)


# ---------------------------------------------------------------------------
# Vectorised transition-matrix construction
# ---------------------------------------------------------------------------
#
# The per-cell loop below is kept as the readable reference; the vectorised
# builder is what runs. _assert_builders_agree() checks they match, on
# `python3 simulate_cascade.py`.

_REASONS = ("hiv_death", "bg_death", "decline", "recover", "diagnose",
            "init_art", "suppress", "ltfu", "vf", "switch", "reengage",
            "to_longterm", "diagnose_sameday")
_RI = {r: i for i, r in enumerate(_REASONS)}
_NR = len(_REASONS)
_SI = {st: i for i, st in enumerate(CASCADE_STATUSES)}
_NS = N_CASCADE_STATUSES
_NB = N_CD4_BANDS

# Which statuses use the untreated mortality curve, the on-ART excess, or the
# LTFU-scaled curve.
# ART1_Ramp is on treatment but not suppressed, so it takes the untreated
# Glaubius rate, not the on-ART SMR excess.
_UNTREATED_MORT = ("Undiagnosed",) + with_base("Diagnosed_PreART", "ART1_Ramp",
                                                "ART1_Failing")
_ONART_MORT = with_base("ART1_Suppressed", "ART2_Suppressed")
_LTFU_MORT = LTFU_STATUSES
# CD4 moves down for anyone not virally suppressed, up for those suppressed and
# still below their recovery ceiling.
_DECLINE_STATUSES = ("Undiagnosed",) + with_base("Diagnosed_PreART", "ART1_Failing",
                                                  "LTFU_Recent", "LTFU_LongTerm")
_RECOVER_STATUSES = with_base("ART1_Suppressed", "ART2_Suppressed")


def _rows(*bases):
    """Status row indices for every ceiling class of the given base statuses."""
    return [_SI[st] for st in with_base(*bases)]


# (status, reason) -> destination status, same CD4 band and SAME ceiling class.
# Every transition here preserves the class; the two that do not -- diagnosis
# (which assigns it) and CD4 decline (which ratchets it) -- are band-dependent
# and are handled in _build_scatter_index instead.
_SAME_BAND_DEST = {}
for _c in CEILING_CLASSES:
    def _t(base, _c=_c):
        return tagged(base, _c)
    _SAME_BAND_DEST.update({
        (_t("Diagnosed_PreART"), "init_art"): _t("ART1_Ramp"),
        (_t("Diagnosed_PreART"), "ltfu"): _t("LTFU_Recent"),
        (_t("ART1_Ramp"), "suppress"): _t("ART1_Suppressed"),
        (_t("ART1_Ramp"), "ltfu"): _t("LTFU_Recent"),
        (_t("ART1_Suppressed"), "vf"): _t("ART1_Failing"),
        (_t("ART1_Suppressed"), "ltfu"): _t("LTFU_Recent"),
        (_t("ART1_Failing"), "switch"): _t("ART2_Suppressed"),
        (_t("ART1_Failing"), "ltfu"): _t("LTFU_Recent"),
        (_t("LTFU_Recent"), "reengage"): _t("Diagnosed_PreART"),
        (_t("LTFU_Recent"), "to_longterm"): _t("LTFU_LongTerm"),
        (_t("LTFU_LongTerm"), "reengage"): _t("Diagnosed_PreART"),
        (_t("ART2_Suppressed"), "ltfu"): _t("LTFU_Recent"),
    })


def _build_scatter_index():
    """(n_statuses, n_bands, n_reasons) destination index, -1 where the
    transition does not exist. Depends only on the state space, so it is
    computed once at import.

    Three destinations are NOT simply "same band, same class":
      diagnose  assigns the ceiling class from the band at diagnosis
      decline   ratchets Full -> Capped on crossing below 200
      recover   does not exist at or above the class's ceiling band
    """
    tgt = np.full((_NS, _NB, _NR), -1, dtype=np.int64)
    for st in CASCADE_STATUSES:
        si = _SI[st]
        cls = class_of(st)
        ceil_b = ceiling_band(cls)
        for b in range(_NB):
            tgt[si, b, _RI["hiv_death"]] = FULL_HIV_DEATH
            tgt[si, b, _RI["bg_death"]] = FULL_BG_DEATH
            if st in _DECLINE_STATUSES and b < _NB - 1:
                nb = b + 1
                if cls is None:                       # Undiagnosed stays untagged
                    dest = st
                else:
                    dest = tagged(base_of(st),
                                   CEILING_CAPPED if nb >= BAND_BELOW_200 else cls)
                tgt[si, b, _RI["decline"]] = full_index(dest, nb)
            if st in _RECOVER_STATUSES and b > ceil_b:
                tgt[si, b, _RI["recover"]] = full_index(st, b - 1)
            if st == "Undiagnosed":
                tgt[si, b, _RI["diagnose"]] = full_index(
                    tagged("Diagnosed_PreART", class_for_band(b)), b)
                # Same-day initiation: diagnosis and ART start in one instant,
                # bypassing Diagnosed_PreART entirely. See params.SAME_DAY_INITIATION.
                tgt[si, b, _RI["diagnose_sameday"]] = full_index(
                    tagged("ART1_Ramp", class_for_band(b)), b)
        for (s_from, reason), s_to in _SAME_BAND_DEST.items():
            if s_from != st:
                continue
            for b in range(_NB):
                tgt[si, b, _RI[reason]] = full_index(s_to, b)
    return tgt


_TGT = _build_scatter_index()
_SRC = np.repeat(np.arange(_NS * _NB, dtype=np.int64), _NR).reshape(_NS, _NB, _NR)
_VALID = _TGT >= 0
_SRC_FLAT = _SRC[_VALID]
_TGT_FLAT = _TGT[_VALID]
_DIAG_IDX = np.arange(_NS * _NB, dtype=np.int64)
# Flattened (row, col) destinations, so the scatter can use np.bincount --
# substantially faster than np.add.at, which is an unbuffered ufunc and the
# single largest remaining cost once the per-cell loop is gone.
_FLAT_DEST = _SRC_FLAT * N_FULL_STATES + _TGT_FLAT
_FLAT_SIZE = N_FULL_STATES * N_FULL_STATES
_BANDS = np.arange(_NB)
# Normalised disease severity per band, the base of the diagnosis ramp.
_SEVERITY_BASE = _BANDS / (_NB - 1)

# Reconstitution depends only on the FIXED recov_rate input, never on a
# calibrated parameter, so it is computed once per distinct rate pair rather
# than on every one of the ~264,000 likelihood evaluations.
_RECOVER_CACHE: dict = {}


def _recover_curve(recov_fast: float, recov_slow: float) -> np.ndarray:
    key = (recov_fast, recov_slow)
    curve = _RECOVER_CACHE.get(key)
    if curve is None:
        curve = np.array([reconstitution_hazard(b, choose_recon_rate(b, recov_fast, recov_slow))
                          for b in range(_NB)])
        _RECOVER_CACHE[key] = curve
    return curve


def build_full_transition_matrix(central: dict, mortality_curve, recov_rate,
                                  bg_mortality=None, dt=1 / 12,
                                  progression_curve=None) -> np.ndarray:
    """Monthly 51x51 transition matrix at ONE age.

    `bg_mortality` is the age-specific non-HIV background hazard; it defaults
    to the value at params.ANCHOR_AGE. For anything spanning more than a
    couple of years use build_full_matrix_stack(), which supplies the correct
    background for each year of the cohort's age.

    Vectorised; numerically identical to _build_full_transition_matrix_loop,
    which is retained as the correctness oracle.
    """
    if bg_mortality is None:
        bg_mortality = background_mortality(ANCHOR_AGE)
    if progression_curve is None:
        progression_curve = progression_row_at_age(ANCHOR_AGE)

    recov_fast, recov_slow = recov_rate

    H = np.zeros((_NS, _NB, _NR))

    # --- mortality ---
    H[:, :, _RI["bg_death"]] = bg_mortality
    untreated = np.asarray(mortality_curve, dtype=float)
    hd = _RI["hiv_death"]
    for st in _UNTREATED_MORT:
        H[_SI[st], :, hd] = untreated
    # Capped at the untreated curve -- see hazards.cap_on_art_at_untreated.
    onart_excess = cap_on_art_at_untreated(
        on_art_excess_curve(bg_mortality), untreated)
    for st in _ONART_MORT:
        H[_SI[st], :, hd] = onart_excess
    H[_rows("ART1_Ramp"), :, _RI["suppress"]] = SUPPRESSION_RATE
    ltfu_curve = ltfu_mortality(untreated, central.get("φ_ltfu"))
    for st in _LTFU_MORT:
        H[_SI[st], :, hd] = ltfu_curve

    # --- CD4 movement ---
    decline = np.array(progression_curve, dtype=float)
    decline[_NB - 1] = 0.0
    for st in _DECLINE_STATUSES:
        H[_SI[st], :, _RI["decline"]] = decline
    recover = _recover_curve(recov_fast, recov_slow)
    for st in _RECOVER_STATUSES:
        H[_SI[st], :, _RI["recover"]] = recover

    # --- cascade movement ---
    # Vectorised form of hazards.diagnosis_hazard across all bands:
    #   delta_bg + delta_symp * (band / (n_bands - 1)) ** gamma_diag
    _dx = (central["δ_bg"] + central["δ_symp"] * _SEVERITY_BASE ** central["γ_diag"])
    _pi = central.get("π_sameday", SAME_DAY_INITIATION)
    H[_SI["Undiagnosed"], :, _RI["diagnose"]] = _dx * (1.0 - _pi)
    H[_SI["Undiagnosed"], :, _RI["diagnose_sameday"]] = _dx * _pi
    H[_rows("Diagnosed_PreART"), :, _RI["init_art"]] = central["λ_init"]
    H[_rows("ART1_Suppressed"), :, _RI["vf"]] = central["μ_vf1"]
    H[_rows("ART1_Failing"), :, _RI["switch"]] = central["μ_switch"]
    H[_rows("Diagnosed_PreART", "ART1_Ramp", "ART1_Suppressed", "ART1_Failing",
            "ART2_Suppressed"), :, _RI["ltfu"]] = central["μ_ltfu"]
    H[_rows("LTFU_Recent"), :, _RI["reengage"]] = central["μ_re-engage"]
    H[_rows("LTFU_Recent"), :, _RI["to_longterm"]] = central.get(
        "μ_perm_diseng", ltfu_perm_disengage(central["μ_re-engage"]))
    H[_rows("LTFU_LongTerm"), :, _RI["reengage"]] = central.get(
        "μ_reengage_late", LTFU_REENGAGE_LATE)

    # Hazards on transitions that do not exist must not enter the totals.
    H *= _VALID

    # --- competing risks, all 49 cells at once ---
    total = H.sum(axis=2)
    p_exit = -np.expm1(-total * dt)          # 1 - exp(-total*dt), stable near 0
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total[:, :, None] > 0, H / total[:, :, None], 0.0)
    P = p_exit[:, :, None] * share

    M = np.bincount(_FLAT_DEST, weights=P[_VALID],
                    minlength=_FLAT_SIZE).reshape(N_FULL_STATES, N_FULL_STATES)
    M[_DIAG_IDX, _DIAG_IDX] = (1.0 - p_exit).ravel()
    M[FULL_HIV_DEATH, FULL_HIV_DEATH] = 1.0
    M[FULL_BG_DEATH, FULL_BG_DEATH] = 1.0

    row_sums = M.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Rows not summing to 1: {np.where(~np.isclose(row_sums, 1.0))}"
    return M


def _assert_builders_agree(n_draws: int = 25, seed: int = 0) -> None:
    """Check the vectorised builder against the readable loop over random
    parameter draws. Cheap insurance -- the vectorised version is the one
    every calibration uses."""
    from params import load_parameters
    import likelihoods as lk

    rng = np.random.default_rng(seed)
    p = load_parameters()
    specs = lk.build_specs(p, lk.SYSTEM2_SYMBOLS, lk.SYSTEM2_KINDS)
    worst = 0.0
    for _ in range(n_draws):
        central = {s.symbol: float(s.sample_prior(rng)) for s in specs}
        # exercise the whole age range of the adopted tables, plus a random
        # level shift, rather than a single age group
        g = int(rng.integers(0, 4))
        mc = mortality_curve_at_age(AGE_GROUP_LOWER[g]) * float(rng.uniform(0.7, 1.4))
        pc = progression_row_at_age(AGE_GROUP_LOWER[g]) * float(rng.uniform(0.8, 1.2))
        bg = float(rng.uniform(0.001, 0.05))
        a = build_full_transition_matrix(central, mc, p.fixed["recov_rate"], bg,
                                          progression_curve=pc)
        b = _build_full_transition_matrix_loop(central, mc, p.fixed["recov_rate"], bg,
                                                progression_curve=pc)
        worst = max(worst, float(np.abs(a - b).max()))
    assert worst < 1e-12, f"builders disagree, max abs diff {worst:.3e}"
    print(f"vectorised builder matches the reference loop over {n_draws} random "
          f"draws (max abs diff {worst:.2e})")


def _build_full_transition_matrix_loop(central: dict, mortality_curve, recov_rate,
                                        bg_mortality=None, dt=1 / 12,
                                        progression_curve=None) -> np.ndarray:
    """Reference implementation, kept ONLY as the correctness oracle for the
    vectorised builder below (see _assert_builders_agree). Readable but slow:
    a Python loop over all 49 status-band cells, each constructing a dict and
    calling competing_risks_step. Do not call it in anything hot.

    Monthly 51x51 transition matrix at ONE age.

    `bg_mortality` is the age-specific non-HIV background hazard; it
    defaults to the value at params.ANCHOR_AGE. For anything spanning more
    than a couple of years, use build_full_matrix_stack() instead, which
    supplies the correct background for each year of the cohort's age.
    """
    if bg_mortality is None:
        bg_mortality = background_mortality(ANCHOR_AGE)
    if progression_curve is None:
        progression_curve = progression_row_at_age(ANCHOR_AGE)
    M = np.zeros((N_FULL_STATES, N_FULL_STATES))
    recov_fast, recov_slow = recov_rate  # [fast, slow], cells/µL/yr

    def decl_cls(cls, to_band):
        """Ceiling-class ratchet: declining past 200 makes the cap permanent."""
        return CEILING_CAPPED if to_band >= BAND_BELOW_200 else cls

    for band in range(N_CD4_BANDS):
        decline = float(progression_curve[band]) if band < N_CD4_BANDS - 1 else 0.0

        # ---------------- Undiagnosed (untagged) ----------------
        # Before any treatment the nadir IS the current band, so the ceiling
        # class is read off the band at the moment of diagnosis.
        i = full_index("Undiagnosed", band)
        _dx = diagnosis_hazard(band, central["δ_bg"], central["δ_symp"],
                                central["γ_diag"])
        _pi = central.get("π_sameday", SAME_DAY_INITIATION)
        h = {
            "hiv_death": mortality_curve[band],
            "bg_death": bg_mortality,
            "diagnose": _dx * (1.0 - _pi),
            "diagnose_sameday": _dx * _pi,
        }
        if decline > 0:
            h["decline"] = decline
        probs = competing_risks_step(h, dt)
        M[i, i] += probs["stay"]
        M[i, FULL_HIV_DEATH] += probs["hiv_death"]
        M[i, FULL_BG_DEATH] += probs["bg_death"]
        M[i, full_index(tagged("Diagnosed_PreART", class_for_band(band)), band)] += probs["diagnose"]
        M[i, full_index(tagged("ART1_Ramp", class_for_band(band)), band)] += probs["diagnose_sameday"]
        if "decline" in probs:
            M[i, full_index("Undiagnosed", band + 1)] += probs["decline"]

        for cls in CEILING_CLASSES:
            T = lambda base, c=cls: tagged(base, c)
            D = lambda base: tagged(base, decl_cls(cls, band + 1))
            ceil_b = ceiling_band(cls)

            # ---------------- Diagnosed_PreART ----------------
            i = full_index(T("Diagnosed_PreART"), band)
            h = {
                "hiv_death": mortality_curve[band],
                "bg_death": bg_mortality,
                "init_art": central["λ_init"],
                "ltfu": central["μ_ltfu"],
            }
            if decline > 0:
                h["decline"] = decline
            probs = competing_risks_step(h, dt)
            M[i, i] += probs["stay"]
            M[i, FULL_HIV_DEATH] += probs["hiv_death"]
            M[i, FULL_BG_DEATH] += probs["bg_death"]
            M[i, full_index(T("ART1_Ramp"), band)] += probs["init_art"]
            M[i, full_index(T("LTFU_Recent"), band)] += probs["ltfu"]
            if "decline" in probs:
                M[i, full_index(D("Diagnosed_PreART"), band + 1)] += probs["decline"]

            # ---------------- ART1_Ramp ----------------
            # On ART, not yet suppressed. CD4 static; untreated mortality.
            i = full_index(T("ART1_Ramp"), band)
            h = {
                "hiv_death": mortality_curve[band],
                "bg_death": bg_mortality,
                "suppress": SUPPRESSION_RATE,
                "ltfu": central["μ_ltfu"],
            }
            probs = competing_risks_step(h, dt)
            M[i, i] += probs["stay"]
            M[i, FULL_HIV_DEATH] += probs["hiv_death"]
            M[i, FULL_BG_DEATH] += probs["bg_death"]
            M[i, full_index(T("ART1_Suppressed"), band)] += probs["suppress"]
            M[i, full_index(T("LTFU_Recent"), band)] += probs["ltfu"]

            # ---------------- ART1_Suppressed ----------------
            i = full_index(T("ART1_Suppressed"), band)
            recov_h = (reconstitution_hazard(band, choose_recon_rate(band, recov_fast, recov_slow))
                       if band > ceil_b else 0.0)
            h = {
                "hiv_death": min(on_art_excess_hazard(band, bg_mortality),
                                 float(mortality_curve[band])),
                "bg_death": bg_mortality,
                "vf": central["μ_vf1"],
                "ltfu": central["μ_ltfu"],
            }
            if recov_h > 0:
                h["recover"] = recov_h
            probs = competing_risks_step(h, dt)
            M[i, i] += probs["stay"]
            M[i, FULL_HIV_DEATH] += probs["hiv_death"]
            M[i, FULL_BG_DEATH] += probs["bg_death"]
            M[i, full_index(T("ART1_Failing"), band)] += probs["vf"]
            M[i, full_index(T("LTFU_Recent"), band)] += probs["ltfu"]
            if "recover" in probs:
                M[i, full_index(T("ART1_Suppressed"), band - 1)] += probs["recover"]

            # ---------------- ART1_Failing ----------------
            i = full_index(T("ART1_Failing"), band)
            h = {
                "hiv_death": mortality_curve[band],
                "bg_death": bg_mortality,
                "switch": central["μ_switch"],
                "ltfu": central["μ_ltfu"],
            }
            if decline > 0:
                h["decline"] = decline
            probs = competing_risks_step(h, dt)
            M[i, i] += probs["stay"]
            M[i, FULL_HIV_DEATH] += probs["hiv_death"]
            M[i, FULL_BG_DEATH] += probs["bg_death"]
            M[i, full_index(T("ART2_Suppressed"), band)] += probs["switch"]
            M[i, full_index(T("LTFU_Recent"), band)] += probs["ltfu"]
            if "decline" in probs:
                M[i, full_index(D("ART1_Failing"), band + 1)] += probs["decline"]

            # ---------------- LTFU_Recent ----------------
            i = full_index(T("LTFU_Recent"), band)
            h = {
                "hiv_death": ltfu_mortality(mortality_curve[band], central.get("φ_ltfu")),
                "bg_death": bg_mortality,
                "reengage": central["μ_re-engage"],
                "to_longterm": central.get("μ_perm_diseng",
                                            ltfu_perm_disengage(central["μ_re-engage"])),
            }
            if decline > 0:
                h["decline"] = decline
            probs = competing_risks_step(h, dt)
            M[i, i] += probs["stay"]
            M[i, FULL_HIV_DEATH] += probs["hiv_death"]
            M[i, FULL_BG_DEATH] += probs["bg_death"]
            M[i, full_index(T("Diagnosed_PreART"), band)] += probs["reengage"]
            M[i, full_index(T("LTFU_LongTerm"), band)] += probs["to_longterm"]
            if "decline" in probs:
                M[i, full_index(D("LTFU_Recent"), band + 1)] += probs["decline"]

            # ---------------- LTFU_LongTerm ----------------
            i = full_index(T("LTFU_LongTerm"), band)
            h = {
                "hiv_death": ltfu_mortality(mortality_curve[band], central.get("φ_ltfu")),
                "bg_death": bg_mortality,
                "reengage": central.get("μ_reengage_late", LTFU_REENGAGE_LATE),
            }
            if decline > 0:
                h["decline"] = decline
            probs = competing_risks_step(h, dt)
            M[i, i] += probs["stay"]
            M[i, FULL_HIV_DEATH] += probs["hiv_death"]
            M[i, FULL_BG_DEATH] += probs["bg_death"]
            M[i, full_index(T("Diagnosed_PreART"), band)] += probs["reengage"]
            if "decline" in probs:
                M[i, full_index(D("LTFU_LongTerm"), band + 1)] += probs["decline"]

            # ---------------- ART2_Suppressed ----------------
            i = full_index(T("ART2_Suppressed"), band)
            recov_h = (reconstitution_hazard(band, choose_recon_rate(band, recov_fast, recov_slow))
                       if band > ceil_b else 0.0)
            h = {
                "hiv_death": min(on_art_excess_hazard(band, bg_mortality),
                                 float(mortality_curve[band])),
                "bg_death": bg_mortality,
                "ltfu": central["μ_ltfu"],
            }
            if recov_h > 0:
                h["recover"] = recov_h
            probs = competing_risks_step(h, dt)
            M[i, i] += probs["stay"]
            M[i, FULL_HIV_DEATH] += probs["hiv_death"]
            M[i, FULL_BG_DEATH] += probs["bg_death"]
            M[i, full_index(T("LTFU_Recent"), band)] += probs["ltfu"]
            if "recover" in probs:
                M[i, full_index(T("ART2_Suppressed"), band - 1)] += probs["recover"]

    M[FULL_HIV_DEATH, FULL_HIV_DEATH] = 1.0
    M[FULL_BG_DEATH, FULL_BG_DEATH] = 1.0

    row_sums = M.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Rows not summing to 1: {np.where(~np.isclose(row_sums, 1.0))}"
    return M


def _hazard_array(central: dict, mortality_curve, recov_rate,
                   progression_curve) -> np.ndarray:
    """(n_statuses, n_bands, n_reasons) annual hazards for ONE Glaubius age
    group, with the background death channel left at ZERO so callers can
    broadcast an age-specific value into it.

    `mortality_curve` and `progression_curve` are that age group's rows from
    the adopted natural-history tables. Everything else here is
    age-independent."""
    recov_fast, recov_slow = recov_rate
    H = np.zeros((_NS, _NB, _NR))

    untreated = np.asarray(mortality_curve, dtype=float)
    hd = _RI["hiv_death"]
    for st in _UNTREATED_MORT:
        H[_SI[st], :, hd] = untreated
    # On-ART excess is (SMR(band) - 1) x background, and background varies by
    # SINGLE YEAR of age while this array is per Glaubius AGE GROUP. So it is
    # left at ZERO here and broadcast in by build_full_matrix_stack alongside
    # the bg_death channel, for exactly the same reason.
    H[_rows("ART1_Ramp"), :, _RI["suppress"]] = SUPPRESSION_RATE
    ltfu_curve = ltfu_mortality(untreated, central.get("φ_ltfu"))
    for st in _LTFU_MORT:
        H[_SI[st], :, hd] = ltfu_curve

    decline = np.array(progression_curve, dtype=float)
    decline[_NB - 1] = 0.0
    for st in _DECLINE_STATUSES:
        H[_SI[st], :, _RI["decline"]] = decline
    recover = _recover_curve(recov_fast, recov_slow)
    for st in _RECOVER_STATUSES:
        H[_SI[st], :, _RI["recover"]] = recover

    _dx = (central["δ_bg"] + central["δ_symp"] * _SEVERITY_BASE ** central["γ_diag"])
    _pi = central.get("π_sameday", SAME_DAY_INITIATION)
    H[_SI["Undiagnosed"], :, _RI["diagnose"]] = _dx * (1.0 - _pi)
    H[_SI["Undiagnosed"], :, _RI["diagnose_sameday"]] = _dx * _pi
    H[_rows("Diagnosed_PreART"), :, _RI["init_art"]] = central["λ_init"]
    H[_rows("ART1_Suppressed"), :, _RI["vf"]] = central["μ_vf1"]
    H[_rows("ART1_Failing"), :, _RI["switch"]] = central["μ_switch"]
    H[_rows("Diagnosed_PreART", "ART1_Ramp", "ART1_Suppressed", "ART1_Failing",
            "ART2_Suppressed"), :, _RI["ltfu"]] = central["μ_ltfu"]
    H[_rows("LTFU_Recent"), :, _RI["reengage"]] = central["μ_re-engage"]
    H[_rows("LTFU_Recent"), :, _RI["to_longterm"]] = central.get(
        "μ_perm_diseng", ltfu_perm_disengage(central["μ_re-engage"]))
    H[_rows("LTFU_LongTerm"), :, _RI["reengage"]] = central.get(
        "μ_reengage_late", LTFU_REENGAGE_LATE)

    H *= _VALID
    return H


def build_full_matrix_stack(central: dict, recov_rate,
                             dt=1 / 12, horizon_years=45,
                             anchor_age=ANCHOR_AGE,
                             kappa_prog=1.0, kappa_mort=1.0,
                             prog_table=None, mort_table=None) -> np.ndarray:
    """Stack of shape (n_years, 51, 51): one transition matrix per year of
    model time, built with the hazards that apply to the cohort at its age
    that year -- the background mortality hazard for its exact age, and the
    CD4-progression and untreated-mortality rows for its Glaubius age group.

    One matrix per YEAR (not per month) because the cause-deleted life table
    is tabulated by single year of age, and the Glaubius natural-history
    tables by age group -- there is no information below annual resolution,
    and per-month rebuilds cost ~12x for no information gain.

    Built as a batched operation rather than a loop over years. The
    age-dependent part now has two pieces with different granularity: the
    background channel varies by single year and is broadcast in, while
    progression and untreated/LTFU mortality change only at Glaubius age-group
    boundaries, so at most four distinct hazard arrays are ever built no
    matter how long the horizon. This is the dominant cost in a System 2
    likelihood evaluation, so the batching is worth the indirection.

    kappa_prog / kappa_mort scale the published schedules and carry System 1's
    uncertainty; prog_table / mort_table substitute an alternative source
    (Mangal 2017) for the sensitivity analysis.
    """
    n_years = int(math.ceil(horizon_years))
    bg = np.array([background_mortality(anchor_age + y) for y in range(n_years)])

    prog = progression_schedule(kappa_prog, n_years, anchor_age, prog_table)
    mort = untreated_mortality_schedule(kappa_mort, n_years, anchor_age, mort_table)

    # Years map onto at most four distinct age groups, so build one hazard
    # array per group actually present and index it, rather than rebuilding
    # per year.
    groups = [age_group_index(anchor_age + y) for y in range(n_years)]
    first_year_of = {}
    for y, g in enumerate(groups):
        first_year_of.setdefault(g, y)
    H_by_group = {g: _hazard_array(central, mort[y0], recov_rate, prog[y0])
                  for g, y0 in first_year_of.items()}

    bgi = _RI["bg_death"]
    Hy = np.stack([H_by_group[g] for g in groups])                   # (Y, S, B, R)
    Hy[:, :, :, bgi] = bg[:, None, None]
    # On-ART suppressed: (SMR(band) - 1) x that year's background. Scales with
    # age exactly as the background channel does, so the two sum to SMR x bg.
    _smr_excess = on_art_excess_curve(1.0)                           # (B,)
    # The untreated curve is already in Hy for every untreated status, indexed
    # by (year, band) -- read it back rather than re-deriving it, so the cap
    # cannot drift from the schedule it is capping against.
    _untr = Hy[:, _SI["Undiagnosed"], :, _RI["hiv_death"]]            # (Y, B)
    _onart = cap_on_art_at_untreated(_smr_excess[None, :] * bg[:, None], _untr)
    for _st in _ONART_MORT:
        Hy[:, _SI[_st], :, _RI["hiv_death"]] = _onart

    total = Hy.sum(axis=3)                                            # (Y, S, B)
    p_exit = -np.expm1(-total * dt)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total[..., None] > 0, Hy / total[..., None], 0.0)
    P = p_exit[..., None] * share

    year_offset = (np.arange(n_years, dtype=np.int64) * _FLAT_SIZE)[:, None]
    flat = (_FLAT_DEST[None, :] + year_offset).ravel()
    stack = np.bincount(flat, weights=P[:, _VALID].ravel(),
                        minlength=n_years * _FLAT_SIZE
                        ).reshape(n_years, N_FULL_STATES, N_FULL_STATES)
    stack[:, _DIAG_IDX, _DIAG_IDX] = 1.0 - p_exit.reshape(n_years, -1)
    stack[:, FULL_HIV_DEATH, FULL_HIV_DEATH] = 1.0
    stack[:, FULL_BG_DEATH, FULL_BG_DEATH] = 1.0

    assert np.allclose(stack.sum(axis=2), 1.0), "Stack rows must sum to 1"
    return stack


def undiagnosed_init() -> np.ndarray:
    """Starting state vector: the cohort enters Undiagnosed, spread across CD4
    bands according to the acquisition distribution (acquisition_cd4).

    Replaces the previous 100%-in-band-0 point mass. See acquisition_cd4 for
    why that mattered: with everyone starting at CD4 >=500, the only route to
    initiating ART below 200 was an implausibly long diagnostic delay.
    """
    from acquisition_cd4 import initial_band_distribution
    init = np.zeros(N_FULL_STATES)
    for b, w in enumerate(initial_band_distribution()):
        init[full_index("Undiagnosed", b)] = w
    return init


def propagate(M, init_vec: np.ndarray, n_steps: int, dt=1 / 12) -> np.ndarray:
    """Trajectory of shape (n_steps+1, n_states).

    `M` is either a single (44, 44) matrix applied at every step, or a stack
    of shape (n_years, 44, 44) indexed by year of model time. Years past the
    end of the stack reuse its final matrix.
    """
    M = np.asarray(M)
    time_varying = M.ndim == 3
    traj = np.zeros((n_steps + 1, M.shape[-1]))
    traj[0] = init_vec
    for t in range(n_steps):
        step_M = M[min(int(t * dt), M.shape[0] - 1)] if time_varying else M
        traj[t + 1] = traj[t] @ step_M
    return traj


def cascade_summary(traj_row: np.ndarray) -> dict:
    """Given a state-probability row, compute UNAIDS-style proportions
    among those alive: % diagnosed, % on ART (of diagnosed), % suppressed
    (of on ART)."""
    from states import full_index
    alive_mass = 1.0 - traj_row[FULL_HIV_DEATH] - traj_row[FULL_BG_DEATH]

    def mass(status):
        return sum(traj_row[full_index(status, b)] for b in range(N_CD4_BANDS))

    undiagnosed = mass("Undiagnosed")
    diagnosed_total = alive_mass - undiagnosed
    def mass_base(*bases):
        return sum(mass(st) for st in with_base(*bases))

    on_art = mass_base("ART1_Ramp", "ART1_Suppressed", "ART1_Failing",
                        "ART2_Suppressed")
    suppressed = mass_base("ART1_Suppressed", "ART2_Suppressed")

    return {
        "alive_mass": alive_mass,
        "pct_diagnosed_of_alive": diagnosed_total / alive_mass if alive_mass else float("nan"),
        "pct_on_art_of_diagnosed": on_art / diagnosed_total if diagnosed_total else float("nan"),
        "pct_suppressed_of_on_art": suppressed / on_art if on_art else float("nan"),
    }


def stationary_cascade_proportions(traj: np.ndarray, dt=1 / 12) -> dict:
    """UNAIDS-style p1/p2/p3 are cross-sectional population snapshots that mix
    people at every duration since infection, under (roughly) constant
    incidence. Our simulator instead follows a single seroconversion cohort
    across its own lifetime. To compare like with like, this integrates the
    cohort trajectory over duration-since-infection under a stationary,
    constant-incidence assumption: the duration-since-infection density among
    the living is proportional to the cohort's own survival curve S(t), so
    population totals are integrals of the cohort's state masses over t
    (ratios of integrated numerators/denominators, not an average of ratios).
    """
    alive = 1.0 - traj[:, FULL_HIV_DEATH] - traj[:, FULL_BG_DEATH]

    def mass(status):
        return sum(traj[:, full_index(status, b)] for b in range(N_CD4_BANDS))

    diagnosed_total = alive - mass("Undiagnosed")
    def mass_base(*bases):
        return sum(mass(st) for st in with_base(*bases))

    on_art = mass_base("ART1_Ramp", "ART1_Suppressed", "ART1_Failing",
                        "ART2_Suppressed")
    suppressed = mass_base("ART1_Suppressed", "ART2_Suppressed")

    A = alive.sum() * dt
    D = diagnosed_total.sum() * dt
    ONART = on_art.sum() * dt
    SUPP = suppressed.sum() * dt

    return {
        "p1_diagnosed_of_alive": D / A if A else float("nan"),
        "p2_onart_of_diagnosed": ONART / D if D else float("nan"),
        "p3_suppressed_of_onart": SUPP / ONART if ONART else float("nan"),
    }


def make_absorbing(M: np.ndarray, indices) -> np.ndarray:
    """Return a copy of M where every state in `indices` is made fully
    absorbing (row -> identity). Used for first-passage analysis: once a
    person first enters the target set, we freeze them there so cumulative
    probability mass in the target set over time equals the cumulative-
    incidence function for "first reached target", correctly netting out
    competing risks (death) and allowing multiple attempts (e.g. LTFU ->
    re-engage -> reached) if the target isn't reached on the first try.
    """
    M2 = np.asarray(M).copy()
    if M2.ndim == 3:                       # per-year stack
        for idx in indices:
            M2[:, idx, :] = 0.0
            M2[:, idx, idx] = 1.0
        return M2
    for idx in indices:
        M2[idx, :] = 0.0
        M2[idx, idx] = 1.0
    return M2


def _first_passage_median(M_absorbing: np.ndarray, target_indices, init_vec: np.ndarray,
                           dt=1 / 12, horizon_years=15) -> float | None:
    n_steps = int(horizon_years / dt)
    traj = propagate(M_absorbing, init_vec, n_steps)
    cum_reached = traj[:, list(target_indices)].sum(axis=1)
    idx = np.argmax(cum_reached >= 0.5)
    if cum_reached[idx] < 0.5:
        return None
    return idx * dt


def time_to_art_median_years(M: np.ndarray, dt=1 / 12) -> float | None:
    """Median time from diagnosis to ART initiation (first entry into any
    ART1_Suppressed band), starting from Diagnosed_PreART/band 0.

    v1 simplification: starts the clock at band 0 regardless of the CD4
    band actually present at diagnosis. Given ART initiation is fast
    relative to CD4 decline, this is a minor approximation -- revisit if
    initiation hazard ends up small in the calibrated posterior."""
    target = [full_index(st, b) for st in with_base("ART1_Ramp")
              for b in range(N_CD4_BANDS)]
    M_abs = make_absorbing(M, target)
    init = np.zeros(N_FULL_STATES)
    init[full_index(tagged("Diagnosed_PreART", class_for_band(0)), 0)] = 1.0
    return _first_passage_median(M_abs, target, init, dt, horizon_years=5)


# Cascade statuses that count as "retained in care" for the retention targets.
# Second-line ART counts: a regimen switch is a treatment decision taken WITHIN
# care, not an exit from it, and the cohort studies these targets come from
# follow patients across switches. The previous definition counted only
# first-line, so anyone switched to second line was scored as having dropped
# out -- a definitional error rather than a modelling choice.
RETAINED_STATUSES = with_base("ART1_Ramp", "ART1_Suppressed", "ART1_Failing",
                              "ART2_Suppressed")


def initiation_within(M, months: int = 1, start_bands=None, dt=1 / 12) -> float:
    """Proportion of a newly-diagnosed cohort that has started first-line ART
    within `months`, allowing for the competing risks of disengaging or dying
    before initiation.

    Replaces time_to_art_median_years as the CALIBRATION statistic. A monthly
    cycle quantises any first-passage median to whole months, so the median
    could only ever return 1, 2, 3... months against a target of 0.5 -- an
    unreachable target contributing a fixed penalty that lambda_init could
    never reduce, and which dragged it from a prior of 8 to 10.05 for nothing.
    A proportion-within-a-window is smooth, monotone in lambda_init and sits
    at exactly the resolution the model has.

    v1 simplification retained from time_to_art_median_years: the cohort
    starts in CD4 band 0 unless `start_bands` is given. Over a one-month
    window CD4 barely moves, so this is immaterial here.
    """
    target = [full_index(st, b) for st in with_base("ART1_Ramp")
              for b in range(N_CD4_BANDS)]
    M_abs = make_absorbing(M, target)
    # The cohort is people at the MOMENT OF DIAGNOSIS, which is now a mixture:
    # a fraction SAME_DAY_INITIATION start ART immediately, the rest wait in
    # Diagnosed_PreART. Starting everyone in Diagnosed_PreART would measure
    # only the non-same-day tail and understate the proportion by ~pi.
    pi = SAME_DAY_INITIATION
    init = np.zeros(N_FULL_STATES)
    bands = [(0, 1.0)] if start_bands is None else list(enumerate(start_bands))
    for b, w in bands:
        init[full_index(tagged("Diagnosed_PreART", class_for_band(b)), b)] += w * (1.0 - pi)
        init[full_index(tagged("ART1_Ramp", class_for_band(b)), b)] += w * pi
    n_steps = int(round(months / 12 / dt))
    traj = propagate(M_abs, init, n_steps, dt)
    return float(traj[n_steps][target].sum())


def retention_at(M, months: int, dt=1 / 12) -> float:
    """Proportion of an ART1-initiation cohort (started 100% in
    ART1_Suppressed/band 0) still in care at `months` post-initiation --
    on either treatment line, suppressed or failing, i.e. not disengaged and
    not dead. See RETAINED_STATUSES.

    Patients who disengage and later re-engage re-enter the numerator, since
    they genuinely are back in care at the measurement point.

    OPEN QUESTION, deliberately left as-is: death currently counts AGAINST
    retention (deaths stay in the denominator). Conventions differ -- some
    cohort analyses censor deaths and report retention among survivors, which
    would raise these figures by roughly 0.6 points at 12 months and 1.0 at
    24. Resolving it means reading the definition used by the GLMM study
    behind the Ret_12m / Ret_24m targets rather than choosing on principle.
    """
    init = np.zeros(N_FULL_STATES)
    # Retention cohorts enrol at ART INITIATION, not at suppression, so the
    # cohort starts in ART1_Ramp. Starting it in ART1_Suppressed skipped the
    # ~12-week ramp, during which people can be lost, and so overstated
    # retention.
    # Band 0 is >=500, so a cohort enrolling there is Full by construction.
    init[full_index(tagged("ART1_Ramp", class_for_band(0)), 0)] = 1.0
    n_steps = int(round(months / 12 / dt))
    traj = propagate(M, init, n_steps, dt)
    row = traj[n_steps]
    return float(sum(row[full_index(st, b)]
                     for st in RETAINED_STATUSES
                     for b in range(N_CD4_BANDS)))


def resuppression_2L_at(M: np.ndarray, months: int = 12, dt=1 / 12) -> float:
    """Proxy for 2nd-line resuppression: proportion of a switch cohort
    (started 100% in ART2_Suppressed/band 0) still in ART2_Suppressed
    (not LTFU, not dead) `months` after the switch.

    Caveat: because the 44-state design has no ART2_Failing status, a
    'failed 2nd line' outcome isn't structurally distinct from LTFU here --
    this proxy will tend to overstate true resuppression if failures on
    2nd line are common. Treat as a rough check, not a precise target."""
    init = np.zeros(N_FULL_STATES)
    init[full_index(tagged("ART2_Suppressed", class_for_band(0)), 0)] = 1.0
    n_steps = int(round(months / 12 / dt))
    traj = propagate(M, init, n_steps)
    row = traj[n_steps]
    return sum(row[full_index(st, b)] for st in with_base("ART2_Suppressed")
               for b in range(N_CD4_BANDS))


def reengagement_at(M, months: int, start_bands=None, dt=1 / 12) -> float:
    """Cumulative proportion of a newly-disengaged cohort that has returned to
    care by `months`, allowing for the competing risk of dying first.

    Started 100% in LTFU_Recent (distributed across CD4 bands by
    `start_bands`, defaulting to band 0), with every Diagnosed_PreART state
    made absorbing so the mass accumulating there is the cumulative-incidence
    function for "first returned to care". Returns via LTFU_LongTerm are
    included, which is the point -- the slow trickle is what makes the
    observed curve plateau rather than stop.

    Compared against Bagayoko et al. 2020 (Mali, n = 3,650): 39.0% at 1 year,
    45.0% at 2 years, 47.0% at 3 years. See likelihoods.REENGAGEMENT_TARGETS.
    """
    target = [full_index(st, b) for st in with_base("Diagnosed_PreART")
              for b in range(N_CD4_BANDS)]
    M_abs = make_absorbing(M, target)
    init = np.zeros(N_FULL_STATES)
    if start_bands is None:
        init[full_index(tagged("LTFU_Recent", class_for_band(0)), 0)] = 1.0
    else:
        for b, w in enumerate(start_bands):
            init[full_index(tagged("LTFU_Recent", class_for_band(b)), b)] = w
    n_steps = int(round(months / 12 / dt))
    traj = propagate(M_abs, init, n_steps, dt)
    return float(traj[n_steps][target].sum())


def ltfu_mortality_hazard_ratio(M, months: int = 12, start_bands=None,
                                 dt=1 / 12) -> float:
    """Ratio of all-cause death risk over `months` between a cohort that has
    just disengaged (LTFU_Recent) and one remaining in care
    (ART1_Suppressed), both started from the SAME CD4 distribution.

    Model analogue of the mortality hazard ratio reported by tracing studies
    -- Brinkhof et al. 2010 put it between 6 and 23 across five sub-Saharan
    ART programmes. Matching the starting CD4 mix is essential: the observed
    ratio is inflated by the fact that people who disengage are sicker to
    begin with, and this statistic deliberately excludes that selection
    effect so the calibration is not asked to reproduce confounding. That is
    also why the target carries a very wide SD -- see
    likelihoods.HR_LTFU_TARGET.

    Risk ratio over a fixed window rather than an instantaneous hazard ratio,
    because that is what the cohort studies actually estimate and it stays
    finite when either arm has near-zero mortality.
    """
    if start_bands is None:
        start_bands = cd4_distribution_at_art_initiation(M)
    n_steps = int(round(months / 12 / dt))

    def death_risk(base):
        init = np.zeros(N_FULL_STATES)
        for b, w in enumerate(start_bands):
            init[full_index(tagged(base, class_for_band(b)), b)] = w
        row = propagate(M, init, n_steps, dt)[n_steps]
        return row[FULL_HIV_DEATH] + row[FULL_BG_DEATH]

    retained = death_risk("ART1_Suppressed")
    if retained <= 0:
        return float("nan")
    return float(death_risk("LTFU_Recent") / retained)


def cd4_distribution_at_art_initiation(M: np.ndarray, dt=1 / 12, horizon_years=25) -> np.ndarray:
    """Flow-weighted distribution (length N_CD4_BANDS, sums to 1) of the CD4
    band a person is in AT THE MOMENT they initiate 1st-line ART -- i.e. the
    Diagnosed_PreART(band b) -> ART1_Ramp(band b) transition -- for a
    cohort entering Undiagnosed/band 0.

    This is deliberately NOT a standing-population snapshot (that would
    answer 'what band are currently-suppressed people in', a different and
    less relevant question). It integrates the actual monthly flow into
    ART1_Ramp over the cohort's lifetime, which is the right quantity
    to compare against published 'CD4 distribution at ART start' cohort
    statistics (e.g. de Waal et al. 2024) -- those describe people AT THE
    MOMENT of initiation, not a cross-sectional mix of durations since.
    """
    M = np.asarray(M)
    time_varying = M.ndim == 3
    n_steps = int(horizon_years / dt)
    init = undiagnosed_init()
    traj = propagate(M, init, n_steps, dt)

    # Per-step initiation probability must come from the matrix in force at
    # that step, which is not constant once background mortality varies with
    # age (it competes with initiation in the same competing-risks step).
    if time_varying:
        year_of_step = np.minimum((np.arange(n_steps) * dt).astype(int), M.shape[0] - 1)

    flow = np.zeros(N_CD4_BANDS)
    for band in range(N_CD4_BANDS):
        for cls in CEILING_CLASSES:
            i = full_index(tagged("Diagnosed_PreART", cls), band)
            j = full_index(tagged("ART1_Ramp", cls), band)
            p_init = M[year_of_step, i, j] if time_varying else M[i, j]
            flow[band] += np.sum(traj[:n_steps, i] * p_init)
        # Same-day initiators never occupy Diagnosed_PreART, so they would be
        # invisible to the loop above -- and they are the MAJORITY of first
        # initiations. Count the Undiagnosed -> ART1_Ramp route too.
        u = full_index("Undiagnosed", band)
        v = full_index(tagged("ART1_Ramp", class_for_band(band)), band)
        p_sd = M[year_of_step, u, v] if time_varying else M[u, v]
        flow[band] += np.sum(traj[:n_steps, u] * p_sd)
    total = flow.sum()
    if total <= 0:
        # Previously this returned a uniform vector, which looks like a real
        # answer (median 225, the midpoint of the middle band) and silently
        # replaced the target. If no initiation flow is observed, something is
        # structurally wrong -- say so.
        raise ValueError(
            "no ART-initiation flow observed: the Diagnosed_PreART -> ART1_Ramp "
            "transition carried zero mass. Check that the initiation destination "
            "in _SAME_BAND_DEST matches the one measured here.")
    return flow / total


def cd4_percentile_at_art_initiation(band_probs: np.ndarray, q: float) -> float:
    """Interpolate the q-th percentile (0 < q < 1) of CD4 count (cells/uL)
    at ART initiation from a per-band probability vector (e.g. the output of
    cd4_distribution_at_art_initiation), assuming CD4 is uniformly
    distributed WITHIN each band. Band 0 (>=500, open-ended in reality) uses
    params.CD4_BAND_OPEN_TOP_ASSUMED as its upper edge -- see that constant's
    docstring for the caveat; only percentiles that land inside band 0 are
    affected."""
    from params import CD4_BAND_LOWER, CD4_BAND_UPPER

    cum = 0.0
    for b in range(N_CD4_BANDS - 1, -1, -1):  # walk lowest CD4 -> highest
        p = band_probs[b]
        if p > 0 and cum + p >= q:
            frac = (q - cum) / p
            return CD4_BAND_LOWER[b] + frac * (CD4_BAND_UPPER[b] - CD4_BAND_LOWER[b])
        cum += p
    return CD4_BAND_UPPER[0]  # q beyond all mass (shouldn't happen if probs sum to ~1)


def cd4_lt200_frac_at_art_initiation(band_probs: np.ndarray) -> float:
    """P(CD4<200 at ART initiation), from a per-band probability vector (e.g.
    the output of cd4_distribution_at_art_initiation). Unlike
    cd4_percentile_at_art_initiation, this needs NO within-band interpolation
    and NO open-top-band assumption -- CD4<200 exactly coincides with the
    model's own band boundary (bands >= BAND_INDEX_CD4_200 are entirely
    <200), so it's just a sum of band probabilities the model already
    computes. Deliberately preferred over the percentile targets it replaces
    for calibration robustness: see README ("CD4 targets: from 5 percentiles
    to 2 robust targets")."""
    from params import BAND_INDEX_CD4_200

    return float(np.sum(band_probs[BAND_INDEX_CD4_200:]))


if __name__ == "__main__":
    from params import load_parameters

    _assert_builders_agree()

    p = load_parameters()
    # build_specs rather than p.central_estimates(), because the LTFU-split
    # parameters live in likelihoods.FALLBACK_PRIORS until they are added to
    # data/parameters.xlsx.
    import likelihoods as _lk
    central = {sp.symbol: sp.central
               for sp in _lk.build_specs(p, _lk.SYSTEM2_SYMBOLS, _lk.SYSTEM2_KINDS)}
    mortality_curve = mortality_curve_at_age(ANCHOR_AGE)

    M = build_full_transition_matrix(central, mortality_curve, p.fixed["recov_rate"])
    print(f"Row-sum-to-1 check passed for all {M.shape[0]} states.\n")

    init = undiagnosed_init()

    n_steps = int(20 / (1 / 12))
    traj = propagate(M, init, n_steps)

    print("Cascade proportions (among alive) at years 5, 10, 20:")
    for yr in (5, 10, 20):
        idx = int(yr / (1 / 12))
        s = cascade_summary(traj[idx])
        print(f"  Year {yr}: alive={s['alive_mass']:.3f}  "
              f"%diag={s['pct_diagnosed_of_alive']:.3f}  "
              f"%onART|diag={s['pct_on_art_of_diagnosed']:.3f}  "
              f"%supp|onART={s['pct_suppressed_of_on_art']:.3f}")
    print("\n(UNAIDS targets: p1=0.93 diagnosed, p2=0.91 on ART, p3=0.95 suppressed "
          "-- not expected to match yet, these are central-estimate priors, not a "
          "calibrated fit.)")

    stat = stationary_cascade_proportions(traj)
    print(f"\nStationary (cross-sectional-equivalent) cascade proportions:")
    print(f"  p1 (diagnosed): {stat['p1_diagnosed_of_alive']:.3f}  (target 0.93)")
    print(f"  p2 (on ART|diagnosed): {stat['p2_onart_of_diagnosed']:.3f}  (target 0.91)")
    print(f"  p3 (suppressed|onART): {stat['p3_suppressed_of_onart']:.3f}  (target 0.95)")

    t_art = time_to_art_median_years(M)
    ret12 = retention_at(M, 12)
    ret24 = retention_at(M, 24)
    supp2l = resuppression_2L_at(M)
    print("\nMicro targets (central-estimate, uncalibrated):")
    print(f"  Median time diagnosis->ART: {t_art*12:.2f} months  (target 0.5 months)")
    print(f"  12-month retention: {ret12:.3f}  (target 0.796)")
    print(f"  24-month retention: {ret24:.3f}  (target 0.812)")
    print(f"  2nd-line resuppression proxy (12mo): {supp2l:.3f}  (target 0.85)")

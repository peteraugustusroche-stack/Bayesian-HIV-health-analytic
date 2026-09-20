"""Priors and log-likelihoods for both systems, shared by the numpy sampler and
the PyMC models so both calibrate the same posterior. Positive parameters get a
LogNormal with median at the spreadsheet central (mu = ln(central), sigma =
sd/central), which also sidesteps the "SD > central" problem several of them have
under a Normal. Each target is a noisy observation of a simulator-derived summary,
Normal(model_implied, target_sd)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from params import load_parameters, HORIZON_YEARS
import simulate_natural_history as nh
from natural_history_params import (
    SYSTEM1_SYMBOLS as NH_SYSTEM1_SYMBOLS,
    KAPPA_PROG_SIGMA, KAPPA_MORT_SIGMA,
)
import simulate_cascade as casc


@dataclass
class ParamSpec:
    symbol: str
    kind: str          # "lognormal" or "normal"
    central: float
    sd: float

    @property
    def log_mu(self):
        """mu of the underlying normal, for lognormal params."""
        return math.log(self.central)

    @property
    def log_sigma(self):
        sigma = self.sd / self.central
        return max(sigma, 0.05)  # floor to avoid a degenerate/zero-width prior

    def log_prior(self, value: float) -> float:
        if self.kind == "lognormal":
            if value <= 0:
                return -np.inf
            z = (math.log(value) - self.log_mu) / self.log_sigma
            return -0.5 * z ** 2 - math.log(value * self.log_sigma * math.sqrt(2 * math.pi))
        else:  # normal
            z = (value - self.central) / self.sd
            return -0.5 * z ** 2 - math.log(self.sd * math.sqrt(2 * math.pi))

    def sample_prior(self, rng: np.random.Generator) -> float:
        if self.kind == "lognormal":
            return float(rng.lognormal(self.log_mu, self.log_sigma))
        return float(rng.normal(self.central, self.sd))

    def propose(self, current: float, rng: np.random.Generator, step_scale: float = 0.3) -> float:
        """Random-walk proposal in the natural support of the parameter."""
        if self.kind == "lognormal":
            return float(current * math.exp(rng.normal(0, step_scale * self.log_sigma)))
        return float(current + rng.normal(0, step_scale * self.sd))


# Replaces the spreadsheet's Lodi et al. 2011 values (78% male, 55% MSM, subtype B
# -- poor ESA match) with Pantazis et al. 2012 estimates for sub-Saharan African
# seroconverters at the model's sex mix and anchor age; SDs propagate Pantazis's 95%
# CIs assuming independent coefficients, which likely OVERSTATES them. T_survival
# stays on the spreadsheet (Pantazis has no mortality); T_<500 is out, ~41% of the
# cohort starting below 500 already.
SYSTEM1_TARGET_OVERRIDES = {
    "T_<350": (4.14, 0.64),
    "T_<200": (8.11, 1.11),
}


def system1_targets(params) -> dict:
    """symbol -> (central, sd), spreadsheet values with Pantazis overrides."""
    out = {k: params.targets[k] for k in ("T_<350", "T_<200", "T_survival")}
    out.update(SYSTEM1_TARGET_OVERRIDES)
    return out


SYSTEM1_SYMBOLS = list(NH_SYSTEM1_SYMBOLS)
SYSTEM1_KINDS = {"κ_prog": "lognormal", "κ_mort": "lognormal"}

# mu_perm_diseng and mu_reengage_late are NOT free: they are fixed from the Mali
# re-engagement shape (params.LTFU_PERM_DISENGAGE / LTFU_REENGAGE_LATE), while
# mu_re-engage stays free and sets the level from UNAIDS coverage.
SYSTEM2_SYMBOLS = ["δ_bg", "δ_symp", "γ_diag", "λ_init", "μ_vf1", "μ_ltfu",
                   "μ_re-engage", "φ_ltfu", "μ_switch"]
SYSTEM2_KINDS = {s: "lognormal" for s in SYSTEM2_SYMBOLS}

# Priors for parameters with no row in data/parameters.xlsx yet.
FALLBACK_PRIORS = {
    # LTFU mortality multiplier on the untreated CD4-specific curve, centred at
    # 1.0 (LTFU = never-treated mortality), wide, identified by HR_LTFU.
    "φ_ltfu": (1.0, 0.75),

    # Median 1: the published Glaubius tables are the central case. Spreads live in
    # natural_history_params; kappa_mort is deliberately wider than Glaubius's own
    # credible intervals, between-source disagreement exceeding either's precision.
    "κ_prog": (1.0, KAPPA_PROG_SIGMA),
    "κ_mort": (1.0, KAPPA_MORT_SIGMA),
}


# mu_ltfu: the spreadsheet prior (0.2 +/- 0.1) is retention-derived and puts the value
# reproducing UNAIDS p2 (~0.04/yr) over three prior SDs out. Re-centred weakly
# informative, p2 being what identifies it.
PRIOR_OVERRIDES = {
    "μ_ltfu": (0.08, 0.08),
}


def build_specs(params, symbols, kinds) -> list[ParamSpec]:
    specs = []
    for sym in symbols:
        if sym in PRIOR_OVERRIDES:
            central, sd = PRIOR_OVERRIDES[sym]
        elif sym in params.free:
            fp = params.free[sym]
            central, sd = fp.central, fp.sd
        elif sym in FALLBACK_PRIORS:
            central, sd = FALLBACK_PRIORS[sym]
        else:
            raise KeyError(
                f"No prior for {sym!r}: absent from data/parameters.xlsx and "
                f"from likelihoods.FALLBACK_PRIORS.")
        specs.append(ParamSpec(symbol=sym, kind=kinds[sym], central=central, sd=sd))
    return specs


def all_central_estimates(params) -> dict:
    """Central (prior) value for EVERY free parameter. params.central_estimates()
    knows only the spreadsheet and silently omits parameters defined in code."""
    out = {}
    for symbols, kinds in ((SYSTEM1_SYMBOLS, SYSTEM1_KINDS),
                           (SYSTEM2_SYMBOLS, SYSTEM2_KINDS)):
        for spec in build_specs(params, symbols, kinds):
            out[spec.symbol] = spec.central
    return out


def normal_logpdf(x, mean, sd):
    if x is None or sd <= 0:
        return -np.inf
    z = (x - mean) / sd
    return -0.5 * z ** 2 - math.log(sd * math.sqrt(2 * math.pi))


# System 1: Natural History

# System 1 is NOT fitted: schedules come from Glaubius et al. 2021 and the three targets
# (CD4<350/CD4<200 from Pantazis et al. 2012, untreated survival from Isingo et al.
# 2007) are OUT-OF-SAMPLE VALIDATION. True fits kappa_prog/kappa_mort to them instead,
# forfeiting that claim.
USE_SYSTEM1_LIKELIHOOD = False


def system1_log_likelihood(kappa_prog: float, kappa_mort: float, params) -> float:
    if kappa_prog <= 0 or kappa_mort <= 0:
        return -np.inf
    if not USE_SYSTEM1_LIKELIHOOD:
        return 0.0

    imp = system1_model_implied(kappa_prog, kappa_mort, params)
    _t = system1_targets(params)
    return sum(normal_logpdf(imp[k], c, sd) for k, (c, sd) in _t.items())


def system1_log_posterior(theta: np.ndarray, specs: list[ParamSpec], params) -> float:
    kappa_prog, kappa_mort = theta
    lp = specs[0].log_prior(kappa_prog) + specs[1].log_prior(kappa_mort)
    if not np.isfinite(lp):
        return -np.inf
    ll = system1_log_likelihood(kappa_prog, kappa_mort, params)
    return lp + ll


def system1_model_implied(kappa_prog: float = 1.0, kappa_mort: float = 1.0,
                           params=None) -> dict:
    """The three validation quantities, in years. Age-varying background
    mortality and Glaubius age-group steps are resolved inside the engine from
    ANCHOR_AGE + model time."""
    return {
        "T_<350": nh.first_passage_median_years(2, kappa_prog, kappa_mort),
        "T_<200": nh.first_passage_median_years(4, kappa_prog, kappa_mort),
        "T_survival": nh.overall_survival_median_years(kappa_prog, kappa_mort),
    }


# System 2: Care Cascade

# CD4-at-ART-initiation targets (de Waal et al. 2024 supplementary data, pooled
# Southern Africa excl. RSA + East Africa, 2016-2019, N-weighted). Only 2 of the 5
# published percentiles are used: q1/q3 correlate with the median but would score as
# independent, p95 depends on open-top-band interpolation, p5 sits where HIV
# mortality is steepest. CD4_lt200_ART needs no interpolation and is separate below.
CD4_QUANTILE_TARGETS = {
    "CD4_median_ART": 0.50,
}

# Re-engagement after disengagement, Bagayoko et al. 2020 (PLoS ONE 15(9):e0238687),
# Mali, n = 3,650 LTFU after ART initiation: KM cumulative return to care 39.0% at
# 1y, 45.0% at 2y, 47.0% at 3y. The 0.06 SDs are NOT the published CIs (~1pp, which
# would dominate every other System 2 target) but transportability from West to ESA.
REENGAGEMENT_TARGETS = {
    "Reeng_1y": (0.390, 0.06, 12),
    "Reeng_2y": (0.450, 0.06, 24),
    "Reeng_3y": (0.470, 0.06, 36),
}

# Diagnostic only: retention (~17%/yr disengagement), Bagayoko (~47% ever return) and
# UNAIDS ESA (83% of diagnosed on treatment) are incompatible in a closed cohort --
# fitting all three missed all three and pushed ~31% of person-time into LTFU (~30%
# more DALYs per acquisition), so p1/p2/p3 anchor the cascade instead.
USE_REENGAGEMENT_LIKELIHOOD = False

# Cohort retention likewise diagnostic only: it conflicts with UNAIDS coverage through
# mu_ltfu alone (p2 met near 0.05/yr, Ret_12m near 0.25, neither by any single value).
# Retention is FACILITY attrition counting silent transfers as lost, so a model with
# no facility dimension should read it high.
USE_RETENTION_LIKELIHOOD = False

# Time from diagnosis to ART initiation, as a PROPORTION within one month: a median
# target is unusable at a monthly cycle, where first-passage medians quantise to whole
# months and the model returns 1.0 whatever lambda_init is. Source IeDEA Southern
# Africa, N = 29,017, universal test-and-treat: median 0 days (IQR 0-7), same-day
# 18,584/29,017 = 64.0%. Central 0.85 extrapolates from those and the >=75% by day
# seven the IQR implies; SD 0.07 keeps that floor inside 2 SD.
ART_INITIATION_TARGET = ("ART_1m", 0.85, 0.07, 1)

# Mortality HR, recently-disengaged vs in care, over 12 months. Brinkhof et al. 2010
# (PLoS ONE 5(11):e14149) report 6 to 23 across five sub-Saharan ART programmes;
# central 12 is the midpoint programme and SD 4.34 = (23-6)/3.92 treats that range as
# a 95% interval -- wide, the published ratio being confounded by selection and the
# model's statistic not.
HR_LTFU_TARGET = ("HR_LTFU", 12.0, 4.34)

# Diagnostic only: the model cannot reach the target -- sweeping phi_ltfu 1 to 32 moves
# the implied HR only 2.94 to 8.99, the LTFU pool sitting mostly at high CD4 where
# untreated mortality is 0.00011/yr. Neither cause is fixable by phi: that curve is
# likely too flat (five of seven bands interpolated) and mu_ltfu is CD4-independent.
USE_HR_LTFU_LIKELIHOOD = False

# Second-line resuppression: diagnostic only, structurally. With no ART2_Failing state
# in the 44-state design the only exits from ART2_Suppressed are LTFU and a little
# death, so the proxy returns exp(-0.05) = 0.951 almost regardless of parameters, and
# it measures maintenance from an already-suppressed cohort where the 0.85 target is
# achieve-AND-maintain after switching.
USE_SUPP2L_LIKELIHOOD = False

# Spreadsheet symbols feeding the p1/p2/p3 coverage terms, by base year. 2019
# (UNAIDS Data 2020, p.40) is the base case, closer in time to the de Waal
# CD4-at-initiation data (2016-2019) than 2024 (UNAIDS 2025 Global AIDS Update).
CASCADE_YEAR_KEYS = {
    "2019": {"p1": "p1_2019", "p2": "p2_2019", "p3": "p3_2019"},
    "2024": {"p1": "p1", "p2": "p2", "p3": "p3"},
    # Counterfactual arm: the UNAIDS 2025 target, 95% diagnosed, 95% of those
    # treated, 95% of those suppressed -- 85.7% of everyone living with HIV
    # virally suppressed. Not an observed cascade, so the CD4-at-initiation
    # targets are dropped for it (see CD4_TARGET_ARMS): a cohort found this
    # much earlier would necessarily start treatment healthier, and holding
    # CD4 at its observed value would make the two sets of targets fight.
    "95-95-95": {"p1": "p1_959595", "p2": "p2_959595", "p3": "p3_959595"},
    "WC2023": {"p1": "p1_wc2023", "p2": "p2_wc2023", "p3": "p3_wc2023"},
}

# Cascade targets absent from data/parameters.xlsx. WC2023 is the Western Cape cyclical
# cascade (Euvrard et al., PLOS Med 2024, doi:10.1371/journal.pmed.1004407), 31 Dec
# 2023, 494,370 people: the same point-in-time estimand as UNAIDS TX_CURR_NAT but on
# individually-linked records, 90-day interruption threshold, reading 67% of PLHIV on
# ART against 84% for ESA. p1 matches UNAIDS 2024, so the pair isolates p2/p3.
#   p1 = 0.93   494,370 diagnosed / Thembisa PLHIV. SD 0.07: the first pass gave
#               115% before excluding people not seen for 2 years.
#   p2 = 0.72   67% of PLHIV on ART / 93% diagnosed. SD 0.08.
#   p3 = 0.89   of 305,699 on long-term ART, 253,229 (83%) <50 copies plus 18,506
#               (6%) <1,000. SD 0.06: suppression among LONG-TERM ART only (no
#               viral load expected before 4 months), not everyone on ART.
#: Arms whose CD4-at-initiation targets enter the likelihood. The counterfactual
#: arm is excluded so that CD4 at initiation is an OUTPUT of hitting 95-95-95
#: rather than a constraint on it.
CD4_TARGET_ARMS = ("2019", "2024", "WC2023")

CASCADE_TARGET_OVERRIDES = {
    # SD 0.02 is tight because these are a stated target, not an estimate with
    # sampling error; it keeps the fitted cascade at the target rather than
    # letting the priors pull it away.
    "p1_959595": (0.950, 0.020),
    "p2_959595": (0.950, 0.020),
    "p3_959595": (0.950, 0.020),
    "p1_wc2023": (0.930, 0.070),
    "p2_wc2023": (0.720, 0.080),
    "p3_wc2023": (0.890, 0.060),
}


def cascade_target(params, symbol):
    """(central, sd) for a cascade symbol: spreadsheet, else the overrides."""
    if symbol in params.targets:
        return params.targets[symbol]
    if symbol in CASCADE_TARGET_OVERRIDES:
        return CASCADE_TARGET_OVERRIDES[symbol]
    raise KeyError(f"No target value for cascade symbol {symbol!r} in either "
                   f"data/parameters.xlsx or CASCADE_TARGET_OVERRIDES")


def system2_log_likelihood(cascade_vals: dict, kappa_prog: float, kappa_mort: float,
                            params, horizon_years: float = HORIZON_YEARS,
                            cascade_year: str = "2019") -> float:
    central = dict(cascade_vals)

    # Per-year stack: background mortality varies with cohort age (~29-54 over the
    # 25-year window) and progression steps at the Glaubius boundary at 35.
    stack = casc.build_full_matrix_stack(central, params.fixed["recov_rate"],
                                          horizon_years=horizon_years,
                                          kappa_prog=kappa_prog,
                                          kappa_mort=kappa_mort)
    # Short-horizon targets use the anchor-age matrix: over <=2 years the
    # background hazard moves a fraction of a percent, noise next to the 15-25%
    # lost to LTFU and treatment failure.
    M_anchor = stack[0]

    init = casc.undiagnosed_init()
    n_steps = int(horizon_years / (1 / 12))
    traj = casc.propagate(stack, init, n_steps)

    stat = casc.stationary_cascade_proportions(traj)
    t_art = casc.time_to_art_median_years(M_anchor)
    ret12 = casc.retention_at(M_anchor, 12)
    ret24 = casc.retention_at(M_anchor, 24)
    supp2l = casc.resuppression_2L_at(M_anchor)
    cd4_dist = casc.cd4_distribution_at_art_initiation(stack, horizon_years=horizon_years)
    # Both start from the model's own CD4-at-initiation mix, so the HR excludes
    # the selection effect in the published estimate.
    reeng = {k: casc.reengagement_at(M_anchor, months, cd4_dist)
             for k, (_, _, months) in REENGAGEMENT_TARGETS.items()}
    hr_ltfu = casc.ltfu_mortality_hazard_ratio(M_anchor, 12, cd4_dist)

    year_keys = CASCADE_YEAR_KEYS[cascade_year]
    ll = 0.0
    p1c, p1sd = cascade_target(params, year_keys["p1"])
    p2c, p2sd = cascade_target(params, year_keys["p2"])
    p3c, p3sd = cascade_target(params, year_keys["p3"])
    # T_ART is not a likelihood term (see ART_INITIATION_TARGET), nor retention.
    supp2c, supp2sd = params.targets["Supp_2L"]   # diagnostic only unless fitted

    ll += normal_logpdf(stat["p1_diagnosed_of_alive"], p1c, p1sd)
    ll += normal_logpdf(stat["p2_onart_of_diagnosed"], p2c, p2sd)
    ll += normal_logpdf(stat["p3_suppressed_of_onart"], p3c, p3sd)
    _art_name, art_c, art_sd, art_months = ART_INITIATION_TARGET
    ll += normal_logpdf(casc.initiation_within(M_anchor, art_months), art_c, art_sd)
    if USE_RETENTION_LIKELIHOOD:
        ret12c, ret12sd = params.targets["Ret_12m"]
        ret24c, ret24sd = params.targets["Ret_24m"]
        ll += normal_logpdf(ret12, ret12c, ret12sd)
        ll += normal_logpdf(ret24, ret24c, ret24sd)
    if USE_SUPP2L_LIKELIHOOD:
        ll += normal_logpdf(supp2l, supp2c, supp2sd)
    if USE_REENGAGEMENT_LIKELIHOOD:
        for key, (tc, tsd, _months) in REENGAGEMENT_TARGETS.items():
            ll += normal_logpdf(reeng[key], tc, tsd)
    if USE_HR_LTFU_LIKELIHOOD:
        _hr_name, hr_c, hr_sd = HR_LTFU_TARGET
        ll += normal_logpdf(hr_ltfu, hr_c, hr_sd)
    if cascade_year in CD4_TARGET_ARMS:
        for symbol, q in CD4_QUANTILE_TARGETS.items():
            if symbol not in params.targets:
                continue
            tc, tsd = params.targets[symbol]
            model_val = casc.cd4_percentile_at_art_initiation(cd4_dist, q)
            ll += normal_logpdf(model_val, tc, tsd)
        if "CD4_lt200_ART" in params.targets:
            tc, tsd = params.targets["CD4_lt200_ART"]
            model_val = casc.cd4_lt200_frac_at_art_initiation(cd4_dist)
            ll += normal_logpdf(model_val, tc, tsd)
    return ll


def system2_model_implied(cascade_vals: dict, kappa_prog: float, kappa_mort: float,
                           params, horizon_years: float = HORIZON_YEARS) -> dict:
    """The model's own p1/p2/p3 etc. cascade_year changes only what these are
    compared to in the likelihood, never what is computed here."""
    central = dict(cascade_vals)

    stack = casc.build_full_matrix_stack(central, params.fixed["recov_rate"],
                                          horizon_years=horizon_years,
                                          kappa_prog=kappa_prog,
                                          kappa_mort=kappa_mort)
    M_anchor = stack[0]
    init = casc.undiagnosed_init()
    n_steps = int(horizon_years / (1 / 12))
    traj = casc.propagate(stack, init, n_steps)
    stat = casc.stationary_cascade_proportions(traj)
    t_art = casc.time_to_art_median_years(M_anchor)
    cd4_dist = casc.cd4_distribution_at_art_initiation(stack, horizon_years=horizon_years)
    reeng = {k: casc.reengagement_at(M_anchor, months, cd4_dist)
             for k, (_, _, months) in REENGAGEMENT_TARGETS.items()}
    hr_ltfu = casc.ltfu_mortality_hazard_ratio(M_anchor, 12, cd4_dist)
    cd4_quantiles = {symbol: casc.cd4_percentile_at_art_initiation(cd4_dist, q)
                      for symbol, q in CD4_QUANTILE_TARGETS.items()}
    cd4_quantiles["CD4_lt200_ART"] = casc.cd4_lt200_frac_at_art_initiation(cd4_dist)

    return {
        "p1": stat["p1_diagnosed_of_alive"],
        "p2": stat["p2_onart_of_diagnosed"],
        "p3": stat["p3_suppressed_of_onart"],
        "T_ART": t_art * 12 if t_art is not None else None,
        ART_INITIATION_TARGET[0]: casc.initiation_within(M_anchor, ART_INITIATION_TARGET[3]),
        "Ret_12m": casc.retention_at(M_anchor, 12),
        "Ret_24m": casc.retention_at(M_anchor, 24),
        **cd4_quantiles,
        "Supp_2L": casc.resuppression_2L_at(M_anchor),
        **reeng,
        HR_LTFU_TARGET[0]: hr_ltfu,
    }


def system2_log_posterior(theta: np.ndarray, specs: list[ParamSpec],
                           nh_draw: tuple[float, float], params,
                           cascade_year: str = "2019") -> float:
    """theta = cascade free params only. nh_draw is one (kappa_prog, kappa_mort)
    draw supplied externally from the Glaubius-derived prior, per the cut-model
    design: cascade data must not feed back into natural history."""
    cascade_vals = {spec.symbol: theta[i] for i, spec in enumerate(specs)}
    lp = sum(spec.log_prior(theta[i]) for i, spec in enumerate(specs))
    if not np.isfinite(lp):
        return -np.inf
    kappa_prog, kappa_mort = nh_draw
    ll = system2_log_likelihood(cascade_vals, kappa_prog, kappa_mort, params,
                                 cascade_year=cascade_year)
    return lp + ll

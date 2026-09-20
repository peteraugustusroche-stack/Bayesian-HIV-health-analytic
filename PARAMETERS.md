# Parameter provenance

Sourcing, derivations and caveats for the fixed parameters defined in code, keyed by the
constant name. Parameters read from `data/parameters.xlsx` carry their own source column;
this file covers the ones that live in `params.py` and `health_econ_params.py` because
they are not in the spreadsheet.

Items marked **CAVEAT** are for the write-up's limitations section. Items marked
**UNVERIFIED** have an attribution that could not be confirmed against a publication.

---

## Cohort and horizon

### `ANCHOR_AGE = 29.0`
Age at cohort entry (seroconversion). The cohort is age-synchronised — everyone enters at
this age simultaneously — which is what lets the model use an age-varying background
mortality schedule with **no age dimension in the state space**: age is deterministically
`ANCHOR_AGE + (years since entry)`. See `life_table.py`.

29 rather than 28 is the mean age at HIV acquisition among adults in eastern and southern
Africa (UNAIDS). It does more work than a round number should: it indexes both the
cause-deleted background mortality schedule and the remaining-life-expectancy YLL
reference, so both move with it.

It also sits inside the 15–29 reference band of the Pantazis mixed model used for
acquisition CD4 (`acquisition_cd4.py`), where age enters the CD4 decline slope only — so
no age adjustment is needed there. **CAVEAT:** 29 is the *top* of that band; a mean age any
higher would require one.

### `HORIZON_YEARS = 71`
One horizon, used by **both** the calibration and the health-economic run. Previously the
likelihood integrated over 25 years while `health_economics.py` accounted over 71 —
parameters estimated under one window and applied under another.

71 years (age 29 → 100, the last age in the cause-deleted life table, by which point the
cohort is extinct) is not a free choice. `stationary_cascade_proportions` derives p1/p2/p3
by integrating the cohort trajectory over duration-since-infection, on the argument that
under constant incidence the duration density among the living is proportional to the
cohort's survival curve. That identity requires integrating to extinction. Truncating at 25
years drops the long-duration survivors, who are disproportionately diagnosed and on
treatment, and therefore biases p1 and p2 **downward**.

Measured effect at the arm-A posterior mean: p1 0.763 → 0.848 (z −2.10 → −0.43), p2 0.838 →
0.814. CD4-at-initiation moves the other way (median 347 → 384), so this does not improve
overall fit — it removes an inconsistency and relocates the residual misfit onto the CD4
distribution, where it belongs.

---

## CD4 band geometry

`CD4_BAND_NAMES` runs healthiest (0, ≥500) to most advanced (6, <50).

`CD4_BAND_WIDTH` is used only to convert the cells/yr reconstitution rate into a
band-crossing hazard. Band 0 is open-ended above and is never a "width being left" for
decline; band 6 has no lower bound, so it takes a pseudo-width matching band 5 for symmetry
when converting reconstitution into the 6 → 5 hazard.

`CD4_BAND_LOWER` / `CD4_BAND_UPPER` support within-band linear interpolation when comparing
the model's discrete band distribution against continuous CD4 percentiles (the de Waal et
al. 2024 CD4-at-ART-initiation targets).

`CD4_BAND_OPEN_TOP_ASSUMED = 1500.0` is a **documented modelling assumption, not sourced
data**, for band 0's upper edge. It affects only a percentile estimate that happens to fall
inside band 0.

---

## Mortality

### Background mortality
Not a scalar. Comes from the cause-deleted ESA life table — see `life_table.py` and
`hazards.background_mortality(age)`. The previous flat 0.006/yr stand-in was ~3× too high at
age 28 and too low beyond ~50, and implied 76% non-HIV survival from age 28 to 73 against
the table's 54%.

### `EXCESS_HAZARD_ON_ART = 0.001`
HIV-attributable excess mortality on suppressive ART, framed as an **additive excess
hazard** (a constant absolute rate on top of the age-specific background), not as a
multiplicative SMR.

That choice became load-bearing once background was made age-varying: under a multiplicative
SMR the implied excess would rise ~21-fold between age 28 (bg 0.00202/yr) and age 70 (bg
0.04276/yr) — a strong claim to defend over a 45-year horizon. The HIV cohort literature has
moved toward reporting excess mortality *rates* for the same reason: SMRs are unstable
across age because the denominator moves so much.

This also fixed an accounting bug. The old code returned `background × SMR` into the
`hiv_death` channel while a separate `bg_death` hazard of `background` was **also** applied,
so suppressed patients faced 2× background mortality with half of it mislabelled as HIV
death (and hence accruing YLL). Total mortality on ART is now background + excess, with only
the excess attributed to HIV.

**Source:** Rodger AJ, et al. Mortality in well controlled HIV in the continuous
antiretroviral therapy arms of the SMART and ESPRIT trials compared with the general
population. *AIDS* 2013;27(6):973–979. doi:10.1097/QAD.0b013e32835cae9c

*Why this source and not an ESA cohort.* The hazard is applied only to `ART1_Suppressed` and
`ART2_Suppressed` (`simulate_cascade._ONART_MORT`), so its estimand is the excess mortality
of someone virally suppressed with a recovered CD4 count — not the excess of everyone on
ART. Rodger et al. restricted to exactly that group: non-IDU adults on continuous ART with
suppressed viral load and CD4 ≥350 in the previous six months.

*Derivation.* 62 deaths over 12,357 person-years is a **total** mortality rate of 5.02 per
1000 py. The reported SMRs bound how much is excess:

| Stratum | SMR (95% CI) | Implied excess per 1000 py |
|---|---|---|
| CD4 >500 | 1.00 (0.69–1.40) | 0 |
| CD4 350–499 | 1.77 (1.17–2.55) | 5.02 × (1 − 1/1.77) = 2.18 |

So the person-time-weighted excess lies in [0, 2.2] per 1000 py depending on CD4 mix.
0.001/yr sits inside that range.

*Why the previous 0.005/yr was wrong.* It reached for the ~3–10 per 1000 py excess reported
for ESA ART cohorts as a whole — a different quantity, which includes deaths among people
failing treatment, presenting with advanced disease, or disengaged. This model already
carries all three in separate compartments (`ART1_Failing`, the untreated mortality curve,
the two LTFU states), so using it here double-counted them. 0.005 is also close to Rodger's
*total* rate, i.e. it conflated total with excess.

**CAVEAT:** SMART/ESPRIT is a high-income, non-IDU, trial-enrolled population. Whether the
same excess holds for virally suppressed people in ESA is untested. Well-matched on
estimand, weakly transportable; an ESA-specific estimate would be a material improvement.

*Sensitivity.* Influential. At posterior means, moving 0.005 → 0.001 shifts DALYs per
acquisition 5.87 → 5.05 (−14%) and DALYs averted 11.29 → 12.11 (+7%). Set to 0.0 for a "no
excess once suppressed" scenario. Re-run the pipeline after changing it.

### `ART_SMR = [1.0, 1.8, 2.3, 2.3, 4.6, 7.0, 10.0]`
On-ART mortality while virally suppressed, as SMR on **current** CD4 × background.
`EXCESS_HAZARD_ON_ART` is retained as the ≥500 anchor and for the sensitivity script, but
applying it flat across every CD4 band was the error it replaced.

SMART/ESPRIT measures excess among suppressed patients who were largely **reconstituted**
(median CD4 ~600), so 0.001/yr is the excess at *high* CD4. It is not the excess for someone
suppressed at CD4 30, who in this model recovers through the low bands over several years.

**Source, verbatim from `model_functions_v3.R` `art_smr()`:** ≥500 → 1.0, ≥350 → 1.8, ≥200 →
2.3, ≥100 → 4.6, ≥50 → 7.0, <50 → 10.0, then `max(s, art_smr_floor)`. The source breaks at
200, not 250, so this model's 250–349 and 200–249 bands both take 2.3.

**UNVERIFIED:** the CD4 gradient is attributed in `CALIBRATION_NOTES.md` to PISCIS (achieved
CD4 200–500 ~1.95×, <200 ~4.6×). No PISCIS publication reporting SMR by achieved CD4 with
these values could be located; 1.95 is not implemented anywhere; the single 200–500 stratum
is implemented here as two values (1.8 and 2.3); and the upper of those is Rodger's 1.77
rather than anything from PISCIS. Treat 2.3 and 4.6 as modelling assumptions until a source
is found. `CALIBRATION_NOTES.md` already records that 7.0 and 10.0 are extrapolated.

Applied as excess over background, `(SMR − 1) × bg(age)`, so the separate `bg_death` channel
plus this excess sum to `SMR × bg` with no double counting.

*Why it matters:* `CALIBRATION_NOTES.md` v3.6 records that the previous treated-arm hazard
"gave ~0.02%/yr at high CD4, so recovered patients were effectively immortal — understating
the with-care per-acquisition DALY". The same flaw was present here in a different form, and
bites hardest on the low-CD4 on-treatment person-time that LTFU-and-return generates.

### `ART_SMR_FLOOR = 1.0` (removed from the base case)
Was 1.10, carried from `param_table.csv` and justified there as the "persistent excess
mortality of treated HIV vs the general population even at full reconstitution (residual
non-AIDS events, immune activation)", cited to ART-CC / D:A:D — cohort names rather than
publications.

*Why removed.* Rodger et al. 2013 report SMR 1.00 (95% CI 0.69–1.40) at CD4 ≥500, and their
deaths were overwhelmingly non-AIDS: cardiovascular or sudden death 31%, non-AIDS malignancy
19%, AIDS 3%. The "residual non-AIDS events" the floor represented are already *inside*
Rodger's 1.00; counting them again as a floor above the point estimate double-counts them.
The era argument points the same way — SMART/ESPRIT ran on d4T/AZT/EFV-era regimens, so any
drug-attributable component should be smaller under DTG, not larger. Choosing 1.10 from
inside the confidence interval was preferring a number to using the point estimate.

*The one argument that does point above 1.0* is transportability, and it is not the one the
floor claimed. Rodger's comparator is a high-income general population; in a high-TB-burden
setting residual infectious risk persists after full reconstitution — TB incidence 2.7 per
100py at CD4 >700 on ART against 0.62 in the HIV-uninfected population of the same
community, a 4.4-fold excess (Gupta et al. 2012, doi:10.1371/journal.pone.0034156). At 5–10%
case fatality that is ~1–2 excess deaths per 1000py against a background of ~3.1 per 1000py
at age 35, i.e. an SMR of order 1.3–1.7.

Not adopted, because it is a sketch rather than a derived estimate and the Cape Town cohort
is high-burden even for ESA. The constant is left in place, set to 1.0, so the mechanism can
still be exercised in sensitivity analysis — raising it is the correct way to test an
ESA-specific residual-risk assumption. Worth about **+0.11 DALYs** per acquisition at 1.10
and **+0.31** at 1.30.

### `ART_SMR_LOG_SD = 0.19`
Multiplicative uncertainty on the whole on-ART SMR ladder, log-scale SD.

The ladder itself is fixed, so before this the model carried **no** uncertainty at all in
on-ART mortality: removing the floor took out the only parameter that had any, leaving the
nine cascade hazards and two natural-history scale factors — which contribute under 1% of
variance — as the entire uncertainty budget.

Spread taken from the two rungs Rodger et al. actually estimate:

| Stratum | SMR (95% CI) | log-scale SD (lower/upper) |
|---|---|---|
| CD4 ≥500 | 1.00 (0.69–1.40) | 0.171 / 0.189 |
| CD4 350–499 | 1.77 (1.17–2.55) | 0.186 / 0.211 |

Two independently estimated bands implying almost the same log-scale spread is the argument
for carrying it as a **common multiplicative factor** across the ladder rather than as
per-band intervals; 0.19 is the round number those four figures bracket.

Applied to the SMR, **not** to the excess — see `hazards.smr_excess_for_multiplier` for why
that distinction decides whether this propagates anything at all.

Drawn once per posterior draw in `run_posterior_analysis.py`. **Not** a calibration
parameter: the cascade targets barely inform it, and adding a tenth weakly identified
parameter to a gradient-free sampler is the mistake the `gamma_diag` ridge analysis warns
against.

**CAVEAT:** 0.19 understates the uncertainty on the 7.0 and 10.0 rungs, which
`CALIBRATION_NOTES.md` records as extrapolations rather than estimates. Treat as a lower
bound on the ladder's uncertainty.

### `LTFU_MORTALITY_MULTIPLIER = 1.0` — **PLACEHOLDER**
Disengaged patients face the untreated CD4-specific mortality curve scaled by this. φ = 1.0
reproduces "LTFU treated as if never treated".

This matters more than it looks: LTFU is ~27% of person-time and generated ~59% of **all**
HIV deaths in the model at φ = 1.0, so this single assumption dominates the mortality model
far more than the on-ART excess does.

Two unsourced forces push opposite ways:
- treatment-experienced disengagers retain immune benefit, often silently transfer, and
  frequently self-restart — the full untreated curve **overstates** their mortality;
- tracing studies find a large share of recorded LTFU are already dead, so cohort-derived
  LTFU rates already embed mortality — pushing the other way.

φ = 1.0 sits at the extreme of the first. Pending a sourced value; run the sensitivity
before settling. Note φ is now a free parameter (`phi_ltfu`) — this constant is the fallback
when it is absent.

---

## Cascade dynamics

### `SUPPRESSION_RATE = 9.0`
Exit hazard from `ART1_Ramp`; its dwell time is the interval between starting ART and
suppressing. While in the ramp a person is on treatment but not suppressed: they carry the
untreated Glaubius mortality rate and their CD4 does not recover.

**Corrected from 21.0.** That value was carried from the microsim's `lambda_ART`, which is
*not* a transition rate — there it is the exponential decay constant of viral load:
`ind$VL <- max(50, ind$VL_sp * exp(-lambda_ART * t_ART))`. At 21/yr viral load halves every
ln2/21 ≈ 12 days, and from a set-point near 30,000 copies takes ln(30000/50)/21 ≈ 0.30 yr ≈
16 weeks to cross 50 copies — hence the microsim's "~12-week suppression" comment. Reused as
a compartment exit hazard, 21/yr means something entirely different: mean dwell 17 days,
median 12 days. On a monthly cycle 83% of people left the ramp in the first cycle, i.e. the
model suppressed people in about two weeks.

The ramp represents the virological time course among people who **remain** on treatment —
disengagement (`mu_ltfu`) and virological failure (`mu_vf1`) are separate states — so the
anchor is trial/clinical time-to-suppression on modern first-line ART rather than
programme-level suppression rates, which bundle adherence, retention and viral-load
monitoring coverage.

9.0/yr gives a median of 28 days, the value most commonly reported for dolutegravir-based
first-line regimens, and 98.9% suppressed by six months among those still on treatment —
consistent with ADVANCE (South Africa, n = 1035, TLD) reporting 85% <50 copies/mL at week 48
on intention-to-treat, which additionally counts discontinuations this model handles
elsewhere (Venter et al. 2019, doi:10.1056/NEJMoa1902824).

**CAVEAT:** the exponential is a poor shape here. Suppression is closer to a fixed delay than
a memoryless process, so no single rate reproduces both the median and the tail — the same
objection that motivated the same-day mixture below. Not remedied.

*Effect of the correction at posterior means:* p3 0.9703 → 0.9673 (target 0.9028, a small
move toward it) and DALYs per acquisition +0.06. A correctness fix, not a headline change.

### `SAME_DAY_INITIATION = 0.64`
Time from diagnosis to initiation is **not** exponential. IeDEA Southern Africa (N = 29,017,
universal test-and-treat): median 0 days, IQR 0–7, 18,584/29,017 = 64.0% initiating same
day, at least 75% within a week. That is a large point mass at zero plus a short tail, and a
single exponential hazard cannot represent it — fitted to the one-month proportion it puts
the mass in the wrong place, which is why `ART_1m` sat at 0.71 against 0.85 while
`lambda_init` stayed pinned at its prior.

Modelled as a **split of the diagnosis flow** rather than a faster hazard, because same-day
initiation is not a fast transition through `Diagnosed_PreART` — it is diagnosis and
initiation in the same instant:

```
Undiagnosed --diagnose-->  π      -> ART1_Ramp        (same day)
                           (1-π)  -> Diagnosed_PreART (then lambda_init)
```

Costs no new states. Sourced, not fitted: 0.64 is the observed proportion.

**SCOPE:** applied to first diagnosis only, not to re-engagement from LTFU. Returners
re-enter via `Diagnosed_PreART` and initiate within weeks at `lambda_init`; routing them
around that state would remove them from the CD4-at-ART-initiation flow measurement. State
as a limitation. If relaxed, `cd4_distribution_at_art_initiation` must count that route too.

### `LTFU_RETURN_FRACTION = 0.437`, `LTFU_REENGAGE_LATE = 0.0314`
Shape of the re-engagement process, fixed from Bagayoko et al. 2020 (Mali, n = 3,650):
cumulative return to care 39.0% at 1 yr, 45.0% at 2, 47.0% at 3.

Fixed rather than calibrated. Fitting all three LTFU hazards to the Mali curve reproduces it
exactly (residual ~4e-13) and yields `mu_re-engage` 0.941, `mu_perm_diseng` 1.212,
`mu_reengage_late` 0.031, eventual returning fraction 43.7%. Holding the latter two fixed
preserves the **shape** Mali pins down — a steep early rise onto a plateau — while leaving
`mu_re-engage` free to set the **level** from UNAIDS coverage.

*Why not calibrate all three:* once the Mali figures were demoted from targets (they measure
clinic re-engagement, not population coverage, and are West African), nothing identified
them. Four free LTFU hazards against a single effective constraint would leave the
parameters that dominate the sensitivity analysis determined by their priors.

*Why a fraction, not a fixed companion hazard:* the plateau is `r / (r + d)`, so fixing
`mu_perm_diseng` directly and freeing `mu_re-engage` does **not** hold the shape — a
calibration run drove `mu_re-engage` to 0.287, dropping the eventual-return fraction from
0.437 to 0.192. Holding the fraction fixed and deriving the companion hazard
(`ltfu_perm_disengage`) keeps Mali's plateau exactly while letting overall speed vary:

```
mu_perm_diseng = mu_re-engage × (1 − π) / π
```

---

## Natural history — System 1 (`natural_history_params.py`)

System 1 is a **sourced input, not a calibrated module**. The natural history of untreated
HIV has been estimated repeatedly, on far more data than this study has access to, and
published in exactly this model's CD4 stratification. The progression and mortality tables
are therefore adopted wholesale, and the three former calibration targets become
**out-of-sample validation checks the model can fail**.

### What this replaced
System 1 previously estimated two parameters — `lambda_base` and `alpha_shape` — defining a
parametric hazard `lambda_base × exp(alpha_shape × band)` fitted against three targets.
Untreated mortality was a fixed input, but only two of its seven bands were sourced (0.002/yr
at 250–349 and 0.712/yr at <50); `hazards.py` log-linearly interpolated the other five, a
curve the code itself flagged as "a placeholder curve shape, not sourced data".

Both anchors turned out to be Glaubius values read off the **15–24** age row (71.2 per 100 py
at <50 is that row exactly), while the model's cohort is anchored at age 29 — the 25–34 row.
So the model ran the youngest-adult mortality schedule against a cohort ten years older, and
the interpolation understated even that row in the middle bands (100–199 by 21%, 50–99 by
28%).

### It does not fail the checks
With **no free parameters at all**, the adopted tables give:

| Check | Model | Target | z |
|---|---|---|---|
| Median time to CD4<350 | 4.50 yr | 4.14 ± 0.64 | +0.56 |
| Median time to CD4<200 | 8.42 yr | 8.11 ± 1.11 | +0.28 |
| Median untreated survival | 11.42 yr | 11.50 ± 0.50 | −0.17 |

The previous fitted two-parameter version scored z = −1.00, −0.10 and −0.50 on the same
three. A parameterisation the model did *not* fit reproduces the targets better than the one
it did — and the targets are independent of it, being Pantazis-derived (CD4 thresholds) and
Isingo-derived (survival).

### Source
> Glaubius R, Stover J, Johnson LF, et al. Disease progression and mortality with untreated
> HIV infection: evidence synthesis of HIV seroconverter cohorts, antiretroviral treatment
> clinical cohorts and population-based survey data. *J Int AIDS Soc*
> 2021;24(Suppl 5):e25784. doi:10.1002/jia2.25784 — Table 3.

These are the values UNAIDS Spectrum/AIM itself uses, which matters for internal
consistency: the cascade denominators System 2 calibrates against (UNAIDS p1/p2/p3) are
produced by Spectrum runs using these same natural-history inputs.

The CD4 stratification is identical to this model's, boundary for boundary. Glaubius tested
a sex-stratified variant and reported it "did not improve model fit appreciably", so the
published inputs are not sex-specific and no sex assumption is needed.

**Note the non-monotonicity** at 200–249 → 100–199: 200–249 is a 50-cell band and 100–199 a
100-cell band, so the per-band exit rate drops even though CD4 decline itself does not slow.
A hazard monotone in band index — as `lambda_base × exp(alpha_shape × band)` was —
structurally cannot reproduce this.

Mortality is exactly **zero in the top two bands** in the published table; Glaubius
attributes no AIDS mortality above CD4 350. The previous interpolated curve put small
non-zero hazards there (1e-4, 5e-4), which meant booking HIV-attributable deaths, and
therefore YLL, at CD4 counts where the evidence records none.

`GLAUBIUS_INITIAL_CD4` is **not adopted** — the model keeps
`acquisition_cd4.initial_band_distribution()`, built from Pantazis 2012, which is
SSA-specific where Glaubius is not. Retained for the annex comparison: the two agree closely
(model 0.589/0.243/0.108 against 0.574/0.219/0.118 in the top three bands), an independent
check on the acquisition distribution.

### CD4 at acquisition (`acquisition_cd4.py`)

Seroconverters start from a CD4 **distribution**, not a point mass at the median. With a
point mass the only route to initiating ART below 200 was slow diagnosis, which forced
`delta_bg` down far enough to wreck the UNAIDS diagnosis-coverage targets. Late
presentation should arise from where people *start* as well as how slowly they are found.

**Source.** Pantazis N, et al. Differences in HIV Natural History among African and
Non-African Seroconverters in Europe and Seroconverters in Sub-Saharan Africa. *PLoS ONE*
2012;7(3):e32369 (CASCADE Collaboration + ANRS 1220 Primo-CI), Table 3 adjusted model — a
linear mixed model on sqrt(CD4) with random intercept and slope. Chosen over Lodi et al.
2011, whose cohort (78% male, 55% MSM, 90% subtype B) is a poor match for ESA. Lodi is
retained **only** for the spread, on the grounds that between-individual variance
transports better than a mean trajectory does.

**Structure.** sqrt(CD4) ~ Normal(mu, sigma0), truncated below at
`TRUNCATION_FLOOR_CELLS`, discretised onto the 7 bands. `mu` comes from the Pantazis
coefficients weighted to the ESA sex mix (`PCT_FEMALE = 0.63`); sex enters the **intercept
only**, so reweighting does not touch the decline rate.

**Age.** `ANCHOR_AGE` (29) sits inside Pantazis's 15–29 reference band and age enters the
**slope** only, so no age adjustment is applied. The 30–39 slope coefficient is −0.10
(p = 0.284), so the boundary is immaterial in any case.

#### `TRUNCATION_FLOOR_CELLS = 100.0` — a modelling judgement, not a sourced value
Seroconverting already severely immunosuppressed is not clinically credible, but the exact
floor is arbitrary. Vary it in sensitivity.

100 rather than 200, for two reasons:

- *Substantive.* A floor of 200 forces P(CD4<200 at acquisition) to exactly zero and caps
  band 3 (200–249) at ~3%, removing most of the mechanism this distribution exists to
  provide.
- *Numerical.* A floor of 200 lands on a band boundary and makes P(CD4≥500)
  **non-monotonic** in sigma0 (64.5% → 60.6% → 61.3% → 62.6% across sigma0 = 3, 5, 7, 9),
  because the truncated normal flattens toward uniform as sigma0 grows. A badly behaved
  likelihood surface for a free parameter. At a floor of 100 both P(≥500) and P(<200) are
  monotone in sigma0 across the plausible range.

A floor of 100 still excludes acquisition at AIDS-level immunosuppression (mass below 100
is <0.1%) while leaving ~2.8% in 100–199 at sigma0 = 5, consistent with a minority
presenting in acute infection with transiently depressed counts.

#### `ACQUISITION_SIGMA0 = 5.0` — FIXED, not calibrated
Pantazis fits random intercepts but *PLoS ONE* publishes fixed effects only, so the
variance component is not recoverable from the paper. Derived instead from Lodi et al.
2011's CD4 distribution one year post-seroconversion (median 510, IQR 341–721), which on
the sqrt scale implies ~6.2 sqrt-cells. That **overstates** the acquisition spread, since
it contains a year of heterogeneous decline plus measurement error on top of the intercept
variance — hence 5.0 rather than 6.2.

Fixed rather than free for the same reason the Mali re-engagement shape is fixed: nothing
in the target set identifies it cleanly, and inferring the acquisition spread from the
CD4-at-ART-initiation target would come close to fitting the input to the output. Treat it
in sensitivity analysis instead — `band_distribution()` takes `sigma0` as an argument.

> Superseded note: earlier comments described this as a prior centre identified by
> `LODI_PROPORTION_TARGETS`. It is not free, and those two descriptions sat contradicting
> each other in the source until 2026-09.

#### `LODI_PROPORTION_TARGETS`
Lodi et al. 2011 (*Clin Infect Dis* 53:817) Table 2 — proportion below each CD4 threshold
at 1, 2 and 5 years post-seroconversion. Used **only** to identify the shape of the
acquisition distribution, not its level or slope, which come from Pantazis.

Published 95% CIs are sampling error from n = 18,495 and imply SDs of ~0.3 percentage
points — far too tight to use directly. `LODI_TARGET_SD_FRAC = 0.25` widens them to a
fraction of the point estimate, reflecting the transportability of a European cohort's
spread to an ESA population.

#### `derive_target_sd()` — **CAVEAT** for wherever these SDs are used
The fixed-effect covariance matrix is not published, so the coefficients are treated as
**independent**. Intercept and slope estimates in mixed models are typically negatively
correlated, so this naive propagation probably **overstates** the resulting SD — the same
caveat already documented for the p2/p3 derivations in `data/parameters.xlsx`.

Time-to-threshold is a ratio of coefficients, so the Monte Carlo mean sits slightly above
the point estimate by Jensen's inequality. Use `time_to_threshold()` for the central value
and `derive_target_sd()` only for the SD.

#### Validation
The reconstruction is checked on import against three quantities Pantazis reports
independently (run `python3 acquisition_cd4.py`):

| Check | Model | Paper |
|---|---|---|
| Median CD4 at seroconversion | 607 / 470 / 570 | 607 / 469 / 570 |
| CD4 lost over first four years | 258 / 155 / 199 | 259 / 155 / 199 |
| SSA delay reaching CD4<350 | 6.2 months | "almost 6 months" |

### Cross-check (sensitivity only)
> Mangal TD and the UNAIDS Working Group on CD4 Progression and Mortality Amongst HIV
> Seroconverters. Joint estimation of CD4+ cell progression and survival in untreated
> individuals with HIV-1 infection. *AIDS* 2017;31(8):1073–1082.
> doi:10.1097/QAD.0000000000001437

A hidden Markov model on 16,373 seroconverters (3,341 in Africa), same seven CD4 states,
reporting Africa-specific rates. Reproduces the CD4<350 target at 4.17 against 4.14, again
with nothing fitted. Retained as `MANGAL_*` for sensitivity rather than base case because
(a) Glaubius is the parameterisation behind the UNAIDS cascade estimates this model
calibrates to, and (b) Mangal's African stratum is a minority of its sample.

Mangal reports mortality age effects as hazard ratios rather than a full age × band table,
so the age dimension is applied multiplicatively. Progression is sex-averaged across the
published male and female tables; the sex effect is small (F vs M 0.92, 0.86–0.99).

### System 1 uncertainty
No longer a posterior. Carried as two multiplicative factors on the tables — `kappa_prog` on
progression, `kappa_mort` on mortality — each lognormal with median 1. This preserves the
published **shape** of both schedules (what the evidence constrains well) and puts the
uncertainty on their **level** (what transports least well between populations). It also
keeps System 1's draw two-dimensional, so the cut machinery downstream is unchanged.

- **`KAPPA_PROG_SIGMA = 0.05`.** Glaubius's 95% CrIs on the progression cells have relative
  half-widths of 3.8–10.5% (mean 5.7%) at ages 25–34, so σ ≈ 0.057/1.96 ≈ 0.03. Rounded up
  to 0.05 to allow for a common factor standing in for seven separately-estimated cells.
- **`KAPPA_MORT_SIGMA = 0.15`.** Glaubius's mortality intervals imply σ ≈ 0.09.
  **Deliberately widened**, because the honest uncertainty is not Glaubius's internal
  precision but the disagreement *between* sources on where untreated mortality sits: Mangal
  puts substantially less in 100–199 and 50–99 and more in <50, and reports an
  Africa-vs-Europe mortality HR of 2.68 while finding progression rates statistically
  indistinguishable between regions. Between-source disagreement exceeds either source's
  internal uncertainty, and the prior should say so.

---

## Health economics (`health_econ_params.py`)

### Costs — one source

All unit costs are adopted wholesale from:

> Hyle EP, Maphosa T, Rangaraj A, et al. Clinical impact and cost-effectiveness of the
> WHO-recommended advanced HIV disease package of care. *Lancet Global Health*
> 2025;13(8):e1436–e1447. doi:10.1016/S2214-109X(25)00190-1
> Supplementary appendix 2, Table S6 ("Detailed costs parameters, USD 2023").

A CEPAC-International analysis for Malawi in 2023 USD.

**Why a single source.** The previous set mixed Tagar 2014 (five-country facility ART
costs, nominal 2009–2011), a four-tier Menzies/Hyle CD4 table whose attribution could not
be verified, CHAI catalogue drug prices, and an unsourced $50 initiation cost. Three of
those five rows had provenance problems and the price years spanned fourteen years. Hyle et
al. give every quantity this model needs, in one price year, from one setting, with a
stated derivation.

**Their derivation, which is the reason to prefer it.** CD4-stratified routine care is not
asserted — it is built from resource utilisation by CD4 (average inpatient days and
outpatient visits per month) from Holmes et al. 2006 (Cape Town, *JAIDS* 42:464–9),
multiplied by Malawi unit costs for an inpatient day and an outpatient visit from
Maheswaran et al. 2017 (*JAIDS* 75:280–9) and 2018 (*PLoS One* 13:e0192991). The outpatient
visit cost is itself the average cost of an HIV clinic visit in Malawi — non-clinical
personnel time, equipment, space, overhead — plus 20 minutes of nurse time. Auditable end
to end, which the previous flat service-delivery figure was not.

**CAVEAT — scope.** This is a Malawi cost set, and Malawi sits at the low end of the
eastern and southern African range. Adopting it makes the estimate conservative on cost and
therefore favourable on cost-effectiveness. A one-way sensitivity spans Malawi to the CEPAC
Côte d'Ivoire routine-care costs.

| Constant | Value (2023 USD) | Note |
|---|---|---|
| `SRC_COST_1L_ART_PER_YEAR` | $3.60/month | TDF/3TC + DTG |
| `SRC_COST_2L_ART_PER_YEAR` | $18.57/month | AZT/3TC + LPV/r |
| `SRC_ROUTINE_CARE_BY_BAND` | 2.86 / 3.96 / 4.90 / 4.90 / 10.77 / 18.51 / 26.24 per month | by CD4 band |
| `SRC_COST_DEATH` | $104.26 one-off | any cause; terminal care |

Hyle's six routine-care strata map onto this model's seven bands exactly: their 200–349
stratum spans this model's 250–349 and 200–249 bands, and every other boundary coincides.

The routine-care schedule **replaces both** previous non-drug terms — the flat ART
service-delivery cost and the CD4-tiered clinical care cost. It has to: Hyle's routine care
already contains both, since the outpatient visit cost is the service-delivery component
and the inpatient days are the advanced-disease component. Keeping the old service-delivery
term alongside it would double-count the clinic visit. `COST_SERVICE_DELIVERY_PER_YEAR` and
`COST_ART_STARTUP` are therefore retired (held at 0.0, names kept so imports don't break).

**`ROUTINE_CARE_STATUSES` — the one deliberate departure.** Hyle charge routine care to
people *in care*; this model extends it to `Undiagnosed`. The underlying resource
utilisation (Holmes 2006) is inpatient days and outpatient visits driven by
immunosuppression, and people present and are treated for the consequences of advanced HIV
before anyone records them as having it. Excluding the undiagnosed would understate the
cost of late diagnosis, which is one of the quantities this model exists to estimate.
Charging in-care states only is reported as a scenario.

**NOT ADOPTED**, because the model has no state to attach them to: Hyle's per-incident
opportunistic infection costs (malaria $185.49, serious bacterial infection $142.80, other
WHO stage 3/4 disease $86.72), their diagnostic test costs (CD4 $5.80, viral load $19.58),
and their TB and cryptococcal modules. In their model these are charged *on top of* routine
care. Omitting them makes this model's cost per acquisition a **lower bound** relative to a
full Hyle implementation — state in the limitations.

### Price-year adjustment

CHEERS 2022 item 15 requires the price date and conversion method to be stated; before this
they were not. Every unit cost is recorded at its source's price year and converted by US
CPI-U (BLS all-items annual average, 1982–84 = 100). `set_price_year()` recomputes the
effective values; the base case matches the price year to the cascade year, so the
2019-anchored analysis reports 2019 USD and the 2024-anchored analysis 2024 USD.

A US index is the pragmatic choice: the inputs are either internationally traded
commodities priced in USD (antiretrovirals, procured through pooled mechanisms) or costs
already converted to USD by their source at the collection-year exchange rate. Deflating in
local currency and reconverting — the Global Health Cost Consortium reference-case
preference — is not possible without the original local-currency values, which the sources
do not report. Hyle's own conversion did use the Malawi inflation index and the 2023 average
exchange rate, so values arrive here already in 2023 USD and the CPI-U step only moves them
between USD price years.

### Discounting and YLL

`DISCOUNT_RATE = 0.03`, WHO-CHOICE reference case, discrete compounding, applied per model
month: `discount_factor(t years) = (1 + r) ** (-t)`.

`REFERENCE_LIFE_EXPECTANCY_LEGACY = 72.0` is **no longer used for YLL**. Years of life lost
now come from the cause-deleted ESA life table (`life_table.remaining_life_expectancy`), so
the YLL reference and the model's background mortality share a source. Retained as a
documented comparator: the old `72 − age_at_death` convention happens to agree with the
table at the anchor age (non-HIV e₂₈ = 44.4 → mean age at death 72.4) but diverges sharply
after — at 65 it scored 7 years lost against a true eₓ of 14.0.

### Disability weights — reconstructed, not adopted

**Not** from Hyle, who use quality-of-life multipliers by opportunistic infection, which
this model has no states for. Specified in the source table as *continuous* functions of
CD4 rather than discrete stage weights:

| | Min | Max | Slope | GBD benchmark |
|---|---|---|---|---|
| On-ART | 0.04 | 0.19 | 0.010 | HIV on ART ~0.053 |
| Off-ART | 0.08 | 0.57 | 0.008 | Early HIV ~0.08; Symptomatic 0.22–0.27; AIDS ~0.55 |

**The exact original formula and units for "Slope" were not specified.** Reconstructed here
as linear in CD4 deficit from a 500 cells/µL anchor, in units of 8 cells per slope-step:

```
DW(CD4) = clip(Min + Slope × (500 − CD4) / 8, Min, Max)
```

The divisor of 8 was chosen so the off-ART curve reproduces the table's own cited GBD AIDS
benchmark (~0.55) at low CD4 — it is a fit to the benchmark anchors, **not** the original
source formula. If the actual formula surfaces, swap it in; everything downstream reads
through `disability_weight()`.

A structural sensitivity analysis substitutes GBD's three published health states directly.

`CD4_BAND_REPRESENTATIVE` evaluates the continuous function per band. Band 0 (≥500,
open-ended) uses 575 to match the acquisition-CD4 median used elsewhere; the rest are band
midpoints.

---

Every parameter, its source and its price year is also listed in Appendix B of the report,
generated by `writeup/table_inputs.py`, which prints the inputs whose provenance is
unresolved.

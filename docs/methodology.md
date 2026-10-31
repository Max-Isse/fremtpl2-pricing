# Methodology

Evidence tables and figures are generated in `reports/` by `fremtpl2 report`; numbers quoted here are copied
from those files (regenerate before citing). Sections for phases 2–5 are filled in as those phases run.

## 0. Validation design (no time dimension)

freMTPL2 is a one-year cross-section with no policy dates, so **out-of-time validation is not possible**.
Substitute (D-007):

- **Grouped holdout (20%)**: untouched until final model comparison; plays the role of "unseen book".
- **Grouped 5-fold CV** on the remaining 80% for model selection and tuning.
- **Groups = identical rating-factor profiles** (D-006). 38% of rows share their full profile with another row,
  and the evidence (median IDpol span 2 within a group, combined exposure ≤ 1 year in 94% of groups) says these
  are fragments of one policy. A row-level split would leak. `tests/test_leakage.py` asserts that no group
  straddles the holdout or two folds, on fixtures and (locally) on the full data.
- **Bootstrap CIs** on deviance and Gini (phase 4).

What this cannot show: performance under drift, claims inflation or the portfolio change caused by our own
re-pricing. Phase 6 simulates the latter under stated assumptions; the rest is a monitoring-plan item.

## 1. Exposure

Source: `reports/exposure_analysis.md` (training split; models fitted on folds 1–4, scored on fold 0).

**Claim counts scale with time at risk.** The mean claims per record rises from 0.009 for records under one
month to 0.065 for 0.75–1 year. A one-year record cannot be compared to a one-month record without
accounting for time on cover. The natural model is a Poisson process: N ~ Poisson(λ · t) with t = exposure,
so log E[N] = log t + x'β — **log(exposure) is an offset with coefficient fixed at 1**, and exp(x'β) is the
annual frequency that the premium is built on.

**What goes wrong if exposure is ignored.** The model then predicts claims per record, whatever its
length:
- the implied annual frequency is 0.039 against an observed 0.074 claims per exposure-year: pricing a full
  year on it would charge roughly half the expected claims;
- relativities absorb the correlation between factors and exposure. Older drivers have longer average
  exposure (0.68 years for 71+ vs 0.38 for 17–20), so the "ignored" model gives the 71+ band a relativity of 1.14
  against the youngest band, where the offset model gives 0.79. A
  factor's price would then reflect how long its policies happened to be on cover, not its risk;
- out-of-fold A/E by exposure band ranges from 0.22 to 1.70.

**What happens with log(exposure) as a free feature.** The fitted coefficient is **0.67 (95% CI 0.65–0.69)**,
not 1, and this model has the lowest out-of-fold deviance of the three. The data say short records have
higher annualised frequency: 0.164 claims per exposure-year under one month vs 0.054 for full-year records.
Two mechanisms are visible in the data:
1. *Split records:* 72% of records under one month belong to duplicate-profile groups (vs 3% of full-year
   records), i.e. they are fragments of a policy whose other fragments sit elsewhere.
2. *Early termination / selection:* short single records (not in any duplicate group) have the highest
   annualised frequency (0.105 vs 0.055 for full-year singletons), consistent with policies that end
   because of, or shortly after, a claim — a known feature of this dataset.

Neither mechanism is a property of the *risk per unit time* that the premium should price. A free coefficient
would bake them into the tariff: a one-month policy would cost 1/12^0.67 ≈ 19% of the annual premium instead of
8.3%, a rule that cannot be justified to a regulator or customer and that would reward splitting or
cancellation. This is a first instance of **better deviance ≠ better pricing** (see §3).

**Decision (D-008):** offset for all frequency models (GLM: `offset=log(exposure)`; LightGBM: `init_score`).
Consequences we carry forward:
- The offset model under-predicts short records (A/E 1.85 under one month) and over-predicts full-year records
  (0.84). Because the premium is quoted for a full year, the relevant calibration check is on exposure-weighted
  totals and on full-year business; phase 4 reports A/E by exposure band alongside the segment checks.
- Phase 2 tested an exposure-band *control* fitted next to the offset. Removing it moves the base frequency by 14%
  and some BonusMalus and Region relativities by up to 17% (`reports/frequency.md` §6), which is material. It was
  adopted as D-012 (see §2a). The control is fitted but pinned at full year when pricing, so the offset still
  carries the pro-rata scaling for time on cover.

## 2. Frequency vs severity

### 2a. Frequency model (phase 2)
Source: `reports/frequency.md` (training split, grouped 5-fold CV, holdout untouched).

**Why a Poisson GLM with a log link.** Claim counts are non-negative integers whose expectation scales with time
at risk. The log link makes factor effects multiplicative (relativities), which is how tariffs are filed and
explained, and makes log(exposure) an additive offset. The Poisson score equations make actual and expected
claims balance within every level of every factor in the training data. The mean is consistently estimated even
when the variance is not Poisson, so the overdispersion question below is about *inference*, not about the fitted
premium.

**Model structure.**
- *Tariff factors (D-009):* DrivAge, VehAge, VehPower, BonusMalus and log-density in bands; VehBrand, VehGas and
  Region as given; and the AgeMalus interaction (D-011). Each reference level is the largest-exposure level.
- *Control (D-012):* an exposure-band factor is fitted but **not charged**. Every policy is priced at the
  full-year level, pro rata to its time on cover. The control absorbs the short-record effect from §1, so it no
  longer leaks into factor relativities or the base rate.
- *Area* is not used. Its boundaries are nested cuts of Density, and given log-density bands it adds nothing
  (LR χ² = 9.4 on 5 df, p = 0.093; quasi-F p = 0.19).
- Every tariff factor is strongly significant under all three tests: Poisson LR, dispersion-scaled quasi-F and
  cluster-robust Wald.

**Full-year calibration.** This is the headline check, since full-year cover is what is quoted. Out of fold,
charging full-year records at the tariff rate gives:

| Tariff | A/E on full-year records |
|---|---|
| Portfolio-average rate | 0.74 |
| BonusMalus only | 0.83 |
| GLM, offset only (option A) | 0.82 |
| **GLM with exposure-band control (option B, adopted)** | **1.000 (95% CI 0.977–1.023)** |

Under option A every full-year customer would be charged about 18% more than their expected claims.

**The cost of getting full-year right.** Charging every record pro rata at the full-year rate collects expected
claims of 16,542 against 21,152 actual in the training book (A/E 1.28). Short records really do claim more per
year, whether because records split around mid-term changes or because policies end after claims. A pro-rata
tariff does not fund that excess, so it has to come from somewhere else: a short-period or cancellation rate
scale, or a book-level loading. Phase 6 carries it into the technical premium as an explicit, configurable loading.
Option A was hiding the same cost by spreading it over every full-year customer.

**Overdispersion.**
- Without the control, Pearson χ²/df was 1.73, with 40% of it from records under one month, where tiny expected
  counts turn a single claim into a huge Pearson residual.
- With the control, Pearson χ²/df is 1.27, and Cameron–Trivedi is significant (α = 0.71, one-sided p = 0.0009;
  auxiliary regression of ((y−μ)²−y)/μ on μ).
- NB2 is preferred by likelihood (α = 0.73; LR = 289; boundary-mixture p ≈ 5e-65).
- So overdispersion is real but modest. NB2 relativities differ from Poisson by at most 1.8%, and NB2 standard
  errors are 2% larger on average. Quasi-Poisson would scale every SE by 1.13.
- Randomised quantile residuals (Dunn–Smyth) are close to N(0,1). NB2 fits the extreme upper tail better.

**Decision (D-010).** Keep the **Poisson mean**, since NB2 would change the tariff by at most 1.8%. Report
**cluster-robust standard errors with profile groups as clusters**. They are valid whatever the variance function
and also cover correlation between split records of one policy (D-006). They average 3.5% above the model-based
SEs, between NB2 and quasi-Poisson.

**Interaction found by the segment check (D-011).** Out-of-fold A/E on segments that are *not* in the model showed
two problems: drivers 61+ with a malus (BM>50) were under-predicted (A/E 1.21–1.33), and those at BM=50 were
over-predicted (0.88–0.91). One reading is that a malus says much more about recent claims for an experienced
driver than for a young one, who starts at BM 100. The `AgeMalus` factor (levels: 18–30 & malus, 61+ & malus,
base) has relativities of 0.73 and 1.52 and LR χ² = 113 on 2 df. It improves CV deviance in all 5 folds and brings
the older-driver segments to A/E 0.96–1.06. Some residual pattern remains, e.g. 51–60 with a malus at 1.07
(CI 1.02–1.13), so a gradient-boosting model may find more of this interaction (phase 4).

**Against the baselines (out of fold, grouped 5-fold).**

| Model | Mean Poisson deviance per policy | Deviance explained vs portfolio mean |
|---|---|---|
| Portfolio-average rate | 0.2532 | — |
| BonusMalus only | 0.2438 | 3.7% |
| GLM, offset only | 0.2405 | 5.0% |
| GLM with control (final) | 0.2384 | 5.8% |

The final GLM beats both the BonusMalus-only tariff and the offset-only GLM in 5/5 folds (paired improvement
0.0054 and 0.0021 respectively). Deviance explained is small in absolute terms, as expected for rare events.
Phase 4 adds Gini/Lorenz and lift, which are easier to interpret for pricing. The BonusMalus-only tariff badly
misprices young drivers (out-of-fold A/E 1.35 for ages 18–20).

**Where the model can fail.**
- *Non-monotone BonusMalus relativities* (58–67: 2.27; 68–79: 1.98). The 58–67 band contains the levels a driver
  reaches after a single recent claim from the 50 floor. The pattern is statistically real, but a non-monotone
  malus scale is hard to file and explain (input to phase 5, §3).
- *Thin levels:* small regions (R43: 445 exposure-years, CI 0.57–1.24) and the 18–20 age band (CI 1.12–1.74) have
  wide intervals. Phase 5 considers credibility weighting or grouping.
- *The short-record loading* above is a single book-level number. If the mix of short-duration business changes,
  that loading has to be re-estimated (monitoring-plan item).
- *Data decision D-003 dominates:* refitting on the recorded `ClaimNb` changes the base frequency by 81% and the
  new-vehicle relativity by +204%. This is because 42% of policies with claims but no amounts are VehAge 0,
  against 5% of matched claims in the same IDpol block, which supports treating that block as an artefact.

### 2b. Severity model (phase 3)
Source: `reports/severity.md` (training split; grouped 5-fold CV; holdout untouched).

**Why a Gamma GLM with log link, weighted by claim count.** Claim amounts are positive and right-skewed, and
their spread grows with their level. The Gamma family encodes Var(Y) = φμ², i.e. a constant coefficient of
variation. With the log link, severity relativities multiply with frequency relativities. The unit of analysis is
the policy: the response is the *average* capped claim amount and the weight is the number of claims, since an
average of n claims has variance φμ²/n. As with frequency, the quasi-likelihood estimate of the mean holds
without the full Gamma distribution. What matters for the tariff is the mean and variance function, and the
distributional checks below matter for simulation.

**How different the data are from frequency.**

| | Frequency | Severity |
|---|---|---|
| Observations | 542,600 policies | 19,998 policies with claims (3.7%), 21,191 claims |
| Coefficient of variation of the outcome | — | 14.0 raw, 1.85 after the large-loss cap |
| Largest single observation | 4 claims | 4.08 million (capped at 32,162) |
| Flat-rate amounts | — | 41% of claims on four values (1,204; 1,128.12; 1,172; 1,128) |

The flat amounts look like fixed-cost settlements under an inter-insurer agreement. That reading can't be
verified from the data, but the point masses are real and visible in the distribution figure.

**Severity depends on how long the claiming record ran (D-014).** Mean capped severity falls from 2,512 on records
under one month to 1,369 on full-year records. Over the same range, the share of flat-rate amounts rises from 23%
to 48%. This is the same selection effect as in frequency: records that end early carry larger claims. It is
the strongest severity signal in the data (quasi-F = 39.7 on 5 df; on its own it explains 2.6% of CV deviance).
Severity therefore gets the same exposure-band control as frequency, fitted but pinned at full year when
pricing. Without the control, frequency × severity would over-price full-year records by about 18% even though
frequency alone is calibrated.

**Which factors matter (D-013).** Factors are coarser than for frequency (3–4 bands instead of 8) because of
roughly 27× less data and much noisier outcomes.
- Every factor is strongly significant for frequency.
- For severity, given the control, only VehAge is clearly significant (p = 2e-5). Region (21 df, p = 0.006),
  BonusMalus (p = 0.04) and VehPower (p = 0.05) are borderline. DrivAge, LogDensity, VehBrand, VehGas and AgeMalus
  are not significant.
- *Selection method:* in-sample F-tests would keep five factors, including Region with 21 df. Grouped CV along the
  backward-elimination path shows that every model with more than two factors is *worse* out of fold than
  {DrivAge, VehAge}. The paired one-standard-error rule picks **VehAge only**: 0.90 for vehicles aged 10+ (CI
  0.85–0.95) and 1.01 for 0–2 years.
- Out of fold, the selected model explains 2.8% of severity deviance against the mean-severity baseline (5/5
  folds). Most of that comes from the control (2.6%), and the all-factor model does worse (2.4%).

**Why severity needs fewer, coarser factors and is far more volatile.**
1. *Information:* the severity likelihood sees 21k claims, not 543k policy records. Standard errors scale
   accordingly, so a 22-level factor that is estimable for frequency is mostly noise for severity.
2. *Noise per observation:* a single claim's amount has a CV near 1.85 even after capping, and 14 before.
   Frequency noise per policy-year is Poisson with a mean around 0.07, and pooling across many policies averages
   it out. Severity relies on far fewer, much noisier observations.
3. *Drivers differ:* frequency is about *whether* an accident happens (driver experience, claims history,
   traffic density). Severity is about *what it costs*: vehicle value and age, settlement conventions, injury
   mix. Here the rating factors barely separate costs once flat-rate settlements and record duration are
   accounted for.
4. *Consequence:* frequency explains 98.7% of the variance of log pure premium across policies and severity
   1.1% (`reports/pure_premium.md` §2). The tariff's shape is almost entirely a frequency tariff with a
   near-flat severity scale.

**Gamma assumption checks.**
- *Variance function:* a log-log regression of binned w(y−μ)² on fitted μ gives a slope of 2.41
  (95% CI 1.92–2.89), consistent with Gamma's μ² and well away from Poisson-like (1). This is what the
  mean/inference uses.
- *Distribution shape:* the quantile residuals have mean 0.38, SD 0.54 and skewness 1.5, far from N(0,1). The
  flat-rate point masses and a tight centre with a long tail are not Gamma-shaped. A single Gamma with φ ≈ 3.2
  puts too much mass near zero and too little at the settlement values.
- *Implication:* the Gamma GLM is fit for the *mean* (the tariff). It is **not** a valid generative model of
  individual claims. The phase 6 Monte Carlo therefore resamples empirical capped amounts (by exposure band)
  instead of drawing from a fitted Gamma. A two-part model (point mass + continuous part) is the alternative
  if per-claim distributions are needed.

### 2c. Pure premium and Tweedie cross-check (phase 3)
Source: `reports/pure_premium.md`.

**Pure premium** = annual frequency (full-year) × expected severity (full-year claim) × (1 + large-loss loading
0.378). The training-book mean is 108.74 per policy-year. In the training book, frequency × severity reproduces
total capped losses (A/E 1.002 with each record's own exposure band) and prices full-year records correctly
(A/E 1.005).

**Short-record cost.** Charged pro rata at the full-year rate, the tariff funds only 64% of capped losses
(A/E 1.57), because short records have both higher frequency and higher severity. This is the D-012
consequence, now on cost rather than counts. Phase 6 adds it as an explicit, configurable short-record loading.

**Tweedie cross-check (D-015).**
- *Setup:* a Tweedie GLM on loss cost per exposure-year with the same factors and control, power
  p = (ν+2)/(ν+1) = 1.76 from the severity dispersion.
- *Agreement:* it agrees closely in rank (correlation of log premiums 0.97).
- *Totals:* it does not reproduce total losses (A/E 0.967 before a base-level correction), because the log link
  with p ≠ 1 does not balance actual and expected totals.
- *Full-year bias:* even after rebalancing, it under-prices full-year records by 11% (A/E 1.12). Its estimating
  equations weight low-cost records more heavily, so the short-record cost is attributed differently.
- *Per-policy differences:* with both models rescaled to the same book mean, 56% of policies differ by more than
  10%. The Tweedie route lets every frequency factor also carry severity effects, which the severity CV shows do
  not generalise.
- *Power:* the choice of p moves premium shapes only modestly (5th–95th percentile ratio 0.97–1.03 for p ± 0.1).
- *Conclusion:* the frequency × severity route is preferred for the tariff. Phase 4 compares both on the holdout.

## 3. Better accuracy ≠ better pricing
*Phases 4–6 build the main evidence. Examples found so far:*
- §1: the free log(exposure) coefficient has the lowest deviance but gives a non-pro-rata rule.
- §2b: in-sample F-tests keep five severity factors, but the model is worse out of fold than a
  one-factor model.
- §2c: the Tweedie GLM tracks in-sample loss cost per level more closely, yet it mis-prices the full-year
  business that is actually quoted.

## 4. Large losses
Claims above the training 99.5th percentile (32,162) are capped and the excess (37.8% of capped training losses)
is spread as a flat loading (D-005). The severity model is therefore a model of *attritional* severity; the
loading's size is a pricing assumption carried to the technical premium and included in sensitivity analysis.
**Large losses are concentrated on short records.** In the training data, 0.29% of claims on full-year records
exceed the cap, against 0.62% on shorter records. The excess is 11.6% of capped losses on full-year records but
48.2% on shorter ones. The flat 37.8% loading is a book average, so applied to full-year business it would
over-charge heavily. Phase 6 must split it: about 11.6% on the full-year rate, with the rest going into the
short-record loading (D-012/D-014). Until then, the pure premium in `reports/pure_premium.md` uses the flat loading
as specified in D-005.

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
- **Bootstrap CIs** on deviance and Gini: grouped bootstrap on the holdout, 200 replicates (phase 4, §5).
- The holdout was used once, in phase 4, after all modelling and tuning decisions were fixed.

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
The brief asks for four demonstrations. Phases 1–5 provide the first two and part of the third; phase 6 adds
the effect of rate caps and price elasticity.

**(a) Lower deviance can still be miscalibrated in segments** (`reports/model_comparison.md` §5,
`reports/tables/cmp_segments_frequency.csv`). On the holdout:
- *Random forest:* its frequency deviance is lower than the GLM's (0.2347 vs 0.2364), yet 17 of 73 rating-factor
  segments have A/E outside their exact 95% Poisson interval (each factor's levels counted separately). Examples:
  B12 0.81 (13k of the 71k holdout exposure-years), B11 1.37, 18–20-year-olds 1.34, BonusMalus 115+ 1.58. The GLM
  has 3 such segments.
- *LightGBM:* it has the lowest deviance and only 4 miscalibrated segments, but one is the largest segment in the
  book: drivers at BonusMalus 50 (45k of 71k exposure-years) are over-predicted by 4% (A/E 0.959). Commercially that
  means overcharging the best customers, the group most likely to leave for a competitor.
- *Other examples from earlier phases:*
  - the free log(exposure) coefficient has the lowest deviance but gives a non-pro-rata rule (§1);
  - in-sample F-tests keep five severity factors, but that model is worse out of fold than a one-factor
    model (§2b);
  - the Tweedie GLM tracks in-sample loss cost per level more closely, yet it mis-prices full-year business
    (§2c).

**(b) Unstable or non-monotonic relativities** (`reports/explainability_fairness.md` §2–3).
- *The GLM is the less monotone model.* Its BonusMalus scale falls from 2.27 (58–67) to 1.98 (68–79) in the full
  model and in all five refits. The 58–67 band contains the levels reached after a recent claim from the 50 floor.
  A regulator or customer would ask why a worse bonus-malus coefficient costs less. LightGBM's marginal
  BonusMalus effect is monotone in the full model (a reversal appears in at least one of the five refits).
- *Where the GLM's relativities move most:* thin levels such as Region (up to 20% between refits), VehBrand
  (17%) and the 18–20 age band (14%).
- *LightGBM is worse on what customers feel.* Its level-average SHAP effects are at least as stable as the
  GLM's coefficients, but each policy's premium is not. Across five refits the median policy's LightGBM
  premium moves 20% (GLM 7.5%), and 90% of policies move by more than 10% (GLM 26%) (§6).
- *The two models disagree on what drives risk.* The GLM charges 18–20-year-olds 1.40× the reference age, while
  LightGBM attributes almost all of that to their BonusMalus (SHAP relativity 0.97 for the age band). VehBrand
  carries 9% of GLM premium variance but almost none in LightGBM. Same data, different stories: the GLM's story is
  what a pricing committee can file and defend.

**(c) Premium moves that cause adverse selection or lose profitable customers** (§5, §7).
- *How different the prices are:* LightGBM and GLM premiums differ by more than 25% for 55% of holdout policies.
- *Where the gaps sit:* in segments that are almost never full-year, where neither premium can be validated.
- *Phase 6 (§7):* a flat rate keeps 95% of the riskiest decile and 15% of the safest. Uncapped GLM and LightGBM
  tariffs move the average customer's premium by 27% and 35%.

**(d) Behaviour under rate caps and elasticity** (§7). Under the assumed elasticity, LightGBM's holdout ranking
advantage does not turn into profit: its realised profit is 0.05M uncapped and 0.04M capped, against 0.26M for
the capped GLM. Capping changes the ranking of strategies, and the capped GLM stays best under every truth
assumption tested.

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

## 5. Benchmarks and holdout comparison (phase 4)
Source: `reports/model_comparison.md`. This is the first and only use of the holdout (135,413 policies). All
modelling and tuning decisions were made on the training folds. CIs are 95% percentile intervals from 200
grouped bootstrap replicates (whole rating-profile groups resampled, D-006).

**Benchmarks (D-018).**
- *LightGBM:* Poisson objective with log(exposure) as init_score for frequency, Gamma with claim-count weights
  for severity, and Tweedie (p = 1.76, as for the GLM) with exposure weights for pure premium. Each is tuned by a
  random search over 6 configurations with grouped 5-fold CV and early stopping.
- *Random forest:* Poisson split criterion on claim rate, weighted by exposure, with regularised defaults. It
  was not tuned and serves as a reference only.
- *Same treatment as the GLMs:* every tree model sees the same rating factors plus the exposure band as an
  ordinal *control* feature, pinned at full year when pricing (D-012).

**Results on the holdout.**

| | GLM | LightGBM | Difference (95% CI) |
|---|---|---|---|
| Frequency, Poisson deviance explained vs portfolio mean | 5.7% | 7.9% | deviance −0.0055 (−0.0063, −0.0048) |
| Frequency Gini (priced) | 0.298 | 0.330 | +0.032 (0.023, 0.043) |
| Severity, Gamma deviance explained vs mean | 3.1% | 3.0% | no difference (CI −0.011, 0.010) |
| Pure-premium Gini, all records | 0.303 | 0.346 | +0.043 (0.026, 0.065) |
| Pure-premium Gini, **full-year records** | 0.396 | 0.467 | +0.070 (0.048, 0.096) |
| Full-year A/E of the priced premium | 1.037 | 1.006 | |

- *Other models:* the random forest falls between the two (frequency deviance explained 6.4%). The
  BonusMalus-only tariff explains 3.8%.
- *Frequency is where the trees win.* Severity is equally weak for both, which confirms §2b.
- *Double lift:* in the decile where LightGBM charges about half the GLM premium, actual losses are 0.54 of the
  GLM's expectation and 0.98 of LightGBM's. In the decile where LightGBM charges 1.75× the GLM, actual losses
  are 1.98× the GLM's expectation and 1.13× LightGBM's. LightGBM is closer to the truth where the two disagree.
- *Calibration:* LightGBM freq × sev is calibrated in total (fitted A/E 1.016) and on full-year records (1.006).
  The LightGBM Tweedie model is not: fitted A/E is 1.13 (D-020).

**What the comparison does *not* show.** The priced premiums of the two models differ by more than 25% for 55%
of holdout policies. LightGBM's book-average priced premium is 92.3 against 79.2 for the GLM. Much of that gap
sits in segments that are almost never observed at full year:

| Segment | Share of records that are part-year | LightGBM / GLM priced premium |
|---|---|---|
| VehBrand B12 | 97% | 1.66 |
| New vehicles (VehAge 0) | 99% | 1.55 |
| BonusMalus 100–114 | 95% | 1.57 |

- *Extrapolation, not evidence:* for these profiles a "full-year premium" is an extrapolation. The GLM
  extrapolates multiplicatively; LightGBM pins a feature to a value those profiles barely have. Neither can be
  validated on this data.
- *Why LightGBM looks better on the book:* its pro-rata book A/E is 1.35, against 1.57 for the GLM. It moves part
  of the short-record cost into rating factors, so customers in those segments would pay for duration effects
  through their risk profile. That improves book-level adequacy but is hard to justify in a filing, and is a
  fairness question for phase 5.
- *On full-year records,* where validation is possible, LightGBM is still clearly better at ranking. There its
  median priced premium is 14% below the GLM's: it charges a minority of high-risk full-year customers much more.

**Status (D-019).** On accuracy and ranking, LightGBM wins clearly for frequency and pure premium. Whether it
should set prices is not decided here. Phase 5 assesses stability, explainability, monotonicity and proxy risk,
and phase 6 the commercial effect under rate caps and elasticity.

## 6. Explainability, stability and fairness (phase 5)
Source: `reports/explainability_fairness.md` (training split only; no holdout).

**Method.**
- *SHAP:* LightGBM's exact TreeSHAP (`pred_contrib`) gives additive contributions on the log-rate scale, the same
  scale as GLM coefficients. Averaging a feature's SHAP value within each GLM level and dividing by the
  reference level gives a "SHAP relativity". It is a marginal attribution, with interactions shared out between
  features, not a tariff cell.
- *Refits:* both frequency models are refitted on each of the five grouped-CV training sets (80% of the training
  split). This measures the spread of level effects and of each policy's own priced frequency.

**Side by side.**
- *Where they agree:* BonusMalus dominates both models (64% of GLM log-premium variance, 58% of LightGBM's), and
  above BonusMalus 80 the two scales agree within a few percent. VehAge, density, fuel and vehicle power look
  similar too.
- *Where they disagree:*
  - Young drivers: the GLM charges 18–20-year-olds 1.40 where LightGBM's marginal age effect is 0.97, because
    LightGBM explains them through their BonusMalus.
  - VehBrand: the GLM gives it a 9% share of premium variance; LightGBM gives it about 0.
  - BonusMalus 58–67: GLM 2.27, SHAP 1.85.

**Stability.**
- *Level effects:* the most volatile levels belong to thin factor levels in both models (Region, VehBrand, young
  drivers). LightGBM's largest level spread across refits is 14% and the GLM's 20%.
- *Policy-level premiums:* what an individual customer would feel is the change in their own priced frequency between refits.
  The picture reverses here:

  | | Median max/min − 1 | 95th percentile | Policies moving >10% |
  |---|---|---|---|
  | GLM | 7.5% | 15% | 26% |
  | LightGBM | 20% | 41% | 90% |

  LightGBM's level averages are stable because they average over thousands of policies. Each policy's tree
  path, however, changes with the training sample. A tariff built on it would move most customers' premiums
  by double digits at every refresh, with nothing about their risk having changed. That is a problem for
  customers, for complaint volumes and for rate filings, and phase 6's rate caps have to absorb it.

**Monotonicity.** The GLM's BonusMalus scale has a reversal (58–67 above 68–79) in every refit. A filed tariff
would merge or smooth these bands, at a small cost in fit. LightGBM's marginal effect is monotone in the full
model; monotone constraints are available if it is used.

**Fairness: geography as a proxy (D-021, D-022).**
- *What the data can't test:* there are no protected attributes, so discrimination against a protected group
  can't be tested. Region and density are the plausible socioeconomic proxies.
- *How much of the premium is geography:* 13.5% of log-premium variance for the GLM (density 8.3%, Region 5.2%) and
  10.4% for LightGBM.
- *Does it follow experience?*
  - By density band, the GLM premium index rises from 0.69 (most rural) to 1.32 (densest), and LightGBM's from
    0.66 to 1.37. Observed loss cost, including large losses, rises from 0.58 to 1.39, with noise in between.
  - By region, premiums correlate with observed loss cost at 0.54 (GLM) and 0.64 (LightGBM). They are compressed
    relative to the noisy regional experience (premium index 0.62–1.29 vs observed 0.37–1.73).
- *What removing it costs (grouped CV):*
  - Region alone: Gini −0.001 and deviance +0.0003. Region adds 22 levels, some very thin, for almost no
    out-of-sample accuracy.
  - Region and density: Gini −0.017. Density is the factor that carries real risk information.
- *Who pays if geography is removed:* refitting without both and rebalancing to the same book premium moves
  52% of exposure by more than 10%. The most rural band pays 34% more and the densest 15% less, a sizeable
  cross-subsidy from rural to urban drivers.
- *Implication:* Region is a candidate to drop, or to shrink by credibility weighting, on fairness and
  explainability grounds at negligible cost. Density is justified by risk but should be monitored as a proxy.

## 7. Pricing strategy simulation (phase 6)
Source: `reports/strategy_simulation.md`. Everything that depends on customer behaviour is **SYNTHETIC** and rests
on **assumed** parameters (`pricing` in `configs/config.yaml`). No price, lapse or competitor data exist for
freMTPL2, so retention is a labelled assumption, not an estimate. All monetary results are estimates under these
assumptions.

**Set-up.**
- *Book:* the 135,413 holdout policies renew for one year with their observed exposure.
- *Technical premium:* true cost / (1 − expense ratio 25% − margin 5%).
- *Comparison basis:* every strategy collects the same total from the full book, so strategies differ only in
  how they spread it (D-023).
- *Strategies:*
  - flat rate;
  - the current tariff (BonusMalus only, D-023);
  - GLM;
  - LightGBM;
  - GLM and LightGBM with changes capped at ±15% of the current premium;
  - my own design: a *filable GLM* (Region dropped, the two non-monotone BonusMalus bands merged, ±15% cap; D-026).

**Duration terms resolve D-012, D-014 and D-017.**
- *Short-period scale:* premium = full-year rate × exposure × a short-period factor by exposure band. The factor
  is each model's own fitted / full-year-priced cost on the training split: for the GLM, 4.75 for records under a
  month down to 1.48 for 0.75–1 year. LightGBM needs smaller factors (3.32 to 1.19) because its full-year rate
  already carries part of the duration effect.
- *Large-loss loading:* 11.6% on full-year records and 48.2% on part-year records, pooled because per-band
  loadings were set by a handful of huge claims.
- *Effect:* together these fund the short-record cost without charging it to full-year customers.

**"Truth" and competitors (D-024).**
- *Truth:* expected cost per policy is the geometric mean of the GLM and LightGBM fitted costs, calibrated to
  actual total holdout losses. The competitor premium is that cost / (1 − 30%).
- *Two profit views:*
  - *Expected:* model-based cost.
  - *Realised:* actual holdout losses of each policy weighted by its retention probability. This is model-free
    and unbiased under the retention assumption, but noisy.
- *Robustness:* truth = GLM and truth = LightGBM are run as sensitivities.

**Retention (D-025, ASSUMED).** logit P(renew) = logit(0.85) − 3·log(new/current) − 4·log(new/competitor).

**Results (base case).**

| Strategy | Retention (exposure) | Loss ratio, expected | Loss ratio, realised | Realised profit | P(loss), Monte Carlo |
|---|---|---|---|---|---|
| flat | 51% | 0.89 | 0.88 | −1.10M | 0.82 |
| current | 78% | 0.77 | 0.76 | −0.17M | 0.49 |
| GLM | 80% | 0.72 | 0.74 | 0.10M | 0.33 |
| LightGBM | 80% | 0.79 | 0.74 | 0.05M | 0.53 |
| GLM capped ±15% | 80% | 0.73 | 0.73 | **0.26M** | 0.36 |
| LightGBM capped ±15% | 81% | 0.74 | 0.75 | 0.04M | 0.40 |
| filable GLM capped ±15% | 80% | 0.73 | 0.73 | 0.19M | 0.38 |

The total premium in play is about 12M a year.

**What it shows.**
1. *Adverse selection is real and large.* A flat rate keeps 95% of the riskiest decile and 15% of the safest
   (retained risk mix 1.36× the book), and loses 1.1M. The BonusMalus-only status quo also loses money.
2. *Rate caps change the ranking (brief §3d).*
   - Uncapped, the GLM and LightGBM move the average customer's premium by 27% and 35%. 23% and 19% of customers
     face increases above 25%, and change-driven lapses eat the gain from better risk differentiation.
   - Capping at ±15% (average change 12%) raises the GLM's realised profit from 0.10M to 0.26M.
   - LightGBM gains nothing from the cap: its ranking advantage (phase 4) does not survive the price constraints
     and elasticity. Its expected profit under the blended truth is even negative, partly because its premiums are
     more dispersed than the blend.
3. *The cap level matters.* Realised profit of the capped GLM rises with the cap: ±5% 0.03M, ±10% 0.18M,
   ±15% 0.26M, ±25% 0.30M. Uncapped, it is 0.10M. A moderate cap (15–25%) is best under these assumptions.
4. *Robust to the truth assumption.*
   - The capped GLM has the highest realised profit under all three truths: blend 0.26M, GLM-truth 0.55M,
     LightGBM-truth −0.06M.
   - When competitors price like LightGBM, every strategy loses money, the capped GLM least.
   - On *expected* profit the uncapped GLM is highest under the blend (0.41M vs 0.23M capped). The two views
     disagree on cap vs no cap, but not on GLM vs LightGBM.
5. *Elasticity and expenses dominate the level of profit, not the ranking.*
   - Halving elasticity lifts every strategy. Doubling it makes all of them lose money, the capped GLM least
     (−0.08M).
   - Actual expenses at 30% instead of the priced 25% turn every strategy negative (capped GLM −0.36M); at 20%
     the capped GLM earns 0.87M.
6. *One year is noisy.* The Monte Carlo profit SD is about 1.5M (12% of premium). The 99.5% one-year VaR is about
   4.9–5.0M for the GLM strategies (40–41% of premium), driven by very large claims on a small book. The differences
   between the GLM strategies (0.1–0.2M) are far smaller than one year's noise, so the choice has to rest on
   expected value, robustness and structural arguments, not on a single year's result.

**Limitations.**
- Retention elasticities and the competitor model are assumptions.
- One renewal year, no new business, retention treated as expected rather than simulated.
- Claim counts are Poisson with a systemic shock; no parameter risk on the models themselves.
- The holdout informs the final model choice a second time (after phase 4); no parameters are tuned on it here.

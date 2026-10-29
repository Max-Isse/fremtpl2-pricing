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
- Phase 2 will test whether adding an exposure-band *indicator* changes other relativities materially; if not,
  the offset model stands without it.

## 2. Frequency vs severity
*Phase 2–3.*

## 3. Better accuracy ≠ better pricing
*Phases 4–6. First example in §1: the free log(exposure) coefficient.*

## 4. Large losses
Claims above the training 99.5th percentile (32,162) are capped and the excess (37.8% of capped training losses)
is spread as a flat loading (D-005). The severity model is therefore a model of *attritional* severity; the
loading's size is a pricing assumption carried to the technical premium and included in sensitivity analysis.

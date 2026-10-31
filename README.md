# freMTPL2 motor pricing

Frequency–severity pricing for French motor third-party liability on the public freMTPL2 data. The project
covers GLMs with proper exposure handling, LightGBM benchmarks, calibration and stability checks, fairness
review, a pricing-strategy simulation and a rating-engine API.

**Status:** phases 1–5 of 8 are complete (data, SQL layer, cleaning, exposure analysis, frequency, severity,
pure premium, benchmarks, holdout comparison, explainability and fairness). See
[docs/phase_plan.md](docs/phase_plan.md).

## Decision supported

The rate charged per risk profile: a premium that covers expected claims, expenses and margin without
unjustified price differences. See [docs/business_case.md](docs/business_case.md).

## Phase 1 findings

All numbers come from [reports/data_quality.md](reports/data_quality.md) and
[reports/exposure_analysis.md](reports/exposure_analysis.md).

- **27% of recorded claims have no amount.** 9,657 claims across 9,116 policies have no amount records,
  and almost all of them sit in the first IDpol block. Claim counts are taken from the amount records so
  that frequency × severity is consistent (D-003).
- **38% of rows are fragments of split policies.** These rows share all nine rating factors with another
  row, and the train/test split is grouped on them to avoid leakage (D-006).
- **Large losses are a material load.** Capping claims at the 99.5th percentile (32,162) moves 27% of
  training losses into a flat 37.8% loading (D-005).
- **Exposure must be an offset.** Ignoring exposure halves the implied annual frequency (0.039 vs 0.074
  observed). Using log(exposure) as a free feature fits better (coefficient 0.67) but gives a non-filable
  pro-rata rule (D-008, [methodology §1](docs/methodology.md)).

## Phase 2 findings (frequency GLM)

All numbers come from [reports/frequency.md](reports/frequency.md).

- **Full-year cover is priced correctly.** An exposure-band control is fitted but not charged (D-012). Out of
  fold, full-year A/E is 1.000; with exposure only as an offset it was 0.82, overcharging full-year customers by
  about 18%. Short records still claim more per year than a pro-rata tariff collects (book A/E 1.28), so that
  cost becomes an explicit loading in phase 6.
- **The GLM beats the baselines out of fold.** It explains 5.8% of Poisson deviance, against 3.7% for a
  BonusMalus-only tariff, and wins in 5/5 grouped folds.
- **Overdispersion is modest.** Pearson χ²/df is 1.27 and NB2 changes relativities by at most 1.8%. Inference uses
  standard errors clustered by rating profile (D-010).
- **An age × malus interaction was added.** Older drivers with a malus were under-predicted by up to 33%; a
  two-level AgeMalus factor fixes it (D-011).

## Phase 3 findings (severity and pure premium)

All numbers come from [reports/severity.md](reports/severity.md) and [reports/pure_premium.md](reports/pure_premium.md).

- **Severity needs far fewer factors.** It has 21k claims against 543k policy records for frequency. In-sample
  F-tests would keep five factors, but grouped CV keeps only VehAge (D-013). Frequency explains 98.7% of the
  variation in pure premium across policies.
- **Short records have more expensive claims.** Mean claim cost falls from 2,512 (records under one month) to
  1,369 (full-year records), so severity gets the same full-year control as frequency (D-014). The pure premium
  then prices full-year records correctly (A/E 1.005).
- **The Gamma mean fits, the Gamma distribution doesn't.** 41% of claims sit on four flat-rate amounts. The
  variance-vs-mean check supports the Gamma variance function, but quantile residuals are far from normal, so the
  phase 6 Monte Carlo will resample empirical amounts (D-016).
- **Tweedie is a cross-check, not the tariff.** It agrees on ranking (correlation 0.97) but isn't balanced and
  under-prices full-year records by 11% (D-015).

## Phase 4 findings (benchmarks on the holdout)

All numbers come from [reports/model_comparison.md](reports/model_comparison.md), with grouped-bootstrap 95% CIs.

- **LightGBM ranks risk better than the GLM.** The pure-premium Gini is 0.346 vs 0.303 (+0.043, CI 0.026–0.065),
  and 0.467 vs 0.396 on the full-year records that are actually quoted. Frequency deviance explained is 7.9% vs
  5.7%. The gain comes from frequency; for severity the two are equal.
- **Where they disagree, LightGBM is closer to actual losses.** In the double-lift deciles at the extremes, the
  GLM's A/E is 0.54 and 1.98 while LightGBM's is 0.98 and 1.13.
- **But much of the disagreement can't be validated.** Priced premiums differ by more than 25% for 55% of
  policies, concentrated in segments that are almost never full-year (B12: 97% part-year; new vehicles: 99%), and
  LightGBM moves short-record cost into rating factors. Which model sets prices is left open for phases 5–6
  (D-019).

## Phase 5 findings (explainability, stability, fairness)

All numbers come from [reports/explainability_fairness.md](reports/explainability_fairness.md) (training split only).

- **The two models agree on what matters most and disagree on the rest.** BonusMalus drives about 60% of premium
  variation in both. The GLM charges 18–20-year-olds 1.40× where LightGBM attributes their risk to BonusMalus
  (SHAP 0.97), and VehBrand matters to the GLM but hardly at all to LightGBM.
- **Better deviance, worse segments.** The random forest beats the GLM on holdout deviance yet miscalibrates 17 of
  73 segments, against 3 for the GLM. LightGBM over-predicts the largest segment, bonus-50 drivers, by 4%.
- **Stability and monotonicity.** The GLM's BonusMalus scale is non-monotone (58–67 above 68–79) in every refit.
  Individual premiums are far less stable for LightGBM: across five refits on 80% subsamples, the median
  policy's premium moves 20% (GLM 7.5%), and 90% of policies move by more than 10% (GLM 26%).
- **Geography.** It drives 13.5% of GLM premium variation. Dropping Region costs almost nothing (CV Gini −0.001);
  dropping density as well costs 0.017 Gini and moves 52% of exposure by more than 10%, the most rural band paying
  34% more. Proposal: drop Region, keep density with monitoring (D-022).

## How to run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv installs it if missing).

```bash
uv sync                    # install pinned dependencies from uv.lock
uv run fremtpl2 all        # fetch -> build -> report -> frequency -> severity -> pure_premium -> benchmarks -> compare -> explain (~30 min)
uv run pytest              # tests (the full-data leakage test runs when the DB exists)
uv run ruff check .        # lint
```

`make all` / `make test` / `make lint` wrap the same commands where `make` is available. The individual
steps are `uv run fremtpl2 fetch | build | report | frequency | severity | pure_premium | benchmarks | compare | explain`. Every threshold is set in
[configs/config.yaml](configs/config.yaml).

## Layout

| Path | Contents |
|---|---|
| `sql/` | numbered DuckDB transforms (raw → policy / exposure / claims → clean) |
| `src/fremtpl2_pricing/` | importable package: data, analysis, reporting (models, pricing, API to follow) |
| `tests/` | pytest on hand-built fixtures; leakage checks |
| `docs/` | business case, methodology, decisions, data lineage, phase plan |
| `reports/` | generated reports, CSV tables and figures |

## Data

OpenML 41214 / 41215 (freMTPL2freq / freMTPL2sev, originally from CASdatasets). The data has no policy
dates, so validation uses a grouped holdout plus 5-fold CV instead of an out-of-time split. See
[data/README.md](data/README.md) and [docs/data_lineage.md](docs/data_lineage.md).

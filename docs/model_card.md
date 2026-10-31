# Model card: freMTPL2 GLM motor TPL tariff

| | |
|---|---|
| **Model** | Frequency GLM (Poisson, log link, log-exposure offset) × severity GLM (Gamma, log link), with an exposure-band control pinned at full year and a large-loss loading split full-year / part-year |
| **Deployed artefact** | `artifacts/tariff.json`, model version `fremtpl2-glm-tariff-<hash>`. The hash covers coefficients, duration terms and assumptions; see `/model-info` |
| **Variants** | `full` (all factors) and `filable` (Region dropped, monotone BonusMalus scale; recommended, D-026); set with `scoring.variant` |
| **Owner / status** | Portfolio project; not a production tariff |
| **Code** | `src/fremtpl2_pricing/` (models, scoring engine, API), tests in `tests/` |

## Intended use

- **For:** quoting the annual technical premium and the premium for a contract term for French motor third-party
  liability policies with the rating factors below. It also serves as a reference tariff for strategy analysis.
- **Not for:**
  - individual underwriting decisions;
  - risks outside the training ranges (rejected by validation);
  - other countries or other covers;
  - pricing without a rate-level, expense and reinsurance review.

## Inputs

| Factor | Type | Accepted values |
|---|---|---|
| DrivAge | integer | 18–100 |
| VehAge | integer | 0–100 |
| VehPower | integer | 4–15 |
| BonusMalus | integer | 50–230 |
| Density | number | 1–27,000 inhabitants/km² |
| VehBrand | category | levels in the tariff |
| VehGas | category | levels in the tariff |
| Region | category | levels in the tariff (full variant); accepted but not used by the filable variant |
| exposure | number | contract term in years, (0, 1], default 1 |
| current_premium | number | optional; enables the ±15% change cap |

## Output

- annual frequency;
- severity per claim;
- risk premium;
- technical premium (risk / (1 − 25% expenses − 5% margin));
- premium for the term;
- whether the cap applied;
- the three largest factor effects;
- the model version.

## Training data

- **Source:** OpenML 41214 and 41215 (freMTPL2freq and freMTPL2sev, CASdatasets), version 1. SHA-256 hashes are in
  `data/raw_manifest.json`.
- **Cleaning:** exposure capped at 1; claim counts taken from the amount records (D-003) and capped at 4; claims
  capped at the training 99.5th percentile (32,162) (`reports/data_quality.md`).
- **Split:** 80% training (542,600 records, grouped 5-fold CV) and a 20% holdout grouped on identical rating
  profiles (D-006).

## Performance (holdout, 95% grouped-bootstrap CIs; `reports/model_comparison.md`)

| Metric | GLM | Reference |
|---|---|---|
| Frequency Poisson deviance explained vs portfolio mean | 5.7% | BonusMalus-only 3.8%, LightGBM 7.9% |
| Pure-premium Gini (all / full-year records) | 0.303 / 0.397 | LightGBM 0.346 / 0.467 |
| Full-year A/E of the priced premium | 1.037 | LightGBM 1.006 |
| Rating-factor segments with A/E outside the 95% CI | 3 of 73 | LightGBM 4, random forest 17 |
| Median change in a policy's premium across refits | 7.5% | LightGBM 20% |

## Assumptions and limitations

- **One observation year with no policy dates:** no out-of-time validation is possible. Drift, claims inflation and
  the effect of the tariff on its own book are untested (`docs/limitations.md`).
- **Short records:** they have much higher annual frequency and severity in this data (split records, early
  terminations). The short-period factors (for example 2.15 for 3–6 months) need business review.
- **Severity:** weakly predictable (2.8% of CV deviance explained). The Gamma variance function fits but its shape
  does not, because 41% of claims are flat-rate amounts. Do not simulate claims from the fitted Gamma.
- **Overdispersion:** modest. Inference uses cluster-robust standard errors.
- **Interactions:** AgeMalus is the only interaction. LightGBM's double lift shows some remaining structure.
- **Behaviour figures:** expected-profit and retention figures depend on assumed elasticities (D-025).

## Fairness and ethical considerations

- **No direct use of protected attributes:** there is no gender or other protected attribute in the data. EU unisex
  pricing (Test-Achats) is respected by construction; indirect proxies cannot be tested with this data.
- **Geography:** Region and density are possible socioeconomic proxies. Density tracks observed loss cost and
  carries most of the geographic signal. Region adds almost no accuracy and is dropped in the filable variant
  (`reports/explainability_fairness.md` §4).
- **Change cap:** the ±15% cap limits disruption for individual customers. The size of the cross-subsidy it creates
  should be monitored.

## Monitoring and maintenance

See `docs/monitoring_plan.md`. Retrain annually, or when full-year A/E leaves the range 0.90–1.10 for two
consecutive quarters. Any refit produces a new model version.

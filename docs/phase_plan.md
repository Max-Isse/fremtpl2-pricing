# Phase plan

Each phase ends with: tests green, lint clean, a summary of what was built, key results, assumptions and
open questions. Status is updated at the end of each phase.

| Phase | Scope | Main outputs | Status |
|---|---|---|---|
| 0 | Repo structure, standards, phase plan, business case | this file, `docs/business_case.md`, `docs/project_standards.md` | done |
| 1 | Data loading (OpenML → Parquet), DuckDB SQL layer, cleaning with documented impact, near-duplicate grouping, grouped holdout + 5 folds, exposure analysis | `sql/`, `src/.../data/`, `reports/data_quality.md`, `reports/exposure_analysis.md`, methodology §1 | done |
| 2 | Frequency GLM: Poisson with offset, overdispersion tests (Pearson χ²/df, Cameron–Trivedi), NB2, coefficients/CIs/relativities, LR tests per factor, residual checks. Baselines (portfolio average, BonusMalus-only). | `features.py`, `models/glm.py`, `models/frequency.py`, `reports/frequency.md`, methodology §2a, D-009–D-012 | done |
| 3 | Severity GLM (Gamma, log link, weighted by claim count), variance-vs-mean check, factor selection; pure premium = freq × sev × large-loss loading; Tweedie GLM cross-check | `models/severity.py`, `models/pure_premium.py`, `reports/severity.md`, `reports/pure_premium.md`, methodology §2b–2c, D-013–D-016 | done |
| 4 | Benchmarks: RF on claim rate (exposure-weighted), LightGBM Poisson / Gamma / Tweedie with CV tuning. Evaluation: deviance, ordered Lorenz/Gini, lift and double-lift, calibration by decile, A/E by segment with exposure, bootstrap CIs | `models/benchmarks.py`, `evaluation/metrics.py`, `evaluation/compare.py`, `reports/model_comparison.md`, methodology §5, D-018–D-020 | done |
| 5 | Explainability and fairness: GLM relativity plots with CIs, SHAP vs relativities, proxy-risk review of region/density, premium dispersion, regulatory context; "better accuracy ≠ better pricing" evidence (methodology §3) | `analysis/explain.py`, `analysis/fairness.py`, `reports/explainability_fairness.md`, methodology §3 and §6, D-021–D-022 | done |
| 6 | Pricing strategy simulation: technical premium, five strategies, assumed elasticity retention model, risk-mix shift, Monte Carlo (≥5,000 runs) profit / VaR / TVaR, sensitivity on elasticity, expense ratio, cap | `pricing/`, `configs/pricing.yaml`, `reports/strategy_simulation.md` | |
| 7 | Deployment: batch scoring with validation and model version, FastAPI rating engine with schemas and tests, Dockerfile, optional Streamlit | `scoring/`, `api/`, `Dockerfile` | |
| 8 | Documentation pack: `reports/pricing_report.md`, `reports/recommendations.md`, `docs/model_card.md`, monitoring plan, limitations, final README | | |

## Repository layout

```
configs/config.yaml          all thresholds, paths, seeds (pricing assumptions added in phase 6)
data/raw/                    OpenML Parquet cache (git-ignored); data/raw_manifest.json pins the version
data/processed/              DuckDB database (git-ignored, rebuilt by `fremtpl2 build`)
sql/                         numbered SQL transforms run by src/.../data/build.py
src/fremtpl2_pricing/
  data/                      fetch, DuckDB runner, split assignment, quality report
  analysis/                  descriptive analyses (exposure)
  reporting/                 generated Markdown reports and figures
  models/ evaluation/ pricing/ scoring/ api/   (phases 2-7)
tests/                       pytest on small hand-built fixtures; full-data leakage test runs if DB exists
docs/                        business case, methodology, decisions, lineage, model card
reports/                     generated reports, tables (CSV) and figures
notebooks/                   narrative only, importing from src/ (added when there are model results)
```

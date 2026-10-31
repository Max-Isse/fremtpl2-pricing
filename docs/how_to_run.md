# How to run and reproduce this project

A guide for another analyst picking the project up. It assumes no knowledge of the code.

## 1. Set up

- Install [uv](https://docs.astral.sh/uv/). uv installs the pinned Python 3.11 and every dependency from `uv.lock`.
- Windows, macOS and Linux all work; `make` is optional because every step is a `uv run fremtpl2 …` command.

```bash
uv sync
```

## 2. Reproduce everything from raw data (about 35 minutes)

```bash
uv run fremtpl2 all
```

The steps, in order (each can be run on its own):

| Step | Command | Reads | Writes |
|---|---|---|---|
| Download | `fremtpl2 fetch` | OpenML 41214/41215 | `data/raw/*.parquet`, `data/raw_manifest.json` |
| Build tables | `fremtpl2 build` | raw Parquet, `sql/*.sql` | `data/processed/fremtpl2.duckdb` |
| Data quality, exposure | `fremtpl2 report` | DuckDB | `reports/data_quality.md`, `reports/exposure_analysis.md` |
| Frequency GLM | `fremtpl2 frequency` | DuckDB | `artifacts/frequency_glm.json`, `reports/frequency.md` |
| Severity GLM | `fremtpl2 severity` | DuckDB, frequency tables | `artifacts/severity_glm.json`, `reports/severity.md` |
| Pure premium, Tweedie | `fremtpl2 pure_premium` | artefacts | `artifacts/pure_premium.json`, `reports/pure_premium.md` |
| LightGBM, random forest | `fremtpl2 benchmarks` | DuckDB, severity artefact | `artifacts/lgbm_*.txt`, `rf_*.joblib`, tuning tables |
| Holdout comparison | `fremtpl2 compare` | all artefacts, holdout | `reports/model_comparison.md` |
| Explainability, fairness | `fremtpl2 explain` | artefacts, training data | `reports/explainability_fairness.md` |
| Strategy simulation | `fremtpl2 simulate` | artefacts, holdout | `reports/strategy_simulation.md`, `SYNTHETIC_*` tables |
| Deployable tariff | `fremtpl2 package` | artefacts | `artifacts/tariff.json` |
| Batch scoring | `fremtpl2 score --input in.csv --output out.csv` | `tariff.json` | quotes and rejects |

The CSV behind every table in `reports/` is in `reports/tables/`. Generated Markdown starts with a "GENERATED" banner
and should not be edited by hand. The narrative documents (`docs/`, `reports/pricing_report.md`,
`reports/recommendations.md`) quote those tables; re-check them after changing anything upstream.

## 3. Tests and lint

```bash
uv run pytest
```

```bash
uv run ruff check .
```

```bash
uv run mypy src
```

- The tests use small hand-built and simulated data and do not need the download. One leakage test also runs on the
  full database when it exists.
- CI (`.github/workflows/ci.yml`) runs lint, types and tests on every push.

## 4. Change an assumption

- All thresholds, seeds and business assumptions are in `configs/config.yaml`. Nothing is hard-coded in `src/`.
- Typical changes:
  - expense ratio, margin, cap or retention elasticities: `pricing` → re-run `simulate` and `package`;
  - large-loss cap quantile: `cleaning.large_loss_quantile` → re-run from `build`;
  - deployed tariff variant: `scoring.variant` (`filable` or `full`) → re-run `package`.

## 5. Reproducibility

- **Dependencies:** pinned in `uv.lock` (Python 3.11).
- **Data version:** OpenML dataset version 1. `data/raw_manifest.json` records row counts and SHA-256 hashes of the
  cached files. A new download with different hashes means results are not comparable.
- **Randomness:** one seed (`seed` in config) feeds the split, CV folds, bootstrap, LightGBM (`deterministic: true`),
  the random forest and the Monte Carlo, each with a fixed offset.
- **Determinism check:** deleting `data/processed/` and `artifacts/` and re-running `fremtpl2 all` regenerates every
  table in `reports/tables/` identically. CSVs are written with 10 significant digits to absorb last-digit
  floating-point noise from parallel sums. Only the generation dates in the Markdown headers and the tariff build
  timestamp change.
- **Verified on 2026-10-31.** After deleting `data/processed/` and `artifacts/`, a full `fremtpl2 all` run
  (27 minutes) reproduced every table and the tariff coefficients exactly, except the Monte Carlo table. Its
  resampling query had no `ORDER BY`, so DuckDB's parallel scan could change row order. With the query ordered by
  `claim_id`, two consecutive runs give identical output.
- **Model version:** the hash in `tariff.json` changes whenever coefficients or pricing assumptions change, so a
  quote can always be traced to the tariff that produced it.

## 6. Run the rating API

```bash
uv run uvicorn fremtpl2_pricing.api.app:app --port 8000
```

Open `http://localhost:8000/docs` for the interactive schema, or see the curl example in the README. With Docker
(after `package`): `docker build -t fremtpl2-rating .` then `docker run --rm -p 8000:8000 fremtpl2-rating`.

## 7. Where to read

| Question | Document |
|---|---|
| What decision does this support? | `docs/business_case.md` |
| Why each model, and which checks? | `docs/methodology.md` |
| Every judgement call | `docs/decisions.md` |
| Raw → final data, with row counts | `docs/data_lineage.md`, `reports/data_quality.md` |
| What to recommend | `reports/pricing_report.md`, `reports/recommendations.md` |
| Model summary | `docs/model_card.md` |
| Limitations, monitoring | `docs/limitations.md`, `docs/monitoring_plan.md` |

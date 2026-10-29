# freMTPL2 motor pricing

Frequency–severity pricing for French motor third-party liability on the public freMTPL2 data. The project
covers GLMs with proper exposure handling, LightGBM benchmarks, calibration and stability checks, fairness
review, a pricing-strategy simulation and a rating-engine API.

**Status:** phase 1 of 8 is complete (data, SQL layer, cleaning, exposure analysis). See
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

## How to run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv installs it if missing).

```bash
uv sync                    # install pinned dependencies from uv.lock
uv run fremtpl2 all        # fetch -> build DuckDB -> regenerate reports (~1 min after first download)
uv run pytest              # tests (the full-data leakage test runs when the DB exists)
uv run ruff check .        # lint
```

`make all` / `make test` / `make lint` wrap the same commands where `make` is available. The individual
steps are `uv run fremtpl2 fetch | build | report`. Every threshold is set in
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

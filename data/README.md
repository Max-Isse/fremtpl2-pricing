# Data

## Sources

| Dataset | Source | Version | Content |
|---|---|---|---|
| freMTPL2freq | OpenML [41214](https://www.openml.org/d/41214) | 1 | 678,013 MTPL policies observed over one year: exposure, claim count, 9 rating factors |
| freMTPL2sev | OpenML [41215](https://www.openml.org/d/41215) | 1 | 26,639 individual claim amounts keyed by IDpol |

Originally published in the R package CASdatasets (Dutang & Charpentier), French motor third-party liability.
Region codes are the pre-2016 French administrative regions (R11 … R94).

`data/raw_manifest.json` (committed) records row counts and SHA-256 hashes of the cached Parquet files; if a
fresh download produces different hashes, results are not comparable with the committed reports.

## Layout

- `raw/` — Parquet cache written by `fremtpl2 fetch`. Git-ignored. Never modified after download.
- `processed/fremtpl2.duckdb` — all derived tables, rebuilt by `fremtpl2 build`. Git-ignored.

## Synthetic data

None is used in any reported result. `tests/conftest.py` contains a 25-row hand-built fixture and
`tests/test_exposure.py` simulates a Poisson portfolio with a known rate; both exist only to test code
(CI cannot download the full data on every run) and are never written to `data/`.

Phase 6 will add a SYNTHETIC Monte Carlo of claims on the retained book; it will be listed here when added.

## Optional external data

French regional statistics (e.g. INSEE median income by pre-2016 region) were considered for the proxy-risk
analysis in phase 5. Not yet added: the region codes must first be verified against an official
pre-2016 code list (open question).

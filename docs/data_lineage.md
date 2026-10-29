# Data lineage

Row counts at every step are generated in `reports/data_quality.md` §1 (CSV: `reports/tables/lineage.csv`).
This page describes the flow; it holds no numbers so it cannot drift from the code.

```
OpenML 41214 (freMTPL2freq) ─┐                      ┌─ policy        (rating factors / IDpol)
                             ├─ data/raw/*.parquet ─┼─ exposure      (raw, capped, flags)          ─┐
OpenML 41215 (freMTPL2sev) ──┘   raw_freq, raw_sev  └─ claims        (policy_found, amount_positive)│
                                                         │                                        │
                                   claims_by_policy ─── reconciliation (ClaimNb vs amount records)  │
                                                                                                  │
                                   profile_groups (identical rating factors) ── split (Python, seeded)
                                                                                                  │
                     policy_clean  (frequency / pure-premium base: capped exposure, capped count, ◄┘
                                    claim cost, group_id, is_holdout, fold)
                     claims_clean  (severity base: one row per positive claim, capped amount, excess)
                     large_loss_summary (threshold and loading from training claims)
```

| Step | SQL / code | Rule | Decision |
|---|---|---|---|
| Raw cache | `data/fetch.py` | Download once, normalise dtypes, strip ARFF quotes; SHA-256 in `data/raw_manifest.json` | D-001 |
| Raw tables | `sql/01_load_raw.sql` | Load Parquet unchanged; `claim_id` = source row order | |
| Policy / exposure | `sql/02_policy_exposure.sql` | Exposure capped at 1; non-positive flagged | D-002 |
| Claims | `sql/03_claims.sql` | Flag claims with no matching policy and non-positive amounts | |
| Reconciliation | `sql/04_reconcile.sql` | Status per policy: match / claims_without_amounts / … | D-003 |
| Profile groups | `sql/05_profile_groups.sql` | `dense_rank` over the nine rating factors | D-006 |
| Split | `data/split.py` | Seeded permutation of groups: 20% holdout, 5 folds | D-007 |
| Clean tables | `sql/06_clean_tables.sql` | Count source, count cap, large-loss cap (training quantile) | D-003–D-005 |
| Exposure summary | `sql/07_exposure_summary.sql` | Training split by exposure band | D-008 |

No SYNTHETIC data enters any reported result. Test fixtures in `tests/` are hand-built and used only by tests.

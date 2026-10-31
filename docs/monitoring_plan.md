# Monitoring plan

What to monitor once the tariff is live, how often, the threshold, and the action. The checks run in order: a
data-quality failure stops the statistical checks, because drift or A/E computed on broken inputs is meaningless.
The thresholds are starting points for a book of roughly this size (about 70k policy-years a year) and should be
recalibrated after the first year.

| # | What | Frequency | Threshold (initial) | Action |
|---|---|---|---|---|
| 1 | **Input quality:** rejected-row rate in batch scoring and API 422s, by reason (`reject_reason`) | daily | >1% of quotes, or any new reason | Fix the upstream feed before quoting continues |
| 2 | **Model version** on quotes matches the approved version | daily | any mismatch | Block quoting; investigate the deployment |
| 3 | **Input drift:** PSI of each rating factor (banded as in the tariff) against the training distribution | monthly | PSI > 0.10 investigate, > 0.25 act (rule of thumb until bootstrapped thresholds exist) | Check for mix change vs data error; if real, re-run the phase 6 simulation on the new mix |
| 4 | **Short-record mix:** share of part-year contracts and of each exposure band | monthly | ±5 points vs training | Short-period factors and the part-year large-loss loading assume this mix (R4) |
| 5 | **Full-year A/E (claim counts)**, exposure-weighted, with exact Poisson CI | quarterly | CI excludes 1 two quarters running, or A/E outside 0.90–1.10 | Rate-level review; refit if structural |
| 6 | **A/E by rating-factor level** (all tariff factors, exposure shown) | quarterly | CI excludes 1 for a level with >2,000 exposure-years | Review that factor's relativity; check the LightGBM challenger's view |
| 7 | **Severity and large losses:** mean capped severity (CUSUM), number and size of claims above the cap, share of flat-rate settlements | monthly | Severity CUSUM signal; excess ratio outside 5–25% full-year | Review the cap and large-loss loading; claims inflation check |
| 8 | **Cap impact:** share of quotes capped, book premium vs uncapped target | monthly | Book premium more than 2% below target | Revisit the cap width (R3) or the base rate |
| 9 | **Retention vs assumption:** observed renewal rate by price-change band and risk decile vs the assumed logit | monthly (after renewals) | Outside the assumed curve by more than 5 points in any band | Refit elasticities (price test, R6); rerun the simulation |
| 10 | **Risk mix of the retained book:** mean GLM risk of renewals vs lapses | quarterly | Retained mix index > 1.05 | Adverse selection building: check competitiveness of low-risk segments |
| 11 | **Challenger:** LightGBM vs GLM double lift and Gini on new business, full-year records | quarterly | LightGBM A/E closer to 1 in the outer deciles two quarters running | Investigate the segments for new GLM factors (R5) |
| 12 | **Fairness:** premium index by density band and region vs observed loss cost; premium dispersion | annually | Premium index moves away from experience by more than 10% in a band | Review geographic factors (D-022) |

**Retraining:** annually, or when trigger 5 fires. Refit on the latest full year, re-run the whole pipeline
(`uv run fremtpl2 all`), compare challenger against champion on the latest labelled data (same metrics as
`reports/model_comparison.md`), and promote only if better on full-year A/E and Gini without worse stability. Keep
the previous `tariff.json` for rollback; the model version in every quote identifies which tariff produced it.

**Labels arrive late:** claims for a policy-year are not fully known until months after it ends. A/E for recent
periods should use development factors or be read only for mature periods.

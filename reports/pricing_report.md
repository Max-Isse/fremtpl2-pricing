# Pricing report: which model and which strategy for the motor TPL renewal tariff

*For: Pricing Manager. Basis: freMTPL2 (French MTPL, 678k policy records, one observation year). All monetary
figures are estimates under the assumptions stated in section 5 and in `configs/config.yaml`; the customer-behaviour
figures come from a SYNTHETIC simulation with an assumed price elasticity. Supporting detail: `reports/` and
`docs/methodology.md`.*

## Recommendation

1. **File a frequency × severity GLM, not the LightGBM model.** Use the "filable" version: Region removed, and the
   two BonusMalus bands that rate in the wrong order (58–67 above 68–79) merged so the malus scale is monotone.
2. **Cap premium changes at ±15% of each customer's current premium** for the first renewal cycle, and review the
   cap after a year of observed lapses.
3. **Keep LightGBM as a challenger and diagnostic tool,** not as the price-setter. It ranks risk better, but most of
   that advantage does not survive price caps and customer reactions, and its individual premiums are too unstable
   to file.

Against the current BonusMalus-only tariff, the recommended tariff is estimated to improve the loss ratio from 0.76
to 0.73 and annual profit by about 0.37M on a book of about 12M premium. It does so under all three "true cost"
assumptions tested. The full GLM with Region would earn about 0.06M more (0.5% of premium), well within one year's
noise; the filable version is preferred for regulatory and fairness reasons.

## 1. The decision

The decision is the rate charged to each risk profile: a premium that covers expected claims, expenses (25%
assumed) and margin (5%), without unjustified differences. Underpricing a segment loses money and attracts more of
that risk. Overpricing loses good customers. Large unexplained moves cause complaints and filing problems
(`docs/business_case.md`).

## 2. What was compared

| Criterion | GLM (freq × sev) | LightGBM (freq × sev) | Evidence |
|---|---|---|---|
| Ranking accuracy (holdout pure-premium Gini, 95% CI) | 0.303 (0.264–0.344) | **0.346** (0.314–0.380) | `model_comparison.md` §3 |
| Same, full-year policies only (what is quoted) | 0.397 | **0.467**; difference +0.070 (0.048–0.096) | §3 |
| Frequency deviance explained (holdout) | 5.7% | **7.9%** | §1 |
| Calibration on full-year business (A/E) | 1.037 | **1.006** | §4 |
| Rating-factor segments miscalibrated (outside 95% CI) | **3 of 73** | 4 of 73, incl. the largest (bonus-50, A/E 0.96) | §5 |
| Premium stability: median policy's change between refits | **7.5%** | 20% (90% of policies move >10%) | `explainability_fairness.md` §2 |
| Monotone malus scale | no (fixed in the filable version) | yes (marginal effect) | §3 |
| Explainable, filable relativity table | **yes** | no (SHAP gives attributions, not a tariff) | §1 |
| Simulated profit, capped ±15% (actual holdout losses) | **0.26M** (filable: 0.19M) | 0.04M | `strategy_simulation.md` §2 |

Severity is equally weak in both: claim cost varies far more than any rating factor can explain. Frequency drives about
99% of the premium differentiation.

## 3. Why not LightGBM, despite better accuracy

- **Its gains sit where they can't be checked.** Its premiums differ from the GLM's by more than 25% for 55% of
  policies, concentrated in profiles that are almost never on cover for a full year (VehBrand B12: 97% part-year
  records; new vehicles: 99%). For those, any full-year price is an extrapolation.
- **Its prices are unstable.** Retraining on a slightly different sample would move most customers' premiums by
  double digits with no change in their risk. That means complaints, refiling, and lapses driven by change, not
  by risk.
- **Its ranking advantage does not turn into profit.**
  - With customers reacting to price changes and to competitor prices (assumed elasticity), LightGBM earns
    0.05M uncapped and 0.04M capped, against 0.26M for the capped GLM.
  - Uncapped, its expected profit under the blended "true cost" view is negative.
  - The capped GLM is the most profitable strategy under all three truth assumptions: blended, GLM-like and
    LightGBM-like competitors.
- **It is still valuable as a diagnostic.** Where the two disagree, LightGBM is closer to actual losses (double
  lift: the GLM's A/E is 0.54 and 1.98 in the extreme deciles, LightGBM's 0.98 and 1.13). The AgeMalus interaction
  was found the same way. Use its disagreements to find the next GLM factors.

## 4. The recommended strategy and its impact

Each strategy below collects the same total premium from the book; they differ in how it is spread. Realised
figures use the actual holdout losses of the customers each strategy is expected to keep.

| Strategy | Retention | Loss ratio (realised) | Profit (realised) | P(loss year) | 99.5% VaR, % of premium |
|---|---|---|---|---|---|
| Current (BonusMalus only) | 78% | 0.765 | −0.17M | 49% | 46% |
| Flat rate | 51% | 0.879 | −1.10M | 82% | 72% |
| GLM, uncapped | 80% | 0.742 | 0.10M | 33% | 41% |
| GLM, ±15% cap | 80% | 0.729 | **0.26M** | 36% | 40% |
| **Filable GLM, ±15% cap (recommended)** | 80% | 0.734 | 0.19M | 38% | 40% |
| LightGBM, ±15% cap | 81% | 0.746 | 0.04M | 40% | 45% |

- **Customers:** average premium change 12%; 41% of customers pay more than 10% extra and 35% pay more than 10%
  less, with no change beyond ±15%.
- **Cap level:** for the capped GLM, profit rises from 0.03M at ±5% to 0.26M at ±15% and 0.30M at ±25%.
  Uncapped, it falls to 0.10M because large increases drive lapses. ±15% is the prudent first step; widen it once
  real lapse data are in.
- **Short-term business:** policies on cover for less than a year are charged through a short-period scale and a
  higher large-loss loading (48% vs 12%). Pricing every policy pro rata at the full-year rate would fund only 64%
  of the book's capped losses.

## 5. Risks and what would change this recommendation

- **The behaviour assumptions are not estimates.** Retention uses an assumed logit: base 85%, −3 per log price
  change, −4 per log price gap to competitors.
  - Doubling sensitivity makes every strategy lose money (filable capped −0.16M, current −0.64M).
  - Halving it lifts all of them.
  - The capped GLM stays ahead of the current tariff in every case. **A randomised price test on a sample of
    renewals should precede full rollout.**
- **Expenses.** If expenses run at 30% instead of the 25% priced in, every strategy loses money (filable capped
  −0.42M). The rate level, not the structure, must absorb that.
- **One year is noisy.** Profit varies by about 1.5M from year to year (12% of premium), and very large claims drive
  the tail. Differences between the GLM variants (0.06M) cannot be seen in one year's results.
- **Short-period factors** reflect how this dataset records policies (split records, early terminations): about
  1.4× the annual premium for a half-year term. They need business review against the insurer's cancellation rules
  before use.
- **What would change it:**
  - a price test showing much lower elasticity (wider cap);
  - a stabilised LightGBM (monotone constraints, bagged refits) that wins on full-year business in a
    champion–challenger test;
  - external evidence that density proxies deprivation after controlling for claims (revisit density);
  - a regulator accepting Region (the full GLM earns about 0.06M more).

## 6. Next steps

1. Package the filable GLM as the deployed tariff. The rating engine builds it with `scoring.variant: filable`.
2. Business review of the short-period scale and the large-loss loadings.
3. Design and run a price test (D-025) to replace the assumed elasticities.
4. Go live with the monitoring plan in `docs/monitoring_plan.md`: full-year A/E, segment A/E, cap impact on book
   premium, lapse rates against assumption, large-loss experience.
5. Run LightGBM as a shadow challenger and review its largest disagreements quarterly for new GLM factors.

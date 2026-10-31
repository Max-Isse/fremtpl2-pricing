# Recommendations

Each recommendation states what to do, the estimated effect, the assumptions it rests on, the main risks, and what
would change it. Every estimate comes from code in this repository (sources in brackets); behaviour-dependent figures
come from the SYNTHETIC simulation and an assumed price elasticity (`pricing` in `configs/config.yaml`). The total
premium in the simulated book is about 12M a year.

## R1. Replace the BonusMalus-only tariff with a frequency × severity GLM tariff

- **What:** frequency GLM with eight rating factors plus the AgeMalus interaction, and a severity GLM with VehAge
  only, both fitted with an exposure-band control and priced at full-year cover.
- **Estimated effect:** realised loss ratio 0.765 → 0.734 and profit −0.17M → +0.19M a year (filable version,
  ±15% cap; `strategy_simulation.md` §2).
- **Robustness:** the gain holds under every "true cost" assumption tested: +0.37M (blended truth), +0.41M
  (GLM-like), +0.27M (LightGBM-like) (`SYNTHETIC_sim_sensitivity.csv`).
- **Assumptions:**
  - retention logit: base 85%, −3 per log change against the current premium, −4 per log gap to competitors;
  - expenses 25%, margin 5%;
  - competitors price at the blended true cost.
- **Risks:** elasticity is unknown. If it is twice as strong, every strategy loses money, but the new tariff still
  beats the current one by 0.48M.
- **Would change if:** a price test showed customers do not react to price differences relative to competitors.
  In that case adverse selection, the main source of the gain, would be smaller.

## R2. File the "filable" version: drop Region, merge BonusMalus 58–67 with 68–79

- **What:** remove Region and merge the two non-monotone BonusMalus bands (D-022, D-026).
- **Why:**
  - Region adds almost nothing out of sample (cross-validated Gini −0.001 when removed).
  - It has 22 levels, several of them very thin, with relativities that swing up to 20% between refits.
  - It is a plausible socioeconomic proxy.
  - The unmerged malus scale charges some drivers with a worse coefficient less.
- **Cost:** 0.06M a year in realised profit against the full GLM (0.5% of premium; 0.04M on expected profit),
  smaller than one year's noise (SD about 1.5M).
- **Keep density:** removing it as well would cost 0.017 Gini and move 52% of exposure by more than 10%; the most
  rural band would pay 34% more (`explainability_fairness.md` §4).
- **Would change if:** the regulator accepts Region, or external data show density acting as a deprivation proxy
  after controlling for claims.

## R3. Cap premium changes at ±15% for the first renewal cycle

- **Estimated effect (GLM):** realised profit 0.26M with the cap against 0.10M uncapped. Uncapped, 23% of customers
  would face increases above 25%.
- **Sensitivity:** ±5% 0.03M, ±10% 0.18M, ±25% 0.30M. A wider cap is better *if* elasticity is as assumed;
  ±15% leaves room for elasticity to be worse than assumed.
- **Risk:** the cap is applied per quote without rebalancing, so the book premium can drift from target. Monitor
  capped share and book premium monthly (`docs/monitoring_plan.md`).
- **Would change if:** observed lapse rates on capped increases are well below the assumed curve (widen to ±25%),
  or well above it (tighten to ±10%).

## R4. Price short-term cover with an explicit short-period scale and split large-loss loading

- **What:**
  - short-period factors by exposure band: 4.75 for a month or less, down to 1.48 for 9–12 months;
  - large-loss loading 11.6% for full-year cover and 48.2% for shorter terms.
- **Why:** charged pro rata at the full-year rate, short records would fund only 64% of the book's capped losses
  (`pure_premium.md` §1). Loading that cost onto full-year customers instead would overcharge them by about 18%.
- **Risk:** the factors partly reflect how this dataset records policies (split records, early terminations).
- **Action:** business review against the insurer's cancellation and short-term rules before use.

## R5. Keep LightGBM as a shadow challenger, not the price-setter

- **Why:**
  - Better ranking on the holdout (Gini +0.043; +0.070 on full-year business), but only 0.04M profit when capped
    against 0.26M for the GLM.
  - Policy premiums that move 20% between refits (GLM 7.5%).
  - Its gains concentrate in profiles almost never seen at full year.
- **Use:** quarterly double-lift review of where it beats the GLM to find new GLM factors or interactions (as with
  AgeMalus).
- **Would change if:** a stabilised LightGBM (monotone constraints, bagged refits, a premium-change penalty) wins a
  champion–challenger test on full-year renewals, including stability.

## R6. Run a randomised price test before full rollout

- **Why:** every behaviour figure above rests on assumed elasticities. There is no price or lapse data, so true
  uplift and elasticity cannot be estimated from freMTPL2.
- **Design:** randomise small price changes (e.g. −5%, 0, +5%) across about 10% of renewals, stratified by risk
  decile, for one renewal cycle. Fit the retention logit on the outcome, then rerun `fremtpl2 simulate` with the
  fitted values.
- **Would change:** the cap width (R3) and possibly the size of the gain (R1).

## R7. Hold capital and rate-level discipline for the tail

- **Evidence:** one-year profit varies by about 1.5M around the mean. The 99.5% VaR is about 40% of premium, driven
  by individual claims up to 4M on a book of about 70k policy-years (`strategy_simulation.md` §3).
- **Expenses:** if they run at 30% instead of the 25% priced in, every strategy loses money (filable capped
  −0.42M).
- **Action:** a rate-level review and reinsurance for large losses matter as much as the structure choice.

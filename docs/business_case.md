# Business case

## The decision

A French motor third-party-liability (MTPL) insurer must set the **rate charged per risk profile**. For each
policy the premium has to cover expected claims, expenses and a profit margin:

    technical premium = expected claim cost / (1 − expense ratio − profit margin)

where expected claim cost per policy year = claim frequency × average claim severity (plus a loading for
large losses that are capped in the modelling data).

**Who decides:** the pricing team proposes the rating structure (factors, relativities, base rate). The
pricing manager and the underwriting/product committee approve it, subject to regulatory and
fair-pricing constraints and commercial targets such as volume and loss ratio.

**What the model changes:** it supplies expected claim cost per profile, and therefore how much each
segment pays relative to the portfolio average.

## Baseline (what happens without a model)

1. **Flat rate:** every policy pays the portfolio-average premium. Simple, but it mis-prices nearly
   everyone.
2. **One-factor tariff:** premium varies by the bonus-malus coefficient only. This is close to how a
   no-claims-discount tariff works on its own.

Both are implemented as baselines in phase 2, and every model is compared against them.

## Cost of errors

| Error | What happens | Who pays |
|---|---|---|
| **Underpricing a segment** | Premiums do not cover claims, so the segment is loss-making. Worse, competitors that price it correctly push those customers to us (**adverse selection**), and the mispriced segment grows. | Insurer's underwriting result; eventually all customers, through rate increases |
| **Overpricing a segment** | Good risks are offered cheaper premiums elsewhere and leave (**lost profitable business**). The book shrinks towards the risks we under-price. | Lost margin and volume; fixed expenses spread over fewer policies |
| **Unjustified price differences** | Premium differences that cannot be explained by risk, or that proxy protected characteristics, create regulatory and reputational exposure. Examples: the EU gender directive, and fair-pricing expectations such as no penalising of loyal customers. | Regulatory sanctions, remediation, brand damage |
| **Unstable rates** | Large year-on-year premium swings for the same customer cause complaints and churn, even when the expected cost is right. | Retention, customer trust |

Phase 6 quantifies these trade-offs. Errors in either direction feed a retention model (assumed price
elasticity, clearly labelled) and a loss-ratio and profit simulation. Phase 6 also adds the expense
ratio, profit margin, elasticity and rate-cap assumptions to `configs/`, with sensitivity analysis.
Until then, no monetary figure is claimed.

## Data constraints that shape the case

- freMTPL2 holds one year of observations and has no policy dates, so true out-of-time validation is not
  possible. See `docs/methodology.md` §0.
- There is no price, competitor or lapse data. Retention effects can only be simulated under assumed
  elasticities, never estimated.
- There is no gender field. Proxy-discrimination risk (region and density vs socioeconomic status) is
  assessed in phase 5.

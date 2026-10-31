# Limitations

**Data**
- **One observation year, no policy or claim dates.**
  - No out-of-time validation is possible; the holdout is a random grouped split (D-007).
  - Claims inflation, trend, seasonality and reporting delay are invisible.
- **Missing claim amounts.** 27% of recorded claims (9,657) have no amount; almost all are in the first block of
  policy IDs. Claim counts are taken from the amount records (D-003). If those claims were real, the frequency level
  is understated, and the alternative definition moves relativities materially (phase 2 §6).
- **Split records.** 38% of rows look like fragments of one policy (identical profiles, near-consecutive IDs).
  Splits are grouped on this, but true policy identity is not in the data.
- **Short records.** Records with short exposure have much higher annual frequency and severity. The model
  separates this with an exposure-band control and a short-period scale, but the mechanism (split records, early
  terminations after claims, mid-year new business) cannot be identified.
- **No protected attributes, no external data.** Proxy discrimination can only be assessed through geography; the
  region codes could not be verified against an official list (D-021).
- **Flat-rate amounts.** 41% of claims sit on four flat amounts (likely inter-insurer settlements), so per-claim
  distributions are not Gamma.

**Models**
- **Severity is close to unpredictable** from the rating factors: 2.8% of CV deviance explained, and VehAge is
  the only factor kept. Pure-premium differentiation comes almost entirely from frequency.
- **The large-loss loading** comes from a few hundred capped claims and is volatile: 11.6% full-year, 48.2%
  part-year. The cap level matters: the loading is 21–47% for caps at the 99.9th–99th percentile.
- **Model uncertainty is not in the premium.** Standard errors are reported for relativities, but the technical
  premium has no parameter-uncertainty margin.
- **LightGBM was tuned lightly** (6 random configurations per objective) and without monotone constraints. A more
  heavily engineered tree model could narrow the stability gap.

**Simulation (phase 6)**
- **Behaviour is assumed, not estimated:** retention elasticities, the competitor price and the base retention
  rate (D-025).
- **Scope:** one renewal year, no new business, retention as an expectation, and Poisson claim counts with a single
  systemic shock. No reinsurance, investment income or capital cost.
- **Truth and holdout:** "true cost" is a calibrated blend of the two models (D-024); realised results use actual
  holdout losses and are noisy. The holdout informs the model choice twice, in phases 4 and 6, with no parameters
  tuned on it.

**Deployment**
- **The Docker image is large** (about 1.3 GB): it installs the full locked dependency set, including the
  training libraries. An API-only dependency group would slim it. The image was built and smoke-tested
  (`/health`, `/quote`, a 422 on invalid input, non-root user); the API is also covered by TestClient tests.
- **No authentication, rate limiting or request logging** in the API: not production-ready as is.
- **The ±15% cap is applied per quote** without rebalancing the book premium. Book-level drift must be monitored.

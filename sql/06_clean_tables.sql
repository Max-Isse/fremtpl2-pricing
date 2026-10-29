-- Modelling tables. Requires table `split(group_id, is_holdout, fold)` written from Python.

CREATE OR REPLACE TABLE policy_clean_stage AS
SELECT
    p.*,
    e.exposure,
    e.exposure_raw,
    e.exposure_capped_flag,
    r.claimnb_recorded,
    r.n_sev_records,
    r.recon_status,
    CASE getvariable('claim_count_source')
        WHEN 'recorded' THEN r.claimnb_recorded
        ELSE r.n_sev_records
    END                       AS claim_count_uncapped,
    g.group_id,
    s.is_holdout,
    s.fold
FROM policy p
JOIN exposure e        USING (IDpol)
JOIN reconciliation r  USING (IDpol)
JOIN profile_groups g  USING (IDpol)
JOIN split s           USING (group_id)
WHERE NOT e.exposure_nonpositive_flag;

-- Large-loss threshold is estimated on TRAINING claims only, so the holdout does not inform it.
CREATE OR REPLACE TABLE large_loss_threshold AS
SELECT quantile_cont(c.claim_amount, getvariable('large_loss_quantile')) AS threshold
FROM claims c
JOIN policy_clean_stage p USING (IDpol)
WHERE c.amount_positive AND NOT p.is_holdout;

-- Severity base: every positive claim amount on a retained policy, capped at the threshold.
CREATE OR REPLACE TABLE claims_clean AS
SELECT
    c.claim_id,
    c.IDpol,
    c.claim_amount                              AS claim_amount_raw,
    LEAST(c.claim_amount, t.threshold)          AS claim_amount,
    c.claim_amount > t.threshold                AS large_loss_flag,
    GREATEST(c.claim_amount - t.threshold, 0)   AS large_loss_excess,
    p.* EXCLUDE (IDpol)
FROM claims c
JOIN policy_clean_stage p USING (IDpol)
CROSS JOIN large_loss_threshold t
WHERE c.amount_positive;

-- Excess above the cap is spread back as a flat loading on capped losses (training claims only).
CREATE OR REPLACE TABLE large_loss_summary AS
SELECT
    any_value(t.threshold)                       AS threshold,
    count(*) FILTER (WHERE large_loss_flag)      AS n_large_train,
    count(*)                                     AS n_claims_train,
    sum(claim_amount_raw)                        AS amount_raw_train,
    sum(claim_amount)                            AS amount_capped_train,
    sum(large_loss_excess)                       AS excess_train,
    sum(large_loss_excess) / sum(claim_amount)   AS large_loss_loading
FROM claims_clean
CROSS JOIN large_loss_threshold t
WHERE NOT is_holdout;

-- Frequency / pure-premium base: one row per policy with capped count and capped claim cost.
CREATE OR REPLACE TABLE policy_clean AS
SELECT
    p.*,
    LEAST(p.claim_count_uncapped, getvariable('claim_count_cap'))   AS claim_count,
    p.claim_count_uncapped > getvariable('claim_count_cap')         AS claim_count_capped_flag,
    coalesce(a.claim_amount, 0.0)                                   AS claim_amount,
    coalesce(a.claim_amount_raw, 0.0)                               AS claim_amount_raw
FROM policy_clean_stage p
LEFT JOIN (
    SELECT IDpol, sum(claim_amount) AS claim_amount, sum(claim_amount_raw) AS claim_amount_raw
    FROM claims_clean
    GROUP BY IDpol
) a USING (IDpol);

DROP TABLE policy_clean_stage;

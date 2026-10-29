-- Claims table: one row per claim from freMTPL2sev, flagged for whether its policy exists.

CREATE OR REPLACE TABLE claims AS
SELECT
    s.claim_id,
    s.IDpol,
    s.ClaimAmount               AS claim_amount,
    f.IDpol IS NOT NULL         AS policy_found,
    s.ClaimAmount > 0           AS amount_positive
FROM raw_sev s
LEFT JOIN (SELECT DISTINCT IDpol FROM raw_freq) f USING (IDpol);

CREATE OR REPLACE TABLE claims_by_policy AS
SELECT
    IDpol,
    count(*)                                  AS n_sev_records,
    count(*) FILTER (WHERE amount_positive)   AS n_sev_positive,
    sum(claim_amount)                         AS sev_amount
FROM claims
WHERE policy_found
GROUP BY IDpol;

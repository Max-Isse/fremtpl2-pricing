-- Reconcile the recorded claim count (ClaimNb) with the number of positive claim-amount records.

CREATE OR REPLACE TABLE reconciliation AS
SELECT
    f.IDpol,
    f.ClaimNb                          AS claimnb_recorded,
    coalesce(c.n_sev_positive, 0)      AS n_sev_records,
    coalesce(c.sev_amount, 0.0)        AS sev_amount,
    CASE
        WHEN f.ClaimNb = coalesce(c.n_sev_positive, 0) THEN 'match'
        WHEN coalesce(c.n_sev_positive, 0) = 0         THEN 'claims_without_amounts'
        WHEN f.ClaimNb = 0                             THEN 'amounts_without_claims'
        WHEN f.ClaimNb > c.n_sev_positive              THEN 'fewer_amounts_than_claims'
        ELSE 'more_amounts_than_claims'
    END                                AS recon_status
FROM raw_freq f
LEFT JOIN claims_by_policy c USING (IDpol);

CREATE OR REPLACE TABLE reconciliation_summary AS
SELECT
    recon_status,
    count(*)                  AS n_policies,
    sum(claimnb_recorded)     AS claims_recorded,
    sum(n_sev_records)        AS claims_with_amount,
    sum(sev_amount)           AS amount
FROM reconciliation
GROUP BY recon_status
ORDER BY n_policies DESC;

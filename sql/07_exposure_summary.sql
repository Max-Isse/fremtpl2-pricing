-- Exposure-band summary on the TRAINING split.
-- Requires table `exposure_band_edges(lo, hi, label)` written from Python (bands from config).

CREATE OR REPLACE TABLE exposure_band_summary AS
SELECT
    b.label                                   AS exposure_band,
    count(*)                                  AS n_policies,
    sum(p.exposure)                           AS exposure_years,
    sum(p.claim_count)                        AS claims,
    sum(p.claim_count) / count(*)             AS claims_per_policy,
    sum(p.claim_count) / sum(p.exposure)      AS claims_per_year,
    avg(p.DrivAge)                            AS mean_driver_age,
    avg(p.BonusMalus)                         AS mean_bonus_malus,
    avg((g.n_rows > 1)::INT)                  AS share_in_duplicate_group
FROM policy_clean p
JOIN exposure_band_edges b ON p.exposure > b.lo AND p.exposure <= b.hi
JOIN profile_group_sizes g USING (group_id)
WHERE NOT p.is_holdout
GROUP BY b.label, b.lo
ORDER BY b.lo;

-- Policy/customer table (rating factors only) and exposure table, both keyed by IDpol.

CREATE OR REPLACE TABLE policy AS
SELECT IDpol, Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region
FROM raw_freq;

-- Exposure is capped at one policy year (the observation window). Non-positive values are
-- flagged rather than silently dropped, so the lineage shows how many rows they affect.
CREATE OR REPLACE TABLE exposure AS
SELECT
    IDpol,
    Exposure                                       AS exposure_raw,
    LEAST(Exposure, getvariable('exposure_cap'))   AS exposure,
    Exposure > getvariable('exposure_cap')         AS exposure_capped_flag,
    (Exposure IS NULL OR Exposure <= 0)            AS exposure_nonpositive_flag
FROM raw_freq;

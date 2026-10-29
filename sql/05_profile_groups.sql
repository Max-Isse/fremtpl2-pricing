-- Near-duplicate detection: policies with identical rating factors share a profile group.
-- The train/holdout split is drawn at group level so identical profiles never straddle it.

CREATE OR REPLACE TABLE profile_groups AS
SELECT
    IDpol,
    dense_rank() OVER (
        ORDER BY Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region
    ) AS group_id
FROM policy;

CREATE OR REPLACE TABLE profile_group_sizes AS
SELECT group_id, count(*) AS n_rows
FROM profile_groups
GROUP BY group_id;

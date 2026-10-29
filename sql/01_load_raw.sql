-- Raw layer: load the cached OpenML Parquet files unchanged.
-- claim_id preserves the source row order of freMTPL2sev (DuckDB preserves insertion order).
CREATE OR REPLACE TABLE raw_freq AS
SELECT * FROM read_parquet(getvariable('raw_freq_path'));

CREATE OR REPLACE TABLE raw_sev AS
SELECT row_number() OVER () AS claim_id, *
FROM read_parquet(getvariable('raw_sev_path'));

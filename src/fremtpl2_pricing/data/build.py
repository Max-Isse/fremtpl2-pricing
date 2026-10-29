"""Build the DuckDB database from the raw Parquet cache: raw -> policy/exposure/claims -> clean."""

from __future__ import annotations

import duckdb

from fremtpl2_pricing.config import Config
from fremtpl2_pricing.data.db import connect, query_df, run_sql_file, set_variables
from fremtpl2_pricing.data.fetch import RAW_FILES
from fremtpl2_pricing.data.split import assign_split


def build_database(cfg: Config, con: duckdb.DuckDBPyConnection | None = None) -> duckdb.DuckDBPyConnection:
    con = con or connect(cfg)
    cleaning = cfg.section("cleaning")
    if cleaning["claim_count_source"] not in {"severity_records", "recorded"}:
        raise ValueError(f"unknown claim_count_source {cleaning['claim_count_source']!r}")
    raw_dir = cfg.path("raw_dir")
    set_variables(
        con,
        {
            "raw_freq_path": (raw_dir / RAW_FILES["freq"]).as_posix(),
            "raw_sev_path": (raw_dir / RAW_FILES["sev"]).as_posix(),
            "exposure_cap": float(cleaning["exposure_cap"]),
            "claim_count_source": str(cleaning["claim_count_source"]),
            "claim_count_cap": int(cleaning["claim_count_cap"]),
            "large_loss_quantile": float(cleaning["large_loss_quantile"]),
        },
    )
    for name in (
        "01_load_raw.sql",
        "02_policy_exposure.sql",
        "03_claims.sql",
        "04_reconcile.sql",
        "05_profile_groups.sql",
    ):
        run_sql_file(con, name)

    split_cfg = cfg.section("split")
    groups = query_df(con, "SELECT DISTINCT group_id FROM profile_groups")["group_id"]
    split = assign_split(groups, split_cfg["holdout_fraction"], split_cfg["n_folds"], cfg.seed)
    con.register("split_df", split)
    con.execute("CREATE OR REPLACE TABLE split AS SELECT * FROM split_df")
    con.unregister("split_df")

    run_sql_file(con, "06_clean_tables.sql")
    return con

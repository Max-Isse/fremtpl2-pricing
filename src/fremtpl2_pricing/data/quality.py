"""Cleaning impact, reconciliation, duplicate-profile and lineage tables for the data-quality report.

Every number in reports/data_quality.md is produced here from the DuckDB tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import pandas as pd

from fremtpl2_pricing.config import Config
from fremtpl2_pricing.data.db import query_df, table_count


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> float:
    row = con.execute(sql).fetchone()
    assert row is not None
    return float(row[0]) if row[0] is not None else 0.0


def lineage(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = [
        ("raw_freq", "freMTPL2freq as downloaded", table_count(con, "raw_freq")),
        ("raw_sev", "freMTPL2sev as downloaded (one row per claim)", table_count(con, "raw_sev")),
        ("policy", "rating factors per IDpol", table_count(con, "policy")),
        ("exposure", "exposure per IDpol, capped and flagged", table_count(con, "exposure")),
        ("claims", "claims with policy-match and amount flags", table_count(con, "claims")),
        (
            "claims (policy found, amount > 0)",
            "claims usable for severity",
            table_count(con, "claims", "policy_found AND amount_positive"),
        ),
        ("reconciliation", "ClaimNb vs amount records per IDpol", table_count(con, "reconciliation")),
        (
            "profile_groups",
            "distinct rating-factor profiles",
            int(_scalar(con, "SELECT count(DISTINCT group_id) FROM profile_groups")),
        ),
        ("policy_clean", "frequency / pure-premium base", table_count(con, "policy_clean")),
        ("policy_clean (train)", "", table_count(con, "policy_clean", "NOT is_holdout")),
        ("policy_clean (holdout)", "", table_count(con, "policy_clean", "is_holdout")),
        ("claims_clean", "severity base (capped amounts)", table_count(con, "claims_clean")),
        ("claims_clean (train)", "", table_count(con, "claims_clean", "NOT is_holdout")),
        ("claims_clean (holdout)", "", table_count(con, "claims_clean", "is_holdout")),
    ]
    return pd.DataFrame(rows, columns=["table", "description", "rows"])


def cleaning_impact(con: duckdb.DuckDBPyConnection, cfg: Config) -> pd.DataFrame:
    """One row per cleaning decision with what it changes in policies, exposure, claims and amounts."""
    c = cfg.section("cleaning")
    s = _scalar
    rows = [
        {
            "step": "Cap exposure at {:g} year".format(c["exposure_cap"]),
            "rows_affected": s(con, "SELECT sum(exposure_capped_flag::INT) FROM exposure"),
            "exposure_change": s(con, "SELECT sum(exposure) - sum(exposure_raw) FROM exposure"),
            "claims_change": 0.0,
            "amount_change": 0.0,
        },
        {
            "step": "Drop non-positive exposure",
            "rows_affected": s(con, "SELECT sum(exposure_nonpositive_flag::INT) FROM exposure"),
            "exposure_change": -s(
                con, "SELECT coalesce(sum(exposure), 0) FROM exposure WHERE exposure_nonpositive_flag"
            ),
            "claims_change": 0.0,
            "amount_change": 0.0,
        },
        {
            "step": "Drop claims whose IDpol is not in freMTPL2freq",
            "rows_affected": s(con, "SELECT count(*) FROM claims WHERE NOT policy_found"),
            "exposure_change": 0.0,
            "claims_change": 0.0,
            "amount_change": -s(con, "SELECT coalesce(sum(claim_amount), 0) FROM claims WHERE NOT policy_found"),
        },
        {
            "step": "Drop non-positive claim amounts",
            "rows_affected": s(con, "SELECT count(*) FROM claims WHERE policy_found AND NOT amount_positive"),
            "exposure_change": 0.0,
            "claims_change": 0.0,
            "amount_change": -s(
                con, "SELECT coalesce(sum(claim_amount), 0) FROM claims WHERE policy_found AND NOT amount_positive"
            ),
        },
        {
            "step": f"Claim count source = {c['claim_count_source']}",
            "rows_affected": s(con, "SELECT count(*) FROM policy_clean WHERE claim_count_uncapped <> claimnb_recorded"),
            "exposure_change": 0.0,
            "claims_change": s(con, "SELECT sum(claim_count_uncapped) - sum(claimnb_recorded) FROM policy_clean"),
            "amount_change": 0.0,
        },
        {
            "step": f"Cap claim count at {c['claim_count_cap']}",
            "rows_affected": s(con, "SELECT sum(claim_count_capped_flag::INT) FROM policy_clean"),
            "exposure_change": 0.0,
            "claims_change": s(con, "SELECT sum(claim_count) - sum(claim_count_uncapped) FROM policy_clean"),
            "amount_change": 0.0,
        },
        {
            "step": "Cap claim amounts at the q={:g} training quantile (excess -> loading)".format(
                c["large_loss_quantile"]
            ),
            "rows_affected": s(con, "SELECT sum(large_loss_flag::INT) FROM claims_clean"),
            "exposure_change": 0.0,
            "claims_change": 0.0,
            "amount_change": -s(con, "SELECT sum(large_loss_excess) FROM claims_clean"),
        },
    ]
    return pd.DataFrame(rows)


def totals(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    raw = {
        "stage": "raw",
        "policies": table_count(con, "raw_freq"),
        "exposure": _scalar(con, "SELECT sum(Exposure) FROM raw_freq"),
        "claims": _scalar(con, "SELECT sum(ClaimNb) FROM raw_freq"),
        "claim_amount": _scalar(con, "SELECT sum(ClaimAmount) FROM raw_sev"),
    }
    clean = {
        "stage": "clean",
        "policies": table_count(con, "policy_clean"),
        "exposure": _scalar(con, "SELECT sum(exposure) FROM policy_clean"),
        "claims": _scalar(con, "SELECT sum(claim_count) FROM policy_clean"),
        "claim_amount": _scalar(con, "SELECT sum(claim_amount) FROM claims_clean"),
    }
    return pd.DataFrame([raw, clean])


def reconciliation_by_idpol_decile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return query_df(
        con,
        """
        WITH r AS (SELECT *, ntile(10) OVER (ORDER BY IDpol) AS decile FROM reconciliation)
        SELECT decile, min(IDpol) AS idpol_min, max(IDpol) AS idpol_max, count(*) AS policies,
               sum(claimnb_recorded) AS claims_recorded, sum(n_sev_records) AS claims_with_amount,
               sum(claimnb_recorded) - sum(n_sev_records) AS claims_without_amount
        FROM r GROUP BY decile ORDER BY decile
    """,
    )


def duplicate_profiles(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    size_dist = query_df(
        con,
        """
        SELECT CASE WHEN n_rows >= 5 THEN '5+' ELSE n_rows::VARCHAR END AS group_size,
               count(*) AS groups, sum(n_rows) AS rows
        FROM profile_group_sizes GROUP BY 1 ORDER BY 1
    """,
    )
    # Evidence on what the duplicates are: consecutive IDpols and combined exposure within one year
    # point to one policy split into several records rather than distinct customers.
    nature = query_df(
        con,
        """
        WITH g AS (
            SELECT group_id, count(*) AS n, max(IDpol) - min(IDpol) AS idpol_span,
                   sum(exposure) AS exposure
            FROM policy_clean GROUP BY group_id HAVING count(*) > 1
        )
        SELECT count(*) AS duplicate_groups,
               median(idpol_span) AS median_idpol_span,
               avg((idpol_span < n * 10)::INT) AS share_near_consecutive_idpol,
               avg((exposure <= 1.0 + 1e-9)::INT) AS share_total_exposure_le_1
        FROM g
    """,
    )
    straddle = query_df(
        con,
        """
        SELECT count(*) AS groups_straddling_split FROM (
            SELECT group_id FROM policy_clean GROUP BY group_id
            HAVING count(DISTINCT is_holdout) > 1
        )
    """,
    )
    return {"size_distribution": size_dist, "nature": nature, "straddle": straddle}


def large_loss_sensitivity(con: duckdb.DuckDBPyConnection, quantiles: list[float]) -> pd.DataFrame:
    """Threshold, claims capped and implied loading for alternative cap quantiles (training claims)."""
    out = []
    for q in quantiles:
        df = query_df(
            con,
            """
            WITH t AS (SELECT quantile_cont(claim_amount_raw, ?) AS thr FROM claims_clean WHERE NOT is_holdout)
            SELECT ? AS quantile, any_value(t.thr) AS threshold,
                   sum((claim_amount_raw > t.thr)::INT) AS claims_capped,
                   sum(GREATEST(claim_amount_raw - t.thr, 0)) / sum(claim_amount_raw) AS excess_share_of_losses,
                   sum(GREATEST(claim_amount_raw - t.thr, 0)) / sum(LEAST(claim_amount_raw, t.thr)) AS loading
            FROM claims_clean CROSS JOIN t WHERE NOT is_holdout
        """,
            [q, q],
        )
        out.append(df)
    return pd.concat(out, ignore_index=True)


@dataclass
class QualityReport:
    lineage: pd.DataFrame
    impact: pd.DataFrame
    totals: pd.DataFrame
    reconciliation: pd.DataFrame
    reconciliation_decile: pd.DataFrame
    duplicates: dict[str, pd.DataFrame]
    large_loss: pd.DataFrame
    large_loss_sensitivity: pd.DataFrame
    split_summary: pd.DataFrame
    extras: dict[str, float] = field(default_factory=dict)


def build_quality_report(con: duckdb.DuckDBPyConnection, cfg: Config) -> QualityReport:
    q = float(cfg.section("cleaning")["large_loss_quantile"])
    return QualityReport(
        lineage=lineage(con),
        impact=cleaning_impact(con, cfg),
        totals=totals(con),
        reconciliation=query_df(con, "SELECT * FROM reconciliation_summary"),
        reconciliation_decile=reconciliation_by_idpol_decile(con),
        duplicates=duplicate_profiles(con),
        large_loss=query_df(con, "SELECT * FROM large_loss_summary"),
        large_loss_sensitivity=large_loss_sensitivity(con, sorted({0.99, q, 0.999})),
        split_summary=query_df(
            con,
            """
            SELECT CASE WHEN is_holdout THEN 'holdout' ELSE 'fold ' || fold END AS part,
                   count(DISTINCT group_id) AS groups, count(*) AS policies,
                   sum(exposure) AS exposure, sum(claim_count) AS claims,
                   sum(claim_count) / sum(exposure) AS frequency
            FROM policy_clean GROUP BY 1 ORDER BY 1
        """,
        ),
    )

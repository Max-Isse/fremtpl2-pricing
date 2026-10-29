"""SQL transform tests on the fixture database: each cleaning rule and totals reconciliation."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from fremtpl2_pricing.config import Config
from fremtpl2_pricing.data.build import build_database
from fremtpl2_pricing.data.fetch import normalise_freq
from fremtpl2_pricing.data.quality import build_quality_report
from tests.conftest import fixture_freq, fixture_sev, make_cfg


def one(con: duckdb.DuckDBPyConnection, sql: str) -> object:
    row = con.execute(sql).fetchone()
    assert row is not None
    return row[0]


def policy(con: duckdb.DuckDBPyConnection, idpol: int) -> pd.Series:
    df = con.execute("SELECT * FROM policy_clean WHERE IDpol = ?", [idpol]).df()
    assert len(df) == 1
    return df.iloc[0]


def test_normalise_strips_arff_quotes() -> None:
    raw = fixture_freq().assign(VehGas="'Regular'", IDpol=lambda d: d["IDpol"].astype(float))
    out = normalise_freq(raw)
    assert set(out["VehGas"]) == {"Regular"}
    assert out["IDpol"].dtype == "int64"


def test_exposure_capped_and_nonpositive_excluded(con: duckdb.DuckDBPyConnection) -> None:
    assert one(con, "SELECT max(exposure) FROM policy_clean") == 1.0
    assert policy(con, 2)["exposure_raw"] == 1.5
    assert bool(policy(con, 2)["exposure_capped_flag"])
    assert one(con, "SELECT count(*) FROM policy_clean WHERE IDpol = 3") == 0
    assert one(con, "SELECT count(*) FROM policy_clean WHERE exposure <= 0") == 0


def test_reconciliation_statuses(con: duckdb.DuckDBPyConnection) -> None:
    status = dict(con.execute("SELECT IDpol, recon_status FROM reconciliation").fetchall())
    assert status[1] == "match"
    assert status[4] == "claims_without_amounts"
    assert status[8] == "fewer_amounts_than_claims"
    assert status[5] == "match"
    assert one(con, "SELECT count(*) FROM claims WHERE NOT policy_found") == 1  # IDpol 999


def test_claim_count_from_severity_records_and_capped(con: duckdb.DuckDBPyConnection) -> None:
    assert policy(con, 4)["claim_count"] == 0  # no amounts -> no counted claims
    assert policy(con, 8)["claim_count"] == 1
    p5 = policy(con, 5)
    assert p5["claim_count_uncapped"] == 6 and p5["claim_count"] == 4
    assert bool(p5["claim_count_capped_flag"])


def test_claim_count_source_recorded(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, claim_count_source="recorded")
    con = build_database(cfg, duckdb.connect())
    assert one(con, "SELECT claim_count FROM policy_clean WHERE IDpol = 4") == 2
    assert one(con, "SELECT claim_count FROM policy_clean WHERE IDpol = 8") == 2


def test_large_loss_cap_conserves_amounts(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    df = con.execute("SELECT * FROM claims_clean").df()
    assert np.allclose(df["claim_amount"] + df["large_loss_excess"], df["claim_amount_raw"])
    sev = fixture_sev()
    usable = sev[sev["IDpol"].isin(fixture_freq().query("Exposure > 0")["IDpol"]) & (sev["ClaimAmount"] > 0)]
    assert np.isclose(df["claim_amount_raw"].sum(), usable["ClaimAmount"].sum())
    # policy-level amounts add up to claim-level amounts
    assert np.isclose(one(con, "SELECT sum(claim_amount) FROM policy_clean"), df["claim_amount"].sum())


def test_large_loss_threshold_uses_training_claims_only(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    q = cfg.section("cleaning")["large_loss_quantile"]
    train = con.execute("SELECT claim_amount_raw FROM claims_clean WHERE NOT is_holdout").df()
    expected = np.quantile(train["claim_amount_raw"], q)  # linear interpolation == quantile_cont
    assert np.isclose(one(con, "SELECT threshold FROM large_loss_threshold"), expected)
    loading = one(con, "SELECT large_loss_loading FROM large_loss_summary")
    t = con.execute(
        "SELECT sum(large_loss_excess), sum(claim_amount) FROM claims_clean WHERE NOT is_holdout"
    ).fetchone()
    assert t is not None and np.isclose(loading, t[0] / t[1])


def test_quality_report_lineage_and_impact(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    rep = build_quality_report(con, cfg)
    rows = dict(zip(rep.lineage["table"], rep.lineage["rows"], strict=True))
    assert rows["raw_freq"] == len(fixture_freq())
    assert rows["policy_clean"] == len(fixture_freq()) - 1
    assert rows["policy_clean (train)"] + rows["policy_clean (holdout)"] == rows["policy_clean"]
    impact = rep.impact.set_index("step")
    assert impact.filter(like="non-positive exposure", axis=0)["rows_affected"].iloc[0] == 1
    assert impact.filter(like="not in freMTPL2freq", axis=0)["rows_affected"].iloc[0] == 1
    # recorded claims + count-source change + count cap = claims in policy_clean (excluded policy has 0)
    recorded = fixture_freq().query("Exposure > 0")["ClaimNb"].sum()
    changes = impact["claims_change"].sum()
    assert recorded + changes == one(con, "SELECT sum(claim_count) FROM policy_clean")

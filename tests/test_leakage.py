"""Leakage checks: identical rating profiles never straddle the holdout or two CV folds.

The fixture tests always run. The same checks run against the full database when it has been built
locally (`fremtpl2 build`); in CI, where only fixtures exist, those are skipped.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from fremtpl2_pricing.config import load_config
from fremtpl2_pricing.data.split import HOLDOUT_FOLD, assign_split

FULL_DB = load_config().path("duckdb")


def assert_no_group_leakage(con: duckdb.DuckDBPyConnection) -> None:
    straddle = con.execute("""
        SELECT count(*) FROM (SELECT group_id FROM policy_clean GROUP BY group_id
                              HAVING count(DISTINCT is_holdout) > 1 OR count(DISTINCT fold) > 1)
    """).fetchone()
    assert straddle is not None and straddle[0] == 0
    # identical rating factors <=> same group
    mismatched = con.execute("""
        SELECT count(*) FROM (
            SELECT Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region
            FROM policy_clean
            GROUP BY ALL HAVING count(DISTINCT group_id) > 1)
    """).fetchone()
    assert mismatched is not None and mismatched[0] == 0
    # holdout rows carry the holdout fold marker and nothing else
    bad_fold = con.execute(f"SELECT count(*) FROM policy_clean WHERE is_holdout <> (fold = {HOLDOUT_FOLD})").fetchone()
    assert bad_fold is not None and bad_fold[0] == 0
    # the severity base inherits the policy's split
    sev_mismatch = con.execute("""
        SELECT count(*) FROM claims_clean c JOIN policy_clean p USING (IDpol)
        WHERE c.is_holdout <> p.is_holdout OR c.fold <> p.fold
    """).fetchone()
    assert sev_mismatch is not None and sev_mismatch[0] == 0


def test_assign_split_is_deterministic_and_complete() -> None:
    ids = np.arange(1000)
    a = assign_split(ids, 0.2, 5, seed=7)
    b = assign_split(ids, 0.2, 5, seed=7)
    assert a.equals(b)
    assert sorted(a["group_id"]) == list(ids)
    assert a["is_holdout"].sum() == 200
    train_folds = a.loc[~a["is_holdout"], "fold"].value_counts()
    assert set(train_folds.index) == set(range(5)) and train_folds.max() - train_folds.min() <= 1
    assert not assign_split(ids, 0.2, 5, seed=8).equals(a)


def test_assign_split_validates_arguments() -> None:
    with pytest.raises(ValueError):
        assign_split(np.arange(10), 1.2, 5, seed=1)
    with pytest.raises(ValueError):
        assign_split(np.arange(10), 0.2, 1, seed=1)


def test_fixture_duplicates_share_group(con: duckdb.DuckDBPyConnection) -> None:
    groups = dict(con.execute("SELECT IDpol, group_id FROM policy_clean WHERE IDpol IN (6, 7)").fetchall())
    assert groups[6] == groups[7]
    assert_no_group_leakage(con)


@pytest.mark.skipif(not FULL_DB.exists(), reason="full database not built (run `fremtpl2 build`)")
def test_full_data_has_no_group_leakage() -> None:
    con = duckdb.connect(str(FULL_DB), read_only=True)
    try:
        assert_no_group_leakage(con)
    finally:
        con.close()

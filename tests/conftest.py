"""Small hand-built fixtures that exercise every cleaning rule. Not used for any reported result."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from fremtpl2_pricing.config import Config, load_config, with_overrides
from fremtpl2_pricing.data.build import build_database
from fremtpl2_pricing.data.fetch import RAW_FILES

BASE = {
    "Area": "C",
    "VehPower": 6,
    "VehAge": 5,
    "DrivAge": 40,
    "BonusMalus": 50,
    "VehBrand": "B12",
    "VehGas": "Regular",
    "Density": 500,
    "Region": "R24",
}


def _policy(idpol: int, claimnb: int, exposure: float, **factors: object) -> dict[str, object]:
    return {"IDpol": idpol, "ClaimNb": claimnb, "Exposure": exposure, **BASE, **factors}


def fixture_freq() -> pd.DataFrame:
    rows = [
        _policy(1, 1, 0.5),  # match: 1 claim, 1 amount
        _policy(2, 0, 1.5, DrivAge=22),  # exposure capped at 1
        _policy(3, 0, 0.0, DrivAge=23),  # non-positive exposure -> excluded
        _policy(4, 2, 0.8, DrivAge=24),  # claims without amounts
        _policy(5, 6, 1.0, DrivAge=25),  # 6 claims, 6 amounts -> capped at 4
        _policy(6, 0, 0.3, DrivAge=60),  # duplicate profile of 7 (split record)
        _policy(7, 1, 0.7, DrivAge=60),  # duplicate profile of 6
        _policy(8, 2, 1.0, DrivAge=30),  # 2 claims but only 1 amount
        _policy(9, 1, 1.0, DrivAge=31),  # one very large loss
    ]
    rows += [_policy(100 + i, 0, 1.0, DrivAge=70 + i, Region="R11") for i in range(16)]
    df = pd.DataFrame(rows)
    return df.astype({"IDpol": "int64", "ClaimNb": "int64", "Exposure": "float64"})


def fixture_sev() -> pd.DataFrame:
    rows = [(1, 1000.0)] + [(5, 500.0 + i) for i in range(6)] + [(7, 2000.0), (8, 800.0), (9, 1_000_000.0), (999, 50.0)]
    return pd.DataFrame(rows, columns=["IDpol", "ClaimAmount"]).astype({"IDpol": "int64"})


def make_cfg(tmp_path: Path, **cleaning: object) -> Config:
    cfg = with_overrides(load_config(), root=tmp_path)
    if cleaning:
        raw = {**cfg.raw, "cleaning": {**cfg.raw["cleaning"], **cleaning}}
        cfg = Config(raw=raw, root=tmp_path)
    raw_dir = cfg.path("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    fixture_freq().to_parquet(raw_dir / RAW_FILES["freq"], index=False)
    fixture_sev().to_parquet(raw_dir / RAW_FILES["sev"], index=False)
    return cfg


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return make_cfg(tmp_path)


@pytest.fixture
def con(cfg: Config) -> Iterator[duckdb.DuckDBPyConnection]:
    c = duckdb.connect()
    build_database(cfg, c)
    yield c
    c.close()

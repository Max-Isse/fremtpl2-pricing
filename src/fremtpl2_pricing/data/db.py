"""Thin helpers for running the SQL files in sql/ against a DuckDB database."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from fremtpl2_pricing.config import SQL_DIR, Config


def connect(cfg: Config, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    db_path = cfg.path("duckdb")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path), read_only=read_only)


def set_variables(con: duckdb.DuckDBPyConnection, variables: dict[str, Any]) -> None:
    """Expose config values to SQL as DuckDB variables, read there with getvariable('name')."""
    for name, value in variables.items():
        con.execute(f"SET VARIABLE {name} = ?", [value])


def run_sql_file(con: duckdb.DuckDBPyConnection, name: str, sql_dir: Path = SQL_DIR) -> None:
    con.execute((sql_dir / name).read_text(encoding="utf-8"))


def query_df(con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    return con.execute(sql, params or []).df()


def table_count(con: duckdb.DuckDBPyConnection, table: str, where: str = "TRUE") -> int:
    row = con.execute(f"SELECT count(*) FROM {table} WHERE {where}").fetchone()
    assert row is not None
    return int(row[0])

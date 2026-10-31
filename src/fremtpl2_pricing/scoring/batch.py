"""Batch scoring: policy file (CSV or Parquet) -> validated -> premiums with model version.

Usage: `fremtpl2 score --input policies.csv --output quotes.csv` (rejected rows go to `<output stem>_rejected.csv`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fremtpl2_pricing.scoring.engine import RatingEngine, score_frame

log = logging.getLogger(__name__)


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input format {path.suffix!r}: use .csv or .parquet")


def write_any(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def score_file(engine: RatingEngine, input_path: Path, output_path: Path) -> tuple[int, int]:
    df = read_any(input_path)
    scored, rejected = score_frame(engine, df)
    write_any(scored, output_path)
    rej_path = output_path.with_name(f"{output_path.stem}_rejected{output_path.suffix}")
    write_any(rejected, rej_path)
    log.info(
        "scored %s rows (%s rejected) with %s -> %s", len(scored), len(rejected), engine.model_version, output_path
    )
    return len(scored), len(rejected)

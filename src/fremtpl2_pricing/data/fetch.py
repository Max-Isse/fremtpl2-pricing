"""Download freMTPL2freq / freMTPL2sev from OpenML and cache them as Parquet in data/raw/.

The raw cache is written once and never modified; all cleaning happens downstream in DuckDB.
A manifest (OpenML id/version, row counts, SHA-256 of each Parquet file) pins the data version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from fremtpl2_pricing.config import Config

RAW_FILES = {"freq": "freMTPL2freq.parquet", "sev": "freMTPL2sev.parquet"}


def _strip_quotes(s: pd.Series) -> pd.Series:
    # OpenML's ARFF export wraps some nominal values in single quotes ('Regular', 'Diesel').
    return s.astype(str).str.strip().str.strip("'")


def normalise_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Give the frequency table stable dtypes: integer IDpol, plain-string categoricals."""
    out = df.copy()
    out["IDpol"] = out["IDpol"].astype("int64")
    for col in ("Area", "VehBrand", "VehGas", "Region"):
        out[col] = _strip_quotes(out[col])
    for col in ("ClaimNb", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"):
        out[col] = out[col].astype("int64")
    out["Exposure"] = out["Exposure"].astype("float64")
    return out


def normalise_sev(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["IDpol"] = out["IDpol"].astype("int64")
    out["ClaimAmount"] = out["ClaimAmount"].astype("float64")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_raw(cfg: Config, force: bool = False) -> dict[str, Path]:
    raw_dir = cfg.path("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    ids = cfg.section("openml")
    targets = {"freq": (ids["freq_id"], normalise_freq), "sev": (ids["sev_id"], normalise_sev)}
    manifest: dict[str, dict[str, object]] = {}
    paths: dict[str, Path] = {}
    for name, (data_id, normalise) in targets.items():
        path = raw_dir / RAW_FILES[name]
        if force or not path.exists():
            bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
            df = normalise(bunch.frame)
            df.to_parquet(path, index=False)
            version = bunch.details.get("version")
        else:
            df = pd.read_parquet(path)
            version = _previous_version(cfg, name)
        manifest[name] = {
            "openml_id": data_id,
            "openml_version": version,
            "file": f"{cfg.raw['paths']['raw_dir']}/{RAW_FILES[name]}",
            "rows": len(df),
            "columns": list(df.columns),
            "sha256": sha256(path),
        }
        paths[name] = path
    cfg.path("manifest").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths


def _previous_version(cfg: Config, name: str) -> object:
    mpath = cfg.path("manifest")
    if mpath.exists():
        return json.loads(mpath.read_text(encoding="utf-8")).get(name, {}).get("openml_version")
    return None

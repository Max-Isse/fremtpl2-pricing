"""Load the YAML config and resolve paths relative to the project root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
SQL_DIR = PROJECT_ROOT / "sql"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.root / p

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw[name])

    @property
    def rating_factors(self) -> list[str]:
        rf = self.raw["rating_factors"]
        return [*rf["categorical"], *rf["numeric"]]


def load_config(path: Path | str | None = None, root: Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, root=root or PROJECT_ROOT)


def with_overrides(cfg: Config, root: Path, **paths: str) -> Config:
    """Copy of ``cfg`` with a new root and optionally replaced path entries (used by tests)."""
    raw = {**cfg.raw, "paths": {**cfg.raw["paths"], **paths}}
    return Config(raw=raw, root=root)

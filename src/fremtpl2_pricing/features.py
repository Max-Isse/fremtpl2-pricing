"""GLM rating-factor design: banding, level order, reference levels and the design matrix.

The same `GLMDesign` is used for fitting, cross-validation and (later) batch scoring and the API, so
training and serving cannot bin a policy differently. It serialises to a plain dict for model artefacts.

Control factors (kind "control", e.g. ExposureBand) are fitted like any other factor but are not part of the
tariff: `matrix(df, pricing=True)` pins them at their reference level, so every policy is priced at the
full-year rate and charged pro rata for its time on cover (decisions.md D-012).
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from fremtpl2_pricing.config import Config

INTERCEPT = "Intercept"
DERIVED_SOURCES = {"LogDensity": "Density"}
DERIVED_BASE_LEVEL = "base"


def derived_levels(name: str, params: dict[str, Any]) -> list[str]:
    if name == "AgeMalus":
        return [DERIVED_BASE_LEVEL, *[f"{label} & malus" for label in params["age_labels"]]]
    raise KeyError(name)


def derive(df: pd.DataFrame, name: str, params: dict[str, Any]) -> pd.Series:
    """Derived categorical factors built from raw rating factors (same code for training and scoring).

    AgeMalus: driver-age band x (BonusMalus above the malus threshold). Only the age bands listed in
    `age_labels` get their own level; everything else (no malus, or a malus in the middle age band) is
    the base level, which keeps the factor identifiable alongside the BonusMalus main effect.
    """
    if name == "AgeMalus":
        cuts = params["age_cuts"]
        labels = params["all_age_labels"]
        band = pd.cut(df["DrivAge"], [*cuts, np.inf], right=False, labels=labels).astype(str)
        malus = df["BonusMalus"] > params["malus_threshold"]
        out = pd.Series(DERIVED_BASE_LEVEL, index=df.index)
        for label in params["age_labels"]:
            out[(band == label) & malus] = f"{label} & malus"
        return out
    raise KeyError(name)


def band_labels(name: str, edges: list[float]) -> list[str]:
    fmt = "{:g}"
    prefix = "ln " if name.startswith("Log") else ""
    labels = [f"{prefix}{fmt.format(lo)}-<{fmt.format(hi)}" for lo, hi in itertools.pairwise(edges)]
    labels.append(f"{prefix}{fmt.format(edges[-1])}+")
    return labels


def control_labels(edges: list[float]) -> list[str]:
    return [f"({lo:g}, {hi:g}]" for lo, hi in itertools.pairwise(edges)]


def source_values(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "LogDensity":
        return pd.Series(np.log(df["Density"].astype(float)), index=df.index)
    return df[name]


@dataclass
class FactorSpec:
    name: str
    kind: str  # "banded", "categorical", "derived" or "control"
    levels: list[str]
    reference: str
    edges: list[float] | None = None
    params: dict[str, Any] | None = None

    def bin(self, df: pd.DataFrame) -> pd.Series:
        if self.kind == "banded":
            assert self.edges is not None
            values = source_values(df, self.name)
            if (values < self.edges[0]).any():
                bad = values[values < self.edges[0]].iloc[0]
                raise ValueError(f"{self.name}={bad} is below the lowest band edge {self.edges[0]}")
            return pd.Series(
                pd.cut(values, [*self.edges, np.inf], right=False, labels=self.levels).astype(str), index=df.index
            )
        if self.kind == "derived":
            assert self.params is not None
            return derive(df, self.name, self.params)
        if self.kind == "control":
            assert self.edges is not None and self.params is not None
            values = df[self.params["source"]]
            out = pd.cut(values, self.edges, right=True, labels=self.levels).astype(str)
            if (out == "nan").any():
                bad = values[out == "nan"].iloc[0]
                raise ValueError(
                    f"{self.name}: {self.params['source']}={bad} outside ({self.edges[0]}, {self.edges[-1]}]"
                )
            return pd.Series(out, index=df.index)
        out = df[self.name].astype(str)
        unknown = set(out.unique()) - set(self.levels)
        if unknown:
            raise ValueError(f"unknown {self.name} level(s): {sorted(unknown)}")
        return out


@dataclass
class GLMDesign:
    factors: list[FactorSpec]

    @classmethod
    def fit(cls, df: pd.DataFrame, cfg: Config, factors: list[str] | None = None) -> GLMDesign:
        """Learn levels and reference levels (largest exposure) from training data."""
        gcfg = cfg.section("glm")
        controls = gcfg.get("controls", {})
        names = factors if factors is not None else [*gcfg["factors"], *controls]
        specs = []
        for name in names:
            if name in controls:
                c = controls[name]
                edges = [float(e) for e in c["edges"]]
                levels = control_labels(edges)
                if c["reference"] not in levels:
                    raise ValueError(f"{name}: reference {c['reference']!r} not in {levels}")
                specs.append(FactorSpec(name, "control", levels, c["reference"], edges, {"source": c["source"]}))
                continue
            if name in gcfg.get("derived", {}):
                params = dict(gcfg["derived"][name])
                levels = derived_levels(name, params)
                specs.append(FactorSpec(name, "derived", levels, DERIVED_BASE_LEVEL, params=params))
                continue
            if name in gcfg["bands"]:
                edges = [float(e) for e in gcfg["bands"][name]]
                levels = band_labels(name, edges)
                spec = FactorSpec(name, "banded", levels, levels[0], edges)
            else:
                levels = sorted(df[name].astype(str).unique())
                spec = FactorSpec(name, "categorical", levels, levels[0])
            binned = spec.bin(df)
            exposure = df["exposure"].groupby(binned).sum().reindex(spec.levels, fill_value=0.0)
            if (exposure == 0).any():
                empty = list(exposure[exposure == 0].index)
                raise ValueError(f"{name}: no training exposure in level(s) {empty}; adjust the bands")
            spec.reference = str(exposure.idxmax())
            specs.append(spec)
        return cls(specs)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.factors]

    @property
    def tariff_factors(self) -> list[FactorSpec]:
        return [f for f in self.factors if f.kind != "control"]

    @property
    def control_names(self) -> list[str]:
        return [f.name for f in self.factors if f.kind == "control"]

    def factor(self, name: str) -> FactorSpec:
        return next(f for f in self.factors if f.name == name)

    def binned(self, df: pd.DataFrame, pricing: bool = False) -> pd.DataFrame:
        """Level of every factor per row; with pricing=True controls are pinned at their reference level."""
        return pd.DataFrame(
            {
                f.name: (pd.Series(f.reference, index=df.index) if pricing and f.kind == "control" else f.bin(df))
                for f in self.factors
            },
            index=df.index,
        )

    def columns(self, factors: list[str] | None = None) -> list[str]:
        keep = factors if factors is not None else self.names
        cols = [INTERCEPT]
        for f in self.factors:
            if f.name in keep:
                cols += [f"{f.name}[{lvl}]" for lvl in f.levels if lvl != f.reference]
        return cols

    def matrix(self, df: pd.DataFrame, factors: list[str] | None = None, pricing: bool = False) -> pd.DataFrame:
        """Treatment-coded design matrix with an intercept; reference levels are dropped.

        pricing=True gives the matrix for quoting: control factors at their reference (full-year) level.
        """
        keep = factors if factors is not None else self.names
        binned = self.binned(df, pricing=pricing)
        parts: list[np.ndarray] = [np.ones((len(df), 1))]
        for f in self.factors:
            if f.name not in keep:
                continue
            others = [lvl for lvl in f.levels if lvl != f.reference]
            values = binned[f.name].to_numpy()
            parts.append(np.column_stack([values == lvl for lvl in others]).astype(float))
        return pd.DataFrame(np.hstack(parts), columns=self.columns(keep), index=df.index)

    def to_dict(self) -> dict[str, Any]:
        return {"factors": [asdict(f) for f in self.factors]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GLMDesign:
        return cls([FactorSpec(**f) for f in d["factors"]])

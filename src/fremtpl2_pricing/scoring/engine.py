"""Rating engine: one versioned tariff artefact, validation, and quotes (shared by batch scoring and the API).

Tariff (decisions D-012, D-014, D-017, D-023):
    annual frequency f   = exp(X_f beta)          GLM frequency, exposure-band control pinned at full year
    severity per claim s = exp(X_s gamma)         GLM severity, same
    annual risk premium  = f x s x (1 + LL_full_year)
    annual technical     = risk premium / (1 - expense ratio - profit margin)
    premium for a term t = annual technical x t x SP_band(t) x (1 + LL_group(t)) / (1 + LL_full_year)
    optional cap         = term premium clipped to current premium x (1 +- cap)
SP = short-period factor by exposure band, LL = large-loss loading (full-year vs part-year), both from training.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from fremtpl2_pricing import __version__
from fremtpl2_pricing.config import Config
from fremtpl2_pricing.features import GLMDesign
from fremtpl2_pricing.models import glm

TARIFF_FILE = "tariff.json"
REQUIRED = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]


def build_tariff(con: duckdb.DuckDBPyConnection, cfg: Config) -> dict[str, Any]:
    """Assemble the deployable tariff from the phase 2-3 artefacts and training-split duration terms."""
    from fremtpl2_pricing.models.pure_premium import PurePremiumModel, load_training
    from fremtpl2_pricing.pricing.simulation import large_loss_loadings, short_period_factors

    art = cfg.path("artifacts_dir")
    fa = json.loads((art / "frequency_glm.json").read_text(encoding="utf-8"))
    sa = json.loads((art / "severity_glm.json").read_text(encoding="utf-8"))
    ppm = PurePremiumModel.from_artifacts(art / "frequency_glm.json", art / "severity_glm.json", 0.0)
    band = ppm.freq_design.factor("ExposureBand")
    train = load_training(con)

    def cost(pricing: bool) -> np.ndarray:
        return glm.predict_log_linear(
            ppm.freq_design, ppm.freq_params, train, pricing=pricing
        ) * glm.predict_log_linear(ppm.sev_design, ppm.sev_params, train, pricing=pricing)

    sp = short_period_factors(band.bin(train).to_numpy(), cost(False), cost(True), train["exposure"].to_numpy())
    ll = large_loss_loadings(con, lambda d: band.bin(d).to_numpy(), band.reference)
    pc, sc = cfg.section("pricing"), cfg.section("scoring")
    body = {
        "frequency": {"design": fa["design"], "params": fa["params"]},
        "severity": {"design": sa["design"], "params": sa["params"]},
        "short_period_factors": {str(k): float(v) for k, v in sp.items()},
        "large_loss_loading": {"full_year": float(ll["full-year"]), "part_year": float(ll["part-year"])},
        "expense_ratio": float(pc["expense_ratio"]),
        "profit_margin": float(pc["profit_margin"]),
        "cap": float(pc["cap"]),
        "ranges": sc["ranges"],
        "exposure_min": float(sc["exposure_min"]),
        "top_factors": int(sc["top_factors"]),
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=float).encode()).hexdigest()[:12]
    return {
        "model_name": sc["model_name"],
        "model_version": f"{sc['model_name']}-{digest}",
        "package_version": __version__,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_sha256": fa.get("data_sha256", {}),
        **body,
    }


def save_tariff(tariff: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tariff, indent=2, default=float), encoding="utf-8")
    return path


class ValidationError(ValueError):
    pass


@dataclass
class RatingEngine:
    tariff: dict[str, Any]

    def __post_init__(self) -> None:
        self.freq_design = GLMDesign.from_dict(self.tariff["frequency"]["design"])
        self.sev_design = GLMDesign.from_dict(self.tariff["severity"]["design"])
        self.freq_params = pd.Series(self.tariff["frequency"]["params"], dtype=float)
        self.sev_params = pd.Series(self.tariff["severity"]["params"], dtype=float)
        self.band = self.freq_design.factor("ExposureBand")

    @classmethod
    def load(cls, artifacts_dir: Path) -> RatingEngine:
        return cls(json.loads((artifacts_dir / TARIFF_FILE).read_text(encoding="utf-8")))

    @property
    def model_version(self) -> str:
        return str(self.tariff["model_version"])

    def categories(self) -> dict[str, list[str]]:
        return {f.name: list(f.levels) for f in self.freq_design.factors if f.kind == "categorical"}

    # ---------------------------------------------------------------- validation
    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split rows into valid and rejected (with a reason). Same rules for batch files and the API."""
        df = df.copy()
        if "exposure" not in df:
            df["exposure"] = 1.0
        reasons = pd.Series("", index=df.index, dtype=object)

        def flag(mask: pd.Series, msg: str) -> None:
            reasons[mask & (reasons == "")] = msg

        missing = [c for c in REQUIRED if c not in df]
        if missing:
            raise ValidationError(f"missing required column(s): {missing}")
        for c in REQUIRED:
            flag(df[c].isna(), f"{c} is missing")
        for c, (lo, hi) in self.tariff["ranges"].items():
            v = pd.to_numeric(df[c], errors="coerce")
            flag(v.isna() | (v < lo) | (v > hi), f"{c} outside [{lo}, {hi}]")
        e = pd.to_numeric(df["exposure"], errors="coerce")
        flag(e.isna() | (e < self.tariff["exposure_min"]) | (e > 1.0), "exposure outside (0, 1]")
        for c, levels in self.categories().items():
            flag(~df[c].astype(str).isin(levels), f"{c} not one of the rated levels")
        if "current_premium" in df:
            cp = pd.to_numeric(df["current_premium"], errors="coerce")
            flag(df["current_premium"].notna() & ~(cp > 0), "current_premium must be positive")
        ok = reasons == ""
        rejected = df[~ok].assign(reject_reason=reasons[~ok])
        valid = df[ok].copy()
        for c in ("VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"):
            valid[c] = pd.to_numeric(valid[c])
        valid["exposure"] = pd.to_numeric(valid["exposure"])
        return valid, rejected

    # ---------------------------------------------------------------- pricing
    def quote(self, df: pd.DataFrame) -> pd.DataFrame:
        """Quote validated policies. Returns one row per input row (same index)."""
        t = self.tariff
        freq = glm.predict_log_linear(self.freq_design, self.freq_params, df, pricing=True)
        sev = glm.predict_log_linear(self.sev_design, self.sev_params, df, pricing=True)
        ll_full, ll_part = t["large_loss_loading"]["full_year"], t["large_loss_loading"]["part_year"]
        risk = freq * sev * (1 + ll_full)
        annual = risk / (1 - t["expense_ratio"] - t["profit_margin"])
        band = self.band.bin(df)
        full = (band == self.band.reference).to_numpy()
        sp = band.map(t["short_period_factors"]).to_numpy(float)
        term = annual * df["exposure"].to_numpy() * sp * np.where(full, 1.0, (1 + ll_part) / (1 + ll_full))
        out = pd.DataFrame(
            {
                "frequency": freq,
                "severity": sev,
                "risk_premium_annual": risk,
                "technical_premium_annual": annual,
                "premium_for_term": term,
            },
            index=df.index,
        )
        out["capped"] = False
        if "current_premium" in df:
            cur = pd.to_numeric(df["current_premium"], errors="coerce").to_numpy(float)
            has = ~np.isnan(cur)
            capped = np.clip(term, cur * (1 - t["cap"]), cur * (1 + t["cap"]))
            out.loc[has, "premium_for_term"] = capped[has]
            out.loc[has, "capped"] = ~np.isclose(capped[has], term[has])
        out["model_version"] = self.model_version
        return out

    def top_factors(self, df: pd.DataFrame, k: int | None = None) -> list[list[dict[str, Any]]]:
        """Largest rating-factor effects per policy: relativity of the policy's level (frequency x severity),
        excluding controls and reference levels, sorted by |log relativity|."""
        k = k or int(self.tariff["top_factors"])
        bf, bs = self.freq_design.binned(df, pricing=True), self.sev_design.binned(df, pricing=True)
        out: list[list[dict[str, Any]]] = []
        for i in range(len(df)):
            effects: dict[str, tuple[str, float]] = {}
            for design, params, binned in (
                (self.freq_design, self.freq_params, bf),
                (self.sev_design, self.sev_params, bs),
            ):
                for f in design.tariff_factors:
                    lvl = str(binned[f.name].iloc[i])
                    b = float(params.get(f"{f.name}[{lvl}]", 0.0))
                    prev = effects.get(f.name, (lvl, 0.0))[1]
                    effects[f.name] = (lvl, prev + b)
            ranked = sorted(((n, lv, b) for n, (lv, b) in effects.items() if b != 0.0), key=lambda x: -abs(x[2]))
            out.append(
                [{"factor": n, "level": lv, "relativity": round(float(np.exp(b)), 4)} for n, lv, b in ranked[:k]]
            )
        return out

    def info(self) -> dict[str, Any]:
        t = self.tariff
        return {
            "model_name": t["model_name"],
            "model_version": t["model_version"],
            "package_version": t["package_version"],
            "built_at": t["built_at"],
            "expense_ratio": t["expense_ratio"],
            "profit_margin": t["profit_margin"],
            "cap": t["cap"],
            "frequency_factors": [f.name for f in self.freq_design.tariff_factors],
            "severity_factors": [f.name for f in self.sev_design.tariff_factors],
            "categories": self.categories(),
            "ranges": t["ranges"],
            "data_sha256": t["data_sha256"],
        }


def score_frame(engine: RatingEngine, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batch scoring: validate, quote, attach top factors and a timestamp. Returns (scored, rejected)."""
    valid, rejected = engine.validate(df)
    if len(valid) == 0:
        return valid, rejected
    q = engine.quote(valid)
    tops = engine.top_factors(valid)
    q["top_factors"] = ["; ".join(f"{t['factor']}={t['level']} x{t['relativity']}" for t in row) for row in tops]
    q["scored_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return pd.concat([valid, q], axis=1), rejected

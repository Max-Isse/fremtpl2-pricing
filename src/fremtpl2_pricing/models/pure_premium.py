"""Phase 3b: pure premium = frequency x severity x (1 + large-loss loading), cross-checked with a Tweedie GLM.

Pure premium here is the expected annual claim cost of a full-year policy (frequency controls pinned at
full year, D-012), including the flat large-loss loading (D-005). Expenses, profit and the short-record
loading are added in phase 6. All comparisons are on the training split; the holdout is used in phase 4.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from fremtpl2_pricing.config import Config
from fremtpl2_pricing.data.db import query_df
from fremtpl2_pricing.features import GLMDesign
from fremtpl2_pricing.models import glm
from fremtpl2_pricing.models.frequency import POLICY_COLUMNS

log = logging.getLogger(__name__)


@dataclass
class PurePremiumModel:
    """Scores frequency, severity and pure premium from the saved GLM artefacts (training = serving code)."""

    freq_design: GLMDesign
    freq_params: dict[str, float]
    sev_design: GLMDesign
    sev_params: dict[str, float]
    large_loss_loading: float

    @classmethod
    def from_artifacts(cls, freq_path: Path, sev_path: Path, large_loss_loading: float) -> PurePremiumModel:
        f = json.loads(freq_path.read_text(encoding="utf-8"))
        s = json.loads(sev_path.read_text(encoding="utf-8"))
        return cls(
            GLMDesign.from_dict(f["design"]),
            f["params"],
            GLMDesign.from_dict(s["design"]),
            s["params"],
            large_loss_loading,
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annual frequency (full-year cover), expected severity per claim, and annual pure premium."""
        freq = glm.predict_log_linear(self.freq_design, self.freq_params, df, pricing=True)
        sev = glm.predict_log_linear(self.sev_design, self.sev_params, df, pricing=True)
        return pd.DataFrame(
            {"frequency": freq, "severity": sev, "pure_premium": freq * sev * (1 + self.large_loss_loading)},
            index=df.index,
        )


def large_loss_loading(con: duckdb.DuckDBPyConnection) -> float:
    return float(query_df(con, "SELECT large_loss_loading FROM large_loss_summary")["large_loss_loading"].iloc[0])


def load_training(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return query_df(
        con,
        f"SELECT {POLICY_COLUMNS}, claim_amount, claim_amount_raw "
        "FROM policy_clean WHERE NOT is_holdout ORDER BY IDpol",
    )


def fit_tweedie(X: pd.DataFrame, df: pd.DataFrame, power: float) -> Any:
    """Tweedie GLM on loss cost per exposure-year, var_weights = exposure (log link)."""
    y = (df["claim_amount"] / df["exposure"]).to_numpy()
    family = sm.families.Tweedie(var_power=power, link=sm.families.links.Log())
    return sm.GLM(y, X, family=family, var_weights=df["exposure"].to_numpy()).fit()


@dataclass
class PurePremiumResults:
    tweedie_power: float
    loading: float
    reconciliation: pd.DataFrame
    agreement: pd.DataFrame
    implied_relativities: pd.DataFrame
    tweedie_power_sensitivity: pd.DataFrame
    variance_decomposition: pd.DataFrame
    scatter: pd.DataFrame
    tweedie_params: pd.Series
    tweedie_design: GLMDesign
    extras: dict[str, Any] = field(default_factory=dict)


def implied_relativities(df: pd.DataFrame, design: GLMDesign, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per factor level: exposure-weighted mean predicted annual cost / book mean, and the same for observed
    loss cost. Compares models that use different factor sets or bandings on one scale."""
    binned = design.binned(df)
    e = df["exposure"].to_numpy()
    observed_rate = df["claim_amount"].sum() / e.sum()
    frames = []
    for f in design.tariff_factors:
        g = (
            pd.DataFrame(
                {
                    "level": binned[f.name],
                    "exposure": e,
                    "cost": df["claim_amount"],
                    **{m: p * e for m, p in preds.items()},
                }
            )
            .groupby("level")
            .sum()
            .reindex(f.levels)
        )
        out = pd.DataFrame({"factor": f.name, "level": f.levels, "exposure": g["exposure"].to_numpy()})
        out["observed"] = (g["cost"] / g["exposure"]).to_numpy() / observed_rate
        for m, p in preds.items():
            book = float((p * e).sum() / e.sum())
            out[m] = (g[m] / g["exposure"]).to_numpy() / book
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def run_pure_premium(con: duckdb.DuckDBPyConnection, cfg: Config) -> PurePremiumResults:
    art = cfg.path("artifacts_dir")
    freq_path, sev_path = art / "frequency_glm.json", art / "severity_glm.json"
    for path in (freq_path, sev_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `fremtpl2 frequency` and `fremtpl2 severity` first")
    loading = large_loss_loading(con)
    model = PurePremiumModel.from_artifacts(freq_path, sev_path, loading)
    sev_art = json.loads(sev_path.read_text(encoding="utf-8"))
    power = glm.tweedie_power_from_gamma_dispersion(float(sev_art["dispersion"]))

    df = load_training(con)
    e = df["exposure"].to_numpy()
    priced = model.predict(df)
    # "fitted" = both components at the record's own exposure band (model calibration);
    # "priced" = both pinned at full year (what the tariff charges, pro rata).
    freq_fitted = glm.predict_log_linear(model.freq_design, model.freq_params, df, pricing=False)
    sev_fitted = glm.predict_log_linear(model.sev_design, model.sev_params, df, pricing=False)
    cost_fs_priced = (priced["frequency"] * priced["severity"]).to_numpy()  # capped, per year
    cost_fs_fitted = freq_fitted * sev_fitted

    log.info("Tweedie GLM, power %.3f", power)
    design = GLMDesign.fit(df, cfg)
    X = design.matrix(df)
    tw = fit_tweedie(X, df, power)
    actual_capped, actual_raw = float(df["claim_amount"].sum()), float(df["claim_amount_raw"].sum())
    # A log-link Tweedie GLM with p != 1 does not reproduce total losses; apply the usual base-level
    # (intercept) correction so fitted total = actual total, and report the uncorrected A/E.
    tw_raw_ae = actual_capped / float((np.asarray(tw.fittedvalues) * e).sum())
    tw_params = tw.params.copy()
    tw_params["Intercept"] += np.log(tw_raw_ae)
    tw_priced = np.exp(design.matrix(df, pricing=True).to_numpy() @ tw_params.to_numpy())
    tw_fitted = np.asarray(tw.fittedvalues) * tw_raw_ae
    full = (design.factor("ExposureBand").bin(df) == design.factor("ExposureBand").reference).to_numpy()
    rows = []
    for label, fitted, pr in (("freq x sev", cost_fs_fitted, cost_fs_priced), ("tweedie", tw_fitted, tw_priced)):
        rows += [
            {
                "model": label,
                "basis": "fitted (record's own exposure band), capped losses",
                "expected": float((fitted * e).sum()),
                "actual": actual_capped,
            },
            {
                "model": label,
                "basis": "priced at full-year rate, pro rata, capped losses",
                "expected": float((pr * e).sum()),
                "actual": actual_capped,
            },
            {
                "model": label,
                "basis": "priced, full-year records only, capped losses",
                "expected": float((pr * e)[full].sum()),
                "actual": float(df["claim_amount"].to_numpy()[full].sum()),
            },
            {
                "model": label,
                "basis": "priced x (1 + large-loss loading), all losses incl. excess",
                "expected": float((pr * e).sum() * (1 + loading)),
                "actual": actual_raw,
            },
        ]
    recon = pd.DataFrame(rows)
    recon["ae"] = recon["actual"] / recon["expected"]

    raw_ratio = tw_priced / cost_fs_priced
    # shape-only comparison: both models rescaled to the same exposure-weighted book average
    shape_ratio = raw_ratio * np.average(cost_fs_priced, weights=e) / np.average(tw_priced, weights=e)
    agreement = pd.DataFrame(
        [
            {
                "comparison": label,
                "corr_log_premium": float(np.corrcoef(np.log(tw_priced), np.log(cost_fs_priced))[0, 1]),
                "ratio_p05": float(np.quantile(r, 0.05)),
                "ratio_median": float(np.median(r)),
                "ratio_p95": float(np.quantile(r, 0.95)),
                "share_policies_differ_gt_10pct": float(np.mean(np.abs(r - 1) > 0.10)),
                "share_policies_differ_gt_20pct": float(np.mean(np.abs(r - 1) > 0.20)),
            }
            for label, r in (
                ("tweedie / freq x sev (as priced)", raw_ratio),
                ("same, both rescaled to equal book mean", shape_ratio),
            )
        ]
    )

    implied = implied_relativities(df, design, {"freq_x_sev": cost_fs_priced, "tweedie": tw_priced})

    sens_rows = []
    for off in cfg.section("pure_premium")["tweedie_power_sensitivity"]:
        alt = fit_tweedie(X, df, power + float(off))
        alt_priced = np.exp(design.matrix(df, pricing=True).to_numpy() @ alt.params.to_numpy())
        # compare premium *shapes*: each model rescaled to the same book average
        r = (alt_priced / np.average(alt_priced, weights=e)) / (tw_priced / np.average(tw_priced, weights=e))
        sens_rows.append(
            {
                "power": power + float(off),
                "base_power": power,
                "max_abs_relativity_change": float(
                    np.max(np.abs(np.exp(alt.params.drop("Intercept") - tw.params.drop("Intercept")) - 1))
                ),
                "premium_ratio_p05": float(np.quantile(r, 0.05)),
                "premium_ratio_p95": float(np.quantile(r, 0.95)),
            }
        )

    lf, ls = np.log(priced["frequency"].to_numpy()), np.log(priced["severity"].to_numpy())
    var_pp = np.var(lf + ls)
    decomp = pd.DataFrame(
        [
            {
                "component": "log frequency",
                "variance": float(np.var(lf)),
                "share_of_log_premium_variance": float(np.var(lf) / var_pp),
            },
            {
                "component": "log severity",
                "variance": float(np.var(ls)),
                "share_of_log_premium_variance": float(np.var(ls) / var_pp),
            },
            {
                "component": "2 x covariance",
                "variance": float(2 * np.cov(lf, ls)[0, 1]),
                "share_of_log_premium_variance": float(2 * np.cov(lf, ls)[0, 1] / var_pp),
            },
        ]
    )

    rng = np.random.default_rng(cfg.seed + 3)
    idx = rng.choice(len(df), size=min(20_000, len(df)), replace=False)
    scatter = pd.DataFrame({"freq_x_sev": cost_fs_priced[idx], "tweedie": tw_priced[idx]})

    return PurePremiumResults(
        tweedie_power=power,
        loading=loading,
        reconciliation=recon,
        agreement=agreement,
        implied_relativities=implied,
        tweedie_power_sensitivity=pd.DataFrame(sens_rows),
        variance_decomposition=decomp,
        scatter=scatter,
        tweedie_params=tw_params,
        tweedie_design=design,
        extras={
            "mean_pure_premium": float(np.average(priced["pure_premium"], weights=e)),
            "n_train": len(df),
            "tweedie_uncorrected_ae": tw_raw_ae,
        },
    )


def save_artifact(res: PurePremiumResults, cfg: Config, path: Path) -> Path:
    art = {
        "model": "pure_premium = frequency_glm x severity_glm x (1 + large_loss_loading)",
        "frequency_artifact": "frequency_glm.json",
        "severity_artifact": "severity_glm.json",
        "large_loss_loading": res.loading,
        "tweedie_crosscheck": {
            "power": res.tweedie_power,
            "design": res.tweedie_design.to_dict(),
            "params": res.tweedie_params.to_dict(),
        },
        "training": res.extras,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(art, indent=2, default=float), encoding="utf-8")
    return path

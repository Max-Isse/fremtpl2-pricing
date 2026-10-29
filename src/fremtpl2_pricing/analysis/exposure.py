"""Exposure analysis: why claim counts need log(Exposure) as an offset.

Three Poisson GLMs with the same small factor set are compared, fitted on training folds 1..K-1
and evaluated on fold 0 (the holdout is never touched here):

* offset   - log(exposure) enters with coefficient fixed at 1 (claims proportional to time at risk);
* ignored  - exposure left out (the model predicts claims per record, whatever its duration);
* feature  - log(exposure) enters as a free regressor (the data choose how claims scale with time).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from fremtpl2_pricing.config import Config
from fremtpl2_pricing.data.db import query_df, run_sql_file

MODELS = ("offset", "ignored", "feature")
VALIDATION_FOLD = 0


def band_labels(edges: list[float]) -> list[str]:
    return [f"({lo:g}, {hi:g}]" for lo, hi in itertools.pairwise(edges)]


def add_bands(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ec = cfg.section("exposure_analysis")
    out = df.copy()
    for col, key in (("BonusMalus", "bonusmalus_bands"), ("DrivAge", "drivage_bands")):
        # Left-closed bands; the last band is open-ended so the data maximum is always included.
        lows = ec[key][:-1]
        labels = [f"{lo:03d}-{hi - 1:03d}" for lo, hi in itertools.pairwise(lows)]
        labels.append(f"{lows[-1]:03d}+")
        out[f"{col}_band"] = pd.cut(out[col], [*lows, np.inf], right=False, labels=labels).astype(str)
    edges = ec["exposure_bands"]
    out["exposure_band"] = pd.cut(out["exposure"], edges, labels=band_labels(edges), right=True).astype(str)
    out["log_exposure"] = np.log(out["exposure"])
    return out


def exposure_band_summary(con: duckdb.DuckDBPyConnection, cfg: Config) -> pd.DataFrame:
    edges = cfg.section("exposure_analysis")["exposure_bands"]
    bands = pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "label": band_labels(edges)})
    con.register("bands_df", bands)
    con.execute("CREATE OR REPLACE TABLE exposure_band_edges AS SELECT * FROM bands_df")
    con.unregister("bands_df")
    run_sql_file(con, "07_exposure_summary.sql")
    return query_df(con, "SELECT * FROM exposure_band_summary")


def load_training(con: duckdb.DuckDBPyConnection, cfg: Config) -> pd.DataFrame:
    df = query_df(
        con,
        """
        SELECT IDpol, fold, exposure, claim_count, BonusMalus, DrivAge, Area
        FROM policy_clean WHERE NOT is_holdout
    """,
    )
    return add_bands(df, cfg)


@dataclass
class ExposureResults:
    band_summary: pd.DataFrame
    ae_by_band: pd.DataFrame
    log_exposure_coef: pd.DataFrame
    relativities: pd.DataFrame
    annual_frequency: pd.DataFrame
    deviance: pd.DataFrame
    exposure_hist: pd.DataFrame


def fit_models(train: pd.DataFrame, factors: list[str]) -> dict[str, sm.GLM]:
    rhs = " + ".join(f"C({f})" for f in factors)
    poisson = sm.families.Poisson()
    return {
        "offset": smf.glm(f"claim_count ~ {rhs}", train, family=poisson, offset=np.log(train["exposure"])).fit(),
        "ignored": smf.glm(f"claim_count ~ {rhs}", train, family=poisson).fit(),
        "feature": smf.glm(f"claim_count ~ {rhs} + log_exposure", train, family=poisson).fit(),
    }


def predict(results: dict[str, sm.GLM], df: pd.DataFrame, annual: bool = False) -> dict[str, np.ndarray]:
    """Expected claims per record; with annual=True, expected claims for one full year on cover."""
    exposure = np.ones(len(df)) if annual else df["exposure"].to_numpy()
    frame = df.assign(log_exposure=np.log(exposure))
    return {
        "offset": results["offset"].predict(frame, offset=np.log(exposure)).to_numpy(),
        "ignored": results["ignored"].predict(frame).to_numpy(),
        "feature": results["feature"].predict(frame).to_numpy(),
    }


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.sum(term - (y - mu)))


def run_exposure_analysis(con: duckdb.DuckDBPyConnection, cfg: Config) -> ExposureResults:
    ec = cfg.section("exposure_analysis")
    factors = list(ec["demo_factors"])
    band_summary = exposure_band_summary(con, cfg)
    data = load_training(con, cfg)
    fit_df = data[data["fold"] != VALIDATION_FOLD]
    val = data[data["fold"] == VALIDATION_FOLD].copy()
    results = fit_models(fit_df, factors)

    preds = predict(results, val)
    for m in MODELS:
        val[f"pred_{m}"] = preds[m]
    labels = band_labels(ec["exposure_bands"])
    ae = (
        val.groupby("exposure_band")
        .agg(
            policies=("claim_count", "size"),
            exposure=("exposure", "sum"),
            actual=("claim_count", "sum"),
            **{f"expected_{m}": (f"pred_{m}", "sum") for m in MODELS},
        )
        .reindex(labels)
        .reset_index()
    )
    for m in MODELS:
        ae[f"ae_{m}"] = ae["actual"] / ae[f"expected_{m}"]
    total = ae[["policies", "exposure", "actual", *[f"expected_{m}" for m in MODELS]]].sum()
    total_row: dict[str, object] = {
        "exposure_band": "total",
        **{str(k): v for k, v in total.items()},
        **{f"ae_{m}": total["actual"] / total[f"expected_{m}"] for m in MODELS},
    }
    ae = pd.concat([ae, pd.DataFrame([total_row])], ignore_index=True)

    feat = results["feature"]
    ci = feat.conf_int().loc["log_exposure"]
    coef = pd.DataFrame(
        [{"coef": feat.params["log_exposure"], "ci_low": ci[0], "ci_high": ci[1], "se": feat.bse["log_exposure"]}]
    )

    rel_rows = []
    for m in ("offset", "ignored", "feature"):
        params = results[m].params
        for name, value in params.items():
            if name.startswith("C(DrivAge_band)"):
                level = name.split("[T.")[1].rstrip("]")
                rel_rows.append({"model": m, "DrivAge_band": level, "relativity": float(np.exp(value))})
    # Zero-padded labels sort numerically, matching patsy's default (first-level) reference.
    base_level = sorted(fit_df["DrivAge_band"].unique())[0]
    rel = pd.DataFrame(rel_rows)
    rel = pd.concat(
        [pd.DataFrame([{"model": m, "DrivAge_band": base_level, "relativity": 1.0} for m in MODELS]), rel],
        ignore_index=True,
    )
    exposure_by_age = (
        fit_df.groupby("DrivAge_band")
        .agg(policies=("exposure", "size"), mean_exposure=("exposure", "mean"), exposure_years=("exposure", "sum"))
        .reset_index()
    )
    rel = rel.merge(exposure_by_age, on="DrivAge_band", how="left")
    rel = rel.sort_values(["model", "DrivAge_band"]).reset_index(drop=True)

    annual = predict(results, val, annual=True)
    observed_rate = val["claim_count"].sum() / val["exposure"].sum()
    annual_freq = pd.DataFrame(
        [
            {"model": "observed (claims / exposure years)", "annual_frequency": observed_rate},
            *[{"model": m, "annual_frequency": float(annual[m].mean())} for m in MODELS],
        ]
    )

    y = val["claim_count"].to_numpy(dtype=float)
    null_mu = np.full(len(y), y.mean())
    deviance = pd.DataFrame(
        [
            {
                "model": m,
                "validation_deviance": poisson_deviance(y, preds[m]),
                "deviance_vs_intercept_only": poisson_deviance(y, preds[m]) - poisson_deviance(y, null_mu),
                "n_params": int(results[m].df_model) + 1,
            }
            for m in MODELS
        ]
    )

    edges = np.linspace(0, 1, 21)
    counts, _ = np.histogram(data["exposure"], bins=edges)
    hist = pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "policies": counts})
    return ExposureResults(band_summary, ae, coef, rel, annual_freq, deviance, hist)

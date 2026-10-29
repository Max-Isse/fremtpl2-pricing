"""Phase 2: claim-frequency GLM on the training split (the holdout is not touched).

Steps: Poisson GLM with log(exposure) offset -> overdispersion tests -> NB2 -> standard-error comparison
(model-based, quasi-Poisson, NB2, cluster-robust by profile group) -> factor tests -> relativities ->
residual checks -> grouped 5-fold CV against baselines -> sensitivity of relativities to data decisions.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from fremtpl2_pricing.config import Config
from fremtpl2_pricing.data.db import query_df
from fremtpl2_pricing.features import INTERCEPT, GLMDesign
from fremtpl2_pricing.models import glm

log = logging.getLogger(__name__)

POLICY_COLUMNS = """IDpol, group_id, fold, exposure, claim_count, claimnb_recorded,
    Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region"""


def load_training(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return query_df(con, f"SELECT {POLICY_COLUMNS} FROM policy_clean WHERE NOT is_holdout ORDER BY IDpol")


@dataclass
class FrequencyResults:
    design: GLMDesign
    params: pd.Series
    cov_cluster: pd.DataFrame
    base_frequency: float
    overdispersion: pd.DataFrame
    pearson_by_exposure: pd.DataFrame
    nb2_summary: pd.DataFrame
    se_comparison: pd.DataFrame
    factor_tests: pd.DataFrame
    relativities: pd.DataFrame
    nb2_relativity_diff: pd.DataFrame
    cv_metrics: pd.DataFrame
    cv_by_fold: pd.DataFrame
    oof_deciles: pd.DataFrame
    oof_segments: pd.DataFrame
    oof_unmodelled: pd.DataFrame
    full_year: pd.DataFrame
    cv_paired: pd.DataFrame
    sensitivity: pd.DataFrame
    residuals: pd.DataFrame
    extras: dict[str, Any] = field(default_factory=dict)


def _exposure_bands(df: pd.DataFrame, cfg: Config) -> pd.Series:
    edges = cfg.section("exposure_analysis")["exposure_bands"]
    labels = [f"({lo:g}, {hi:g}]" for lo, hi in itertools.pairwise(edges)]
    return pd.Series(pd.cut(df["exposure"], edges, labels=labels).astype(str), index=df.index)


def pearson_by_exposure(df: pd.DataFrame, mu: np.ndarray, cfg: Config) -> pd.DataFrame:
    y = df["claim_count"].to_numpy(float)
    contrib = (y - mu) ** 2 / mu
    out = (
        pd.DataFrame({"band": _exposure_bands(df, cfg), "pearson": contrib, "y": y, "mu": mu})
        .groupby("band", sort=False)
        .agg(policies=("y", "size"), claims=("y", "sum"), expected=("mu", "sum"), pearson_chi2=("pearson", "sum"))
        .reset_index()
    )
    out["pearson_per_policy"] = out["pearson_chi2"] / out["policies"]
    out["share_of_pearson_chi2"] = out["pearson_chi2"] / out["pearson_chi2"].sum()
    return out.sort_values("band", key=lambda s: s.str.extract(r"\(([\d.]+)")[0].astype(float), ignore_index=True)


def wald_tests_cluster(design: GLMDesign, res: Any) -> pd.DataFrame:
    rows = []
    names = list(res.params.index)
    for f in design.factors:
        cols = [c for c in names if c.startswith(f"{f.name}[")]
        R = np.zeros((len(cols), len(names)))
        for i, c in enumerate(cols):
            R[i, names.index(c)] = 1.0
        w = res.wald_test(R, scalar=True)
        rows.append(
            {
                "factor": f.name,
                "test": "cluster-robust Wald",
                "df": len(cols),
                "wald_chi2": float(w.statistic),
                "p_wald": float(w.pvalue),
            }
        )
    return pd.DataFrame(rows)


def cross_validate(
    df: pd.DataFrame, design_full: GLMDesign, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Grouped K-fold CV on the training split. Designs are re-learned inside each fold.

    Returns per-fold metrics, out-of-fold *fitted* expected claims (controls at each record's own level, used
    for deviance) and out-of-fold *pricing* expected claims (controls pinned at full year, i.e. what the
    tariff would charge pro rata). They differ only for models with control factors.
    """
    one = cfg.section("frequency")["baseline_one_factor"]
    tariff = [n for n in design_full.names if n not in design_full.control_names]
    models = ["portfolio_mean", f"one_factor_{one}", "glm_offset_only", "glm_poisson"]
    oof = pd.DataFrame(index=df.index, columns=models, dtype=float)
    oof_pricing = oof.copy()
    rows = []
    for k in sorted(df["fold"].unique()):
        fit_mask = (df["fold"] != k).to_numpy()
        tr, va = df[fit_mask], df[~fit_mask]
        y_tr, e_tr = tr["claim_count"].to_numpy(float), tr["exposure"].to_numpy()
        # portfolio average rate: claims / exposure
        oof.loc[va.index, "portfolio_mean"] = y_tr.sum() / e_tr.sum() * va["exposure"].to_numpy()
        log_e = np.log(va["exposure"].to_numpy())
        for col, factors in ((f"one_factor_{one}", [one]), ("glm_offset_only", tariff), ("glm_poisson", None)):
            d = GLMDesign.fit(tr, cfg, factors=factors)
            r = glm.fit_poisson(d.matrix(tr), y_tr, e_tr)
            oof.loc[va.index, col] = r.predict(d.matrix(va), offset=log_e)
            oof_pricing.loc[va.index, col] = r.predict(d.matrix(va, pricing=True), offset=log_e)
        oof_pricing.loc[va.index, "portfolio_mean"] = oof.loc[va.index, "portfolio_mean"]
        y_va = va["claim_count"].to_numpy(float)
        for col in oof.columns:
            mu = oof.loc[va.index, col].to_numpy(float)
            rows.append(
                {
                    "fold": int(k),
                    "model": col,
                    "policies": len(va),
                    "exposure": va["exposure"].sum(),
                    "mean_deviance": glm.poisson_deviance(y_va, mu) / len(va),
                    "ae": y_va.sum() / mu.sum(),
                }
            )
        log.info("CV fold %s done", k)
    return pd.DataFrame(rows), oof, oof_pricing


def full_year_calibration(df: pd.DataFrame, oof_pricing: pd.DataFrame, design: GLMDesign) -> pd.DataFrame:
    """A/E of what each tariff would charge (pro rata), on full-year records and on the whole book.

    Full-year cover is what is quoted, so A/E there is the headline calibration check (D-012).
    """
    ref = design.factor(design.control_names[0]).reference if design.control_names else None
    full = (design.factor(design.control_names[0]).bin(df) == ref).to_numpy() if ref else np.ones(len(df), bool)
    y = df["claim_count"].to_numpy(float)
    rows = []
    for m in oof_pricing.columns:
        mu = oof_pricing[m].to_numpy(float)
        for scope, mask in (("full-year records", full), ("all records", np.ones(len(df), bool))):
            a, ex = y[mask].sum(), mu[mask].sum()
            rows.append(
                {
                    "model": m,
                    "scope": scope,
                    "policies": int(mask.sum()),
                    "exposure": float(df["exposure"].to_numpy()[mask].sum()),
                    "actual": a,
                    "charged_expected": ex,
                    "ae": a / ex,
                    "ae_ci_low": stats.chi2.ppf(0.025, 2 * a) / 2 / ex,
                    "ae_ci_high": stats.chi2.ppf(0.975, 2 * (a + 1)) / 2 / ex,
                }
            )
    return pd.DataFrame(rows)


def summarise_cv(by_fold: pd.DataFrame, oof: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    y = df["claim_count"].to_numpy(float)
    null_dev = glm.poisson_deviance(y, oof["portfolio_mean"].to_numpy(float))
    rows = []
    for m in oof.columns:
        f = by_fold[by_fold["model"] == m]["mean_deviance"]
        dev = glm.poisson_deviance(y, oof[m].to_numpy(float))
        rows.append(
            {
                "model": m,
                "oof_mean_deviance": dev / len(y),
                "fold_sd": float(f.std(ddof=1)),
                "deviance_explained_vs_portfolio_mean": 1 - dev / null_dev,
                "oof_ae_total": y.sum() / oof[m].sum(),
            }
        )
    return pd.DataFrame(rows)


def oof_deciles(df: pd.DataFrame, oof: pd.DataFrame, model: str = "glm_poisson") -> pd.DataFrame:
    freq = oof[model] / df["exposure"]
    decile = pd.qcut(freq.rank(method="first"), 10, labels=False) + 1
    g = pd.DataFrame(
        {"decile": decile, "exposure": df["exposure"], "actual": df["claim_count"], "expected": oof[model]}
    ).groupby("decile")
    out = g.sum().reset_index()
    out["predicted_frequency"] = out["expected"] / out["exposure"]
    out["observed_frequency"] = out["actual"] / out["exposure"]
    out["ae"] = out["actual"] / out["expected"]
    # Poisson CI on A/E treating expected as fixed (exact, Garwood)
    out["ae_ci_low"] = stats.chi2.ppf(0.025, 2 * out["actual"]) / 2 / out["expected"]
    out["ae_ci_high"] = stats.chi2.ppf(0.975, 2 * (out["actual"] + 1)) / 2 / out["expected"]
    return out


def paired_cv(by_fold: pd.DataFrame, model: str = "glm_poisson") -> pd.DataFrame:
    """Per-fold deviance improvement of `model` over each other model (paired across the same folds)."""
    wide = by_fold.pivot(index="fold", columns="model", values="mean_deviance")
    rows = []
    for other in wide.columns.drop(model):
        diff = wide[other] - wide[model]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        rows.append(
            {
                "comparison": f"{other} minus {model}",
                "mean_deviance_reduction": diff.mean(),
                "fold_sd": diff.std(ddof=1),
                "folds_improved": int((diff > 0).sum()),
                "t_stat": diff.mean() / se,
            }
        )
    return pd.DataFrame(rows)


def oof_unmodelled_segments(df: pd.DataFrame, oof: pd.DataFrame, design: GLMDesign, cfg: Config) -> pd.DataFrame:
    """Out-of-fold A/E on segments that are NOT main effects of the model, where miscalibration can hide.

    (Pooled OOF A/E on a main-effect level is ~1 by construction, so it is not an informative check.)
    """
    binned = design.binned(df)
    bm = np.where(df["BonusMalus"] <= 50, "BM=50", "BM>50")
    segments = {
        "Area (not in model)": df["Area"].astype(str),
        "DrivAge x BonusMalus": binned["DrivAge"] + " | " + bm,
        "VehAge x VehGas": binned["VehAge"] + " | " + df["VehGas"].astype(str),
    }
    frames = []
    for name, seg in segments.items():
        g = (
            pd.DataFrame(
                {
                    "segment": seg,
                    "exposure": df["exposure"],
                    "actual": df["claim_count"],
                    "expected": oof["glm_poisson"],
                }
            )
            .groupby("segment")
            .sum()
            .reset_index()
        )
        g.insert(0, "dimension", name)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["ae"] = out["actual"] / out["expected"]
    out["ae_ci_low"] = stats.chi2.ppf(0.025, 2 * out["actual"]) / 2 / out["expected"]
    out["ae_ci_high"] = stats.chi2.ppf(0.975, 2 * (out["actual"] + 1)) / 2 / out["expected"]
    out["outside_95ci"] = (out["ae_ci_low"] > 1) | (out["ae_ci_high"] < 1)
    return out


def oof_segments(df: pd.DataFrame, oof: pd.DataFrame, design: GLMDesign) -> pd.DataFrame:
    binned = design.binned(df)
    frames = []
    for f in design.factors:
        g = (
            pd.DataFrame(
                {
                    "level": binned[f.name],
                    "exposure": df["exposure"],
                    "actual": df["claim_count"],
                    **{f"exp_{m}": oof[m] for m in oof.columns},
                }
            )
            .groupby("level")
            .sum()
        )
        g = g.reindex(f.levels).reset_index()
        g.insert(0, "factor", f.name)
        for m in oof.columns:
            g[f"ae_{m}"] = g["actual"] / g[f"exp_{m}"]
        frames.append(g.drop(columns=[f"exp_{m}" for m in oof.columns]))
    return pd.concat(frames, ignore_index=True)


def relativity_sensitivity(
    df: pd.DataFrame, design: GLMDesign, X: pd.DataFrame, base: pd.Series, cfg: Config
) -> pd.DataFrame:
    """How much do tariff relativities move under the two data decisions that touch frequency?"""
    e = df["exposure"].to_numpy()
    cap = int(cfg.section("cleaning")["claim_count_cap"])
    y_rec = np.minimum(df["claimnb_recorded"].to_numpy(float), cap)
    variants: dict[str, pd.Series] = {}
    variants["claim count = recorded ClaimNb (D-003)"] = glm.fit_poisson(X, y_rec, e).params
    tariff = [n for n in design.names if n not in design.control_names]
    if design.control_names:
        no_control = glm.fit_poisson(design.matrix(df, tariff), df["claim_count"].to_numpy(float), e)
        variants["exposure-band control removed (option A, D-012)"] = no_control.params
    rows = []
    for name, p in variants.items():
        for f in design.tariff_factors:
            cols = [c for c in base.index if c.startswith(f"{f.name}[")]
            diff = np.abs(np.exp(p[cols] - base[cols]) - 1)
            rows.append(
                {
                    "variant": name,
                    "factor": f.name,
                    "max_abs_relativity_change": float(diff.max()),
                    "mean_abs_relativity_change": float(diff.mean()),
                }
            )
        rows.append(
            {
                "variant": name,
                "factor": "base frequency (intercept)",
                "max_abs_relativity_change": float(abs(np.exp(p[INTERCEPT] - base[INTERCEPT]) - 1)),
                "mean_abs_relativity_change": float(abs(np.exp(p[INTERCEPT] - base[INTERCEPT]) - 1)),
            }
        )
    return pd.DataFrame(rows)


def run_frequency(con: duckdb.DuckDBPyConnection, cfg: Config) -> FrequencyResults:
    fcfg = cfg.section("frequency")
    level = float(cfg.section("glm")["ci_level"])
    df = load_training(con)
    design = GLMDesign.fit(df, cfg)
    X = design.matrix(df)
    y = df["claim_count"].to_numpy(float)
    e = df["exposure"].to_numpy()

    log.info("fitting Poisson GLM on %s rows x %s columns", *X.shape)
    pois = glm.fit_poisson(X, y, e)
    pois_cl = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(e)).fit(
        cov_type="cluster", cov_kwds={"groups": df["group_id"].to_numpy()}
    )
    od = glm.overdispersion_tests(pois, y)
    mu = np.asarray(pois.fittedvalues)

    log.info("fitting NB2")
    nb = glm.fit_nb2(X, y, e, start=pois.params.to_numpy())
    alpha = float(nb.params["alpha"])
    lr, p_lr = glm.nb2_vs_poisson_lr(float(nb.llf), float(pois.llf))
    nb2_summary = pd.DataFrame(
        [
            {
                "model": "Poisson",
                "log_likelihood": float(pois.llf),
                "aic": float(pois.aic),
                "alpha": 0.0,
                "n_params": int(pois.df_model) + 1,
            },
            {
                "model": "NB2",
                "log_likelihood": float(nb.llf),
                "aic": float(nb.aic),
                "alpha": alpha,
                "n_params": int(nb.df_model) + 2,
            },
        ]
    )
    nb2_summary["lr_vs_poisson"] = [np.nan, lr]
    nb2_summary["p_boundary_mixture"] = [np.nan, p_lr]

    nb_params = nb.params.drop("alpha")
    se = pd.DataFrame(
        {
            "poisson_model_based": pois.bse,
            "quasi_poisson": pois.bse * np.sqrt(od.pearson_dispersion),
            "nb2": nb.bse.drop("alpha"),
            "cluster_robust": pois_cl.bse,
        }
    )
    se_ratio = se.div(se["poisson_model_based"], axis=0).drop(index=INTERCEPT)
    se_comparison = se_ratio.describe(percentiles=[0.5]).loc[["mean", "min", "50%", "max"]].T.reset_index()
    se_comparison.columns = ["standard_error", "mean_ratio", "min_ratio", "median_ratio", "max_ratio"]

    nb2_diff = (
        pd.DataFrame(
            {
                "factor": [c.split("[")[0] for c in nb_params.index if c != INTERCEPT],
                "abs_relativity_diff": np.abs(np.exp(nb_params.drop(INTERCEPT) - pois.params.drop(INTERCEPT)) - 1),
            }
        )
        .groupby("factor", sort=False)["abs_relativity_diff"]
        .max()
        .reset_index(name="max_abs_relativity_diff")
    )

    log.info("factor tests")
    extra = {c: GLMDesign.fit(df, cfg, factors=[*design.names, c]) for c in cfg.section("glm")["candidate_factors"]}
    lr_tests = glm.factor_lr_tests(design, df, y, e, pois, od.pearson_dispersion, extra)
    tests = lr_tests.merge(
        wald_tests_cluster(design, pois_cl)[["factor", "wald_chi2", "p_wald"]], on="factor", how="left"
    )

    rel = glm.relativities(design, pois_cl.params, pois_cl.bse, df, level)

    rng = np.random.default_rng(cfg.seed + int(fcfg["residual_seed_offset"]))
    idx = rng.choice(len(df), size=min(int(fcfg["residual_sample"]), len(df)), replace=False)
    nb_mu = np.asarray(nb.predict())
    residuals = pd.DataFrame(
        {
            "poisson": glm.randomized_quantile_residuals(y[idx], mu[idx], rng),
            "nb2": glm.randomized_quantile_residuals(y[idx], nb_mu[idx], rng, alpha=alpha),
            "fitted_frequency": mu[idx] / e[idx],
        }
    )

    log.info("grouped cross-validation")
    by_fold, oof, oof_pricing = cross_validate(df, design, cfg)

    log.info("sensitivity refits")
    sens = relativity_sensitivity(df, design, X, pois.params, cfg)

    return FrequencyResults(
        design=design,
        params=pois.params,
        cov_cluster=pd.DataFrame(pois_cl.cov_params()),
        base_frequency=glm.base_frequency(pois.params),
        overdispersion=od.as_frame(),
        pearson_by_exposure=pearson_by_exposure(df, mu, cfg),
        nb2_summary=nb2_summary,
        se_comparison=se_comparison,
        factor_tests=tests,
        relativities=rel,
        nb2_relativity_diff=nb2_diff,
        cv_metrics=summarise_cv(by_fold, oof, df),
        cv_by_fold=by_fold,
        oof_deciles=oof_deciles(df, oof),
        oof_segments=oof_segments(df, oof, design),
        oof_unmodelled=oof_unmodelled_segments(df, oof, design, cfg),
        full_year=full_year_calibration(df, oof_pricing, design),
        cv_paired=paired_cv(by_fold),
        sensitivity=sens,
        residuals=residuals,
        extras={
            "n_train": len(df),
            "exposure_train": float(e.sum()),
            "claims_train": float(y.sum()),
            "nb2_alpha": alpha,
            "pearson_dispersion": od.pearson_dispersion,
            "controls": design.control_names,
        },
    )


def save_artifact(res: FrequencyResults, cfg: Config, path: Path) -> Path:
    """Model artefact for scoring: design, coefficients, cluster-robust covariance, data version.

    Score with `GLMDesign.from_dict(art["design"]).matrix(df, pricing=True)`.
    """
    manifest = json.loads(cfg.path("manifest").read_text(encoding="utf-8"))
    art = {
        "model": "frequency_glm_poisson",
        "link": "log",
        "offset": "log(exposure)",
        "pricing": "control factors pinned at their reference level (full year); premium pro rata to exposure",
        "design": res.design.to_dict(),
        "params": res.params.to_dict(),
        "cov_cluster": res.cov_cluster.to_numpy().tolist(),
        "base_frequency": res.base_frequency,
        "training": res.extras,
        "data_sha256": {k: v["sha256"] for k, v in manifest.items()},
        "config_cleaning": cfg.section("cleaning"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(art, indent=2, default=float), encoding="utf-8")
    return path

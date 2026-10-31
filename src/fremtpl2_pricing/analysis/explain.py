"""Phase 5a: explainability and stability of the frequency models (training split only).

* GLM relativities with cluster-robust CIs (from the phase 2 artefact).
* LightGBM TreeSHAP (LightGBM's exact `pred_contrib`), aggregated to the same levels as the GLM: for a factor
  level, relativity = exp(mean SHAP in the level - mean SHAP in the GLM reference level). SHAP values are
  additive on the log-rate scale, so this is directly comparable to a GLM log-relativity. Interactions are
  shared out between features, so a SHAP "relativity" is a marginal attribution, not a tariff cell.
* Stability: both models refitted on each of the 5 grouped CV training sets; spread of each level's relativity.
* Monotonicity: adjacent decreases for factors a filed tariff would rate monotonically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from fremtpl2_pricing.config import Config
from fremtpl2_pricing.features import GLMDesign
from fremtpl2_pricing.models import glm
from fremtpl2_pricing.models.benchmarks import LgbmTask, LoadedBenchmarks, fit_final
from fremtpl2_pricing.models.pure_premium import load_training

log = logging.getLogger(__name__)


def _art(cfg: Config, name: str) -> dict[str, Any]:
    return json.loads((cfg.path("artifacts_dir") / name).read_text(encoding="utf-8"))


def glm_relativities(design: GLMDesign, params: pd.Series, cov: np.ndarray | None, factors: list[str]) -> pd.DataFrame:
    z = stats.norm.ppf(0.975)
    se = pd.Series(np.sqrt(np.diag(cov)), index=params.index) if cov is not None else pd.Series(0.0, index=params.index)
    rows = []
    for f in design.factors:
        if f.name not in factors:
            continue
        for order, lvl in enumerate(f.levels):
            col = f"{f.name}[{lvl}]"
            b, s = float(params.get(col, 0.0)), float(se.get(col, 0.0))
            rows.append(
                {
                    "factor": f.name,
                    "level": lvl,
                    "order": order,
                    "relativity": np.exp(b),
                    "ci_low": np.exp(b - z * s),
                    "ci_high": np.exp(b + z * s),
                }
            )
    return pd.DataFrame(rows)


def shap_contributions(booster: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP on the raw (log) scale; last column = expected value."""
    contrib = np.asarray(booster.predict(X, pred_contrib=True))
    return pd.DataFrame(contrib, columns=[*X.columns, "expected_value"], index=X.index)


def shap_level_relativities(
    contrib: pd.DataFrame, binned: pd.DataFrame, design: GLMDesign, factors: list[str]
) -> pd.DataFrame:
    rows = []
    for f in design.factors:
        if f.name not in factors:
            continue
        mean = contrib[f.name].groupby(binned[f.name]).mean()
        ref = mean.get(f.reference, np.nan)
        for order, lvl in enumerate(f.levels):
            m = mean.get(lvl, np.nan)
            rows.append(
                {
                    "factor": f.name,
                    "level": lvl,
                    "order": order,
                    "relativity": float(np.exp(m - ref)),
                    "n_sample": int((binned[f.name] == lvl).sum()),
                }
            )
    return pd.DataFrame(rows)


def monotonicity(rel: pd.DataFrame, factors: list[str], label: str) -> pd.DataFrame:
    rows = []
    for f in factors:
        r = rel[rel["factor"] == f].sort_values("order")["relativity"].to_numpy()
        d = np.diff(r)
        rows.append(
            {
                "model": label,
                "factor": f,
                "adjacent_decreases": int((d < 0).sum()),
                "largest_decrease": float(-d.min()) if (d < 0).any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ExplainResults:
    comparison: pd.DataFrame  # GLM vs SHAP relativity per level, with fold ranges
    importance: pd.DataFrame  # spread of each factor's log-contribution, GLM terms vs SHAP
    stability: pd.DataFrame  # per factor summary of fold spreads
    monotone: pd.DataFrame
    premium_volatility: pd.DataFrame  # per-policy premium spread across refits
    extras: dict[str, Any]


def run_explain(con: duckdb.DuckDBPyConnection, cfg: Config) -> ExplainResults:
    ec = cfg.section("explain")
    factors = list(ec["factors"])
    fa = _art(cfg, "frequency_glm.json")
    design = GLMDesign.from_dict(fa["design"])
    params = pd.Series(fa["params"], dtype=float)
    glm_rel = glm_relativities(design, params, np.asarray(fa["cov_cluster"]), factors)

    train = load_training(con)
    rng = np.random.default_rng(cfg.seed + 30)
    sample = train.iloc[np.sort(rng.choice(len(train), size=min(int(ec["shap_sample"]), len(train)), replace=False))]
    bench = LoadedBenchmarks.load(cfg)
    Xs = bench.features.transform(sample, pricing=True)
    binned = design.binned(sample, pricing=True)
    booster = bench.boosters["lgbm_poisson_frequency"]
    contrib = shap_contributions(booster, Xs)
    shap_rel = shap_level_relativities(contrib, binned, design, factors)

    # GLM term contributions on the same sample (log scale), for an importance comparison
    X_glm = design.matrix(sample, pricing=True)
    terms = {
        f: X_glm[[c for c in X_glm.columns if c.startswith(f"{f}[")]].to_numpy()
        @ params[[c for c in X_glm.columns if c.startswith(f"{f}[")]].to_numpy()
        for f in factors
    }
    terms["AgeMalus (GLM interaction)"] = (
        X_glm.filter(like="AgeMalus[").to_numpy() @ params.filter(like="AgeMalus[").to_numpy()
    )
    importance = pd.DataFrame(
        [
            {
                "factor": f,
                "glm_sd_log_contribution": float(np.std(terms[f])),
                "lgbm_sd_shap": float(np.std(contrib[f])) if f in contrib else np.nan,
                "lgbm_mean_abs_shap": float(np.mean(np.abs(contrib[f]))) if f in contrib else np.nan,
            }
            for f in terms
        ]
    ).sort_values("glm_sd_log_contribution", ascending=False, ignore_index=True)

    # Fold refits
    meta = _art(cfg, "benchmarks.json")
    p_lgbm, rounds = meta["params"]["lgbm_poisson_frequency"], int(meta["rounds"]["lgbm_poisson_frequency"])
    X_all = bench.features.transform(train)
    e = train["exposure"].to_numpy()
    folds = train["fold"].to_numpy()
    glm_fold, shap_fold = [], []
    glm_prem, lgbm_prem = [], []  # priced annual frequency on the sample, per refit
    for k in sorted(np.unique(folds)):
        idx = np.where(folds != k)[0]
        tr = train.iloc[idx]
        d = GLMDesign.fit(tr, cfg)
        r = glm.fit_poisson(d.matrix(tr), tr["claim_count"].to_numpy(float), tr["exposure"].to_numpy())
        g = glm_relativities(d, r.params, None, factors).assign(fold=int(k))
        glm_fold.append(g)
        glm_prem.append(glm.predict_log_linear(d, r.params, sample, pricing=True))
        task = LgbmTask(
            "frequency",
            "poisson",
            X_all.iloc[idx],
            train["claim_count"].to_numpy(float)[idx],
            None,
            np.log(e[idx]),
            folds[idx],
            {},
        )
        b = fit_final(task, p_lgbm, rounds)
        shap_fold.append(
            shap_level_relativities(shap_contributions(b, Xs), binned, design, factors).assign(fold=int(k))
        )
        lgbm_prem.append(np.asarray(b.predict(Xs)))
        log.info("explain: fold %s refits done", k)

    def ranges(frames: list[pd.DataFrame], label: str) -> pd.DataFrame:
        a = pd.concat(frames)
        g = a.groupby(["factor", "level", "order"])["relativity"].agg(["min", "max"]).reset_index()
        return g.rename(columns={"min": f"{label}_fold_min", "max": f"{label}_fold_max"})

    comp = glm_rel.rename(columns={"relativity": "glm", "ci_low": "glm_ci_low", "ci_high": "glm_ci_high"}).merge(
        shap_rel.rename(columns={"relativity": "lgbm_shap"}), on=["factor", "level", "order"]
    )
    comp = comp.merge(ranges(glm_fold, "glm"), on=["factor", "level", "order"]).merge(
        ranges(shap_fold, "lgbm"), on=["factor", "level", "order"]
    )
    comp["glm_fold_spread"] = comp["glm_fold_max"] / comp["glm_fold_min"] - 1
    comp["lgbm_fold_spread"] = comp["lgbm_fold_max"] / comp["lgbm_fold_min"] - 1
    comp["lgbm_over_glm"] = comp["lgbm_shap"] / comp["glm"]
    comp["_f"] = comp["factor"].map({f: i for i, f in enumerate(factors)})
    comp = comp.sort_values(["_f", "order"], ignore_index=True).drop(columns="_f")

    stability = (
        comp.groupby("factor", sort=False)
        .agg(
            levels=("level", "size"),
            glm_median_spread=("glm_fold_spread", "median"),
            glm_max_spread=("glm_fold_spread", "max"),
            lgbm_median_spread=("lgbm_fold_spread", "median"),
            lgbm_max_spread=("lgbm_fold_spread", "max"),
            max_abs_log_gap_lgbm_vs_glm=("lgbm_over_glm", lambda s: float(np.max(np.abs(np.log(s))))),
        )
        .reset_index()
    )

    mono = pd.concat(
        [
            monotonicity(glm_rel, ec["monotone_increasing"], "glm (full training)"),
            monotonicity(shap_rel, ec["monotone_increasing"], "lgbm SHAP (full training)"),
            *[monotonicity(g, ec["monotone_increasing"], f"glm fold {int(g['fold'].iloc[0])}") for g in glm_fold],
            *[monotonicity(s, ec["monotone_increasing"], f"lgbm fold {int(s['fold'].iloc[0])}") for s in shap_fold],
        ],
        ignore_index=True,
    )

    # What a customer would feel: how much their own premium moves between refits on 80% subsamples
    vol_rows = []
    for label, preds in (("glm", glm_prem), ("lgbm", lgbm_prem)):
        a = np.vstack(preds)
        ratio = a.max(axis=0) / a.min(axis=0) - 1
        cv = a.std(axis=0, ddof=1) / a.mean(axis=0)
        vol_rows.append(
            {
                "model": label,
                "median_max_min_spread": float(np.median(ratio)),
                "p95_max_min_spread": float(np.quantile(ratio, 0.95)),
                "share_policies_spread_gt_10pct": float(np.mean(ratio > 0.10)),
                "median_cv": float(np.median(cv)),
            }
        )
    volatility = pd.DataFrame(vol_rows)

    return ExplainResults(
        comp,
        importance,
        stability,
        mono,
        volatility,
        {"shap_sample": len(sample), "lgbm_rounds": rounds, "factors": factors},
    )

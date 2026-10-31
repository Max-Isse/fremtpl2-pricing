"""Phase 5b: fairness review: proxy risk from geography (training split only).

freMTPL2 has no gender, income or other protected/socioeconomic fields, so direct discrimination tests are
impossible. Region and population density are the plausible proxies (urban density and region correlate with
income and other socioeconomic characteristics). Without verified external statistics (D-021) the review asks:
1. How much of each model's premium differentiation comes from geography?
2. How dispersed are premiums across regions and density bands, and does that follow claims experience?
3. What would removing geography cost in accuracy, and who would pay more or less (the cross-subsidy)?
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from fremtpl2_pricing.analysis.explain import shap_contributions
from fremtpl2_pricing.config import Config
from fremtpl2_pricing.evaluation import metrics as M
from fremtpl2_pricing.features import GLMDesign
from fremtpl2_pricing.models import glm
from fremtpl2_pricing.models.benchmarks import LoadedBenchmarks
from fremtpl2_pricing.models.pure_premium import PurePremiumModel, load_training

log = logging.getLogger(__name__)


@dataclass
class FairnessResults:
    geography_share: pd.DataFrame
    dispersion: pd.DataFrame
    removal_cv: pd.DataFrame
    cross_subsidy: pd.DataFrame
    extras: dict[str, Any]


def variance_shares(parts: dict[str, np.ndarray], total: np.ndarray) -> dict[str, float]:
    """Share of var(total) attributable to each part: cov(part, total) / var(total) (shares sum to 1)."""
    v = float(np.var(total))
    return {k: float(np.cov(p, total, bias=True)[0, 1] / v) for k, p in parts.items()}


def run_fairness(con: duckdb.DuckDBPyConnection, cfg: Config) -> FairnessResults:
    ec = cfg.section("explain")
    geo = list(ec["geography"])
    art = cfg.path("artifacts_dir")
    fa = json.loads((art / "frequency_glm.json").read_text(encoding="utf-8"))
    design = GLMDesign.from_dict(fa["design"])
    params = pd.Series(fa["params"], dtype=float)
    pa = json.loads((art / "pure_premium.json").read_text(encoding="utf-8"))
    ppm = PurePremiumModel.from_artifacts(
        art / "frequency_glm.json", art / "severity_glm.json", float(pa["large_loss_loading"])
    )
    bench = LoadedBenchmarks.load(cfg)
    train = load_training(con)
    e = train["exposure"].to_numpy()

    # 1. Geography's share of premium differentiation (log pure premium, priced, on a sample)
    rng = np.random.default_rng(cfg.seed + 31)
    s = train.iloc[np.sort(rng.choice(len(train), size=min(int(ec["shap_sample"]), len(train)), replace=False))]
    X_glm = design.matrix(s, pricing=True)
    glm_terms = {
        f.name: X_glm.filter(like=f"{f.name}[").to_numpy() @ params.filter(like=f"{f.name}[").to_numpy()
        for f in design.tariff_factors
    }
    X_sev = ppm.sev_design.matrix(s, pricing=True)
    sev_params = pd.Series(ppm.sev_params, dtype=float)
    for f in ppm.sev_design.tariff_factors:  # severity terms add to the same factor's share
        glm_terms[f.name] = glm_terms.get(f.name, 0.0) + (
            X_sev.filter(like=f"{f.name}[").to_numpy() @ sev_params.filter(like=f"{f.name}[").to_numpy()
        )
    log_glm = np.log(ppm.predict(s)["pure_premium"].to_numpy())
    Xt = bench.features.transform(s, pricing=True)
    cf = shap_contributions(bench.boosters["lgbm_poisson_frequency"], Xt)
    cs = shap_contributions(bench.boosters["lgbm_gamma_severity"], Xt)
    lgbm_parts = {c: (cf[c] + cs[c]).to_numpy() for c in Xt.columns if c != bench.features.control}
    log_lgbm = (cf.sum(axis=1) + cs.sum(axis=1)).to_numpy()
    sh_glm = variance_shares(glm_terms, log_glm)
    sh_lgbm = variance_shares(lgbm_parts, log_lgbm)
    rows = [
        {
            "factor": f,
            "glm_share_of_log_premium_variance": sh_glm.get(f, np.nan),
            "lgbm_share_of_log_premium_variance": sh_lgbm.get(f, np.nan),
        }
        for f in sorted(set(sh_glm) | set(sh_lgbm))
    ]
    share = pd.DataFrame(rows)
    share["geography"] = share["factor"].isin(geo)
    share = share.sort_values("glm_share_of_log_premium_variance", ascending=False, ignore_index=True)

    # 2. Dispersion of priced premium across geography (full training book) vs observed loss cost
    pp_glm = ppm.predict(train)["pure_premium"].to_numpy()
    b = bench.predict(train, pricing=True)
    pp_lgbm = b["lgbm_frequency"] * b["lgbm_severity"] * (1 + ppm.large_loss_loading)
    binned = design.binned(train, pricing=True)
    obs = train["claim_amount_raw"].to_numpy()
    disp = []
    for f in geo:
        g = (
            pd.DataFrame({"level": binned[f], "exposure": e, "glm": pp_glm * e, "lgbm": pp_lgbm * e, "obs": obs})
            .groupby("level")
            .sum()
            .reindex(design.factor(f).levels)
            .reset_index()
        )
        g.insert(0, "factor", f)
        for c, tot in (("glm", (pp_glm * e).sum()), ("lgbm", (pp_lgbm * e).sum()), ("obs", obs.sum())):
            g[f"{c}_index"] = (g[c] / g["exposure"]) / (tot / e.sum())
        disp.append(g[["factor", "level", "exposure", "glm_index", "lgbm_index", "obs_index"]])
    dispersion = pd.concat(disp, ignore_index=True)

    # 3. Accuracy cost of removing geography (frequency GLM, grouped CV) and the cross-subsidy it creates
    tariff = [f.name for f in design.tariff_factors]
    variants = {
        "full tariff": tariff,
        "without Region": [f for f in tariff if f != "Region"],
        "without Region and density": [f for f in tariff if f not in geo],
    }
    y = train["claim_count"].to_numpy(float)
    folds = train["fold"].to_numpy()
    cv_rows = []
    for label, fs in variants.items():
        devs, ginis = [], []
        for k in sorted(np.unique(folds)):
            tr, va = train[folds != k], train[folds == k]
            d = GLMDesign.fit(tr, cfg, factors=[*fs, *design.control_names])
            r = glm.fit_poisson(d.matrix(tr), tr["claim_count"].to_numpy(float), tr["exposure"].to_numpy())
            mu = glm.predict_log_linear(d, r.params, va, pricing=False) * va["exposure"].to_numpy()
            score = glm.predict_log_linear(d, r.params, va, pricing=True)
            devs.append(M.poisson_deviance(va["claim_count"].to_numpy(float), mu) / len(va))
            ginis.append(M.gini(va["claim_count"].to_numpy(float), score, va["exposure"].to_numpy()))
        cv_rows.append(
            {
                "variant": label,
                "cv_mean_deviance": float(np.mean(devs)),
                "cv_deviance_sd": float(np.std(devs, ddof=1)),
                "cv_gini": float(np.mean(ginis)),
                "cv_gini_sd": float(np.std(ginis, ddof=1)),
            }
        )
        log.info("fairness CV: %s done", label)
    removal = pd.DataFrame(cv_rows)
    removal["gini_change_vs_full"] = removal["cv_gini"] - removal["cv_gini"].iloc[0]

    # Cross-subsidy: refit on the full training split without geography; change in priced frequency
    # (severity has no geographic factor) by density band and region, revenue-neutral on the book.
    d0 = GLMDesign.fit(train, cfg, factors=[*variants["without Region and density"], *design.control_names])
    r0 = glm.fit_poisson(d0.matrix(train), y, e)
    f_full = glm.predict_log_linear(design, params, train, pricing=True)
    f_nogeo = glm.predict_log_linear(d0, r0.params, train, pricing=True)
    f_nogeo *= np.average(f_full, weights=e) / np.average(f_nogeo, weights=e)
    ratio = f_nogeo / f_full
    subs = []
    for f in geo:
        g = (
            pd.DataFrame(
                {
                    "level": binned[f],
                    "exposure": e,
                    "ratio_w": ratio * e,
                    "gt10": (ratio > 1.10) * e,
                    "lt10": (ratio < 0.90) * e,
                }
            )
            .groupby("level")
            .sum()
        )
        g = g.reindex(design.factor(f).levels).reset_index()
        g.insert(0, "factor", f)
        g["mean_premium_change"] = g["ratio_w"] / g["exposure"] - 1
        g["share_up_more_than_10pct"] = g["gt10"] / g["exposure"]
        g["share_down_more_than_10pct"] = g["lt10"] / g["exposure"]
        subs.append(
            g[
                [
                    "factor",
                    "level",
                    "exposure",
                    "mean_premium_change",
                    "share_up_more_than_10pct",
                    "share_down_more_than_10pct",
                ]
            ]
        )
    return FairnessResults(
        share,
        dispersion,
        removal,
        pd.concat(subs, ignore_index=True),
        {"geography": geo, "share_book_changing_gt_10pct": float(np.average(np.abs(ratio - 1) > 0.10, weights=e))},
    )

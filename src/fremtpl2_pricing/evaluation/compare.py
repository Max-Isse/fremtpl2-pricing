"""Phase 4: compare all models on the untouched holdout.

Two views of every model:
* fitted  - control factors at each record's own exposure band: is the model right about these records?
            (deviance, calibration, A/E by segment)
* priced  - controls pinned at full year: what the tariff would charge, pro rata (ranking/Gini, lift on the
            book, full-year calibration, GLM vs LightGBM disagreement)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from fremtpl2_pricing.config import Config
from fremtpl2_pricing.evaluation import metrics as M
from fremtpl2_pricing.features import GLMDesign
from fremtpl2_pricing.models import glm
from fremtpl2_pricing.models.benchmarks import LoadedBenchmarks
from fremtpl2_pricing.models.pure_premium import load_training
from fremtpl2_pricing.models.severity import WEIGHT, load_severity_training

log = logging.getLogger(__name__)

FREQ_MODELS = ["portfolio_mean", "one_factor_BonusMalus", "glm", "rf", "lgbm"]
SEV_MODELS = ["mean", "glm", "lgbm"]
PP_MODELS = ["portfolio_mean", "glm_freq_x_sev", "glm_tweedie", "lgbm_freq_x_sev", "lgbm_tweedie"]


@dataclass
class Predictions:
    """Annual rates (frequency) and annual costs (pure premium, capped) per holdout record."""

    freq_fitted: dict[str, np.ndarray] = field(default_factory=dict)
    freq_priced: dict[str, np.ndarray] = field(default_factory=dict)
    sev_fitted: dict[str, np.ndarray] = field(default_factory=dict)  # per holdout policy with claims
    pp_fitted: dict[str, np.ndarray] = field(default_factory=dict)
    pp_priced: dict[str, np.ndarray] = field(default_factory=dict)


def _load_art(cfg: Config, name: str) -> dict[str, Any]:
    return json.loads((cfg.path("artifacts_dir") / name).read_text(encoding="utf-8"))


def predict_holdout(
    hold: pd.DataFrame, sev_hold: pd.DataFrame, train: pd.DataFrame, sev_mean: float, cfg: Config
) -> Predictions:
    P = Predictions()
    fa, sa, pa = (
        _load_art(cfg, "frequency_glm.json"),
        _load_art(cfg, "severity_glm.json"),
        _load_art(cfg, "pure_premium.json"),
    )
    fd, sd = GLMDesign.from_dict(fa["design"]), GLMDesign.from_dict(sa["design"])
    td = GLMDesign.from_dict(pa["tweedie_crosscheck"]["design"])
    bench = LoadedBenchmarks.load(cfg)

    rate_train = train["claim_count"].sum() / train["exposure"].sum()
    cost_train = train["claim_amount"].sum() / train["exposure"].sum()
    d1 = GLMDesign.fit(train, cfg, factors=["BonusMalus"])
    r1 = glm.fit_poisson(d1.matrix(train), train["claim_count"].to_numpy(float), train["exposure"].to_numpy())
    one = glm.predict_log_linear(d1, r1.params, hold, pricing=False)

    for view, pricing in (("fitted", False), ("priced", True)):
        b = bench.predict(hold, pricing=pricing)
        freq = {
            "portfolio_mean": np.full(len(hold), rate_train),
            "one_factor_BonusMalus": one,
            "glm": glm.predict_log_linear(fd, fa["params"], hold, pricing=pricing),
            "rf": b["rf_frequency"],
            "lgbm": b["lgbm_frequency"],
        }
        sev_glm = glm.predict_log_linear(sd, sa["params"], hold, pricing=pricing)
        pp = {
            "portfolio_mean": np.full(len(hold), cost_train),
            "glm_freq_x_sev": freq["glm"] * sev_glm,
            "glm_tweedie": glm.predict_log_linear(td, pa["tweedie_crosscheck"]["params"], hold, pricing=pricing),
            "lgbm_freq_x_sev": b["lgbm_frequency"] * b["lgbm_severity"],
            "lgbm_tweedie": b["lgbm_tweedie_cost"],
        }
        setattr(P, f"freq_{view}", freq)
        setattr(P, f"pp_{view}", pp)

    bs = bench.predict(sev_hold, pricing=False)
    P.sev_fitted = {
        "mean": np.full(len(sev_hold), sev_mean),
        "glm": glm.predict_log_linear(sd, sa["params"], sev_hold, pricing=False),
        "lgbm": bs["lgbm_severity"],
    }
    return P


@dataclass
class ComparisonResults:
    frequency: pd.DataFrame
    severity: pd.DataFrame
    pure_premium: pd.DataFrame
    paired: pd.DataFrame
    full_year: pd.DataFrame
    freq_lift: dict[str, pd.DataFrame]
    pp_lift: dict[str, pd.DataFrame]
    double_lift: pd.DataFrame
    segments_freq: pd.DataFrame
    segments_pp: pd.DataFrame
    disagreement: pd.DataFrame
    disagreement_segments: pd.DataFrame
    lorenz: dict[str, tuple[np.ndarray, np.ndarray]]
    extras: dict[str, Any] = field(default_factory=dict)


def _bootstrap_freq_pp(hold: pd.DataFrame, P: Predictions, power: float, full: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Bootstrap of all frequency / pure-premium metrics. `full` marks full-year records: the business that
    is actually quoted, and the only records on which a full-year premium can be validated."""
    y = hold["claim_count"].to_numpy(float)
    cost = hold["claim_amount"].to_numpy(float)
    e = hold["exposure"].to_numpy()

    def stat(idx: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in FREQ_MODELS:
            out[f"freq_deviance|{m}"] = M.poisson_deviance(y[idx], P.freq_fitted[m][idx] * e[idx]) / len(idx)
            out[f"freq_gini|{m}"] = M.gini(y[idx], P.freq_priced[m][idx], e[idx])
            f = idx[full[idx]]
            out[f"freq_gini_full_year|{m}"] = M.gini(y[f], P.freq_priced[m][f], e[f])
        for m in PP_MODELS:
            out[f"pp_deviance|{m}"] = (
                M.tweedie_deviance(cost[idx] / e[idx], P.pp_fitted[m][idx], power, e[idx]) / e[idx].sum()
            )
            out[f"pp_gini|{m}"] = M.gini(cost[idx], P.pp_priced[m][idx], e[idx])
            f = idx[full[idx]]
            out[f"pp_gini_full_year|{m}"] = M.gini(cost[f], P.pp_priced[m][f], e[f])
        out["diff|freq_deviance glm - lgbm"] = out["freq_deviance|glm"] - out["freq_deviance|lgbm"]
        out["diff|pp_gini lgbm_freq_x_sev - glm_freq_x_sev"] = (
            out["pp_gini|lgbm_freq_x_sev"] - out["pp_gini|glm_freq_x_sev"]
        )
        out["diff|pp_gini lgbm_tweedie - glm_freq_x_sev"] = out["pp_gini|lgbm_tweedie"] - out["pp_gini|glm_freq_x_sev"]
        out["diff|freq_gini lgbm - glm"] = out["freq_gini|lgbm"] - out["freq_gini|glm"]
        out["diff|pp_gini_full_year lgbm_freq_x_sev - glm_freq_x_sev"] = (
            out["pp_gini_full_year|lgbm_freq_x_sev"] - out["pp_gini_full_year|glm_freq_x_sev"]
        )
        return out

    ev = cfg.section("evaluation")
    return M.grouped_bootstrap(hold["group_id"].to_numpy(), stat, int(ev["n_boot"]), cfg.seed + 20)


def _bootstrap_sev(sev_hold: pd.DataFrame, P: Predictions, cfg: Config) -> pd.DataFrame:
    y, w = sev_hold["avg_severity"].to_numpy(), sev_hold[WEIGHT].to_numpy(float)

    def stat(idx: np.ndarray) -> dict[str, float]:
        out = {
            f"sev_deviance|{m}": M.gamma_deviance(y[idx], P.sev_fitted[m][idx], w[idx]) / w[idx].sum()
            for m in SEV_MODELS
        }
        out |= {f"sev_gini|{m}": M.gini(y[idx] * w[idx], P.sev_fitted[m][idx], w[idx]) for m in SEV_MODELS}
        out["diff|sev_deviance glm - lgbm"] = out["sev_deviance|glm"] - out["sev_deviance|lgbm"]
        return out

    return M.grouped_bootstrap(
        sev_hold["group_id"].to_numpy(), stat, int(cfg.section("evaluation")["n_boot"]), cfg.seed + 21
    )


def _metric_table(boot: pd.DataFrame, prefix: str, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        row: dict[str, Any] = {"model": m}
        for metric in ("deviance", "gini", "gini_full_year"):
            r = boot[boot["statistic"] == f"{prefix}_{metric}|{m}"]
            if len(r):
                row[metric] = r["estimate"].iloc[0]
                row[f"{metric}_ci_low"] = r["ci_low"].iloc[0]
                row[f"{metric}_ci_high"] = r["ci_high"].iloc[0]
        rows.append(row)
    out = pd.DataFrame(rows)
    base = out["deviance"].iloc[0]
    out.insert(4, "deviance_explained_vs_first", 1 - out["deviance"] / base)
    return out


def run_comparison(con: duckdb.DuckDBPyConnection, cfg: Config) -> ComparisonResults:
    ev = cfg.section("evaluation")
    hold = load_training(con, holdout=True)
    sev_hold = load_severity_training(con, holdout=True)
    train = load_training(con)
    log.info("holdout: %s policies, %s policies with claims", len(hold), len(sev_hold))
    sev_train = load_severity_training(con)
    sev_mean = float(sev_train["amount"].sum() / sev_train[WEIGHT].sum())
    P = predict_holdout(hold, sev_hold, train, sev_mean, cfg)
    power = LoadedBenchmarks.load(cfg).tweedie_power

    y = hold["claim_count"].to_numpy(float)
    cost = hold["claim_amount"].to_numpy(float)
    e = hold["exposure"].to_numpy()

    band = GLMDesign.from_dict(_load_art(cfg, "frequency_glm.json")["design"]).factor("ExposureBand")
    full = (band.bin(hold) == band.reference).to_numpy()
    log.info("bootstrap (%s replicates)", ev["n_boot"])
    boot = _bootstrap_freq_pp(hold, P, power, full, cfg)
    sboot = _bootstrap_sev(sev_hold, P, cfg)

    freq_tab = _metric_table(boot, "freq", FREQ_MODELS)
    freq_tab["ae_total_fitted"] = [y.sum() / (P.freq_fitted[m] * e).sum() for m in FREQ_MODELS]
    sev_tab = _metric_table(sboot, "sev", SEV_MODELS)
    w = sev_hold[WEIGHT].to_numpy(float)
    sev_tab["ae_total"] = [sev_hold["amount"].sum() / (P.sev_fitted[m] * w).sum() for m in SEV_MODELS]
    pp_tab = _metric_table(boot, "pp", PP_MODELS)
    pp_tab["ae_total_fitted"] = [cost.sum() / (P.pp_fitted[m] * e).sum() for m in PP_MODELS]
    paired = pd.concat([boot, sboot])
    paired = paired[paired["statistic"].str.startswith("diff|")].reset_index(drop=True)

    fy_rows = []
    for kind, models, preds, outcome in (
        ("frequency", FREQ_MODELS, P.freq_priced, y),
        ("pure premium", PP_MODELS, P.pp_priced, cost),
    ):
        for m in models:
            fy_rows.append(
                {
                    "target": kind,
                    "model": m,
                    "ae_full_year_records": outcome[full].sum() / (preds[m] * e)[full].sum(),
                    "ae_all_records_pro_rata": outcome.sum() / (preds[m] * e).sum(),
                }
            )
    full_year = pd.DataFrame(fy_rows)

    n = int(ev["n_bins"])
    freq_lift = {m: M.lift_table(y, P.freq_fitted[m] * e, e, n) for m in ("glm", "lgbm", "rf")}
    pp_lift = {
        m: M.lift_table(cost, P.pp_fitted[m] * e, e, n, counts=False)
        for m in ("glm_freq_x_sev", "lgbm_freq_x_sev", "lgbm_tweedie")
    }
    dl = M.double_lift(cost, P.pp_fitted["glm_freq_x_sev"] * e, P.pp_fitted["lgbm_freq_x_sev"] * e, e, n)
    lorenz = {m: M.lorenz_curve(cost, P.pp_priced[m], e) for m in PP_MODELS}

    fd = GLMDesign.from_dict(_load_art(cfg, "frequency_glm.json")["design"])
    binned = fd.binned(hold)
    seg_f, seg_p = [], []
    for s in ev["segments"]:
        sf = M.ae_by_segment(binned[s], y, {m: P.freq_fitted[m] * e for m in ("glm", "lgbm", "rf")}, e)
        sp = M.ae_by_segment(
            binned[s],
            cost,
            {m: P.pp_fitted[m] * e for m in ("glm_freq_x_sev", "glm_tweedie", "lgbm_freq_x_sev", "lgbm_tweedie")},
            e,
            counts=False,
        )
        sf.insert(0, "factor", s)
        sp.insert(0, "factor", s)
        seg_f.append(sf)
        seg_p.append(sp)

    ratio = P.pp_priced["lgbm_freq_x_sev"] / P.pp_priced["glm_freq_x_sev"]
    disagreement = pd.DataFrame(
        [
            {
                "comparison": "LightGBM freq x sev / GLM freq x sev (priced)",
                "ratio_p05": float(np.quantile(ratio, 0.05)),
                "ratio_p25": float(np.quantile(ratio, 0.25)),
                "ratio_median": float(np.median(ratio)),
                "ratio_p75": float(np.quantile(ratio, 0.75)),
                "ratio_p95": float(np.quantile(ratio, 0.95)),
                "share_gt_10pct_apart": float(np.mean(np.abs(ratio - 1) > 0.10)),
                "share_gt_25pct_apart": float(np.mean(np.abs(ratio - 1) > 0.25)),
                "ratio_median_full_year_records": float(np.median(ratio[full])),
                "exposure_weighted_premium_glm": float(np.average(P.pp_priced["glm_freq_x_sev"], weights=e)),
                "exposure_weighted_premium_lgbm": float(np.average(P.pp_priced["lgbm_freq_x_sev"], weights=e)),
            }
        ]
    )
    dseg = []
    for s in ev["segments"]:
        g = (
            pd.DataFrame(
                {
                    "level": binned[s],
                    "exposure": e,
                    "glm": P.pp_priced["glm_freq_x_sev"] * e,
                    "lgbm": P.pp_priced["lgbm_freq_x_sev"] * e,
                    "cost": cost,
                    "records": 1.0,
                    "short_records": (~full).astype(float),
                }
            )
            .groupby("level")
            .sum()
        )
        g = g.reset_index()
        g["share_short_records"] = g.pop("short_records") / g.pop("records")
        g.insert(0, "factor", s)
        g["lgbm_over_glm"] = g["lgbm"] / g["glm"]
        dseg.append(g)
    dis_seg = pd.concat(dseg, ignore_index=True)
    dis_seg = dis_seg.assign(abs_gap=(dis_seg["lgbm_over_glm"] - 1).abs()).sort_values(
        "abs_gap", ascending=False, ignore_index=True
    )

    out_path = cfg.path("artifacts_dir") / "holdout_predictions.parquet"
    pd.DataFrame(
        {
            "IDpol": hold["IDpol"],
            "exposure": e,
            "claim_count": y,
            "claim_amount": cost,
            **{f"freq_priced_{m}": v for m, v in P.freq_priced.items()},
            **{f"pp_priced_{m}": v for m, v in P.pp_priced.items()},
            **{f"pp_fitted_{m}": v for m, v in P.pp_fitted.items()},
        }
    ).to_parquet(out_path, index=False)

    return ComparisonResults(
        frequency=freq_tab,
        severity=sev_tab,
        pure_premium=pp_tab,
        paired=paired,
        full_year=full_year,
        freq_lift=freq_lift,
        pp_lift=pp_lift,
        double_lift=dl,
        segments_freq=pd.concat(seg_f, ignore_index=True),
        segments_pp=pd.concat(seg_p, ignore_index=True),
        disagreement=disagreement,
        disagreement_segments=dis_seg,
        lorenz=lorenz,
        extras={
            "n_holdout": len(hold),
            "n_holdout_claim_policies": len(sev_hold),
            "claims": float(y.sum()),
            "exposure": float(e.sum()),
            "tweedie_power": power,
            "n_boot": int(ev["n_boot"]),
        },
    )

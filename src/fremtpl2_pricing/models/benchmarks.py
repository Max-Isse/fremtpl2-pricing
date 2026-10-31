"""Phase 4 benchmarks: random forest (Poisson criterion) on claim rate, and LightGBM with Poisson
(frequency), Gamma (severity) and Tweedie (pure premium) objectives, tuned by grouped 5-fold CV on the
training split only. The holdout is not used here.

Exposure handling mirrors the GLMs (D-012, D-018): frequency and Tweedie models get log(exposure) as
init_score / exposure weights, and the exposure band enters as an ordinal *control* feature that is pinned at
full year when pricing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from fremtpl2_pricing.config import Config
from fremtpl2_pricing.evaluation import metrics as M
from fremtpl2_pricing.features import control_labels

log = logging.getLogger(__name__)


@dataclass
class TreeFeatures:
    """Integer-coded feature matrix shared by training and scoring (categories frozen at training)."""

    numeric: list[str]
    categorical: dict[str, list[str]]
    control: str
    control_edges: list[float]
    control_reference: int

    @classmethod
    def fit(cls, df: pd.DataFrame, cfg: Config) -> TreeFeatures:
        b = cfg.section("benchmarks")
        ctrl = cfg.section("glm")["controls"][b["control"]]
        edges = [float(e) for e in ctrl["edges"]]
        ref = control_labels(edges).index(ctrl["reference"])
        cats = {c: sorted(df[c].astype(str).unique()) for c in b["categorical"]}
        return cls(list(b["numeric"]), cats, b["control"], edges, ref)

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.categorical, self.control]

    def transform(self, df: pd.DataFrame, pricing: bool = False) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.numeric:
            out[c] = np.log(df["Density"].astype(float)) if c == "LogDensity" else df[c].astype(float)
        for c, levels in self.categorical.items():
            codes = pd.Categorical(df[c].astype(str), categories=levels).codes
            if (codes < 0).any():
                bad = sorted(set(df[c].astype(str)) - set(levels))
                raise ValueError(f"unknown {c} level(s): {bad}")
            out[c] = codes.astype(float)
        if pricing:
            out[self.control] = float(self.control_reference)
        else:
            band = pd.cut(df["exposure"], self.control_edges, labels=False, right=True)
            if band.isna().any():
                raise ValueError("exposure outside the control band edges")
            out[self.control] = band.astype(float)
        return out[self.columns]

    @property
    def categorical_columns(self) -> list[str]:
        return list(self.categorical)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "control": self.control,
            "control_edges": self.control_edges,
            "control_reference": self.control_reference,
        }


@dataclass
class LgbmTask:
    name: str
    objective: str
    X: pd.DataFrame
    y: np.ndarray
    weight: np.ndarray | None
    init_score: np.ndarray | None
    folds: np.ndarray
    search_space: dict[str, list[Any]]
    extra: dict[str, Any] = field(default_factory=dict)

    def score(self, y: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> float:
        """Validation loss on the model's natural scale (lower is better)."""
        if self.objective == "poisson":
            e = np.exp(self.init_score[idx]) if self.init_score is not None else 1.0
            return M.poisson_deviance(y, pred * e) / len(y)
        w = self.weight[idx] if self.weight is not None else None
        if self.objective == "gamma":
            return M.gamma_deviance(y, pred, w) / (w.sum() if w is not None else len(y))
        return M.tweedie_deviance(y, pred, self.extra["tweedie_variance_power"], w) / (
            w.sum() if w is not None else len(y)
        )


def _dataset(task: LgbmTask, idx: np.ndarray) -> lgb.Dataset:
    return lgb.Dataset(
        task.X.iloc[idx],
        label=task.y[idx],
        weight=None if task.weight is None else task.weight[idx],
        init_score=None if task.init_score is None else task.init_score[idx],
        categorical_feature=[c for c in task.X.columns if c in ("VehBrand", "VehGas", "Region")],
        free_raw_data=False,
    )


def tune_lgbm(task: LgbmTask, cfg: Config) -> tuple[dict[str, Any], int, pd.DataFrame]:
    """Random search over `search_space` with grouped K-fold CV and early stopping; returns best params,
    the number of boosting rounds for the final fit (mean best iteration) and the search table."""
    lc = cfg.section("benchmarks")["lgbm"]
    rng = np.random.default_rng(cfg.seed + 10)
    configs = [{k: v[rng.integers(len(v))] for k, v in task.search_space.items()} for _ in range(int(lc["n_search"]))]
    rows = []
    for i, c in enumerate(configs):
        params = {
            "objective": task.objective,
            "learning_rate": lc["learning_rate"],
            "verbosity": -1,
            "seed": cfg.seed,
            "bagging_freq": 1 if c["subsample"] < 1 else 0,
            "deterministic": True,
            "num_threads": 0,
            **task.extra,
            **{k: (v.item() if hasattr(v, "item") else v) for k, v in c.items()},
        }
        params["bagging_fraction"] = params.pop("subsample")
        params["feature_fraction"] = params.pop("colsample_bytree")
        params["min_data_in_leaf"] = params.pop("min_child_samples")
        losses, iters = [], []
        for k in np.unique(task.folds):
            tr, va = np.where(task.folds != k)[0], np.where(task.folds == k)[0]
            booster = lgb.train(
                params,
                _dataset(task, tr),
                num_boost_round=int(lc["max_rounds"]),
                valid_sets=[_dataset(task, va)],
                callbacks=[lgb.early_stopping(int(lc["early_stopping_rounds"]), verbose=False)],
            )
            pred = np.asarray(booster.predict(task.X.iloc[va], num_iteration=booster.best_iteration))
            losses.append(task.score(task.y[va], pred, va))
            iters.append(booster.best_iteration)
        rows.append(
            {
                "config": i,
                **{
                    k: params[k]
                    for k in ("num_leaves", "min_data_in_leaf", "reg_lambda", "feature_fraction", "bagging_fraction")
                },
                "cv_loss": float(np.mean(losses)),
                "cv_loss_sd": float(np.std(losses, ddof=1)),
                "mean_best_iteration": float(np.mean(iters)),
                "_params": params,
            }
        )
        log.info("%s config %d: cv loss %.6f, %.0f rounds", task.name, i, rows[-1]["cv_loss"], np.mean(iters))
    table = pd.DataFrame(rows).sort_values("cv_loss", ignore_index=True)
    best = table.iloc[0]
    return dict(best["_params"]), max(round(float(best["mean_best_iteration"])), 1), table.drop(columns="_params")


def fit_final(task: LgbmTask, params: dict[str, Any], rounds: int) -> lgb.Booster:
    return lgb.train(params, _dataset(task, np.arange(len(task.y))), num_boost_round=rounds)


@dataclass
class BenchmarkResults:
    features: TreeFeatures
    boosters: dict[str, lgb.Booster]
    rf: RandomForestRegressor
    tuning: dict[str, pd.DataFrame]
    params: dict[str, dict[str, Any]]
    rounds: dict[str, int]
    tweedie_power: float


def run_benchmarks(con: Any, cfg: Config) -> BenchmarkResults:
    from fremtpl2_pricing.models.pure_premium import load_training as load_pp_training
    from fremtpl2_pricing.models.severity import WEIGHT, load_severity_training

    lc = cfg.section("benchmarks")["lgbm"]
    pol = load_pp_training(con)
    feats = TreeFeatures.fit(pol, cfg)
    X = feats.transform(pol)
    e = pol["exposure"].to_numpy()
    folds = pol["fold"].to_numpy()
    sev_art = json.loads((cfg.path("artifacts_dir") / "severity_glm.json").read_text(encoding="utf-8"))
    from fremtpl2_pricing.models.glm import tweedie_power_from_gamma_dispersion

    power = tweedie_power_from_gamma_dispersion(float(sev_art["dispersion"]))

    sev = load_severity_training(con)
    tasks = {
        "lgbm_poisson_frequency": LgbmTask(
            "frequency", "poisson", X, pol["claim_count"].to_numpy(float), None, np.log(e), folds, lc["search_space"]
        ),
        "lgbm_gamma_severity": LgbmTask(
            "severity",
            "gamma",
            feats.transform(sev),
            sev["avg_severity"].to_numpy(),
            sev[WEIGHT].to_numpy(float),
            None,
            sev["fold"].to_numpy(),
            lc["severity_search_space"],
        ),
        "lgbm_tweedie_pure_premium": LgbmTask(
            "tweedie",
            "tweedie",
            X,
            (pol["claim_amount"] / pol["exposure"]).to_numpy(),
            e,
            None,
            folds,
            lc["search_space"],
            {"tweedie_variance_power": power},
        ),
    }
    boosters, tuning, params, rounds = {}, {}, {}, {}
    for name, task in tasks.items():
        log.info("tuning %s", name)
        p, r, t = tune_lgbm(task, cfg)
        boosters[name] = fit_final(task, p, r)
        tuning[name], params[name], rounds[name] = t, p, r

    rc = cfg.section("benchmarks")["rf"]
    log.info("random forest (Poisson criterion) on claim rate")
    rf = RandomForestRegressor(
        criterion="poisson",
        n_estimators=int(rc["n_estimators"]),
        min_samples_leaf=int(rc["min_samples_leaf"]),
        max_features=float(rc["max_features"]),
        max_samples=float(rc["max_samples"]),
        n_jobs=-1,
        random_state=cfg.seed,
    )
    rf.fit(X, pol["claim_count"].to_numpy(float) / e, sample_weight=e)
    return BenchmarkResults(feats, boosters, rf, tuning, params, rounds, power)


def model_metadata(res: BenchmarkResults) -> dict[str, Any]:
    return {
        "features": res.features.to_dict(),
        "params": res.params,
        "rounds": res.rounds,
        "tweedie_power": res.tweedie_power,
    }


def save_benchmarks(res: BenchmarkResults, cfg: Config) -> Path:
    art = cfg.path("artifacts_dir")
    art.mkdir(parents=True, exist_ok=True)
    for name, b in res.boosters.items():
        b.save_model(str(art / f"{name}.txt"))
    joblib.dump(res.rf, art / "rf_poisson_frequency.joblib", compress=3)
    path = art / "benchmarks.json"
    path.write_text(json.dumps(model_metadata(res), indent=2, default=float), encoding="utf-8")
    return path


@dataclass
class LoadedBenchmarks:
    features: TreeFeatures
    boosters: dict[str, lgb.Booster]
    rf: RandomForestRegressor
    tweedie_power: float

    @classmethod
    def load(cls, cfg: Config) -> LoadedBenchmarks:
        art = cfg.path("artifacts_dir")
        meta = json.loads((art / "benchmarks.json").read_text(encoding="utf-8"))
        f = meta["features"]
        feats = TreeFeatures(f["numeric"], f["categorical"], f["control"], f["control_edges"], f["control_reference"])
        boosters = {n: lgb.Booster(model_file=str(art / f"{n}.txt")) for n in meta["params"]}
        return cls(feats, boosters, joblib.load(art / "rf_poisson_frequency.joblib"), float(meta["tweedie_power"]))

    def predict(self, df: pd.DataFrame, pricing: bool) -> dict[str, np.ndarray]:
        """Annual rates / severities per record (multiply frequency and cost by exposure for expected values)."""
        X = self.features.transform(df, pricing=pricing)
        return {
            "rf_frequency": self.rf.predict(X),
            "lgbm_frequency": self.boosters["lgbm_poisson_frequency"].predict(X),
            "lgbm_severity": self.boosters["lgbm_gamma_severity"].predict(X),
            "lgbm_tweedie_cost": self.boosters["lgbm_tweedie_pure_premium"].predict(X),
        }

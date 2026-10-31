"""Tree features, LightGBM tuning, and an end-to-end run through the holdout comparison on a simulated book.

Simulated data are test data only (known structure); see data/README.md.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from fremtpl2_pricing.config import Config, load_config, with_overrides
from fremtpl2_pricing.evaluation.compare import FREQ_MODELS, PP_MODELS, run_comparison
from fremtpl2_pricing.models.benchmarks import LgbmTask, LoadedBenchmarks, TreeFeatures, run_benchmarks, save_benchmarks
from fremtpl2_pricing.models.frequency import run_frequency
from fremtpl2_pricing.models.frequency import save_artifact as save_freq
from fremtpl2_pricing.models.pure_premium import run_pure_premium
from fremtpl2_pricing.models.pure_premium import save_artifact as save_pp
from fremtpl2_pricing.models.severity import run_severity
from fremtpl2_pricing.models.severity import save_artifact as save_sev
from fremtpl2_pricing.reporting.model_comparison import write_comparison_report
from fremtpl2_pricing.reporting.phase1 import _write_tables
from tests.test_frequency import simulated_policies
from tests.test_severity import with_claim_amounts


def small_cfg(tmp_path: Path) -> Config:
    cfg = with_overrides(load_config(), root=tmp_path)
    raw = dict(cfg.raw)
    raw["frequency"] = {**raw["frequency"], "residual_sample": 2_000}
    b = raw["benchmarks"]
    raw["benchmarks"] = {
        **b,
        "lgbm": {**b["lgbm"], "n_search": 1, "max_rounds": 200},
        "rf": {**b["rf"], "n_estimators": 20, "min_samples_leaf": 200},
    }
    raw["evaluation"] = {**raw["evaluation"], "n_boot": 10}
    return Config(raw=raw, root=tmp_path)


def test_tree_features_pin_control_and_reject_unknown_levels() -> None:
    df = simulated_policies(2_000, seed=3)
    f = TreeFeatures.fit(df, load_config())
    fitted, priced = f.transform(df), f.transform(df.drop(columns="exposure"), pricing=True)
    assert list(fitted.columns) == f.columns
    assert (priced["ExposureBand"] == f.control_reference).all()
    assert fitted.drop(columns="ExposureBand").equals(priced.drop(columns="ExposureBand"))
    assert np.allclose(fitted["LogDensity"], np.log(df["Density"]))
    with pytest.raises(ValueError, match="unknown Region"):
        f.transform(df.head(2).assign(Region="R99"))


def test_lgbm_task_scores_on_natural_scale() -> None:
    y, pred, e = np.array([0.0, 1.0]), np.array([0.5, 0.5]), np.array([1.0, 0.5])
    t = LgbmTask("f", "poisson", pd.DataFrame({"a": [0, 1]}), y, None, np.log(e), np.zeros(2), {})
    expected = 2 * (0.5 + (np.log(1 / 0.25) - (1 - 0.25))) / 2  # mu = pred * exposure = [0.5, 0.25]
    assert t.score(y, pred, np.arange(2)) == pytest.approx(expected)


def _book(tmp_path: Path) -> tuple[Config, duckdb.DuckDBPyConnection]:
    cfg = small_cfg(tmp_path)
    pol = simulated_policies(40_000, seed=61)
    pol["is_holdout"] = pol["IDpol"] % 5 == 0
    pol.loc[pol["is_holdout"], "fold"] = -1
    sev = with_claim_amounts(pol, seed=62)
    claims = sev.loc[sev.index.repeat(sev["claim_count"])].copy()
    claims["claim_amount"] = claims["avg_severity"]
    claims["claim_amount_raw"] = claims["avg_severity"]
    pol = pol.merge(sev[["IDpol", "amount"]], on="IDpol", how="left").fillna({"amount": 0.0})
    pol["claim_amount"] = pol["amount"]
    pol["claim_amount_raw"] = pol["amount"]
    con = duckdb.connect()
    for name, frame in (("policy_clean", pol), ("claims_clean", claims)):
        con.register("tmp", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp")
        con.unregister("tmp")
    con.execute("CREATE TABLE large_loss_summary AS SELECT 0.2 AS large_loss_loading")
    cfg.path("manifest").parent.mkdir(parents=True, exist_ok=True)
    cfg.path("manifest").write_text('{"freq": {"sha256": "x"}}', encoding="utf-8")
    return cfg, con


def test_pipeline_through_holdout_comparison(tmp_path: Path) -> None:
    cfg, con = _book(tmp_path)
    art = cfg.path("artifacts_dir")
    fr = run_frequency(con, cfg)
    assert fr.extras["n_train"] == 32_000  # holdout rows excluded from training
    save_freq(fr, cfg, art / "frequency_glm.json")
    _write_tables({"freq_factor_tests": fr.factor_tests}, cfg.path("tables_dir"))
    save_sev(run_severity(con, cfg), cfg, art / "severity_glm.json")
    save_pp(run_pure_premium(con, cfg), cfg, art / "pure_premium.json")
    bench = run_benchmarks(con, cfg)
    save_benchmarks(bench, cfg)
    assert set(LoadedBenchmarks.load(cfg).boosters) == set(bench.boosters)

    res = run_comparison(con, cfg)
    assert res.extras["n_holdout"] == 8_000
    assert list(res.frequency["model"]) == FREQ_MODELS and list(res.pure_premium["model"]) == PP_MODELS
    f = res.frequency.set_index("model")
    assert f.loc["glm", "deviance"] < f.loc["portfolio_mean", "deviance"]
    assert f.loc["glm", "gini"] > 0.05
    assert (res.frequency["deviance_ci_low"] <= res.frequency["deviance"]).all()
    assert np.isclose(f["ae_total_fitted"], 1.0, atol=0.15).all()
    assert (art / "holdout_predictions.parquet").exists()
    out = write_comparison_report(res, cfg)
    assert "Model comparison on the holdout" in out.read_text(encoding="utf-8")

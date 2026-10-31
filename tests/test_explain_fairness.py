"""Explainability and fairness helpers on toy inputs, plus the phase 5 step on a simulated book."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fremtpl2_pricing.analysis.explain import (
    glm_relativities,
    monotonicity,
    run_explain,
    shap_level_relativities,
)
from fremtpl2_pricing.analysis.fairness import run_fairness, variance_shares
from fremtpl2_pricing.config import Config, load_config
from fremtpl2_pricing.features import GLMDesign
from fremtpl2_pricing.reporting.explain_fairness import write_explain_fairness_report
from tests.test_frequency import simulated_policies


def test_variance_shares_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=1000), rng.normal(scale=2, size=1000)
    sh = variance_shares({"a": a, "b": b}, a + b)
    assert sum(sh.values()) == pytest.approx(1.0)
    assert sh["b"] > sh["a"]


def test_monotonicity_counts_adjacent_decreases() -> None:
    rel = pd.DataFrame({"factor": "BM", "order": [0, 1, 2, 3], "relativity": [1.0, 2.0, 1.5, 3.0]})
    m = monotonicity(rel, ["BM"], "x").iloc[0]
    assert m["adjacent_decreases"] == 1 and m["largest_decrease"] == pytest.approx(0.5)


def test_shap_level_relativities_match_additive_effects() -> None:
    cfg = load_config()
    df = simulated_policies(3_000, seed=4)
    d = GLMDesign.fit(df, cfg, factors=["VehGas"])
    binned = d.binned(df)
    # an additive log effect of 0.3 for Diesel, as TreeSHAP would report for a one-feature model
    contrib = pd.DataFrame({"VehGas": np.where(df["VehGas"] == "Diesel", 0.3, 0.0) - 0.15})
    rel = shap_level_relativities(contrib, binned, d, ["VehGas"]).set_index("level")["relativity"]
    ref = d.factor("VehGas").reference
    assert rel[ref] == pytest.approx(1.0)
    other = next(lvl for lvl in rel.index if lvl != ref)
    assert rel[other] == pytest.approx(np.exp(0.3 if other == "Diesel" else -0.3))


def test_glm_relativities_ci_from_covariance() -> None:
    cfg = load_config()
    d = GLMDesign.fit(simulated_policies(2_000, seed=5), cfg, factors=["VehGas"])
    cols = d.columns()
    params = pd.Series([0.0, 0.2], index=cols)
    rel = glm_relativities(d, params, np.diag([0.01, 0.04]), ["VehGas"])
    row = rel[rel["relativity"] != 1.0].iloc[0]
    assert row["relativity"] == pytest.approx(np.exp(0.2))
    assert row["ci_low"] == pytest.approx(np.exp(0.2 - 1.959964 * 0.2), rel=1e-4)


def test_phase5_on_simulated_book(tmp_path: Path) -> None:
    from fremtpl2_pricing.models.benchmarks import run_benchmarks, save_benchmarks
    from fremtpl2_pricing.models.frequency import run_frequency
    from fremtpl2_pricing.models.frequency import save_artifact as save_freq
    from fremtpl2_pricing.models.pure_premium import run_pure_premium
    from fremtpl2_pricing.models.pure_premium import save_artifact as save_pp
    from fremtpl2_pricing.models.severity import run_severity
    from fremtpl2_pricing.models.severity import save_artifact as save_sev
    from fremtpl2_pricing.reporting.phase1 import _write_tables
    from tests.test_benchmarks_compare import _book, small_cfg  # noqa: F401  (builds artefacts)

    cfg, con = _book(tmp_path)
    raw = {**cfg.raw, "explain": {**cfg.raw["explain"], "shap_sample": 3_000}}
    cfg = Config(raw=raw, root=cfg.root)
    art = cfg.path("artifacts_dir")
    fr = run_frequency(con, cfg)
    save_freq(fr, cfg, art / "frequency_glm.json")
    _write_tables({"freq_factor_tests": fr.factor_tests}, cfg.path("tables_dir"))
    save_sev(run_severity(con, cfg), cfg, art / "severity_glm.json")
    save_pp(run_pure_premium(con, cfg), cfg, art / "pure_premium.json")
    save_benchmarks(run_benchmarks(con, cfg), cfg)

    ex = run_explain(con, cfg)
    assert set(ex.stability["factor"]) == set(cfg.section("explain")["factors"])
    assert (ex.comparison["glm_fold_spread"] >= 0).all()
    fa = run_fairness(con, cfg)
    shares = fa.geography_share["glm_share_of_log_premium_variance"].sum()
    assert shares == pytest.approx(1.0, abs=0.05)
    assert len(fa.removal_cv) == 3
    out = write_explain_fairness_report(ex, fa, cfg)
    assert "Fairness: geography as a proxy" in out.read_text(encoding="utf-8")

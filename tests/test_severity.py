"""Severity GLM helpers, CV selection and an end-to-end frequency -> severity -> pure premium smoke test.

Simulated portfolios are test data only (known truth); see data/README.md.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fremtpl2_pricing.config import Config, load_config, with_overrides
from fremtpl2_pricing.features import GLMDesign
from fremtpl2_pricing.models import glm
from fremtpl2_pricing.models.frequency import run_frequency
from fremtpl2_pricing.models.frequency import save_artifact as save_freq
from fremtpl2_pricing.models.pure_premium import PurePremiumModel, run_pure_premium
from fremtpl2_pricing.models.severity import WEIGHT, backward_elimination, cv_select, run_severity
from fremtpl2_pricing.models.severity import save_artifact as save_sev
from fremtpl2_pricing.reporting.phase1 import _write_tables
from tests.test_frequency import simulated_policies

SHAPE = 0.5  # Gamma shape per claim => dispersion 2


def with_claim_amounts(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add one Gamma amount per claim; true severity effect: x1.3 for VehAge < 3, nothing else."""
    rng = np.random.default_rng(seed)
    out = df[df["claim_count"] > 0].copy()
    mean = 1500 * np.where(out["VehAge"] < 3, 1.3, 1.0)
    out["n_claims"] = out["claim_count"].astype(float)
    # average of n Gamma(shape, mean/shape) claims ~ Gamma(n*shape, mean/(n*shape))
    out["avg_severity"] = rng.gamma(out["n_claims"] * SHAPE, mean / (out["n_claims"] * SHAPE))
    out["amount"] = out["avg_severity"] * out["n_claims"]
    return out


def test_gamma_helpers_recover_truth() -> None:
    cfg = load_config()
    sev = with_claim_amounts(simulated_policies(150_000, seed=31), seed=32)
    d = GLMDesign.fit(sev, cfg, factors=["VehAge", "VehGas"], bands=cfg.section("severity")["bands"], weight_col=WEIGHT)
    y, w = sev["avg_severity"].to_numpy(), sev[WEIGHT].to_numpy()
    res = glm.fit_gamma(d.matrix(sev), y, w)
    assert abs(res.scale - 1 / SHAPE) < 0.3
    young = [c for c in res.params.index if c.startswith("VehAge[0-")]
    assert young and abs(np.exp(res.params[young[0]]) - 1.3) < 0.1
    r = glm.gamma_quantile_residuals(y, np.asarray(res.fittedvalues), w, float(res.scale))
    assert abs(r.mean()) < 0.05 and abs(r.std() - 1) < 0.05
    assert glm.gamma_deviance(np.array([2.0, 3.0]), np.array([2.0, 3.0])) == pytest.approx(0.0)


def test_tweedie_power_from_dispersion() -> None:
    assert glm.tweedie_power_from_gamma_dispersion(1.0) == pytest.approx(1.5)
    assert glm.tweedie_power_from_gamma_dispersion(1e6) == pytest.approx(2.0, abs=1e-5)
    assert glm.tweedie_power_from_gamma_dispersion(1e-6) == pytest.approx(1.0, abs=1e-5)


def test_cv_selection_drops_noise_factors() -> None:
    cfg = load_config()
    sev = with_claim_amounts(simulated_policies(150_000, seed=41), seed=42)
    scfg = cfg.section("severity")
    factors = ["VehAge", "VehBrand", "Region", "VehGas"]  # only VehAge matters
    d = GLMDesign.fit(sev, cfg, factors=factors, bands=scfg["bands"], weight_col=WEIGHT)
    path, _ = backward_elimination(d, sev, 0.05, controls=[])
    assert path["dropped"].iloc[-1] == "VehAge"  # most important factor survives longest
    selected, cv_path = cv_select(sev, path, cfg, controls=[])
    assert "VehAge" in selected
    assert "Region" not in selected and "VehBrand" not in selected
    assert cv_path["chosen_one_se"].sum() == 1


def _simulated_db(tmp_path: Path) -> tuple[Config, duckdb.DuckDBPyConnection, pd.DataFrame]:
    cfg = with_overrides(load_config(), root=tmp_path)
    raw = {**cfg.raw, "frequency": {**cfg.raw["frequency"], "residual_sample": 2_000}}
    cfg = Config(raw=raw, root=tmp_path)
    pol = simulated_policies(30_000, seed=51)
    sev = with_claim_amounts(pol, seed=52)
    claims = sev.loc[sev.index.repeat(sev["claim_count"])].copy()
    claims["claim_amount"] = claims["avg_severity"]
    claims["claim_amount_raw"] = claims["avg_severity"]
    claims["claim_id"] = np.arange(len(claims))
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
    return cfg, con, pol


def test_frequency_severity_pure_premium_smoke(tmp_path: Path) -> None:
    cfg, con, pol = _simulated_db(tmp_path)
    fr = run_frequency(con, cfg)
    save_freq(fr, cfg, cfg.path("artifacts_dir") / "frequency_glm.json")
    _write_tables({"freq_factor_tests": fr.factor_tests}, cfg.path("tables_dir"))

    sv = run_severity(con, cfg)
    assert set(sv.extras["selected"]) <= set(sv.freq_vs_sev["factor"])  # selection itself: see test above
    assert set(sv.freq_vs_sev["factor"]) >= {"VehAge", "BonusMalus"}
    assert np.isclose(sv.by_exposure_band["ae_fitted"].dropna(), 1.0, atol=0.1).all()
    save_sev(sv, cfg, cfg.path("artifacts_dir") / "severity_glm.json")

    pp = run_pure_premium(con, cfg)
    assert 1.0 < pp.tweedie_power < 2.0
    fitted = pp.reconciliation[pp.reconciliation["basis"].str.startswith("fitted")]
    assert np.isclose(fitted["ae"], 1.0, atol=0.05).all()

    model = PurePremiumModel.from_artifacts(
        cfg.path("artifacts_dir") / "frequency_glm.json", cfg.path("artifacts_dir") / "severity_glm.json", 0.2
    )
    quote = model.predict(pol.head(20).drop(columns="exposure"))  # quoting needs no exposure
    assert np.allclose(quote["pure_premium"], quote["frequency"] * quote["severity"] * 1.2)
    assert (quote > 0).all().all()


def test_quantile_residuals_detect_point_masses() -> None:
    """Flat-rate settlements (a point mass) make Gamma quantile residuals visibly non-normal."""
    rng = np.random.default_rng(7)
    y = np.where(rng.uniform(size=20_000) < 0.4, 1200.0, rng.gamma(0.5, 3000, 20_000))
    mu = np.full_like(y, y.mean())
    r = glm.gamma_quantile_residuals(y, mu, np.ones_like(y), dispersion=float(y.var() / y.mean() ** 2))
    assert stats.kstest(r, "norm").pvalue < 1e-6

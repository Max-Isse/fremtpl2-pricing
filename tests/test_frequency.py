"""Frequency GLM: design matrix, inference helpers on simulated counts, and a pipeline smoke test.

Simulated portfolios here are test data only (known truth to check estimators); see data/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fremtpl2_pricing.config import load_config, with_overrides
from fremtpl2_pricing.features import DERIVED_BASE_LEVEL, GLMDesign
from fremtpl2_pricing.models import glm
from fremtpl2_pricing.models.frequency import run_frequency, save_artifact

REGIONS = ["R11", "R24", "R82", "R93"]
BRANDS = ["B1", "B2", "B12"]


def simulated_policies(n: int, seed: int, alpha: float | None = None) -> pd.DataFrame:
    """Portfolio covering every configured band; claims from a known log-linear rate."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "IDpol": np.arange(n),
            # 40% full-year records, the rest spread over (0.01, 1): every exposure band is populated
            "exposure": np.where(rng.uniform(size=n) < 0.4, 1.0, rng.uniform(0.01, 1.0, n)),
            "DrivAge": rng.integers(18, 85, n),
            "VehAge": rng.integers(0, 25, n),
            "VehPower": rng.integers(4, 15, n),
            "BonusMalus": rng.choice([50, 50, 50, 53, 62, 72, 85, 97, 100, 120], n),
            "Density": np.exp(rng.uniform(0, 10.3, n)).round().clip(1),
            "VehBrand": rng.choice(BRANDS, n),
            "VehGas": rng.choice(["Diesel", "Regular"], n),
            "Region": rng.choice(REGIONS, n),
            "Area": rng.choice(list("ABCDEF"), n),
        }
    )
    rate = 0.06 * np.exp(0.8 * (df["BonusMalus"] > 50) + 0.3 * (df["VehGas"] == "Diesel"))
    mu = rate * df["exposure"]
    y = rng.poisson(mu) if alpha is None else rng.poisson(mu * rng.gamma(1 / alpha, alpha, n))
    df["claim_count"] = y
    df["claimnb_recorded"] = y
    df["group_id"] = df["IDpol"]
    df["fold"] = df["IDpol"] % 5
    df["is_holdout"] = False
    return df


@pytest.fixture(scope="module")
def portfolio() -> pd.DataFrame:
    return simulated_policies(20_000, seed=11)


def test_design_reference_levels_and_columns(portfolio: pd.DataFrame) -> None:
    cfg = load_config()
    d = GLMDesign.fit(portfolio, cfg)
    X = d.matrix(portfolio)
    binned = d.binned(portfolio)
    for f in d.factors:
        if f.kind == "derived":
            assert f.reference == DERIVED_BASE_LEVEL
            continue
        if f.kind == "control":
            assert f.reference == "(0.999, 1]"
            continue
        expo = portfolio["exposure"].groupby(binned[f.name]).sum()
        assert f.reference == expo.idxmax()
        assert f"{f.name}[{f.reference}]" not in X.columns
    assert X.shape[1] == 1 + sum(len(f.levels) - 1 for f in d.factors)
    assert np.allclose(X.drop(columns="Intercept").filter(like="VehGas").sum(axis=1).isin([0, 1]), True)


def test_design_round_trip_and_validation(portfolio: pd.DataFrame) -> None:
    cfg = load_config()
    d = GLMDesign.fit(portfolio, cfg)
    d2 = GLMDesign.from_dict(json.loads(json.dumps(d.to_dict())))
    assert d2.matrix(portfolio.head(50)).equals(d.matrix(portfolio.head(50)))
    with pytest.raises(ValueError, match="unknown Region"):
        d.matrix(portfolio.head(3).assign(Region="R99"))
    with pytest.raises(ValueError, match="below the lowest band"):
        d.matrix(portfolio.head(3).assign(DrivAge=16))


def test_pricing_matrix_pins_controls(portfolio: pd.DataFrame) -> None:
    d = GLMDesign.fit(portfolio, load_config())
    assert d.control_names == ["ExposureBand"]
    fitted = d.matrix(portfolio.head(200))
    priced = d.matrix(portfolio.head(200).drop(columns="exposure"), pricing=True)  # no exposure needed to quote
    control_cols = [c for c in fitted.columns if c.startswith("ExposureBand[")]
    assert control_cols and (priced[control_cols] == 0).all().all()
    assert fitted.drop(columns=control_cols).equals(priced.drop(columns=control_cols))
    with pytest.raises(ValueError, match="outside"):
        d.matrix(portfolio.head(3).assign(exposure=1.5))


def test_control_removes_full_year_bias() -> None:
    """Short records claim more per year (as in freMTPL2); only the controlled fit prices full year right."""
    cfg = load_config()
    df = simulated_policies(60_000, seed=21)
    rng = np.random.default_rng(22)
    uplift = np.where(df["exposure"] < 0.25, 4.0, 1.0)
    rate = 0.06 * np.exp(0.8 * (df["BonusMalus"] > 50) + 0.3 * (df["VehGas"] == "Diesel"))
    df["claim_count"] = rng.poisson(rate * uplift * df["exposure"])
    full = df["exposure"] == 1.0
    y, e = df["claim_count"].to_numpy(float), df["exposure"].to_numpy()
    ae = {}
    for label, factors in (("offset_only", ["BonusMalus", "VehGas"]), ("controlled", None)):
        names = factors or ["BonusMalus", "VehGas", "ExposureBand"]
        d = GLMDesign.fit(df, cfg, factors=factors)
        if factors is None:  # restrict the full design to the simulated factors plus the control
            d = GLMDesign([f for f in d.factors if f.name in names])
        r = glm.fit_poisson(d.matrix(df), y, e)
        charged = np.asarray(r.predict(d.matrix(df, pricing=True), offset=np.log(e)))
        ae[label] = y[full].sum() / charged[full].sum()
    assert ae["offset_only"] < 0.96
    assert abs(ae["controlled"] - 1.0) < 0.05


def test_age_malus_derivation() -> None:
    cfg = load_config()
    spec = GLMDesign.fit(simulated_policies(5_000, seed=2), cfg).factor("AgeMalus")
    df = pd.DataFrame({"DrivAge": [20, 20, 45, 70, 70], "BonusMalus": [50, 90, 90, 50, 60]})
    assert list(spec.bin(df)) == ["base", "18-30 & malus", "base", "base", "61+ & malus"]


def test_poisson_recovers_known_effects(portfolio: pd.DataFrame) -> None:
    cfg = load_config()
    d = GLMDesign.fit(portfolio, cfg, factors=["VehGas", "Region"])
    res = glm.fit_poisson(d.matrix(portfolio), portfolio["claim_count"].to_numpy(float), portfolio["exposure"])
    rel = glm.relativities(d, res.params, res.bse, portfolio)
    diesel = rel[(rel["factor"] == "VehGas") & (rel["level"] == "Diesel")].iloc[0]
    regular = rel[(rel["factor"] == "VehGas") & (rel["level"] == "Regular")].iloc[0]
    ratio = diesel["relativity"] / regular["relativity"]
    assert abs(np.log(ratio) - 0.3) < 0.1
    assert rel.loc[rel["is_reference"], "relativity"].eq(1.0).all()
    assert (rel["ci_low"] <= rel["relativity"]).all() and (rel["relativity"] <= rel["ci_high"]).all()


def test_overdispersion_tests_detect_nb_not_poisson() -> None:
    cfg = load_config()
    for alpha, should_reject in ((None, False), (1.5, True)):
        df = simulated_policies(40_000, seed=5, alpha=alpha)
        d = GLMDesign.fit(df, cfg, factors=["BonusMalus", "VehGas"])
        y = df["claim_count"].to_numpy(float)
        res = glm.fit_poisson(d.matrix(df), y, df["exposure"].to_numpy())
        od = glm.overdispersion_tests(res, y)
        assert (od.ct_p_one_sided < 0.01) is should_reject
        if not should_reject:
            assert abs(od.pearson_dispersion - 1) < 0.1


def test_nb2_estimates_alpha_and_boundary_lr() -> None:
    cfg = load_config()
    df = simulated_policies(30_000, seed=9, alpha=1.0)
    d = GLMDesign.fit(df, cfg, factors=["BonusMalus", "VehGas"])
    X, y, e = d.matrix(df), df["claim_count"].to_numpy(float), df["exposure"].to_numpy()
    pois = glm.fit_poisson(X, y, e)
    nb = glm.fit_nb2(X, y, e, start=pois.params.to_numpy())
    assert 0.5 < nb.params["alpha"] < 1.6
    lr, p = glm.nb2_vs_poisson_lr(nb.llf, pois.llf)
    assert lr > 0 and p < 0.01
    assert glm.nb2_vs_poisson_lr(-10.0, -9.0) == (0.0, 0.5)


def test_randomized_quantile_residuals_are_normal_under_true_model() -> None:
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.01, 0.3, 20_000)
    y = rng.poisson(mu)
    r = glm.randomized_quantile_residuals(y, mu, rng)
    assert stats.kstest(r, "norm").pvalue > 0.01


def test_frequency_pipeline_smoke(tmp_path: Path, portfolio: pd.DataFrame) -> None:
    cfg = with_overrides(load_config(), root=tmp_path)
    raw = {**cfg.raw, "frequency": {**cfg.raw["frequency"], "residual_sample": 2_000}}
    cfg = type(cfg)(raw=raw, root=tmp_path)
    con = duckdb.connect()
    con.register("p", portfolio)
    con.execute("CREATE TABLE policy_clean AS SELECT * FROM p")
    res = run_frequency(con, cfg)
    assert set(res.cv_metrics["model"]) == {"portfolio_mean", "one_factor_BonusMalus", "glm_offset_only", "glm_poisson"}
    assert set(res.full_year["scope"]) == {"full-year records", "all records"}
    glm_dev = res.cv_metrics.set_index("model")["oof_mean_deviance"]
    assert glm_dev["glm_poisson"] < glm_dev["portfolio_mean"]
    assert np.isclose(res.cv_metrics["oof_ae_total"], 1.0, atol=0.02).all()
    assert len(res.relativities) == sum(len(f.levels) for f in res.design.factors)
    assert set(res.factor_tests["factor"]) == {*res.design.names, "Area"}
    cfg.path("manifest").parent.mkdir(parents=True, exist_ok=True)
    cfg.path("manifest").write_text(json.dumps({"freq": {"sha256": "x"}}), encoding="utf-8")
    art = json.loads(save_artifact(res, cfg, tmp_path / "freq.json").read_text(encoding="utf-8"))
    design = GLMDesign.from_dict(art["design"])
    mu = np.exp(design.matrix(portfolio.head(5), pricing=True) @ pd.Series(art["params"])) * portfolio["exposure"].head(
        5
    )
    assert np.all(mu > 0)

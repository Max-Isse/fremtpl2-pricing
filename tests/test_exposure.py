"""Exposure handling: the offset model recovers a known annual rate; bands cover every value."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fremtpl2_pricing.analysis.exposure import add_bands, fit_models, poisson_deviance, predict
from fremtpl2_pricing.config import load_config


def simulated_portfolio(n: int = 20_000, seed: int = 3) -> pd.DataFrame:
    """Claims proportional to exposure with a known rate per band (test data only)."""
    cfg = load_config()
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "exposure": rng.uniform(0.05, 1.0, n),
            "BonusMalus": rng.choice([50, 55, 75, 120], n),
            "DrivAge": rng.choice([19, 30, 45, 80], n),
            "Area": rng.choice(["A", "B"], n),
            "claim_count": 0,
        }
    )
    df = add_bands(df, cfg)
    rate = 0.1 * np.where(df["BonusMalus"] >= 100, 2.0, 1.0)
    df["claim_count"] = rng.poisson(rate * df["exposure"])
    return df


def test_bands_cover_all_values() -> None:
    cfg = load_config()
    df = add_bands(
        pd.DataFrame(
            {
                "exposure": [0.01, 0.5, 1.0],
                "BonusMalus": [50, 100, 230],
                "DrivAge": [18, 70, 100],
                "Area": "A",
                "claim_count": 0,
            }
        ),
        cfg,
    )
    assert not df[["BonusMalus_band", "DrivAge_band", "exposure_band"]].isin(["nan"]).any().any()
    assert df["BonusMalus_band"].iloc[2] == df["BonusMalus_band"].iloc[1]  # 230 lands in the open top band


def test_offset_recovers_rate_and_feature_coef_near_one() -> None:
    df = simulated_portfolio()
    res = fit_models(df, ["BonusMalus_band", "DrivAge_band", "Area"])
    annual = predict(res, df, annual=True)["offset"]
    low = df["BonusMalus"] < 100
    assert abs(annual[low].mean() - 0.1) < 0.01
    assert abs(annual[~low].mean() - 0.2) < 0.02
    assert abs(res["feature"].params["log_exposure"] - 1.0) < 0.1


def test_poisson_deviance_basics() -> None:
    y = np.array([0.0, 1.0, 2.0])
    assert poisson_deviance(y, np.array([1e-12, 1.0, 2.0])) < 1e-6
    assert poisson_deviance(y, np.array([0.5, 0.5, 0.5])) > 0

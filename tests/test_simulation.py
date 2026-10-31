"""Pricing simulation mechanics on a small hand-built book (test data only; no model artefacts needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fremtpl2_pricing.config import load_config
from fremtpl2_pricing.pricing.simulation import (
    STRATEGIES,
    Book,
    Scenario,
    evaluate,
    monte_carlo,
    premiums,
    retention,
    scale_to_target,
)

RC = {"base_rate": 0.85, "beta_change": -3.0, "beta_market": -4.0}


def toy_book(n: int = 4_000, seed: int = 0) -> Book:
    rng = np.random.default_rng(seed)
    risk = rng.gamma(2.0, 50.0, n)  # true annual cost per policy
    full = rng.uniform(size=n) < 0.5
    e = np.where(full, 1.0, rng.uniform(0.1, 0.9, n))
    band = np.where(full, "(0.999, 1]", "(0.25, 0.5]")
    df = pd.DataFrame(
        {
            "exposure": e,
            "band": band,
            "large_loss_loading": np.where(full, 0.1, 0.4),
            "claim_count": rng.poisson(risk / 1500 * e),
            "claim_amount_raw": rng.gamma(0.5, 2 * risk * e),
            "cost_glm": risk * rng.lognormal(0, 0.1, n),
            "cost_lgbm": risk * rng.lognormal(0, 0.2, n),
            "freq_glm": risk / 1500,
            "freq_lgbm": risk / 1500,
            "r_flat": 1.0,
            "r_current": np.sqrt(risk),
            "r_glm": risk,
            "r_lgbm": risk * rng.lognormal(0, 0.3, n),
            "r_filable_glm": risk * rng.lognormal(0, 0.05, n),
        }
    )
    for s in ("flat", "current", "glm", "lgbm", "filable_glm"):
        df[f"short_period_{s}"] = np.where(full, 1.0, 1.5)
    raw = {"(0.999, 1]": rng.gamma(0.5, 3000, 500), "(0.25, 0.5]": rng.gamma(0.5, 3000, 500)}
    return Book(df, pd.DataFrame(), raw)


def scenario(**kw: object) -> Scenario:
    base = {
        "expense_ratio": 0.25,
        "profit_margin": 0.05,
        "cap": 0.15,
        "elasticity_scale": 1.0,
        "reference_truth": "blend",
    }
    return Scenario(**{**base, **kw})  # type: ignore[arg-type]


def test_scale_to_target_is_revenue_neutral_and_respects_cap() -> None:
    rng = np.random.default_rng(1)
    base, current = rng.gamma(2, 1, 1000), rng.gamma(2, 1, 1000)
    current *= 500 / current.sum()
    out = scale_to_target(base, 500.0, current, cap=0.10)
    assert out.sum() == pytest.approx(500.0, rel=1e-6)
    assert np.all(out <= current * 1.10 + 1e-9) and np.all(out >= current * 0.90 - 1e-9)
    assert scale_to_target(base, 500.0).sum() == pytest.approx(500.0)
    with pytest.raises(ValueError):
        scale_to_target(base, 10_000.0, current, cap=0.10)


def test_retention_falls_with_price_and_equals_base_at_parity() -> None:
    p = np.array([90.0, 100.0, 110.0])
    r = retention(p, np.full(3, 100.0), np.full(3, 100.0), RC)
    assert r[1] == pytest.approx(0.85)
    assert r[0] > r[1] > r[2]
    assert retention(p, np.full(3, 100.0), np.full(3, 100.0), RC, scale=2.0)[2] < r[2]


def test_all_strategies_collect_the_same_book_premium() -> None:
    book = toy_book()
    prem, current, market, _ = premiums(book, scenario())
    target = market.sum()
    for s in STRATEGIES:
        assert prem[s].sum() == pytest.approx(target, rel=1e-6)
    for s in ("glm_capped", "lgbm_capped", "filable_glm_capped"):
        assert np.all(np.abs(prem[s] / current - 1) <= 0.15 + 1e-6)


def test_flat_rate_suffers_adverse_selection_and_risk_rating_does_not() -> None:
    book = toy_book()
    t, _ = evaluate(book, scenario(), RC)
    t = t.set_index("strategy")
    assert t.loc["flat", "risk_mix_index"] > 1.05  # retained book riskier than the original
    assert t.loc["flat", "retention_lowest_risk_decile"] < t.loc["flat", "retention_highest_risk_decile"]
    assert t.loc["glm", "loss_ratio"] < t.loc["flat", "loss_ratio"]
    assert t.loc["current", "share_up_gt_10pct"] == 0.0


def test_monte_carlo_is_reproducible_and_centred_on_expected_profit() -> None:
    cfg = load_config()
    raw = {
        **cfg.raw,
        "pricing": {**cfg.raw["pricing"], "monte_carlo": {**cfg.raw["pricing"]["monte_carlo"], "runs": 2_000}},
    }
    cfg = type(cfg)(raw=raw, root=cfg.root)
    book, sc = toy_book(), scenario()
    _, ret = evaluate(book, sc, RC)
    prem, *_ = premiums(book, sc)
    a, dists = monte_carlo(book, sc, ret, prem, cfg)
    b, _ = monte_carlo(book, sc, ret, prem, cfg)
    assert a.equals(b)
    exp, _ = evaluate(book, sc, RC)
    glm_mc = a.set_index("strategy").loc["glm"]
    glm_exp = exp.set_index("strategy").loc["glm"]
    assert glm_mc["mean_profit"] == pytest.approx(glm_exp["profit"], abs=4 * glm_mc["sd_profit"] / np.sqrt(2_000) + 1)
    assert glm_mc["tvar_99_5"] >= glm_mc["var_99_5"]
    assert len(dists["glm"]) == 2_000

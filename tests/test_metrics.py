"""Evaluation metrics on toy inputs with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from fremtpl2_pricing.evaluation import metrics as m


def test_gini_extremes() -> None:
    y = np.array([0, 0, 0, 10.0])
    w = np.ones(4)
    perfect = m.gini(y, np.array([1, 2, 3, 4.0]), w)
    reversed_ = m.gini(y, np.array([4, 3, 2, 1.0]), w)
    flat = m.gini(y, np.ones(4), w)  # ties keep original order
    assert perfect == pytest.approx(0.75)
    assert reversed_ == pytest.approx(-0.75)
    assert -1 < flat < 1
    assert m.gini(np.ones(100), np.arange(100.0), np.ones(100)) == pytest.approx(0.0, abs=1e-9)


def test_gini_respects_exposure_weights() -> None:
    # one heavy-exposure record with no loss ranked lowest: most exposure carries no loss
    y, w, s = np.array([0.0, 5.0]), np.array([9.0, 1.0]), np.array([0.0, 1.0])
    x, c = m.lorenz_curve(y, s, w)
    assert x.tolist() == [0.0, 0.9, 1.0] and c.tolist() == [0.0, 0.0, 1.0]
    assert m.gini(y, s, w) == pytest.approx(0.9)


def test_deviances_zero_at_truth_and_positive_elsewhere() -> None:
    y = np.array([0.0, 1.0, 3.0])
    assert m.poisson_deviance(y, np.where(y > 0, y, 1e-12)) == pytest.approx(0.0, abs=1e-9)
    assert m.poisson_deviance(y, np.full(3, y.mean())) > 0
    s = np.array([100.0, 250.0])
    assert m.gamma_deviance(s, s) == pytest.approx(0.0)
    assert m.tweedie_deviance(np.array([0.0, 2.0]), np.array([0.5, 2.0]), 1.5) > 0
    assert m.tweedie_deviance(np.array([2.0]), np.array([2.0]), 1.5) == pytest.approx(0.0, abs=1e-12)


def test_lift_and_double_lift_tables() -> None:
    rng = np.random.default_rng(0)
    w = np.ones(10_000)
    rate = rng.gamma(2, 0.05, 10_000)
    y = rng.poisson(rate).astype(float)
    lt = m.lift_table(y, rate * w, w, n=10)
    assert len(lt) == 10 and lt["exposure"].sum() == pytest.approx(10_000)
    assert lt["predicted_rate"].is_monotonic_increasing
    assert np.isclose(lt["ae"], 1, atol=0.25).all()
    dl = m.double_lift(y, np.full(10_000, rate.mean()), rate, w, n=5)
    assert dl["ae_b"].sub(1).abs().mean() < dl["ae_a"].sub(1).abs().mean()  # the informative model wins


def test_grouped_bootstrap_keeps_groups_and_is_reproducible() -> None:
    groups = np.repeat(np.arange(50), 2)
    seen: list[np.ndarray] = []

    def stat(idx: np.ndarray) -> dict[str, float]:
        seen.append(idx)
        return {"n": float(len(idx))}

    a = m.grouped_bootstrap(groups, stat, n_boot=20, seed=1)
    assert a.loc[0, "estimate"] == 100
    assert all(len(i) == 100 for i in seen)  # groups of 2 drawn whole
    assert all(np.all(groups[i[0::2]] == groups[i[1::2]]) for i in seen[1:])
    b = m.grouped_bootstrap(groups, lambda i: {"s": float(np.sum(i))}, n_boot=20, seed=1)
    c = m.grouped_bootstrap(groups, lambda i: {"s": float(np.sum(i))}, n_boot=20, seed=1)
    assert b.equals(c)

"""Pricing-model metrics: deviances, ordered Lorenz / Gini, lift and double lift, calibration and a grouped
bootstrap. Pure functions on numpy arrays so they can be tested on toy inputs.

Conventions: `y` is the observed outcome per record (claims, or capped loss amount), `mu` the model's expected
outcome for that record (already multiplied by exposure), `w` the exposure.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.sum(term - (y - mu)))


def gamma_deviance(y: np.ndarray, mu: np.ndarray, w: np.ndarray | None = None) -> float:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    d = 2 * ((y - mu) / mu - np.log(y / mu))
    return float(np.sum(d if w is None else np.asarray(w, float) * d))


def tweedie_deviance(y: np.ndarray, mu: np.ndarray, p: float, w: np.ndarray | None = None) -> float:
    """Unit Tweedie deviance for 1 < p < 2 (y may be 0)."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    d = 2 * (
        np.power(np.maximum(y, 0), 2 - p) / ((1 - p) * (2 - p))
        - y * np.power(mu, 1 - p) / (1 - p)
        + np.power(mu, 2 - p) / (2 - p)
    )
    return float(np.sum(d if w is None else np.asarray(w, float) * d))


def lorenz_curve(y: np.ndarray, score: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ordered Lorenz curve: sort by score ascending; cumulative exposure share vs cumulative outcome share."""
    order = np.lexsort((np.arange(len(score)), score))  # stable tie-break
    cw = np.concatenate([[0.0], np.cumsum(np.asarray(w, float)[order])])
    cy = np.concatenate([[0.0], np.cumsum(np.asarray(y, float)[order])])
    return cw / cw[-1], cy / cy[-1]


def gini(y: np.ndarray, score: np.ndarray, w: np.ndarray) -> float:
    """Gini of the ordered Lorenz curve = 2 x area between the diagonal and the curve (Frees et al.).

    0 = no segmentation; higher = the score concentrates losses in the records it ranks riskiest.
    """
    x, c = lorenz_curve(y, score, w)
    return float(1 - 2 * np.trapezoid(c, x))


def quantile_bins(score: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """Bin index 1..n with (approximately) equal exposure per bin, ordered by score."""
    order = np.lexsort((np.arange(len(score)), score))
    cw = np.cumsum(np.asarray(w, float)[order])
    b = np.minimum((cw / cw[-1] * n).astype(int), n - 1) + 1
    out = np.empty(len(score), dtype=int)
    out[order] = b
    return out


def _garwood(actual: np.ndarray, expected: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    a = (1 - level) / 2
    lo = np.where(actual > 0, stats.chi2.ppf(a, 2 * actual) / 2, 0.0) / expected
    hi = stats.chi2.ppf(1 - a, 2 * (actual + 1)) / 2 / expected
    return lo, hi


def lift_table(y: np.ndarray, mu: np.ndarray, w: np.ndarray, n: int = 10, counts: bool = True) -> pd.DataFrame:
    """Simple lift: bins of predicted rate (mu / w) with equal exposure; actual vs expected per bin.

    With counts=True (claim counts) an exact Poisson CI on A/E is added.
    """
    b = quantile_bins(np.asarray(mu) / np.asarray(w), w, n)
    g = pd.DataFrame({"bin": b, "exposure": w, "actual": y, "expected": mu}).groupby("bin").sum().reset_index()
    g["predicted_rate"] = g["expected"] / g["exposure"]
    g["observed_rate"] = g["actual"] / g["exposure"]
    g["ae"] = g["actual"] / g["expected"]
    if counts:
        g["ae_ci_low"], g["ae_ci_high"] = _garwood(g["actual"].to_numpy(), g["expected"].to_numpy())
    return g


def double_lift(y: np.ndarray, mu_a: np.ndarray, mu_b: np.ndarray, w: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Double lift: sort by mu_b / mu_a; per bin, each model's A/E. The better model has A/E closer to 1
    across bins, i.e. it is right where the two disagree."""
    b = quantile_bins(np.asarray(mu_b) / np.asarray(mu_a), w, n)
    g = (
        pd.DataFrame({"bin": b, "exposure": w, "actual": y, "expected_a": mu_a, "expected_b": mu_b})
        .groupby("bin")
        .sum()
        .reset_index()
    )
    g["ratio_b_over_a"] = g["expected_b"] / g["expected_a"]
    g["ae_a"] = g["actual"] / g["expected_a"]
    g["ae_b"] = g["actual"] / g["expected_b"]
    return g


def ae_by_segment(
    seg: pd.Series, y: np.ndarray, preds: dict[str, np.ndarray], w: np.ndarray, counts: bool = True
) -> pd.DataFrame:
    g = pd.DataFrame({"segment": seg.to_numpy(), "exposure": w, "actual": y, **preds}).groupby("segment").sum()
    out = g[["exposure", "actual"]].reset_index()
    for m in preds:
        out[f"ae_{m}"] = (g["actual"] / g[m]).to_numpy()
    if counts:
        for m in preds:
            lo, hi = _garwood(out["actual"].to_numpy(), g[m].to_numpy())
            out[f"ae_{m}_outside_ci"] = (lo > 1) | (hi < 1)
    return out


def grouped_bootstrap(
    groups: np.ndarray,
    statistic: Callable[[np.ndarray], dict[str, float]],
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> pd.DataFrame:
    """Percentile bootstrap resampling whole groups (profile groups, so split records stay together).

    `statistic(idx)` receives row indices of one resample and returns named statistics.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_by_group = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    point = statistic(np.arange(len(groups)))
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_group[i] for i in pick])
        for k, v in statistic(idx).items():
            draws[k].append(v)
    a = (1 - level) / 2
    return pd.DataFrame(
        [
            {
                "statistic": k,
                "estimate": point[k],
                "ci_low": float(np.quantile(v, a)),
                "ci_high": float(np.quantile(v, 1 - a)),
                "boot_sd": float(np.std(v, ddof=1)),
            }
            for k, v in draws.items()
        ]
    )

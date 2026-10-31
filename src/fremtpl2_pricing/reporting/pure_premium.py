"""Write reports/pure_premium.md, CSV tables and figures from PurePremiumResults (generated; do not edit)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fremtpl2_pricing import plotting
from fremtpl2_pricing.config import Config
from fremtpl2_pricing.models.pure_premium import PurePremiumResults
from fremtpl2_pricing.reporting.phase1 import GENERATED, _write_tables, md


def _scatter_figure(res: PurePremiumResults, path: Path) -> Path:
    fig, (ax,) = plotting.new_figure(width=6, height=5)
    s = res.scatter
    ax.hexbin(s["freq_x_sev"], s["tweedie"], xscale="log", yscale="log", gridsize=60, mincnt=1, cmap="Blues")
    lo, hi = s.min().min(), s.max().max()
    ax.plot([lo, hi], [lo, hi], color=plotting.REFERENCE, linestyle="--", linewidth=1)
    ax.set_xlabel("Frequency × severity (annual, capped)")
    ax.set_ylabel("Tweedie GLM (annual, capped, balanced)")
    ax.set_title("Per-policy pure premium, two routes")
    return plotting.save(fig, path)


def _relativity_figure(res: PurePremiumResults, path: Path) -> Path:
    imp = res.implied_relativities
    factors = list(dict.fromkeys(imp["factor"]))
    ncols = 3
    nrows = int(np.ceil(len(factors) / ncols))
    plotting.apply_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), squeeze=False)
    series = [
        ("observed", "observed loss cost", plotting.REFERENCE),
        ("freq_x_sev", "frequency × severity", plotting.SERIES[0]),
        ("tweedie", "Tweedie", plotting.SERIES[1]),
    ]
    for ax, name in zip(axes.flat, factors, strict=False):
        r = imp[imp["factor"] == name]
        x = np.arange(len(r))
        for col, label, color in series:
            style = {"linestyle": ":", "marker": "x"} if col == "observed" else {"marker": "o"}
            ax.plot(x, r[col], color=color, label=label, markersize=4, linewidth=1.5, **style)
        ax.set_title(name)
        ax.set_xticks(x, r["level"], rotation=45, ha="right", fontsize=7)
        ax.axhline(1.0, color=plotting.GRID, linewidth=1)
    for ax in list(axes.flat)[len(factors) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper right")
    fig.suptitle(
        "Implied relativity per level (exposure-weighted mean annual cost / book mean)",
        color=plotting.TEXT_PRIMARY,
        fontsize=11,
    )
    fig.tight_layout()
    return plotting.save(fig, path)


def write_pure_premium_report(res: PurePremiumResults, cfg: Config) -> Path:
    tables = {
        "pp_reconciliation": res.reconciliation,
        "pp_tweedie_agreement": res.agreement,
        "pp_implied_relativities": res.implied_relativities,
        "pp_tweedie_power_sensitivity": res.tweedie_power_sensitivity,
        "pp_variance_decomposition": res.variance_decomposition,
    }
    _write_tables(tables, cfg.path("tables_dir"))
    fig_dir = cfg.path("figures_dir")
    f_sc = _scatter_figure(res, fig_dir / "pp_tweedie_vs_freqsev.png")
    f_rel = _relativity_figure(res, fig_dir / "pp_implied_relativities.png")
    x = res.extras
    text = f"""{GENERATED}
# Pure premium (phase 3)

Generated {date.today().isoformat()}. Training split ({x["n_train"]:,} policies); holdout comparisons follow in phase 4.
Interpretation: `docs/methodology.md` §2c; decisions D-005, D-012, D-014, D-015.

**Pure premium** = annual frequency (full-year cover) × expected severity per claim (full-year claim) ×
(1 + large-loss loading {res.loading:.4f}). Exposure-weighted mean on the training book: **{x["mean_pure_premium"]:,.2f}**
per policy-year. Expenses, profit margin and the short-record loading are added in phase 6.

**Tweedie cross-check**: Tweedie GLM on capped loss cost per exposure-year, var_weight = exposure, same tariff
factors and exposure-band control as the frequency model. Power p = {res.tweedie_power:.3f}, implied by the
severity dispersion (compound Poisson–Gamma: p = (ν+2)/(ν+1), ν = 1/φ). Uncorrected Tweedie A/E on total
capped losses = {x["tweedie_uncorrected_ae"]:.4f}; its intercept is rebalanced so fitted total = actual.

## 1. Reconciliation to actual losses (training book)

{md(res.reconciliation, ",.4g")}

## 2. Where pure-premium variation comes from

Variance of log pure premium across training policies, split into frequency and severity parts:

{md(res.variance_decomposition, ",.4f")}

## 3. Tweedie vs frequency × severity

{md(res.agreement, ",.4f")}

![scatter]({"figures/" + f_sc.name})

Sensitivity of the Tweedie premium shape to the power parameter:

{md(res.tweedie_power_sensitivity, ",.4f")}

Implied relativities per level (both models and the observed loss cost, each divided by its own book mean):

![implied relativities]({"figures/" + f_rel.name})

{md(res.implied_relativities, ",.3f")}
"""
    out = cfg.path("reports_dir") / "pure_premium.md"
    out.write_text(text, encoding="utf-8")
    return out

"""Write reports/explainability_fairness.md, CSV tables and figures (generated; do not edit)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fremtpl2_pricing import plotting
from fremtpl2_pricing.analysis.explain import ExplainResults
from fremtpl2_pricing.analysis.fairness import FairnessResults
from fremtpl2_pricing.config import Config
from fremtpl2_pricing.reporting.phase1 import GENERATED, _write_tables, md


def _relativity_figure(ex: ExplainResults, path: Path) -> Path:
    comp = ex.comparison
    factors = list(dict.fromkeys(comp["factor"]))
    ncols = 4
    nrows = int(np.ceil(len(factors) / ncols))
    plotting.apply_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows), squeeze=False)
    for ax, f in zip(axes.flat, factors, strict=False):
        r = comp[comp["factor"] == f].sort_values("order")
        x = np.arange(len(r))
        ax.errorbar(
            x - 0.12,
            r["glm"],
            yerr=np.vstack([r["glm"] - r["glm_ci_low"], r["glm_ci_high"] - r["glm"]]),
            fmt="o",
            markersize=4,
            color=plotting.SERIES[0],
            elinewidth=1.2,
            capsize=0,
            label="GLM (95% CI)",
        )
        ax.errorbar(
            x + 0.12,
            r["lgbm_shap"],
            yerr=np.vstack(
                [
                    (r["lgbm_shap"] - r["lgbm_fold_min"]).clip(lower=0),
                    (r["lgbm_fold_max"] - r["lgbm_shap"]).clip(lower=0),
                ]
            ),
            fmt="s",
            markersize=4,
            color=plotting.SERIES[1],
            elinewidth=1.2,
            capsize=0,
            label="LightGBM SHAP (fold range)",
        )
        ax.axhline(1.0, color=plotting.REFERENCE, linestyle="--", linewidth=1)
        ax.set_title(f)
        ax.set_xticks(x, r["level"], rotation=45, ha="right", fontsize=7 if len(r) < 15 else 6)
    for ax in list(axes.flat)[len(factors) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper right")
    fig.suptitle(
        "Frequency relativities: GLM coefficients vs LightGBM TreeSHAP (relative to GLM reference level)",
        color=plotting.TEXT_PRIMARY,
        fontsize=11,
    )
    fig.tight_layout()
    return plotting.save(fig, path)


def _stability_figure(ex: ExplainResults, path: Path) -> Path:
    st = ex.stability
    fig, (ax,) = plotting.new_figure(width=8)
    x = np.arange(len(st))
    ax.bar(x - 0.2, st["glm_max_spread"], width=0.38, color=plotting.SERIES[0], label="GLM")
    ax.bar(x + 0.2, st["lgbm_max_spread"], width=0.38, color=plotting.SERIES[1], label="LightGBM SHAP")
    ax.set_xticks(x, st["factor"], rotation=30, ha="right")
    ax.set_ylabel("Largest fold-to-fold spread of a level's relativity (max/min − 1)")
    ax.set_title("Relativity stability across 5 refits")
    ax.legend()
    return plotting.save(fig, path)


def _geography_figure(fa: FairnessResults, path: Path) -> Path:
    fig, axes = plotting.new_figure(width=12, ncols=2)
    for ax, f in zip(axes, ("LogDensity", "Region"), strict=True):
        d = fa.dispersion[fa.dispersion["factor"] == f].reset_index(drop=True)
        if f == "Region":
            d = d.sort_values("obs_index", ignore_index=True)
        x = np.arange(len(d))
        ax.plot(x, d["obs_index"], linestyle=":", marker="x", color=plotting.REFERENCE, label="observed loss cost")
        ax.plot(x, d["glm_index"], marker="o", color=plotting.SERIES[0], label="GLM premium")
        ax.plot(x, d["lgbm_index"], marker="s", color=plotting.SERIES[1], label="LightGBM premium")
        ax.axhline(1, color=plotting.GRID, linewidth=1)
        ax.set_xticks(x, d["level"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Index vs book average (exposure-weighted)")
        ax.set_title("Premium by density band" if f == "LogDensity" else "Premium by region (sorted by observed)")
    axes[0].legend()
    return plotting.save(fig, path)


REGULATORY = """**Regulatory and fair-pricing context (high level).**
- *Gender.* EU Directive 2004/113/EC as interpreted by the CJEU in *Test-Achats* (C-236/09, 2011) requires
  unisex premiums and benefits for contracts concluded from 21 December 2012. freMTPL2 has no gender field, so the
  tariff cannot use gender directly. Rating factors correlated with gender (vehicle power and brand, for example)
  can still act as indirect proxies; this dataset cannot test for that.
- *Other protected and socioeconomic characteristics.* European anti-discrimination law and supervisory
  expectations (for example EIOPA's 2023 supervisory statement on differential pricing practices in non-life
  insurance) ask insurers to justify price differences by risk and cost, to monitor for proxy discrimination,
  and to avoid practices that treat groups of customers unfairly. Geography is the classic example: it carries
  real risk information (traffic density, theft, repair costs), but it also correlates with income and other
  characteristics customers cannot easily change.
- *What this section can and cannot show.* With no protected attributes in the data, it shows how much each
  model's prices depend on the geographic proxies, whether that dependence follows claims experience, and what
  removing it would cost and redistribute. It cannot show discrimination against any protected group."""


def write_explain_fairness_report(ex: ExplainResults, fa: FairnessResults, cfg: Config) -> Path:
    tables = {
        "xf_relativity_comparison": ex.comparison,
        "xf_importance": ex.importance,
        "xf_stability": ex.stability,
        "xf_monotonicity": ex.monotone,
        "xf_premium_volatility": ex.premium_volatility,
        "xf_geography_share": fa.geography_share,
        "xf_dispersion": fa.dispersion,
        "xf_geography_removal_cv": fa.removal_cv,
        "xf_cross_subsidy": fa.cross_subsidy,
    }
    _write_tables(tables, cfg.path("tables_dir"))
    fd = cfg.path("figures_dir")
    f_rel = _relativity_figure(ex, fd / "xf_relativities_glm_vs_shap.png")
    f_st = _stability_figure(ex, fd / "xf_stability.png")
    f_geo = _geography_figure(fa, fd / "xf_geography.png")
    mono_full = ex.monotone[ex.monotone["model"].str.contains("full")]
    mono_folds = (
        ex.monotone.assign(kind=ex.monotone["model"].str.split(" ").str[0])
        .groupby(["kind", "factor"])["adjacent_decreases"]
        .agg(["min", "max"])
        .reset_index()
    )
    comp_cols = [
        "factor",
        "level",
        "glm",
        "glm_ci_low",
        "glm_ci_high",
        "glm_fold_spread",
        "lgbm_shap",
        "lgbm_fold_min",
        "lgbm_fold_max",
        "lgbm_fold_spread",
        "lgbm_over_glm",
        "n_sample",
    ]
    text = f"""{GENERATED}
# Explainability, stability and fairness (phase 5)

Generated {date.today().isoformat()}. Training split only (no holdout). SHAP values are LightGBM's exact
TreeSHAP (`pred_contrib`) on the log-rate scale, on a sample of {ex.extras["shap_sample"]:,} training policies
priced at full-year cover. Interpretation: `docs/methodology.md` §3 and §6; decisions D-019, D-021, D-022.

## 1. GLM relativities vs LightGBM SHAP (frequency)

A SHAP "relativity" for a level is exp(mean SHAP in that level − mean SHAP in the GLM's reference level): the
marginal effect the tree attributes to the factor, with interactions shared out between features. It is
comparable to a GLM relativity, but it is not a tariff cell. The GLM's AgeMalus interaction has no separate SHAP
counterpart; LightGBM spreads that effect over DrivAge and BonusMalus.

![relativities]({"figures/" + f_rel.name})

{md(ex.comparison[comp_cols], ",.3f")}

How much each factor moves the log premium across the book (standard deviation of its log contribution):

{md(ex.importance, ",.4f")}

## 2. Stability across refits

Both models were refitted on each of the five grouped-CV training sets (80% of the training split). The spread is
max/min − 1 of a level's relativity across the five refits:

{md(ex.stability, ",.3f")}

![stability]({"figures/" + f_st.name})

Level averages smooth over many policies. What an individual customer would see is how much *their own* priced
frequency moves between the five refits (max/min − 1 per policy on the SHAP sample):

{md(ex.premium_volatility, ",.4f")}

## 3. Monotonicity

Adjacent decreases in factors a filed tariff would normally rate monotonically (higher BonusMalus or density →
higher price). Full-training models:

{md(mono_full, ",.3f")}

Range across the five refits:

{md(mono_folds, ",.0f")}

## 4. Fairness: geography as a proxy

{REGULATORY}

### 4a. How much of the premium comes from geography

Share of the variance of log pure premium (priced, full-year) attributable to each factor:
cov(contribution, log premium) / var(log premium), so shares sum to 1 across factors (GLM: frequency and severity
terms; LightGBM: frequency and severity SHAP).

{md(fa.geography_share, ",.3f")}

### 4b. Premium dispersion by geography vs claims experience (training book)

Index = exposure-weighted average premium (or observed loss cost incl. large losses) in the level / book average.

![geography]({"figures/" + f_geo.name})

{md(fa.dispersion, ",.3f")}

### 4c. Cost of removing geography (frequency GLM, grouped 5-fold CV)

{md(fa.removal_cv, ",.5f")}

### 4d. Who would pay more or less without geography

Frequency GLM refitted without Region and density, rescaled to the same book premium (severity has no geographic
factor). {fa.extras["share_book_changing_gt_10pct"]:.1%} of exposure would see its premium move by more than 10%.

{md(fa.cross_subsidy, ",.3f")}
"""
    out = cfg.path("reports_dir") / "explainability_fairness.md"
    out.write_text(text, encoding="utf-8")
    return out

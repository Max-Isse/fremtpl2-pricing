"""Write reports/frequency.md, CSV tables and figures from FrequencyResults (generated; do not edit)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fremtpl2_pricing import plotting
from fremtpl2_pricing.config import Config
from fremtpl2_pricing.models.frequency import FrequencyResults
from fremtpl2_pricing.reporting.phase1 import GENERATED, _write_tables, md


def _relativity_figure(rel: pd.DataFrame, path: Path) -> Path:
    factors = list(dict.fromkeys(rel["factor"]))
    ncols = 4
    nrows = int(np.ceil(len(factors) / ncols))
    plotting.apply_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, factors, strict=False):
        r = rel[rel["factor"] == name].sort_values("order")
        x = np.arange(len(r))
        yerr = np.vstack([r["relativity"] - r["ci_low"], r["ci_high"] - r["relativity"]])
        ax.errorbar(
            x,
            r["relativity"],
            yerr=yerr,
            fmt="o",
            markersize=4,
            color=plotting.SERIES[0],
            ecolor=plotting.SERIES[0],
            elinewidth=1.2,
            capsize=0,
        )
        ref = r["is_reference"].to_numpy()
        ax.plot(
            x[ref],
            r["relativity"].to_numpy()[ref],
            "o",
            markersize=6,
            markerfacecolor=plotting.SURFACE,
            markeredgecolor=plotting.SERIES[0],
        )
        ax.axhline(1.0, color=plotting.REFERENCE, linewidth=1, linestyle="--")
        control = bool(r["is_control"].iloc[0])
        ax.set_title(f"{name} (control, not charged)" if control else name)
        ax.set_xticks(x, r["level"], rotation=45, ha="right", fontsize=7)
    for ax in list(axes.flat)[len(factors) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Frequency relativities with 95% cluster-robust CIs (hollow = reference level)",
        color=plotting.TEXT_PRIMARY,
        fontsize=11,
    )
    fig.tight_layout()
    return plotting.save(fig, path)


def _qq_figure(resid: pd.DataFrame, path: Path) -> Path:
    fig, axes = plotting.new_figure(width=9, ncols=2)
    for ax, (col, label, color) in zip(
        axes, [("poisson", "Poisson", plotting.SERIES[0]), ("nb2", "NB2", plotting.SERIES[1])], strict=True
    ):
        r = np.sort(resid[col].to_numpy())
        q = stats.norm.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
        ax.plot(q, r, ".", markersize=2, color=color)
        lim = [min(q.min(), r.min()), max(q.max(), r.max())]
        ax.plot(lim, lim, color=plotting.REFERENCE, linewidth=1, linestyle="--")
        ax.set_title(f"{label}: randomised quantile residuals")
        ax.set_xlabel("Normal quantile")
        ax.set_ylabel("Residual quantile")
    return plotting.save(fig, path)


def _decile_figure(dec: pd.DataFrame, path: Path) -> Path:
    fig, (ax,) = plotting.new_figure(width=7)
    yerr = np.vstack([dec["ae"] - dec["ae_ci_low"], dec["ae_ci_high"] - dec["ae"]])
    ax.errorbar(
        dec["decile"], dec["ae"], yerr=yerr, fmt="o", color=plotting.SERIES[0], markersize=5, elinewidth=1.2, capsize=0
    )
    ax.axhline(1.0, color=plotting.REFERENCE, linewidth=1, linestyle="--")
    ax.set_xticks(dec["decile"])
    ax.set_xlabel("Decile of out-of-fold predicted frequency (1 = lowest)")
    ax.set_ylabel("Actual / expected")
    ax.set_title("Out-of-fold calibration of the Poisson GLM (95% Poisson CI)")
    return plotting.save(fig, path)


def write_frequency_report(res: FrequencyResults, cfg: Config) -> Path:
    tables = {
        "freq_overdispersion": res.overdispersion,
        "freq_pearson_by_exposure": res.pearson_by_exposure,
        "freq_nb2_vs_poisson": res.nb2_summary,
        "freq_se_comparison": res.se_comparison,
        "freq_factor_tests": res.factor_tests,
        "freq_relativities": res.relativities,
        "freq_nb2_relativity_diff": res.nb2_relativity_diff,
        "freq_cv_metrics": res.cv_metrics,
        "freq_cv_by_fold": res.cv_by_fold,
        "freq_oof_deciles": res.oof_deciles,
        "freq_oof_segments": res.oof_segments,
        "freq_sensitivity": res.sensitivity,
        "freq_oof_unmodelled_segments": res.oof_unmodelled,
        "freq_cv_paired": res.cv_paired,
        "freq_full_year_calibration": res.full_year,
    }
    _write_tables(tables, cfg.path("tables_dir"))
    fig_dir = cfg.path("figures_dir")
    f_rel = _relativity_figure(res.relativities, fig_dir / "freq_relativities.png")
    f_qq = _qq_figure(res.residuals, fig_dir / "freq_quantile_residuals.png")
    f_dec = _decile_figure(res.oof_deciles, fig_dir / "freq_oof_deciles.png")

    def link(p: Path) -> str:
        return f"figures/{p.name}"

    x = res.extras
    rel_cols = [
        "factor",
        "level",
        "is_reference",
        "relativity",
        "ci_low",
        "ci_high",
        "exposure",
        "claims",
        "observed_frequency",
    ]
    seg = res.oof_segments
    one = next(c for c in seg.columns if c.startswith("ae_one_factor"))
    seg_view = seg[
        ["factor", "level", "exposure", "actual", "ae_portfolio_mean", one, "ae_glm_offset_only", "ae_glm_poisson"]
    ]
    tariff_rel = res.relativities[~res.relativities["is_control"]][rel_cols]
    control_rel = res.relativities[res.relativities["is_control"]][rel_cols]
    controls = ", ".join(x["controls"]) or "none"
    tariff_names = ", ".join(f.name for f in res.design.tariff_factors)
    text = f"""{GENERATED}
# Frequency model (phase 2)

Generated {date.today().isoformat()}. Training split only: {x["n_train"]:,} policies,
{x["exposure_train"]:,.0f} exposure-years, {x["claims_train"]:,.0f} claims. The holdout is not used.
Interpretation: `docs/methodology.md` §2; decisions D-009 to D-012.

Model (`glm_poisson`): Poisson GLM, log link, offset log(exposure), tariff factors {tariff_names}; control
factor(s) {controls} fitted but pinned at full-year cover when pricing (D-012). `glm_offset_only` is the same
tariff without the control (option A), kept as a comparator.
Base annual frequency for a full-year policy with all factors at reference level: **{res.base_frequency:.4f}**.

## 1. Overdispersion

{md(res.overdispersion)}

Where the Pearson statistic comes from (by exposure band):

{md(res.pearson_by_exposure, ",.3f")}

NB2 vs Poisson (LR test of alpha = 0; p-value uses the 50:50 chi2(0)/chi2(1) boundary mixture):

{md(res.nb2_summary, ",.4g")}

Largest relativity difference between NB2 and Poisson, per factor:

{md(res.nb2_relativity_diff, ",.4f")}

Standard errors relative to the Poisson model-based SE (all non-intercept coefficients):

{md(res.se_comparison, ",.3f")}

## 2. Factor tests

Drop-one tests for model factors, add-one test for candidate factors. `lr_chi2` is the Poisson deviance
change; `quasi_F` scales it by the Pearson dispersion ({x["pearson_dispersion"]:.3f}); `wald_chi2` uses the
cluster-robust covariance (groups = rating profiles).

{md(res.factor_tests, ",.4g")}

## 3. Relativities (cluster-robust 95% CI)

![relativities]({link(f_rel)})

{md(tariff_rel, ",.3f")}

Control factor (fitted, not charged; every policy is priced at the full-year level):

{md(control_rel, ",.3f")}

## 4. Residual checks

![quantile residuals]({link(f_qq)})

Out-of-fold actual/expected by decile of predicted frequency:

{md(res.oof_deciles, ",.4f")}

![deciles]({link(f_dec)})

## 5. Cross-validation against baselines (grouped 5-fold, out-of-fold)

Deviance uses each record's own fitted expectation (controls at the record's level):

{md(res.cv_metrics, ",.5f")}

**What each tariff would charge** (controls pinned at full year, pro rata to exposure), out-of-fold A/E.
Full-year cover is what is quoted, so the full-year row is the headline calibration check:

{md(res.full_year, ",.4f")}

Paired comparison across the same folds (positive = `glm_poisson` has lower deviance):

{md(res.cv_paired, ",.5f")}

By fold:

{md(res.cv_by_fold, ",.5f")}

Out-of-fold actual/expected on segments the model does **not** contain (exact 95% Poisson CI):

{md(res.oof_unmodelled, ",.3f")}

Out-of-fold actual/expected by model factor level. Pooled across folds this is close to 1 for the GLM by
construction (each fold model nearly balances on its own factors), so it mainly shows what the baselines miss:

{md(seg_view, ",.3f")}

## 6. Sensitivity of relativities to data decisions

Absolute change in relativity (|new/base - 1|) when refitting with an alternative choice:

{md(res.sensitivity, ",.4f")}
"""
    out = cfg.path("reports_dir") / "frequency.md"
    out.write_text(text, encoding="utf-8")
    return out

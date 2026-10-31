"""Write reports/severity.md, CSV tables and figures from SeverityResults (generated; do not edit)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

from fremtpl2_pricing import plotting
from fremtpl2_pricing.config import Config
from fremtpl2_pricing.models.severity import SeverityResults
from fremtpl2_pricing.reporting.phase1 import GENERATED, _write_tables, md


def _distribution_figure(res: SeverityResults, path: Path) -> Path:
    fig, axes = plotting.new_figure(width=10, ncols=2)
    a = res.amounts["claim_amount_raw"].to_numpy()
    bins = np.logspace(np.log10(max(a.min(), 1)), np.log10(a.max()), 80)
    axes[0].hist(a, bins=list(bins), color=plotting.SERIES[0])
    axes[0].set_xscale("log")
    axes[0].set_title("Claim amounts (raw, log scale)")
    axes[0].set_xlabel("Claim amount")
    axes[0].set_ylabel("Claims")
    thr = res.amounts["claim_amount"].max()
    axes[0].axvline(thr, color=plotting.REFERENCE, linestyle="--", linewidth=1)
    axes[0].annotate(
        "large-loss cap", (thr, axes[0].get_ylim()[1] * 0.9), fontsize=7, color=plotting.TEXT_SECONDARY, ha="right"
    )
    vm = res.variance_mean
    axes[1].loglog(vm["mean_fitted"], vm["phi_v"], "o", color=plotting.SERIES[0], markersize=5)
    vp = res.variance_power.iloc[0]
    x = np.linspace(vm["mean_fitted"].min(), vm["mean_fitted"].max(), 50)
    ref = vm["phi_v"].median() * (x / vm["mean_fitted"].median()) ** 2
    axes[1].loglog(x, ref, color=plotting.REFERENCE, linestyle="--", linewidth=1, label="Gamma: variance ∝ mean²")
    axes[1].set_title(f"Variance vs mean (slope {vp['variance_power']:.2f})")
    axes[1].xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    axes[1].xaxis.set_minor_formatter(mticker.NullFormatter())
    axes[1].set_xlabel("Fitted mean severity (bin)")
    axes[1].set_ylabel("Mean of w·(y − μ)²")
    axes[1].legend()
    return plotting.save(fig, path)


def _qq_figure(res: SeverityResults, path: Path) -> Path:
    fig, (ax,) = plotting.new_figure(width=6)
    r = np.sort(res.residuals["quantile_residual"].to_numpy())
    q = stats.norm.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
    ax.plot(q, r, ".", markersize=2, color=plotting.SERIES[0])
    lim = [min(q.min(), r.min()), max(q.max(), r.max())]
    ax.plot(lim, lim, color=plotting.REFERENCE, linewidth=1, linestyle="--")
    ax.set_title("Gamma quantile residuals")
    ax.set_xlabel("Normal quantile")
    ax.set_ylabel("Residual quantile")
    return plotting.save(fig, path)


def write_severity_report(res: SeverityResults, cfg: Config) -> Path:
    tables = {
        "sev_distribution": res.distribution,
        "sev_flat_amounts": res.flat_amounts,
        "sev_selection_path": res.selection_path,
        "sev_cv_path": res.cv_path,
        "sev_factor_tests_full": res.factor_tests_full,
        "sev_factor_tests_final": res.factor_tests_final,
        "sev_relativities": res.relativities,
        "sev_by_exposure_band": res.by_exposure_band,
        "sev_variance_mean": res.variance_mean,
        "sev_variance_power": res.variance_power,
        "sev_cv_metrics": res.cv_metrics,
        "sev_cv_by_fold": res.cv_by_fold,
        "freq_vs_sev_factors": res.freq_vs_sev,
    }
    _write_tables(tables, cfg.path("tables_dir"))
    fig_dir = cfg.path("figures_dir")
    f_dist = _distribution_figure(res, fig_dir / "sev_distribution_variance.png")
    f_qq = _qq_figure(res, fig_dir / "sev_quantile_residuals.png")
    x = res.extras
    resid = res.residuals["quantile_residual"]
    rel_cols = [
        "factor",
        "level",
        "is_reference",
        "is_control",
        "relativity",
        "ci_low",
        "ci_high",
        "claims",
        "amount",
        "observed_severity",
    ]
    rel = res.relativities[rel_cols]
    text = f"""{GENERATED}
# Severity model (phase 3)

Generated {date.today().isoformat()}. Training split only: {x["n_policies"]:,} policies with at least one claim,
{x["n_claims"]:,.0f} claims ({x["n_policies"] / x["n_freq_policies"]:.1%} of the {x["n_freq_policies"]:,} policies
in the frequency model). Response: average *capped* claim amount per policy, var_weight = number of claims.
Interpretation: `docs/methodology.md` §2b; decisions D-013, D-014.

Selected model: Gamma GLM, log link, tariff factor(s) **{", ".join(x["selected"]) or "none"}**, control
{", ".join(x["controls"])} (fitted, pinned at full year when pricing). Pearson dispersion φ = {res.dispersion:.3f}.
Base severity (full-year claim, reference levels): **{res.base_severity:,.0f}**.

## 1. Distribution

{md(res.distribution, ",.4g")}

![distribution and variance]({"figures/" + f_dist.name})

Amounts that recur in at least {cfg.section("severity")["flat_amount_min_share"]:.0%} of claims (flat-rate settlements):

{md(res.flat_amounts, ",.4f")}

## 2. Severity by exposure band of the claiming record

{md(res.by_exposure_band, ",.4g")}

## 3. Factor selection

Drop-one quasi-F tests in the full starting model:

{md(res.factor_tests_full, ",.4g")}

Backward elimination path (each step drops the least significant tariff factor; `f_test_rule` shows what a
pure p < {cfg.section("severity")["significance"]} rule would do):

{md(res.selection_path, ",.4g")}

Grouped 5-fold CV deviance along that path. The chosen model is the smallest whose paired CV deviance difference
to the best model is within one paired standard error:

{md(res.cv_path, ",.5f")}

## 4. Relativities (cluster-robust 95% CI)

{md(rel, ",.3f")}

## 5. Gamma assumption checks

Variance vs mean (log-log slope of binned w·(y−μ)² on fitted μ, full starting model):

{md(res.variance_power, ",.3f")}

Quantile residuals (should be N(0,1) if the Gamma distribution held): mean {resid.mean():.3f},
SD {resid.std():.3f}, skewness {stats.skew(resid):.3f}.

![quantile residuals]({"figures/" + f_qq.name})

## 6. Cross-validation (grouped 5-fold, out-of-fold)

{md(res.cv_metrics, ",.5f")}

## 7. Which factors matter for frequency vs severity

Drop-one quasi-F tests in each full model (frequency: fine bands, {x["n_freq_policies"]:,} policies;
severity: coarse bands, {x["n_policies"]:,} policies with claims):

{md(res.freq_vs_sev, ",.4g")}
"""
    out = cfg.path("reports_dir") / "severity.md"
    out.write_text(text, encoding="utf-8")
    return out

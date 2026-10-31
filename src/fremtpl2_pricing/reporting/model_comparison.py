"""Write reports/model_comparison.md, CSV tables and figures from ComparisonResults (generated; do not edit)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fremtpl2_pricing import plotting
from fremtpl2_pricing.config import Config
from fremtpl2_pricing.evaluation.compare import ComparisonResults
from fremtpl2_pricing.reporting.phase1 import GENERATED, _write_tables, md

LABELS = {
    "glm_freq_x_sev": "GLM freq × sev",
    "lgbm_freq_x_sev": "LightGBM freq × sev",
    "lgbm_tweedie": "LightGBM Tweedie",
    "glm": "GLM",
    "lgbm": "LightGBM",
    "rf": "Random forest",
}


def _lorenz_figure(res: ComparisonResults, path: Path) -> Path:
    fig, (ax,) = plotting.new_figure(width=5.5, height=5)
    for i, m in enumerate(("glm_freq_x_sev", "lgbm_freq_x_sev", "lgbm_tweedie")):
        x, c = res.lorenz[m]
        step = max(len(x) // 2000, 1)
        g = res.pure_premium.set_index("model").loc[m, "gini"]
        ax.plot(x[::step], c[::step], color=plotting.SERIES[i], linewidth=1.8, label=f"{LABELS[m]} (Gini {g:.3f})")
    ax.plot([0, 1], [0, 1], color=plotting.REFERENCE, linestyle="--", linewidth=1, label="no segmentation")
    ax.set_xlabel("Cumulative share of exposure (lowest premium first)")
    ax.set_ylabel("Cumulative share of capped losses")
    ax.set_title("Ordered Lorenz curves, holdout")
    ax.legend(loc="upper left")
    return plotting.save(fig, path)


def _calibration_figure(res: ComparisonResults, path: Path) -> Path:
    fig, axes = plotting.new_figure(width=11, ncols=2)
    for i, m in enumerate(("glm", "lgbm", "rf")):
        t = res.freq_lift[m]
        off = (i - 1) * 0.22
        yerr = np.vstack([t["ae"] - t["ae_ci_low"], t["ae_ci_high"] - t["ae"]])
        axes[0].errorbar(
            t["bin"] + off,
            t["ae"],
            yerr=yerr,
            fmt="o",
            markersize=4,
            color=plotting.SERIES[i],
            elinewidth=1.1,
            capsize=0,
            label=LABELS[m],
        )
    axes[0].axhline(1, color=plotting.REFERENCE, linestyle="--", linewidth=1)
    axes[0].set_xticks(range(1, len(res.freq_lift["glm"]) + 1))
    axes[0].set_xlabel("Decile of predicted frequency (equal exposure)")
    axes[0].set_ylabel("Actual / expected claims")
    axes[0].set_title("Frequency calibration by decile (95% CI)")
    axes[0].legend()
    dl = res.double_lift
    axes[1].plot(dl["bin"], dl["ae_a"], marker="o", color=plotting.SERIES[0], label="GLM freq × sev")
    axes[1].plot(dl["bin"], dl["ae_b"], marker="o", color=plotting.SERIES[1], label="LightGBM freq × sev")
    axes[1].axhline(1, color=plotting.REFERENCE, linestyle="--", linewidth=1)
    axes[1].set_xticks(dl["bin"])
    axes[1].set_xlabel("Decile of LightGBM / GLM premium ratio")
    axes[1].set_ylabel("Actual / expected capped losses")
    axes[1].set_title("Double lift: who is right where they disagree")
    axes[1].legend()
    return plotting.save(fig, path)


def write_comparison_report(res: ComparisonResults, cfg: Config) -> Path:
    tables: dict[str, pd.DataFrame] = {
        "cmp_frequency": res.frequency,
        "cmp_severity": res.severity,
        "cmp_pure_premium": res.pure_premium,
        "cmp_paired_differences": res.paired,
        "cmp_full_year": res.full_year,
        "cmp_double_lift": res.double_lift,
        "cmp_segments_frequency": res.segments_freq,
        "cmp_segments_pure_premium": res.segments_pp,
        "cmp_disagreement": res.disagreement,
        "cmp_disagreement_segments": res.disagreement_segments,
        **{f"cmp_freq_lift_{m}": t for m, t in res.freq_lift.items()},
        **{f"cmp_pp_lift_{m}": t for m, t in res.pp_lift.items()},
    }
    _write_tables(tables, cfg.path("tables_dir"))
    fig_dir = cfg.path("figures_dir")
    f_lor = _lorenz_figure(res, fig_dir / "cmp_lorenz.png")
    f_cal = _calibration_figure(res, fig_dir / "cmp_calibration_double_lift.png")
    x = res.extras
    tuning_dir = cfg.path("tables_dir")
    tuning = []
    for name in ("lgbm_poisson_frequency", "lgbm_gamma_severity", "lgbm_tweedie_pure_premium"):
        p = tuning_dir / f"tuning_{name}.csv"
        if p.exists():
            t = pd.read_csv(p)
            t.insert(0, "model", name)
            tuning.append(t)
    tuning_md = md(pd.concat(tuning, ignore_index=True), ",.5g") if tuning else "(tuning tables not found)"
    seg_p = res.segments_pp.drop(columns=[c for c in res.segments_pp.columns if c.endswith("_outside_ci")])
    text = f"""{GENERATED}
# Model comparison on the holdout (phase 4)

Generated {date.today().isoformat()}. Holdout: {x["n_holdout"]:,} policies, {x["exposure"]:,.0f} exposure-years,
{x["claims"]:,.0f} claims, {x["n_holdout_claim_policies"]:,} policies with claims. The holdout was not used for any
model or tuning decision. CIs are 95% percentile intervals from {x["n_boot"]} grouped bootstrap replicates
(resampling whole rating-profile groups). Interpretation: `docs/methodology.md` §3–4; decisions D-018, D-019.

**Views.** *Fitted* = controls at each record's own exposure band (deviance, calibration, segment A/E).
*Priced* = controls pinned at full year, i.e. what the tariff charges pro rata (Gini, full-year A/E, disagreement).

## 1. Frequency (claim counts)

Mean Poisson deviance per policy (lower is better) and Gini of claims ordered by priced annual frequency:

{md(res.frequency, ",.5f")}

## 2. Severity (average capped claim, policies with claims)

Mean Gamma deviance per claim and Gini of claim cost ordered by predicted severity:

{md(res.severity, ",.5f")}

## 3. Pure premium (capped loss cost)

Mean Tweedie deviance (p = {x["tweedie_power"]:.3f}) per exposure-year and Gini of capped losses ordered by priced
pure premium: `gini` uses all holdout records; `gini_full_year` only full-year records, i.e. the business that is
actually quoted and the only records on which a full-year premium can be validated. Several segments are almost
never seen at full year (see §6), so the all-records Gini partly rewards how a model spreads short-record cost.

{md(res.pure_premium, ",.5f")}

![Lorenz]({"figures/" + f_lor.name})

Paired bootstrap differences (CI excluding 0 = a real difference on this holdout):

{md(res.paired, ",.5f")}

## 4. Calibration

Full-year calibration of what each tariff charges, and the book-level shortfall of pricing every record pro rata
at the full-year rate (the short-record effect, D-012/D-014):

{md(res.full_year, ",.4f")}

![calibration and double lift]({"figures/" + f_cal.name})

Frequency calibration by decile of predicted frequency (equal exposure per decile):

{md(res.freq_lift["glm"].assign(model="glm"), ",.4f")}

{md(res.freq_lift["lgbm"].assign(model="lgbm"), ",.4f")}

Double lift, GLM vs LightGBM pure premium (deciles of the LightGBM / GLM ratio):

{md(res.double_lift, ",.4f")}

## 5. Actual / expected by segment (fitted, exposure shown)

Frequency (claim counts; `outside_ci` = A/E's exact 95% Poisson interval excludes 1):

{md(res.segments_freq, ",.3f")}

Pure premium (capped losses):

{md(seg_p, ",.3f")}

## 6. Where GLM and LightGBM disagree (priced pure premium)

{md(res.disagreement, ",.4f")}

Segments with the largest average gap (LightGBM / GLM premium ratio; actual capped cost shown for reference):

{md(res.disagreement_segments.head(15), ",.3f")}

## 7. LightGBM tuning (grouped 5-fold CV on the training split)

{tuning_md}
"""
    out = cfg.path("reports_dir") / "model_comparison.md"
    out.write_text(text, encoding="utf-8")
    return out

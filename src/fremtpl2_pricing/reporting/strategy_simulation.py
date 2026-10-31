"""Write reports/strategy_simulation.md, CSV tables (SYNTHETIC_* for simulated outputs) and figures."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from fremtpl2_pricing import plotting
from fremtpl2_pricing.config import Config
from fremtpl2_pricing.pricing.simulation import STRATEGIES, SimulationResults
from fremtpl2_pricing.reporting.phase1 import GENERATED, _write_tables, md

KEY = ["current", "glm_capped", "lgbm_capped"]


def _profit_figure(res: SimulationResults, path: Path) -> Path:
    fig, (ax,) = plotting.new_figure(width=8, height=4)
    for i, s in enumerate(STRATEGIES):
        d = res.mc_dists[s] / 1e6
        lo, p05, med, hi = np.quantile(d, [0.005, 0.05, 0.5, 0.995])
        ax.plot([lo, hi], [i, i], color=plotting.REFERENCE, linewidth=1)
        ax.plot([p05, np.quantile(d, 0.95)], [i, i], color=plotting.SERIES[0], linewidth=4, solid_capstyle="butt")
        ax.plot(med, i, "o", color=plotting.TEXT_PRIMARY, markersize=5)
    ax.axvline(0, color=plotting.SERIES[1], linewidth=1, linestyle="--")
    ax.set_yticks(range(len(STRATEGIES)), STRATEGIES)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_xlabel("Simulated annual profit (millions; median, 5–95% bar, 0.5–99.5% line)")
    ax.set_title(f"SYNTHETIC Monte Carlo profit distribution ({res.extras['runs']:,} runs)")
    return plotting.save(fig, path)


def _retention_figure(res: SimulationResults, path: Path) -> Path:
    fig, (ax,) = plotting.new_figure(width=7)
    d = res.deciles
    for i, s in enumerate(KEY):
        ax.plot(d["risk_decile"], d[s], marker="o", color=plotting.SERIES[i], label=s)
    ax.plot(d["risk_decile"], d["flat"], linestyle=":", marker="x", color=plotting.REFERENCE, label="flat")
    ax.set_xticks(d["risk_decile"])
    ax.set_xlabel("Decile of true risk (1 = lowest)")
    ax.set_ylabel("Expected retention (ASSUMED elasticity)")
    ax.set_title("Who stays: adverse selection by strategy")
    ax.legend()
    return plotting.save(fig, path)


def write_simulation_report(res: SimulationResults, cfg: Config) -> Path:
    pc = cfg.section("pricing")
    tables = {
        "sim_duration_terms": res.terms,
        "SYNTHETIC_sim_strategies": res.base,
        "SYNTHETIC_sim_retention_by_decile": res.deciles,
        "SYNTHETIC_sim_monte_carlo": res.mc,
        "SYNTHETIC_sim_sensitivity": res.sensitivity,
    }
    _write_tables(tables, cfg.path("tables_dir"))
    fd = cfg.path("figures_dir")
    f_p = _profit_figure(res, fd / "SYNTHETIC_sim_profit.png")
    f_r = _retention_figure(res, fd / "SYNTHETIC_sim_retention.png")
    sc = res.scenario
    x = res.extras
    rc = pc["retention"]
    headline = res.base[
        [
            "strategy",
            "retention_exposure",
            "premium",
            "loss_ratio",
            "profit",
            "profit_margin",
            "realised_loss_ratio_actual_losses",
            "realised_profit_actual_losses",
            "risk_mix_index",
        ]
    ]
    customer = res.base[
        [
            "strategy",
            "mean_abs_change_vs_current",
            "share_up_gt_10pct",
            "share_up_gt_25pct",
            "share_down_gt_10pct",
            "retention_lowest_risk_decile",
            "retention_highest_risk_decile",
            "premium_to_true_cost_p10",
            "premium_to_true_cost_p90",
        ]
    ]
    sens = res.sensitivity.pivot_table(
        index=["parameter", "value"], columns="strategy", values="realised_profit_actual_losses", sort=False
    )[STRATEGIES]
    sens_exp = res.sensitivity.pivot_table(
        index=["parameter", "value"], columns="strategy", values="profit", sort=False
    )[STRATEGIES]
    text = f"""{GENERATED}
# Pricing strategy simulation (phase 6)

Generated {date.today().isoformat()}. **Everything below that depends on customer behaviour is SYNTHETIC and rests on
ASSUMED parameters** (no price, lapse or competitor data exist for freMTPL2). Figures are estimates under stated
assumptions, not achievable savings. Interpretation: `docs/methodology.md` §7; decisions D-023 to D-026.

**Book.** The {x["n_policies"]:,} holdout policies ({x["exposure"]:,.0f} exposure-years, actual losses incl. large losses
{x["actual_raw_losses"]:,.0f}) renew for one year with their observed exposure. The holdout is independent of every
fitted model.

**Base-case assumptions** (`pricing` in `configs/config.yaml`): expense ratio {sc.expense_ratio:.0%}, profit margin
{sc.profit_margin:.0%}, current tariff = {pc["current_tariff"]}, caps ±{sc.cap:.0%}, reference truth = {sc.reference_truth},
retention logit = logit({rc["base_rate"]}) + {rc["beta_change"]}·log(new/current) + {rc["beta_market"]}·log(new/market).

**Comparison basis.** Every strategy is scaled to collect the same total premium from the full book (the true
expected cost / (1 − expenses − margin)), so strategies differ only in *how* premium is spread. Two profit views:
*expected* (claims = model-based true cost, D-024) and *realised* (claims = the actual holdout losses of the
policies, weighted by their retention probability: model-free, but noisy).

## 1. Duration terms (training split; D-012, D-014, D-017)

{md(res.terms, ",.4f")}

## 2. Strategy outcomes (base case)

{md(headline, ",.4f")}

Customer impact and selection (price change vs current tariff; premium / true cost at the 10th and 90th
percentile of policies):

{md(customer, ",.4f")}

Expected retention by decile of true risk (ASSUMED elasticity):

{md(res.deciles, ",.3f")}

![retention]({"figures/" + f_r.name})

## 3. SYNTHETIC Monte Carlo ({x["runs"]:,} runs)

Claim counts per exposure band ~ Poisson(systemic shock × retained expected claims); severities resampled from
training claim amounts incl. large losses (D-016); systemic shock lognormal with sd
{pc["monte_carlo"]["systemic_sd"]:.0%}. Profit = premium × (1 − expense ratio) − claims.

{md(res.mc, ",.4f")}

![profit]({"figures/" + f_p.name})

## 4. Sensitivity (one assumption at a time)

Realised profit on actual holdout losses:

{md(sens.reset_index(), ",.0f")}

Expected profit under the reference truth:

{md(sens_exp.reset_index(), ",.0f")}

Full sensitivity table: `reports/tables/SYNTHETIC_sim_sensitivity.csv`.
"""
    out = cfg.path("reports_dir") / "strategy_simulation.md"
    out.write_text(text, encoding="utf-8")
    return out

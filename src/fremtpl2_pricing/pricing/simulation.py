"""Phase 6: pricing-strategy simulation on the holdout book (one renewal year).

Every number that is an assumption (expense ratio, margin, retention elasticities, systemic shock) comes from
`pricing` in configs/config.yaml and is varied in the sensitivity analysis. Simulated outputs are SYNTHETIC.

Mechanics (all per holdout policy i, exposure e_i, exposure band b):
* duration terms D = short-period factor SP_b x (1 + large-loss loading LL), estimated on the TRAINING split:
  SP_b = the strategy model's cost at the record's own exposure band / its cost priced at full year (each model
  splits risk between full-year rate and duration differently); LL = excess over the large-loss cap / capped
  losses, pooled for full-year vs part-year records (resolves D-012, D-014, D-017).
* strategy premium P_i = k x r_i x e_i x D, r_i = the strategy's full-year annual rate, k such that the whole
  book pays the same target T = sum(true cost) / (1 - expense - margin): strategies differ only in how they
  spread premium (revenue-neutral comparison). Capped strategies solve for k with the cap applied.
* "true" cost C_i: blend (geometric mean) of the GLM and LightGBM fitted costs at the record's own band, x (1+LL_b),
  calibrated so the book total equals actual holdout losses incl. large losses. Competitor (market) premium =
  C_i / (1 - expense - margin). Sensitivity: truth = GLM or LightGBM alone.
* retention p_i = logistic(logit(base) + beta_change log(P_i / current_i) + beta_market log(P_i / market_i)).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from fremtpl2_pricing.config import Config
from fremtpl2_pricing.data.db import query_df
from fremtpl2_pricing.features import GLMDesign
from fremtpl2_pricing.models import glm
from fremtpl2_pricing.models.benchmarks import LoadedBenchmarks
from fremtpl2_pricing.models.pure_premium import PurePremiumModel, load_training

log = logging.getLogger(__name__)

STRATEGIES = ["flat", "current", "glm", "lgbm", "glm_capped", "lgbm_capped", "filable_glm_capped"]
CAPPED = {"glm_capped": "glm", "lgbm_capped": "lgbm", "filable_glm_capped": "filable_glm"}


def _art(cfg: Config, name: str) -> dict[str, Any]:
    return json.loads((cfg.path("artifacts_dir") / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ book construction


def short_period_factors(band: np.ndarray, fitted: np.ndarray, priced: np.ndarray, e: np.ndarray) -> pd.Series:
    """Per exposure band: model cost at the record's own band / cost priced at full year (training split)."""
    g = pd.DataFrame({"band": band, "fitted": fitted * e, "priced": priced * e}).groupby("band").sum()
    return g["fitted"] / g["priced"]


def large_loss_loadings(con: duckdb.DuckDBPyConnection, band_of: Any, full_year_band: str) -> pd.Series:
    """Excess over the large-loss cap / capped losses, pooled into full-year vs part-year records (training).

    Pooled rather than per band: a handful of very large claims would otherwise set a band's loading (one band
    reaches 124% from a few claims). D-017.
    """
    claims = query_df(con, "SELECT exposure, claim_amount, large_loss_excess FROM claims_clean WHERE NOT is_holdout")
    group = np.where(band_of(claims) == full_year_band, "full-year", "part-year")
    g = claims.assign(group=group).groupby("group")[["claim_amount", "large_loss_excess"]].sum()
    return g["large_loss_excess"] / g["claim_amount"]


@dataclass
class Book:
    df: pd.DataFrame  # holdout policies with rates per strategy and truth components
    terms: pd.DataFrame
    raw_amounts: dict[str, np.ndarray]  # training raw claim amounts by band (Monte Carlo severities)


def build_book(con: duckdb.DuckDBPyConnection, cfg: Config) -> Book:
    pc = cfg.section("pricing")
    art = cfg.path("artifacts_dir")
    pa = _art(cfg, "pure_premium.json")
    ppm = PurePremiumModel.from_artifacts(
        art / "frequency_glm.json", art / "severity_glm.json", float(pa["large_loss_loading"])
    )
    band_spec = ppm.freq_design.factor("ExposureBand")

    def band_of(df: pd.DataFrame) -> np.ndarray:
        return band_spec.bin(df).to_numpy()

    train = load_training(con)
    hold = load_training(con, holdout=True)
    bench = LoadedBenchmarks.load(cfg)
    et = train["exposure"].to_numpy()
    y = train["claim_count"].to_numpy(float)

    def glm_cost(freq_design: GLMDesign, freq_params: Any, df: pd.DataFrame, pricing: bool) -> np.ndarray:
        return glm.predict_log_linear(freq_design, freq_params, df, pricing=pricing) * glm.predict_log_linear(
            ppm.sev_design, ppm.sev_params, df, pricing=pricing
        )

    # strategy models: current (BonusMalus only), GLM, LightGBM, filable GLM (own design, D-026)
    d1 = GLMDesign.fit(train, cfg, factors=["BonusMalus"])
    r1 = glm.fit_poisson(d1.matrix(train), y, et)
    fc = pc["filable_glm"]
    keep = [f.name for f in ppm.freq_design.factors if f.name not in fc["drop_factors"]]
    dfil = GLMDesign.fit(train, cfg, factors=keep, bands={"BonusMalus": fc["bonusmalus_bands"]})
    rfil = glm.fit_poisson(dfil.matrix(train), y, et)

    # short-period factors per model (each model's own split between full-year rate and duration effect)
    bt = band_of(train)
    btr, bpr = bench.predict(train, pricing=False), bench.predict(train, pricing=True)
    sp = pd.DataFrame(
        {
            "glm": short_period_factors(
                bt,
                glm_cost(ppm.freq_design, ppm.freq_params, train, False),
                glm_cost(ppm.freq_design, ppm.freq_params, train, True),
                et,
            ),
            "lgbm": short_period_factors(
                bt, btr["lgbm_frequency"] * btr["lgbm_severity"], bpr["lgbm_frequency"] * bpr["lgbm_severity"], et
            ),
            "filable_glm": short_period_factors(
                bt, glm_cost(dfil, rfil.params, train, False), glm_cost(dfil, rfil.params, train, True), et
            ),
        }
    )
    sp["flat"] = sp["glm"]  # rating structures without a duration control use the GLM's short-period scale
    sp["current"] = sp["glm"]
    ll = large_loss_loadings(con, band_of, band_spec.reference)
    terms = sp.add_prefix("short_period_").reset_index().rename(columns={"index": "band"})
    terms["large_loss_loading"] = np.where(terms["band"] == band_spec.reference, ll["full-year"], ll["part-year"])

    df = hold[
        [
            "IDpol",
            "group_id",
            "exposure",
            "claim_count",
            "claim_amount",
            "claim_amount_raw",
            "BonusMalus",
            "DrivAge",
            "Density",
            "Region",
            "VehBrand",
            "VehAge",
        ]
    ].copy()
    df["band"] = band_of(hold)
    df = df.merge(terms, on="band", how="left")

    # full-year annual capped rates per strategy
    df["r_flat"] = 1.0
    df["r_current"] = glm.predict_log_linear(d1, r1.params, hold, pricing=False)  # severity flat
    df["r_glm"] = glm_cost(ppm.freq_design, ppm.freq_params, hold, True)
    b = bench.predict(hold, pricing=True)
    df["r_lgbm"] = b["lgbm_frequency"] * b["lgbm_severity"]
    df["r_filable_glm"] = glm_cost(dfil, rfil.params, hold, True)

    # truth components: fitted (own band) capped cost per year for both models, frequency for Monte Carlo
    bf = bench.predict(hold, pricing=False)
    df["cost_glm"] = glm_cost(ppm.freq_design, ppm.freq_params, hold, False)
    df["cost_lgbm"] = bf["lgbm_frequency"] * bf["lgbm_severity"]
    df["freq_glm"] = glm.predict_log_linear(ppm.freq_design, ppm.freq_params, hold, pricing=False)
    df["freq_lgbm"] = bf["lgbm_frequency"]

    claims = query_df(con, "SELECT exposure, claim_amount_raw FROM claims_clean WHERE NOT is_holdout")
    cb = band_of(claims)
    raw = {lvl: claims["claim_amount_raw"].to_numpy()[cb == lvl] for lvl in band_spec.levels}
    log.info("book: %s holdout policies; filable GLM factors %s", len(df), keep)
    return Book(df, terms, raw)


def truth(book: Book, which: str) -> tuple[np.ndarray, np.ndarray]:
    """True expected cost (incl. large losses) and claim frequency per policy, calibrated to actual holdout losses
    (cost) and actual claim counts (frequency)."""
    df = book.df
    if which == "blend":
        cost = np.sqrt(df["cost_glm"].to_numpy() * df["cost_lgbm"].to_numpy())
        freq = np.sqrt(df["freq_glm"].to_numpy() * df["freq_lgbm"].to_numpy())
    else:
        cost, freq = df[f"cost_{which}"].to_numpy(), df[f"freq_{which}"].to_numpy()
    e = df["exposure"].to_numpy()
    c = cost * e * (1 + df["large_loss_loading"].to_numpy())
    c *= df["claim_amount_raw"].sum() / c.sum()
    f = freq * e
    f *= df["claim_count"].sum() / f.sum()
    return c, f


# ------------------------------------------------------------------ strategies and retention


def scale_to_target(
    base: np.ndarray, target: float, current: np.ndarray | None = None, cap: float | None = None
) -> np.ndarray:
    """k x base summing to target; with a cap, premiums are clipped to current x (1 +- cap) and k is solved by
    bisection (the cap redistributes premium, it does not create or destroy it)."""
    if cap is None or current is None:
        return base * target / base.sum()
    lo, hi = current * (1 - cap), current * (1 + cap)
    if not lo.sum() <= target <= hi.sum():
        raise ValueError("target premium not reachable within the cap")
    a, b = 1e-6 * target / base.sum(), 1e6 * target / base.sum()
    for _ in range(200):
        k = np.sqrt(a * b)
        total = np.clip(k * base, lo, hi).sum()
        a, b = (k, b) if total < target else (a, k)
    return np.clip(np.sqrt(a * b) * base, lo, hi)


def retention(
    premium: np.ndarray, current: np.ndarray, market: np.ndarray, rc: dict[str, float], scale: float = 1.0
) -> np.ndarray:
    base = np.log(rc["base_rate"] / (1 - rc["base_rate"]))
    z = base + scale * (rc["beta_change"] * np.log(premium / current) + rc["beta_market"] * np.log(premium / market))
    return 1 / (1 + np.exp(-z))


@dataclass
class Scenario:
    expense_ratio: float
    profit_margin: float
    cap: float
    elasticity_scale: float
    reference_truth: str
    actual_expense_ratio: float | None = None  # expenses actually incurred; None = as priced

    @property
    def incurred_expense_ratio(self) -> float:
        return self.expense_ratio if self.actual_expense_ratio is None else self.actual_expense_ratio


def premiums(book: Book, sc: Scenario) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    df = book.df
    e = df["exposure"].to_numpy()
    ll = 1 + df["large_loss_loading"].to_numpy()
    cost, freq = truth(book, sc.reference_truth)
    loading = 1 - sc.expense_ratio - sc.profit_margin
    target = cost.sum() / loading
    market = cost / loading
    base = {
        s: df[f"r_{s}"].to_numpy() * e * df[f"short_period_{s}"].to_numpy() * ll
        for s in ("flat", "current", "glm", "lgbm", "filable_glm")
    }
    current = scale_to_target(base["current"], target)
    out = {
        "flat": scale_to_target(base["flat"], target),
        "current": current,
        "glm": scale_to_target(base["glm"], target),
        "lgbm": scale_to_target(base["lgbm"], target),
    }
    for s, src in CAPPED.items():
        out[s] = scale_to_target(base[src], target, current, sc.cap)
    return out, current, market, np.column_stack([cost, freq])


def evaluate(book: Book, sc: Scenario, rc: dict[str, float]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    prem, current, market, cf = premiums(book, sc)
    cost = cf[:, 0]
    df = book.df
    e = df["exposure"].to_numpy()
    actual = df["claim_amount_raw"].to_numpy()
    risk = cost / e
    decile = pd.qcut(pd.Series(risk).rank(method="first"), 10, labels=False).to_numpy() + 1
    rows, ret = [], {}
    for s in STRATEGIES:
        p = prem[s]
        r = retention(p, current, market, rc, sc.elasticity_scale)
        ret[s] = r
        exp_prem, exp_cost = float((r * p).sum()), float((r * cost).sum())
        change = p / current - 1
        rows.append(
            {
                "strategy": s,
                "retention_policies": float(r.mean()),
                "retention_exposure": float(np.average(r, weights=e)),
                "premium": exp_prem,
                "expected_claims": exp_cost,
                "loss_ratio": exp_cost / exp_prem,
                "expenses": sc.incurred_expense_ratio * exp_prem,
                "profit": exp_prem - exp_cost - sc.incurred_expense_ratio * exp_prem,
                "profit_margin": (exp_prem - exp_cost - sc.incurred_expense_ratio * exp_prem) / exp_prem,
                "realised_loss_ratio_actual_losses": float((r * actual).sum()) / exp_prem,
                "realised_profit_actual_losses": exp_prem
                - float((r * actual).sum())
                - sc.incurred_expense_ratio * exp_prem,
                "risk_mix_index": float(np.average(risk, weights=r * e) / np.average(risk, weights=e)),
                "retention_lowest_risk_decile": float(r[decile == 1].mean()),
                "retention_highest_risk_decile": float(r[decile == 10].mean()),
                "share_up_gt_10pct": float(np.mean(change > 0.10)),
                "share_up_gt_25pct": float(np.mean(change > 0.25)),
                "share_down_gt_10pct": float(np.mean(change < -0.10)),
                "mean_abs_change_vs_current": float(np.mean(np.abs(change))),
                "premium_to_true_cost_p10": float(
                    np.quantile(p / cost * (1 - sc.expense_ratio - sc.profit_margin), 0.10)
                ),
                "premium_to_true_cost_p90": float(
                    np.quantile(p / cost * (1 - sc.expense_ratio - sc.profit_margin), 0.90)
                ),
            }
        )
    return pd.DataFrame(rows), ret


def retention_by_decile(book: Book, sc: Scenario, ret: dict[str, np.ndarray]) -> pd.DataFrame:
    cost, _ = truth(book, sc.reference_truth)
    risk = cost / book.df["exposure"].to_numpy()
    decile = pd.qcut(pd.Series(risk).rank(method="first"), 10, labels=False).to_numpy() + 1
    out = pd.DataFrame({"risk_decile": np.arange(1, 11)})
    for s, r in ret.items():
        out[s] = pd.Series(r).groupby(decile).mean().to_numpy()
    return out


# ------------------------------------------------------------------ Monte Carlo


def monte_carlo(
    book: Book, sc: Scenario, ret: dict[str, np.ndarray], prem: dict[str, np.ndarray], cfg: Config
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """SYNTHETIC: per run, a systemic frequency shock S ~ lognormal(mean 1, sd systemic_sd); claim counts per
    exposure band N_b ~ Poisson(S x sum of retained expected claims); severities resampled from TRAINING raw
    claim amounts of that band (incl. large losses, D-016), rescaled to the true mean severity of the band's
    retained claims. Premium and expenses use expected retention (deterministic)."""
    mc = cfg.section("pricing")["monte_carlo"]
    runs = int(mc["runs"])
    rng = np.random.default_rng(cfg.seed + int(mc["seed_offset"]))
    sd = float(mc["systemic_sd"])
    sigma = np.sqrt(np.log(1 + sd**2))
    shocks = rng.lognormal(-(sigma**2) / 2, sigma, runs)
    cost, freq = truth(book, sc.reference_truth)
    band = book.df["band"].to_numpy()
    rows, dists = [], {}
    for s in STRATEGIES:
        r, p = ret[s], prem[s]
        premium = float((r * p).sum())
        claims_total = np.zeros(runs)
        for lvl, amounts in book.raw_amounts.items():
            m = band == lvl
            lam = float((r[m] * freq[m]).sum())
            if lam == 0 or len(amounts) == 0:
                continue
            true_sev = float((r[m] * cost[m]).sum() / lam)
            scaled = amounts * true_sev / amounts.mean()
            n = rng.poisson(lam * shocks)
            draws = rng.choice(scaled, size=int(n.sum()), replace=True)
            idx = np.repeat(np.arange(runs), n)
            claims_total += np.bincount(idx, weights=draws, minlength=runs)
        profit = premium * (1 - sc.incurred_expense_ratio) - claims_total
        dists[s] = profit
        q005 = np.quantile(profit, 0.005)
        rows.append(
            {
                "strategy": s,
                "premium": premium,
                "mean_profit": float(profit.mean()),
                "sd_profit": float(profit.std(ddof=1)),
                "prob_loss": float(np.mean(profit < 0)),
                "profit_p05": float(np.quantile(profit, 0.05)),
                "profit_p50": float(np.median(profit)),
                "var_99_5": float(-q005),
                "tvar_99_5": float(-profit[profit <= q005].mean()),
                "var_99_5_pct_premium": float(-q005 / premium),
            }
        )
        log.info("monte carlo %s done", s)
    return pd.DataFrame(rows), dists


# ------------------------------------------------------------------ orchestration


@dataclass
class SimulationResults:
    terms: pd.DataFrame
    base: pd.DataFrame
    deciles: pd.DataFrame
    mc: pd.DataFrame
    mc_dists: dict[str, np.ndarray]
    sensitivity: pd.DataFrame
    scenario: Scenario
    extras: dict[str, Any] = field(default_factory=dict)


def base_scenario(cfg: Config) -> Scenario:
    pc = cfg.section("pricing")
    return Scenario(
        float(pc["expense_ratio"]), float(pc["profit_margin"]), float(pc["cap"]), 1.0, str(pc["reference_truth"])
    )


def run_simulation(con: duckdb.DuckDBPyConnection, cfg: Config) -> SimulationResults:
    pc = cfg.section("pricing")
    rc = pc["retention"]
    book = build_book(con, cfg)
    sc = base_scenario(cfg)
    base, ret = evaluate(book, sc, rc)
    prem, *_ = premiums(book, sc)
    deciles = retention_by_decile(book, sc, ret)
    mc, dists = monte_carlo(book, sc, ret, prem, cfg)

    sens_rows = []
    grid = pc["sensitivity"]
    for param, values in (
        ("elasticity_scale", grid["elasticity_scale"]),
        ("actual_expense_ratio", grid["expense_ratio"]),
        ("cap", grid["cap"]),
        ("reference_truth", grid["reference_truth"]),
    ):
        for v in values:
            alt = Scenario(**{**sc.__dict__, param: v})
            t, _ = evaluate(book, alt, rc)
            t.insert(0, "value", v)
            t.insert(0, "parameter", param)
            sens_rows.append(
                t[
                    [
                        "parameter",
                        "value",
                        "strategy",
                        "retention_exposure",
                        "loss_ratio",
                        "profit",
                        "profit_margin",
                        "risk_mix_index",
                        "realised_loss_ratio_actual_losses",
                        "realised_profit_actual_losses",
                    ]
                ]
            )
    sens = pd.concat(sens_rows, ignore_index=True)
    return SimulationResults(
        book.terms,
        base,
        deciles,
        mc,
        dists,
        sens,
        sc,
        {
            "n_policies": len(book.df),
            "exposure": float(book.df["exposure"].sum()),
            "actual_raw_losses": float(book.df["claim_amount_raw"].sum()),
            "runs": int(pc["monte_carlo"]["runs"]),
        },
    )

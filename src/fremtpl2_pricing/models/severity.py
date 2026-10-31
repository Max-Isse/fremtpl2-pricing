"""Phase 3a: claim-severity GLM on the training split (the holdout is not touched).

Unit of analysis: one row per policy with at least one positive claim; response = average capped claim
amount, var_weight = number of claims (an average of n claims has variance phi * mu^2 / n under Gamma).
Steps: distribution facts -> full Gamma GLM -> backward elimination by quasi-F -> Gamma assumption checks
(variance vs mean, quantile residuals, point masses) -> relativities -> grouped CV vs mean-severity baseline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from fremtpl2_pricing.config import Config
from fremtpl2_pricing.data.db import query_df
from fremtpl2_pricing.features import GLMDesign
from fremtpl2_pricing.models import glm

log = logging.getLogger(__name__)

WEIGHT = "n_claims"


def load_severity_training(con: duckdb.DuckDBPyConnection, holdout: bool = False) -> pd.DataFrame:
    """One row per policy with claims; training split by default, the holdout with holdout=True."""
    df = query_df(
        con,
        f"""
        SELECT IDpol, any_value(group_id) AS group_id, any_value(fold) AS fold,
               count(*) AS n_claims, sum(claim_amount) AS amount, sum(claim_amount_raw) AS amount_raw,
               any_value(exposure) AS exposure,
               any_value(Area) AS Area, any_value(VehPower) AS VehPower, any_value(VehAge) AS VehAge,
               any_value(DrivAge) AS DrivAge, any_value(BonusMalus) AS BonusMalus,
               any_value(VehBrand) AS VehBrand, any_value(VehGas) AS VehGas,
               any_value(Density) AS Density, any_value(Region) AS Region
        FROM claims_clean
        WHERE is_holdout = {"TRUE" if holdout else "FALSE"}
        GROUP BY IDpol
        ORDER BY IDpol
        """,
    )
    df["avg_severity"] = df["amount"] / df[WEIGHT]
    return df


def claim_amounts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return query_df(
        con, "SELECT claim_amount, claim_amount_raw FROM claims_clean WHERE NOT is_holdout ORDER BY claim_id"
    )


@dataclass
class SeverityResults:
    design: GLMDesign
    params: pd.Series
    cov_cluster: pd.DataFrame
    dispersion: float
    base_severity: float
    distribution: pd.DataFrame
    flat_amounts: pd.DataFrame
    selection_path: pd.DataFrame
    cv_path: pd.DataFrame
    factor_tests_full: pd.DataFrame
    factor_tests_final: pd.DataFrame
    relativities: pd.DataFrame
    by_exposure_band: pd.DataFrame
    variance_mean: pd.DataFrame
    variance_power: pd.DataFrame
    residuals: pd.DataFrame
    cv_metrics: pd.DataFrame
    cv_by_fold: pd.DataFrame
    freq_vs_sev: pd.DataFrame
    amounts: pd.DataFrame
    extras: dict[str, Any] = field(default_factory=dict)


def distribution_summary(amounts: pd.DataFrame, n_policies_freq: int, n_sev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in (("claim_amount_raw", "raw"), ("claim_amount", "capped")):
        a = amounts[col].to_numpy()
        rows.append(
            {
                "amounts": label,
                "claims": len(a),
                "mean": a.mean(),
                "median": np.median(a),
                "cv": a.std(ddof=1) / a.mean(),
                "skewness": float(stats.skew(a)),
                "p90": np.quantile(a, 0.9),
                "p99": np.quantile(a, 0.99),
                "max": a.max(),
            }
        )
    out = pd.DataFrame(rows)
    out["policies_with_claims"] = len(n_sev)
    out["share_of_frequency_rows"] = len(n_sev) / n_policies_freq
    return out


def flat_amounts(amounts: pd.DataFrame, min_share: float) -> pd.DataFrame:
    vc = amounts["claim_amount"].round(2).value_counts()
    share = vc / vc.sum()
    out = pd.DataFrame({"amount": vc.index, "claims": vc.to_numpy(), "share": share.to_numpy()})
    return out[out["share"] >= min_share].reset_index(drop=True)


def drop_one_f_tests(
    design: GLMDesign, df: pd.DataFrame, full: Any, factors: list[str], always: list[str] | None = None
) -> pd.DataFrame:
    """Quasi-likelihood F tests: (deviance change / df) / Pearson dispersion of the larger model.

    `factors` are tested one at a time; `always` (controls) stay in every model and are tested too."""
    y, w = df["avg_severity"].to_numpy(), df[WEIGHT].to_numpy(float)
    phi = float(full.scale)
    always = always or []
    rows = []
    for name in [*factors, *always]:
        keep = [n for n in [*factors, *always] if n != name]
        reduced = glm.fit_gamma(design.matrix(df, keep), y, w)
        ddf = int(full.df_model - reduced.df_model)
        f_stat = (reduced.deviance - full.deviance) / ddf / phi
        rows.append(
            {
                "factor": name,
                "df": ddf,
                "delta_deviance": reduced.deviance - full.deviance,
                "F": f_stat,
                "p_F": float(stats.f.sf(f_stat, ddf, full.df_resid)),
            }
        )
    return pd.DataFrame(rows).sort_values("p_F", ascending=False, ignore_index=True)


def backward_elimination(
    design: GLMDesign, df: pd.DataFrame, alpha: float, controls: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full backward-elimination path: repeatedly drop the factor with the largest quasi-F p-value, down to
    the intercept-only model. Returns the path (with what an F-test rule at `alpha` would do) and the
    drop-one tests of the full starting model. The path orders factors by importance for `cv_select`."""
    y, w = df["avg_severity"].to_numpy(), df[WEIGHT].to_numpy(float)
    current = [n for n in design.names if n not in controls]
    path: list[dict[str, Any]] = []
    first: pd.DataFrame | None = None
    while current:
        res = glm.fit_gamma(design.matrix(df, [*current, *controls]), y, w)
        tests = drop_one_f_tests(design, df, res, current, always=controls)
        if first is None:
            first = tests
        worst = tests[~tests["factor"].isin(controls)].iloc[0]
        path.append(
            {
                "step": len(path) + 1,
                "factors_before": ", ".join(current),
                "dispersion": float(res.scale),
                "dropped": worst["factor"],
                "p_F": worst["p_F"],
            }
        )
        current.remove(str(worst["factor"]))
    out = pd.DataFrame(path)
    # What a pure F-test rule would do: drop until the first factor with p <= alpha, keep the rest.
    stop = out.index[out["p_F"] <= alpha]
    first_stop = stop[0] if len(stop) else len(out)
    out["f_test_rule"] = np.where(out.index < first_stop, "dropped", "kept")
    assert first is not None
    return out, first


def cv_select(df: pd.DataFrame, path: pd.DataFrame, cfg: Config, controls: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Grouped-CV deviance of every nested model on the elimination path; choose the smallest model whose
    CV deviance is within one standard error of the best, using the *paired* per-fold difference to the
    best model (fold-to-fold noise is shared by all models, so unpaired SEs would swamp real differences)."""
    scfg = cfg.section("severity")
    nested: list[list[str]] = [r.split(", ") for r in path["factors_before"]] + [[]]
    nested = [[*f, *controls] for f in nested]
    if controls:
        nested.append([])  # pure intercept-only reference
    rows = []
    per_fold: list[list[float]] = []
    for factors in nested:
        devs = []
        for k in sorted(df["fold"].unique()):
            tr, va = df[df["fold"] != k], df[df["fold"] == k]
            y_va, w_va = va["avg_severity"].to_numpy(), va[WEIGHT].to_numpy(float)
            if factors:
                d = GLMDesign.fit(tr, cfg, factors=factors, bands=scfg["bands"], weight_col=WEIGHT)
                r = glm.fit_gamma(d.matrix(tr), tr["avg_severity"].to_numpy(), tr[WEIGHT].to_numpy(float))
                mu = np.asarray(r.predict(d.matrix(va)))
            else:
                mu = np.full(len(va), np.average(tr["avg_severity"], weights=tr[WEIGHT]))
            devs.append(glm.gamma_deviance(y_va, mu, w_va) / w_va.sum())
        per_fold.append(devs)
        tariff = [f for f in factors if f not in controls]
        rows.append(
            {
                "n_factors": len(tariff),
                "factors": ", ".join(factors) or "(intercept only)",
                "cv_mean_deviance": float(np.mean(devs)),
                "cv_se": float(np.std(devs, ddof=1) / np.sqrt(len(devs))),
            }
        )
    out = pd.DataFrame(rows)
    out["has_controls"] = [bool(controls) and all(c in r.split(", ") for c in controls) for r in out["factors"]]
    best_i = int(np.argmin(out["cv_mean_deviance"].to_numpy()))
    diffs = np.array(per_fold) - np.array(per_fold[best_i])
    out["diff_vs_best"] = diffs.mean(axis=1)
    out["paired_se_vs_best"] = diffs.std(axis=1, ddof=1) / np.sqrt(diffs.shape[1])
    out["folds_worse_than_best"] = (diffs > 0).sum(axis=1)
    eligible = out[(out["diff_vs_best"] <= out["paired_se_vs_best"]) & (out["has_controls"] | (not controls))]
    chosen = eligible.sort_values(["n_factors", "cv_mean_deviance"]).iloc[0]
    out["best"] = out.index == best_i
    out["chosen_one_se"] = out.index == chosen.name
    selected = [f for f in str(chosen["factors"]).split(", ") if f not in controls and f != "(intercept only)"]
    return selected, out


def variance_vs_mean(df: pd.DataFrame, mu: np.ndarray, n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin by fitted mean; w*(y-mu)^2 estimates phi*V(mu). log-log slope = variance power (Gamma: 2)."""
    w = df[WEIGHT].to_numpy(float)
    y = df["avg_severity"].to_numpy()
    b = pd.qcut(pd.Series(mu).rank(method="first"), n_bins, labels=False)
    g = pd.DataFrame({"bin": b, "mu": mu, "wsq": w * (y - mu) ** 2, "y": y}).groupby("bin")
    out = g.agg(
        policies=("mu", "size"), mean_fitted=("mu", "mean"), mean_observed=("y", "mean"), phi_v=("wsq", "mean")
    ).reset_index()
    X = sm.add_constant(np.log(out["mean_fitted"]))
    fit = sm.OLS(np.log(out["phi_v"]), X).fit()
    ci = fit.conf_int().iloc[1]
    power = pd.DataFrame(
        [
            {
                "variance_power": float(fit.params.iloc[1]),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "r_squared": float(fit.rsquared),
                "reference": "Poisson-like 1, Gamma 2, inverse Gaussian 3",
            }
        ]
    )
    return out, power


def cross_validate(df: pd.DataFrame, design: GLMDesign, selected: list[str], cfg: Config) -> pd.DataFrame:
    scfg = cfg.section("severity")
    rows = []
    for k in sorted(df["fold"].unique()):
        tr, va = df[df["fold"] != k], df[df["fold"] == k]
        y_tr, w_tr = tr["avg_severity"].to_numpy(), tr[WEIGHT].to_numpy(float)
        y_va, w_va = va["avg_severity"].to_numpy(), va[WEIGHT].to_numpy(float)
        controls = design.control_names
        preds = {"mean_severity": np.full(len(va), np.average(y_tr, weights=w_tr))}
        for label, factors in (
            ("gamma_controls_only", controls),
            ("gamma_all_factors", design.names),
            ("gamma_selected", [*selected, *controls]),
        ):
            if not factors:
                continue
            d = GLMDesign.fit(tr, cfg, factors=factors, bands=scfg["bands"], weight_col=WEIGHT)
            r = glm.fit_gamma(d.matrix(tr), y_tr, w_tr)
            preds[label] = np.asarray(r.predict(d.matrix(va)))
        for m, mu in preds.items():
            rows.append(
                {
                    "fold": int(k),
                    "model": m,
                    "policies": len(va),
                    "claims": float(w_va.sum()),
                    "mean_deviance_per_claim": glm.gamma_deviance(y_va, mu, w_va) / w_va.sum(),
                    "ae_amount": float((y_va * w_va).sum() / (mu * w_va).sum()),
                }
            )
    return pd.DataFrame(rows)


def summarise_cv(by_fold: pd.DataFrame) -> pd.DataFrame:
    g = by_fold.groupby("model", sort=False)
    out = g.agg(
        mean_deviance_per_claim=("mean_deviance_per_claim", "mean"),
        fold_sd=("mean_deviance_per_claim", "std"),
        ae_amount=("ae_amount", "mean"),
    ).reset_index()
    base = out.loc[out["model"] == "mean_severity", "mean_deviance_per_claim"].iloc[0]
    out["deviance_explained_vs_mean"] = 1 - out["mean_deviance_per_claim"] / base
    wide = by_fold.pivot(index="fold", columns="model", values="mean_deviance_per_claim")
    out["folds_better_than_mean"] = [int((wide["mean_severity"] > wide[m]).sum()) for m in out["model"]]
    return out


def frequency_vs_severity(sev_tests: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Side-by-side significance of each factor for frequency and for severity (full models)."""
    path = cfg.path("tables_dir") / "freq_factor_tests.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `fremtpl2 frequency` before `fremtpl2 severity`")
    freq = pd.read_csv(path)
    freq = freq[freq["test"] == "drop from model"][["factor", "df", "quasi_F", "p_quasi_F"]]
    freq.columns = ["factor", "freq_df", "freq_quasi_F", "freq_p"]
    sev = sev_tests[["factor", "df", "F", "p_F"]].rename(columns={"df": "sev_df", "F": "sev_quasi_F", "p_F": "sev_p"})
    out = freq.merge(sev, on="factor", how="outer")
    out["matters_for_frequency"] = out["freq_p"] < cfg.section("frequency")["significance"]
    out["matters_for_severity"] = out["sev_p"] < cfg.section("severity")["significance"]
    return out


def severity_by_exposure_band(
    design: GLMDesign, params: pd.Series, df: pd.DataFrame, mu: np.ndarray, controls: list[str]
) -> pd.DataFrame:
    """Observed vs fitted vs priced (full-year rate) severity by exposure band of the claiming record."""
    if not controls:
        return pd.DataFrame()
    w = df[WEIGHT].to_numpy(float)
    y = df["avg_severity"].to_numpy()
    priced = glm.predict_log_linear(design, params, df, pricing=True)
    out = (
        pd.DataFrame(
            {
                "band": design.factor(controls[0]).bin(df),
                "claims": w,
                "amount": y * w,
                "fitted": mu * w,
                "priced": priced * w,
            }
        )
        .groupby("band")
        .sum()
        .reindex(design.factor(controls[0]).levels)
        .reset_index()
    )
    out["observed_mean_severity"] = out["amount"] / out["claims"]
    out["ae_fitted"] = out["amount"] / out["fitted"]
    out["ae_priced_at_full_year"] = out["amount"] / out["priced"]
    return out


def run_severity(con: duckdb.DuckDBPyConnection, cfg: Config) -> SeverityResults:
    scfg = cfg.section("severity")
    level = float(cfg.section("glm")["ci_level"])
    df = load_severity_training(con)
    amounts = claim_amounts(con)
    n_freq = int(query_df(con, "SELECT count(*) AS n FROM policy_clean WHERE NOT is_holdout")["n"].iloc[0])
    y, w = df["avg_severity"].to_numpy(), df[WEIGHT].to_numpy(float)

    controls = list(scfg["controls"])
    design_all = GLMDesign.fit(
        df, cfg, factors=[*scfg["start_factors"], *controls], bands=scfg["bands"], weight_col=WEIGHT
    )
    log.info("severity: %s policies, %s claims; backward elimination", len(df), int(w.sum()))
    path, tests_full = backward_elimination(design_all, df, float(scfg["significance"]), controls)
    log.info("severity: CV along the elimination path")
    selected, cv_path = cv_select(df, path, cfg, controls)
    design = GLMDesign.fit(df, cfg, factors=[*selected, *controls], bands=scfg["bands"], weight_col=WEIGHT)
    X = design.matrix(df)
    res = glm.fit_gamma(X, y, w)
    res_cl = glm.fit_gamma(X, y, w, groups=df["group_id"].to_numpy())
    tests_final = drop_one_f_tests(design, df, res, selected, always=controls)
    mu = np.asarray(res.fittedvalues)
    by_band = severity_by_exposure_band(design, res.params, df, mu, controls)

    rel = glm.relativities(design, res_cl.params, res_cl.bse, df, level, volume=df[WEIGHT], outcome=df["amount"])
    rel = rel.rename(columns={"exposure": "claims", "claims": "amount", "observed_frequency": "observed_severity"})

    # The variance-function check needs spread in fitted means, so it uses the full starting model
    # (the selected model may be intercept-only).
    res_full = glm.fit_gamma(design_all.matrix(df), y, w)
    vm, vpower = variance_vs_mean(df, np.asarray(res_full.fittedvalues), int(scfg["variance_bins"]))
    resid = pd.DataFrame({"quantile_residual": glm.gamma_quantile_residuals(y, mu, w, float(res.scale)), "n_claims": w})

    log.info("severity: grouped CV")
    by_fold = cross_validate(df, design_all, selected, cfg)

    return SeverityResults(
        design=design,
        params=res.params,
        cov_cluster=pd.DataFrame(res_cl.cov_params()),
        dispersion=float(res.scale),
        base_severity=float(np.exp(res.params["Intercept"])),
        distribution=distribution_summary(amounts, n_freq, df),
        flat_amounts=flat_amounts(amounts, float(scfg["flat_amount_min_share"])),
        selection_path=path,
        cv_path=cv_path,
        factor_tests_full=tests_full,
        factor_tests_final=tests_final,
        relativities=rel,
        by_exposure_band=by_band,
        variance_mean=vm,
        variance_power=vpower,
        residuals=resid,
        cv_metrics=summarise_cv(by_fold),
        cv_by_fold=by_fold,
        freq_vs_sev=frequency_vs_severity(tests_full, cfg),
        amounts=amounts,
        extras={
            "n_policies": len(df),
            "n_claims": float(w.sum()),
            "selected": selected,
            "controls": controls,
            "dropped": [f for f in design_all.names if f not in selected],
            "n_freq_policies": n_freq,
        },
    )


def save_artifact(res: SeverityResults, cfg: Config, path: Path) -> Path:
    """Severity artefact: score with exp(GLMDesign.from_dict(design).matrix(df) @ params)."""
    manifest = json.loads(cfg.path("manifest").read_text(encoding="utf-8"))
    art = {
        "model": "severity_glm_gamma",
        "link": "log",
        "response": "average capped claim amount per claim",
        "design": res.design.to_dict(),
        "params": res.params.to_dict(),
        "cov_cluster": res.cov_cluster.to_numpy().tolist(),
        "dispersion": res.dispersion,
        "base_severity": res.base_severity,
        "training": res.extras,
        "data_sha256": {k: v["sha256"] for k, v in manifest.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(art, indent=2, default=float), encoding="utf-8")
    return path

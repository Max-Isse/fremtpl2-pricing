"""Count-GLM fitting and inference: Poisson / NB2 with exposure offset, overdispersion tests,
likelihood-ratio tests per factor, relativities with CIs, and randomised quantile residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from fremtpl2_pricing.features import INTERCEPT, GLMDesign


def poisson_deviance(y: np.ndarray, mu: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Total Poisson deviance 2 * sum[y log(y/mu) - (y - mu)]."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    d = 2 * (term - (y - mu))
    return float(np.sum(d if weights is None else weights * d))


def fit_poisson(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> sm.genmod.generalized_linear_model.GLMResults:
    return sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(exposure)).fit()


def fit_nb2(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray, start: np.ndarray | None = None):  # type: ignore[no-untyped-def]
    """NB2 (Var = mu + alpha mu^2) by maximum likelihood, alpha estimated jointly.

    BFGS optimises over log(alpha), so alpha stays positive; Newton works on alpha directly and can step
    past zero to NaN on near-equidispersed data while still reporting convergence.
    """
    model = sm.NegativeBinomial(y, X, loglike_method="nb2", offset=np.log(exposure))
    start_params = None if start is None else np.append(start, 0.5)
    res = model.fit(start_params=start_params, method="bfgs", maxiter=1000, disp=False)
    if not np.all(np.isfinite(res.params)):
        raise RuntimeError("NB2 fit returned non-finite parameters")
    return res


@dataclass
class OverdispersionTests:
    pearson_chi2: float
    df_resid: float
    pearson_dispersion: float
    deviance_dispersion: float
    ct_alpha: float
    ct_se: float
    ct_t: float
    ct_p_one_sided: float

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "test": "Pearson chi2 / df",
                    "statistic": self.pearson_dispersion,
                    "detail": f"chi2={self.pearson_chi2:,.0f}, df={self.df_resid:,.0f}",
                },
                {"test": "Deviance / df", "statistic": self.deviance_dispersion, "detail": ""},
                {
                    "test": "Cameron-Trivedi alpha (NB2 form)",
                    "statistic": self.ct_alpha,
                    "detail": f"se={self.ct_se:.4f}, t={self.ct_t:.2f}, one-sided p={self.ct_p_one_sided:.2g}",
                },
            ]
        )


def cameron_trivedi(y: np.ndarray, mu: np.ndarray) -> tuple[float, float, float, float]:
    """Cameron & Trivedi (1990) regression test of H0: Var(y)=mu vs H1: Var(y)=mu+alpha*mu^2.

    Auxiliary OLS without intercept: ((y-mu)^2 - y) / mu = alpha * mu + e. Returns alpha, se, t, one-sided p.
    """
    z = ((y - mu) ** 2 - y) / mu
    aux = sm.OLS(z, mu).fit()
    alpha, se = float(aux.params[0]), float(aux.bse[0])
    t = alpha / se
    return alpha, se, t, float(stats.t.sf(t, df=aux.df_resid))


def overdispersion_tests(res: sm.genmod.generalized_linear_model.GLMResults, y: np.ndarray) -> OverdispersionTests:
    mu = np.asarray(res.fittedvalues)
    alpha, se, t, p = cameron_trivedi(np.asarray(y, dtype=float), mu)
    return OverdispersionTests(
        pearson_chi2=float(res.pearson_chi2),
        df_resid=float(res.df_resid),
        pearson_dispersion=float(res.pearson_chi2 / res.df_resid),
        deviance_dispersion=float(res.deviance / res.df_resid),
        ct_alpha=alpha,
        ct_se=se,
        ct_t=t,
        ct_p_one_sided=p,
    )


def nb2_vs_poisson_lr(ll_nb2: float, ll_poisson: float) -> tuple[float, float]:
    """LR test of alpha=0. alpha sits on the boundary, so the null is a 50:50 mix of chi2_0 and chi2_1."""
    lr = max(2 * (ll_nb2 - ll_poisson), 0.0)
    return lr, float(0.5 * stats.chi2.sf(lr, 1))


def factor_lr_tests(
    design: GLMDesign,
    df: pd.DataFrame,
    y: np.ndarray,
    exposure: np.ndarray,
    full: sm.genmod.generalized_linear_model.GLMResults,
    dispersion: float,
    extra: dict[str, GLMDesign] | None = None,
) -> pd.DataFrame:
    """Drop-one LR tests for each factor in the model, and add-one tests for candidate factors.

    Reports the Poisson LR chi2 and a quasi-likelihood F test that scales the deviance change by the
    Pearson dispersion, which is the appropriate test when counts are overdispersed.
    """
    rows = []
    for name in design.names:
        keep = [n for n in design.names if n != name]
        reduced = fit_poisson(design.matrix(df, keep), y, exposure)
        rows.append(
            _lr_row(
                name,
                "drop from model",
                reduced.deviance - full.deviance,
                int(full.df_model - reduced.df_model),
                dispersion,
                full.df_resid,
            )
        )
    for name, ext in (extra or {}).items():
        bigger = fit_poisson(ext.matrix(df), y, exposure)
        rows.append(
            _lr_row(
                name,
                "add to model",
                full.deviance - bigger.deviance,
                int(bigger.df_model - full.df_model),
                dispersion,
                bigger.df_resid,
            )
        )
    return pd.DataFrame(rows)


def _lr_row(name: str, test: str, delta_dev: float, df: int, dispersion: float, df_resid: float) -> dict[str, object]:
    f_stat = (delta_dev / df) / dispersion
    return {
        "factor": name,
        "test": test,
        "df": df,
        "lr_chi2": delta_dev,
        "p_lr": float(stats.chi2.sf(delta_dev, df)),
        "quasi_F": f_stat,
        "p_quasi_F": float(stats.f.sf(f_stat, df, df_resid)),
    }


def relativities(
    design: GLMDesign,
    params: pd.Series,
    bse: pd.Series,
    df: pd.DataFrame,
    level: float = 0.95,
    volume: pd.Series | None = None,
    outcome: pd.Series | None = None,
) -> pd.DataFrame:
    """exp(coef) per level with Wald CI; reference levels have relativity 1.

    Also reports volume and outcome totals per level: by default exposure and claim count (frequency); the
    severity model passes claim counts and amounts.
    """
    z = stats.norm.ppf(0.5 + level / 2)
    binned = design.binned(df)
    volume = df["exposure"] if volume is None else volume
    outcome = df["claim_count"] if outcome is None else outcome
    rows = []
    for f in design.factors:
        expo = volume.groupby(binned[f.name]).sum()
        claims = outcome.groupby(binned[f.name]).sum()
        for order, lvl in enumerate(f.levels):
            col = f"{f.name}[{lvl}]"
            b = float(params.get(col, 0.0))
            s = float(bse.get(col, 0.0))
            rows.append(
                {
                    "factor": f.name,
                    "level": lvl,
                    "order": order,
                    "is_reference": lvl == f.reference,
                    "is_control": f.kind == "control",
                    "coef": b,
                    "se": s,
                    "relativity": np.exp(b),
                    "ci_low": np.exp(b - z * s),
                    "ci_high": np.exp(b + z * s),
                    "exposure": float(expo.get(lvl, 0.0)),
                    "claims": float(claims.get(lvl, 0.0)),
                }
            )
    cols = [
        "factor",
        "level",
        "order",
        "is_reference",
        "is_control",
        "coef",
        "se",
        "relativity",
        "ci_low",
        "ci_high",
        "exposure",
        "claims",
    ]
    out = pd.DataFrame(rows, columns=cols)
    out["observed_frequency"] = out["claims"] / out["exposure"]
    return out


def base_frequency(params: pd.Series) -> float:
    return float(np.exp(params[INTERCEPT]))


def randomized_quantile_residuals(
    y: np.ndarray, mu: np.ndarray, rng: np.random.Generator, alpha: float | None = None
) -> np.ndarray:
    """Dunn & Smyth (1996) residuals: N(0,1) under a correctly specified discrete model."""
    y = np.asarray(y, dtype=float)
    # NB2 with alpha at (or numerically past) the zero boundary is the Poisson model.
    if alpha is None or not np.isfinite(alpha) or alpha < 1e-8:
        lo = stats.poisson.cdf(y - 1, mu)
        hi = stats.poisson.cdf(y, mu)
    else:
        n = 1.0 / alpha
        p = n / (n + mu)
        lo = stats.nbinom.cdf(y - 1, n, p)
        hi = stats.nbinom.cdf(y, n, p)
    u = rng.uniform(lo, hi)
    return np.asarray(stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12)))


# --- Severity (Gamma) and shared helpers -------------------------------------------------------------


def fit_gamma(
    X: pd.DataFrame, y: np.ndarray, weights: np.ndarray, groups: np.ndarray | None = None
) -> sm.genmod.generalized_linear_model.GLMResults:
    """Gamma GLM with log link on average severities y, var_weights = claim counts (Var = phi mu^2 / w).

    With `groups`, the covariance is cluster-robust; otherwise the default Pearson-scaled covariance.
    """
    model = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Log()), var_weights=weights)
    if groups is None:
        return model.fit(scale="X2")
    return model.fit(scale="X2", cov_type="cluster", cov_kwds={"groups": groups})


def gamma_deviance(y: np.ndarray, mu: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Total (unit-scale) Gamma deviance 2 * sum w [ (y - mu)/mu - log(y/mu) ]."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    d = 2 * ((y - mu) / mu - np.log(y / mu))
    return float(np.sum(d if weights is None else weights * d))


def gamma_quantile_residuals(y: np.ndarray, mu: np.ndarray, weights: np.ndarray, dispersion: float) -> np.ndarray:
    """Quantile residuals for an average of `w` Gamma claims: y ~ Gamma(shape = w / phi, scale = mu phi / w)."""
    shape = np.asarray(weights, dtype=float) / dispersion
    u = stats.gamma.cdf(np.asarray(y, dtype=float), a=shape, scale=np.asarray(mu) / shape)
    return np.asarray(stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12)))


def predict_log_linear(
    design: GLMDesign, params: pd.Series | dict[str, float], df: pd.DataFrame, pricing: bool = True
) -> np.ndarray:
    """exp(X beta) for a log-link GLM saved as (design, params); pricing=True pins control factors."""
    p = pd.Series(params, dtype=float)
    X = design.matrix(df, pricing=pricing)
    return np.exp(X[p.index].to_numpy() @ p.to_numpy())


def tweedie_power_from_gamma_dispersion(phi: float) -> float:
    """Compound Poisson-Gamma: Gamma shape nu = 1/phi gives Tweedie power p = (nu + 2) / (nu + 1)."""
    nu = 1.0 / phi
    return (nu + 2.0) / (nu + 1.0)

"""The coursework analysis (items a-g) and the log-log respecification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from linear_models import ols
from linear_models.data import positive_rows

LINEAR_NAMES = ("intercept", "population", "density")
LOG_NAMES = ("intercept", "log_population", "log_density")


def linear_design(df: pd.DataFrame) -> np.ndarray:
    return ols.design_matrix(df["population"], df["density"])


def log_design(df: pd.DataFrame) -> np.ndarray:
    return ols.design_matrix(np.log(df["population"]), np.log(df["density"]))


def fit_linear(df: pd.DataFrame) -> ols.OLSResult:
    """The coursework model: ``area = b0 + b1 population + b2 density``."""
    return ols.fit(linear_design(df), df["land_area"], LINEAR_NAMES)


def fit_loglog(df: pd.DataFrame) -> ols.OLSResult:
    """``log area = b0 + b1 log population + b2 log density`` on rows with all values > 0."""
    df = positive_rows(df)
    return ols.fit(log_design(df), np.log(df["land_area"]), LOG_NAMES)


def top_residuals(df: pd.DataFrame, result: ols.OLSResult, n: int = 5) -> list[dict]:
    """The ``n`` largest absolute residuals and their share of the residual sum of squares."""
    order = np.argsort(-np.abs(result.resid))[:n]
    return [
        {
            "country": str(df["country"].iloc[i]),
            "residual": float(result.resid[i]),
            "share_of_rss": float(result.resid[i] ** 2 / result.rss),
        }
        for i in order
    ]


def replicate(df: pd.DataFrame) -> dict:
    """Items (b)-(g) of the assignment, computed with the fixes described in the README."""
    y = df["land_area"].to_numpy()
    simple = {}
    for name in ("population", "density"):
        x = df[name].to_numpy()
        rho = ols.pearson(x, y)
        res = ols.fit(ols.design_matrix(x), y, ("intercept", name))
        simple[name] = {
            "rho": rho,
            "r2": rho**2,
            "slope": float(res.coef[1]),
            "slope_p": float(res.p[1]),
        }

    X = linear_design(df)
    model = fit_linear(df)
    beta_ne = ols.normal_equations(X, y)
    top = top_residuals(df, model, n=5)
    lm, lm_p = ols.breusch_pagan(X, model.resid)
    return {
        "n": model.n,
        "simple": simple,
        "multiple": model.to_dict(),
        "normal_equations": {
            "coef": [float(b) for b in beta_ne],
            "max_rel_diff_vs_qr": float(np.max(np.abs(beta_ne - model.coef) / np.abs(model.coef))),
            "cond_X": float(np.linalg.cond(X)),
            "cond_XtX": float(np.linalg.cond(X.T @ X)),
        },
        "residuals": {
            **ols.residual_diagnostics(model.resid),
            "breusch_pagan": lm,
            "breusch_pagan_p": lm_p,
            "top": top,
            "top5_share_of_rss": float(sum(t["share_of_rss"] for t in top)),
        },
    }


def loglog(df: pd.DataFrame, min_density: float = 10.0, min_area: float = 1000.0) -> dict:
    """Fit the log-log model and test the identity ``b = (0, 1, -1)``.

    The published density and area are rounded to integers, so the identity
    ``area = population / density`` holds only approximately. The robustness fit drops
    rows where that rounding is large in relative terms (density < ``min_density`` or
    area < ``min_area`` km²).
    """
    pos = positive_rows(df)
    model = fit_loglog(df)
    lm, lm_p = ols.breusch_pagan(log_design(pos), model.resid)
    tests = {
        "intercept_eq_0": model.t_test(0, 0.0),
        "log_population_eq_1": model.t_test(1, 1.0),
        "log_density_eq_-1": model.t_test(2, -1.0),
    }

    precise = pos[(pos["density"] >= min_density) & (pos["land_area"] >= min_area)]
    robust = fit_loglog(precise)
    return {
        "dropped": sorted(set(df["country"]) - set(pos["country"])),
        "model": model.to_dict(),
        "identity_tests": {k: {"t": t, "p": p} for k, (t, p) in tests.items()},
        "residuals": {
            **ols.residual_diagnostics(model.resid),
            "breusch_pagan": lm,
            "breusch_pagan_p": lm_p,
            "top": top_residuals(pos, model, n=5),
        },
        "robust": {
            "min_density": min_density,
            "min_area": min_area,
            "model": robust.to_dict(),
        },
    }

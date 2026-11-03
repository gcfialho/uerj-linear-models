"""Ordinary least squares from first principles.

The fit uses a thin QR decomposition of the design matrix, ``X = QR``, so the
coefficients solve the triangular system ``R beta = Q^T y``. That avoids forming
``X^T X``, whose condition number is the square of ``X``'s. The coursework
formula, ``beta = (X^T X)^{-1} X^T y``, is kept in :func:`normal_equations` for
comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import linalg, stats


@dataclass(frozen=True)
class OLSResult:
    """A fitted linear model ``y = X beta + e`` with classical (homoscedastic) inference."""

    names: tuple[str, ...]
    coef: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    fitted: np.ndarray
    resid: np.ndarray
    sigma2: float
    n: int
    k: int
    r2: float
    adj_r2: float
    f_stat: float
    f_pvalue: float
    cond: float

    @property
    def df_resid(self) -> int:
        return self.n - self.k

    @property
    def rss(self) -> float:
        return float(self.resid @ self.resid)

    def conf_int(self, level: float = 0.95) -> np.ndarray:
        """Two-sided confidence intervals, one row ``(low, high)`` per coefficient."""
        q = stats.t.ppf(0.5 + level / 2, self.df_resid)
        return np.column_stack([self.coef - q * self.se, self.coef + q * self.se])

    def t_test(self, index: int, value: float) -> tuple[float, float]:
        """t statistic and two-sided p-value for ``H0: beta[index] = value``."""
        t = (self.coef[index] - value) / self.se[index]
        return float(t), float(2 * stats.t.sf(abs(t), self.df_resid))

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "r2": self.r2,
            "adj_r2": self.adj_r2,
            "f_stat": self.f_stat,
            "f_pvalue": self.f_pvalue,
            "sigma": float(np.sqrt(self.sigma2)),
            "condition_number": self.cond,
            "coefficients": {
                name: {"coef": float(b), "se": float(s), "t": float(t), "p": float(p)}
                for name, b, s, t, p in zip(
                    self.names, self.coef, self.se, self.t, self.p, strict=True
                )
            },
        }


def design_matrix(*columns: Sequence[float] | np.ndarray, intercept: bool = True) -> np.ndarray:
    """Stack regressors as columns, with a leading column of ones if ``intercept``."""
    cols = [np.asarray(c, dtype=float) for c in columns]
    if intercept:
        cols.insert(0, np.ones_like(cols[0]))
    return np.column_stack(cols)


def fit(
    X: np.ndarray,
    y: Sequence[float] | np.ndarray,
    names: Sequence[str] | None = None,
) -> OLSResult:
    """Fit ``y`` on the columns of ``X`` by least squares (QR).

    ``X`` must already contain the intercept column if one is wanted (see
    :func:`design_matrix`). R² is centred, so it is only meaningful with an intercept.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch: X {X.shape}, y {y.shape}")
    n, k = X.shape
    if n <= k:
        raise ValueError(f"need more observations than parameters (n={n}, k={k})")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("X and y must be finite")

    Q, R = np.linalg.qr(X, mode="reduced")
    diag = np.abs(np.diag(R))
    if diag.min() <= np.finfo(float).eps * diag.max() * max(n, k):
        raise np.linalg.LinAlgError("design matrix is rank deficient")

    coef = linalg.solve_triangular(R, Q.T @ y)
    fitted = X @ coef
    resid = y - fitted
    df = n - k
    rss = float(resid @ resid)
    sigma2 = rss / df

    # (X^T X)^{-1} = R^{-1} R^{-T}; only its diagonal is needed for the standard errors.
    R_inv = linalg.solve_triangular(R, np.eye(k))
    se = np.sqrt(sigma2 * np.sum(R_inv**2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):  # exact fits give se = 0
        t = coef / se
    p = 2 * stats.t.sf(np.abs(t), df)

    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df
    if k > 1:
        f_stat = ((tss - rss) / (k - 1)) / sigma2 if sigma2 > 0 else float("inf")
        f_pvalue = float(stats.f.sf(f_stat, k - 1, df))
    else:
        f_stat, f_pvalue = float("nan"), float("nan")

    if names is None:
        names = [f"x{i}" for i in range(k)]
    if len(names) != k:
        raise ValueError(f"{len(names)} names for {k} columns")

    return OLSResult(
        names=tuple(names),
        coef=coef,
        se=se,
        t=t,
        p=p,
        fitted=fitted,
        resid=resid,
        sigma2=sigma2,
        n=n,
        k=k,
        r2=r2,
        adj_r2=adj_r2,
        f_stat=float(f_stat),
        f_pvalue=f_pvalue,
        cond=float(np.linalg.cond(X)),
    )


def normal_equations(X: np.ndarray, y: Sequence[float] | np.ndarray) -> np.ndarray:
    """The coursework formula ``(X^T X)^{-1} X^T y``, with an explicit inverse.

    Kept only for comparison: it squares the condition number of ``X``.
    """
    X = np.asarray(X, dtype=float)
    return np.linalg.inv(X.T @ X) @ X.T @ np.asarray(y, dtype=float)


def pearson(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Pearson correlation ``S_xy / sqrt(S_xx S_yy)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - x.mean()
    dy = y - y.mean()
    return float(dx @ dy / np.sqrt((dx @ dx) * (dy @ dy)))


def residual_diagnostics(resid: np.ndarray) -> dict[str, float]:
    """Shape statistics of the residuals and the Jarque-Bera normality test."""
    resid = np.asarray(resid, dtype=float)
    jb = stats.jarque_bera(resid)
    return {
        "mean": float(resid.mean()),
        "median": float(np.median(resid)),
        "skewness": float(stats.skew(resid)),
        "excess_kurtosis": float(stats.kurtosis(resid)),
        "jarque_bera": float(jb.statistic),
        "jarque_bera_p": float(jb.pvalue),
    }


def breusch_pagan(X: np.ndarray, resid: np.ndarray) -> tuple[float, float]:
    """Koenker's studentized Breusch-Pagan test: ``n R²`` of ``e²`` on ``X``.

    Returns the LM statistic and its chi-squared p-value (``k - 1`` degrees of freedom).
    """
    X = np.asarray(X, dtype=float)
    aux = fit(X, np.asarray(resid, dtype=float) ** 2)
    lm = aux.n * aux.r2
    return float(lm), float(stats.chi2.sf(lm, X.shape[1] - 1))

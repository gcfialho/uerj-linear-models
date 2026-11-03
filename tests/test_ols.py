import numpy as np
import pytest
from scipy import stats

from linear_models import ols


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(10, 3, n)
    x2 = rng.uniform(-5, 5, n)
    beta = np.array([2.0, -1.5, 0.75])
    X = ols.design_matrix(x1, x2)
    y = X @ beta + rng.normal(0, 0.5, n)
    return X, y, beta


def test_matches_numpy_lstsq(synthetic):
    X, y, _ = synthetic
    res = ols.fit(X, y)
    expected, *_ = np.linalg.lstsq(X, y, rcond=None)
    np.testing.assert_allclose(res.coef, expected, rtol=1e-12, atol=1e-12)


def test_recovers_known_coefficients(synthetic):
    X, y, beta = synthetic
    res = ols.fit(X, y)
    # Within four standard errors of the truth, and the 95% intervals cover it.
    assert np.all(np.abs(res.coef - beta) < 4 * res.se)
    ci = res.conf_int(0.95)
    assert np.all((ci[:, 0] < beta) & (beta < ci[:, 1]))
    assert res.sigma2 == pytest.approx(0.25, rel=0.15)


def test_normal_equations_agree_on_well_conditioned_data(synthetic):
    X, y, _ = synthetic
    np.testing.assert_allclose(ols.normal_equations(X, y), ols.fit(X, y).coef, rtol=1e-9)


def test_residuals_orthogonal_to_columns(synthetic):
    X, y, _ = synthetic
    res = ols.fit(X, y)
    scale = np.abs(X).sum(axis=0) * np.abs(res.resid).max()
    assert np.all(np.abs(X.T @ res.resid) <= 1e-10 * scale)
    assert res.resid.mean() == pytest.approx(0, abs=1e-10)
    np.testing.assert_allclose(res.fitted + res.resid, y)


def test_r2_bounds_and_decomposition(synthetic):
    X, y, _ = synthetic
    res = ols.fit(X, y)
    assert 0 <= res.adj_r2 <= res.r2 <= 1
    tss = np.sum((y - y.mean()) ** 2)
    ess = np.sum((res.fitted - y.mean()) ** 2)
    assert ess + res.rss == pytest.approx(tss, rel=1e-10)


def test_simple_regression_closed_forms():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 2, 60)
    y = 1.0 + 0.3 * x + rng.normal(0, 1, 60)
    res = ols.fit(ols.design_matrix(x), y)
    rho = ols.pearson(x, y)

    assert rho == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-12)
    assert res.r2 == pytest.approx(rho**2, rel=1e-12)
    assert res.coef[1] == pytest.approx(rho * y.std() / x.std(), rel=1e-12)

    sxx = np.sum((x - x.mean()) ** 2)
    se_slope = np.sqrt(res.sigma2 / sxx)
    se_int = np.sqrt(res.sigma2 * (1 / len(x) + x.mean() ** 2 / sxx))
    np.testing.assert_allclose(res.se, [se_int, se_slope], rtol=1e-10)
    # With one regressor the F test is the square of the slope's t test.
    assert res.f_stat == pytest.approx(res.t[1] ** 2, rel=1e-10)
    assert res.f_pvalue == pytest.approx(res.p[1], rel=1e-8)


def test_p_values_and_t_test():
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    y = 0.5 * x + rng.normal(size=40)
    res = ols.fit(ols.design_matrix(x), y)
    assert res.p[1] == pytest.approx(2 * stats.t.sf(abs(res.t[1]), res.df_resid))
    t, p = res.t_test(1, res.coef[1])
    assert t == pytest.approx(0, abs=1e-12)
    assert p == pytest.approx(1.0)


def test_perfect_fit_has_r2_one():
    x = np.arange(10.0)
    y = 3 - 2 * x
    res = ols.fit(ols.design_matrix(x), y)
    np.testing.assert_allclose(res.coef, [3, -2], atol=1e-12)
    assert res.r2 == pytest.approx(1.0)


def test_rank_deficient_design_rejected():
    x = np.arange(10.0)
    with pytest.raises(np.linalg.LinAlgError):
        ols.fit(ols.design_matrix(x, 2 * x), x)


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (np.ones((3, 3)), np.ones(3)),  # n <= k
        (np.ones((5, 2)), np.ones(4)),  # shape mismatch
        (np.column_stack([np.ones(5), [1, 2, np.nan, 4, 5]]), np.ones(5)),  # NaN
    ],
)
def test_invalid_input_rejected(X, y):
    with pytest.raises(ValueError, match=r"."):
        ols.fit(X, y)


def test_diagnostics_on_normal_residuals():
    rng = np.random.default_rng(0)
    diag = ols.residual_diagnostics(rng.normal(size=5000))
    assert abs(diag["skewness"]) < 0.15
    assert abs(diag["excess_kurtosis"]) < 0.3
    assert diag["jarque_bera_p"] > 0.01


def test_breusch_pagan_detects_heteroscedasticity():
    rng = np.random.default_rng(1)
    x = rng.uniform(1, 10, 400)
    X = ols.design_matrix(x)
    homo = 1 + x + rng.normal(0, 1, 400)
    hetero = 1 + x + rng.normal(0, 1, 400) * x
    assert ols.breusch_pagan(X, ols.fit(X, homo).resid)[1] > 0.01
    assert ols.breusch_pagan(X, ols.fit(X, hetero).resid)[1] < 1e-6

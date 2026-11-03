import numpy as np
import pytest

from linear_models import analysis, data


@pytest.fixture(scope="module")
def df():
    return data.load()


def test_dataset_shape(df):
    assert len(df) == 235
    assert list(df.columns) == ["country", "population", "density", "land_area"]
    assert len(data.positive_rows(df)) == 232


def test_replicates_coursework_numbers(df):
    """The values printed in the original 2023 notebook."""
    r = analysis.replicate(df)
    assert r["simple"]["population"]["rho"] == pytest.approx(0.4557444362371723, rel=1e-9)
    assert r["simple"]["density"]["rho"] == pytest.approx(-0.058765963020454726, rel=1e-9)
    assert r["simple"]["population"]["r2"] == pytest.approx(0.20770299116113802, rel=1e-9)
    assert r["simple"]["density"]["r2"] == pytest.approx(0.003453438409721452, rel=1e-9)
    coefs = [c["coef"] for c in r["multiple"]["coefficients"].values()]
    np.testing.assert_allclose(
        coefs, [381688.0823883057, 0.005685266373947227, -34.18806501727921], rtol=1e-9
    )
    assert r["normal_equations"]["max_rel_diff_vs_qr"] < 1e-8


def test_linear_residuals_are_far_from_normal(df):
    res = analysis.replicate(df)["residuals"]
    assert res["skewness"] > 3
    assert res["jarque_bera_p"] < 1e-10
    assert res["top"][0]["country"] == "Russia"


def test_loglog_recovers_the_identity(df):
    r = analysis.loglog(df)
    assert r["dropped"] == ["Falkland Islands", "Greenland", "Holy See"]
    coefs = [c["coef"] for c in r["model"]["coefficients"].values()]
    np.testing.assert_allclose(coefs, [0, 1, -1], atol=0.01)
    assert r["model"]["r2"] > 0.9999
    robust = [c["coef"] for c in r["robust"]["model"]["coefficients"].values()]
    np.testing.assert_allclose(robust, [0, 1, -1], atol=0.002)

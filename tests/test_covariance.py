import pytest
from numpy import array, linspace, sin, isfinite
from numpy.random import default_rng
from inference.gp import SquaredExponential, RationalQuadratic, WhiteNoise, ChangePoint


def covar_error_check(K, dK_analytic, dK_findiff):
    small_element = abs(K / abs(K).max()) < 1e-4
    zero_grads = (dK_analytic == 0.0) & (dK_findiff == 0.0)
    ignore = small_element | zero_grads
    abs_frac_err = abs((dK_findiff - dK_analytic) / K)
    abs_frac_err[ignore] = 0.0

    assert isfinite(abs_frac_err).all()
    assert abs_frac_err.max() < 1e-5


def covar_findiff(cov_func=None, x0=None, delta=1e-6):
    grad = []
    for i in range(x0.size):
        x1 = x0.copy()
        x2 = x0.copy()
        dx = x0[i] * delta

        x1[i] -= dx
        x2[i] += dx

        f1 = cov_func.covariance_and_gradients(x1)[0]
        f2 = cov_func.covariance_and_gradients(x2)[0]
        grad.append(0.5 * (f2 - f1) / dx)
    return grad


def create_data():
    rng = default_rng(2)
    N = 20
    x = linspace(0, 10, N)
    y = sin(x) + rng.normal(loc=0.1, scale=0.1, size=N)
    return x.reshape([N, 1]), y


@pytest.mark.parametrize(
    "cov",
    [
        SquaredExponential(),
        RationalQuadratic(),
        WhiteNoise(),
        RationalQuadratic() + WhiteNoise(),
        ChangePoint(kernels=[SquaredExponential, SquaredExponential]),
        ChangePoint(kernels=[SquaredExponential, SquaredExponential]) + WhiteNoise(),
    ],
)
def test_covariance_and_gradients(cov):
    x, y = create_data()
    cov.pass_spatial_data(x)
    cov.estimate_hyperpar_bounds(y)
    low = array([a for a, b in cov.bounds])
    high = array([b for a, b in cov.bounds])

    # randomly sample positions in the hyperparameter space to test
    rng = default_rng(7)
    for _ in range(100):
        theta = rng.uniform(low=low, high=high, size=cov.n_params)
        K, dK_analytic = cov.covariance_and_gradients(theta)
        dK_findiff = covar_findiff(cov_func=cov, x0=theta)

        for dKa, dKf in zip(dK_analytic, dK_findiff):
            covar_error_check(K, dKa, dKf)

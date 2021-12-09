import pytest
from numpy import linspace, sin, isfinite
from numpy.random import normal, seed
from inference.gp import SquaredExponential, RationalQuadratic, WhiteNoise


def covar_findiff(cov_func=None, x0=None, delta=1e-5):
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
    seed(2)
    N = 20
    S = 0.1
    x = linspace(0, 10, N)
    y = sin(x) + 0.1 + normal(size=N) * S
    return x.reshape([N, 1]), y


@pytest.mark.parametrize(
    "cov",
    [
        SquaredExponential(),
        RationalQuadratic(),
        WhiteNoise(),
        RationalQuadratic() + WhiteNoise(),
    ],
)
def test_covariance_and_gradients(cov):
    x, y = create_data()
    cov.pass_data(x, y)
    theta = normal(size=cov.n_params) + 0.5
    _, grads = cov.covariance_and_gradients(theta)
    fd_grads = covar_findiff(cov_func=cov, x0=theta)

    for ang, fdg in zip(grads, fd_grads):
        zero_inds = (ang == 0.0) & (fdg == 0.0)
        abs_frac_err = abs(fdg / ang - 1.0)
        abs_frac_err[zero_inds] = 0.0
        assert isfinite(abs_frac_err).all()
        assert abs_frac_err.max() < 1e-6

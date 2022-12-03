from numpy import linspace, sin, cos, ndarray, full, zeros
from numpy.random import default_rng
from inference.gp import (
    GpRegressor,
    SquaredExponential,
    ChangePoint,
    WhiteNoise,
    RationalQuadratic,
)
import pytest


def finite_difference(
    func: callable, x0: ndarray, delta=1e-5, vectorised_arguments=False
):
    grad = zeros(x0.size)
    for i in range(x0.size):
        x1 = x0.copy()
        x2 = x0.copy()
        dx = x0[i] * delta

        x1[i] -= dx
        x2[i] += dx

        if vectorised_arguments:
            f1 = func(x1)
            f2 = func(x2)
        else:
            f1 = func(*x1)
            f2 = func(*x2)

        grad[i] = 0.5 * (f2 - f1) / dx
    return grad


def testing_data():
    n = 32
    rng = default_rng(1)
    points = rng.uniform(low=0.0, high=2.0, size=(n, 2))
    values = sin(points[:, 0]) * cos(points[:, 1]) + rng.normal(scale=0.1, size=n)
    errors = full(n, fill_value=0.1)
    return points, values, errors


@pytest.mark.parametrize(
    "kernel",
    [
        SquaredExponential(),
        RationalQuadratic(),
        RationalQuadratic() + WhiteNoise(),
        ChangePoint(kernels=[SquaredExponential, SquaredExponential]),
        ChangePoint(kernels=[SquaredExponential, SquaredExponential]) + WhiteNoise(),
    ],
)
def test_gpr_predictions(kernel):
    points, values, errors = testing_data()
    gpr = GpRegressor(x=points, y=values, y_err=errors, kernel=kernel)
    mu, sig = gpr(points)


def test_marginal_likelihood_gradient():
    points, values, errors = testing_data()
    gpr = GpRegressor(x=points, y=values, y_err=errors)
    # randomly sample some points in the hyperparameter space to test
    rng = default_rng(123)
    n_samples = 20
    theta_vectors = rng.uniform(
        low=[-0.3, -1.5, 0.1, 0.1], high=[0.3, 0.5, 1.5, 1.5], size=[n_samples, 4]
    )
    # check the gradient at each point using finite-difference
    for theta in theta_vectors:
        _, grad_lml = gpr.marginal_likelihood_gradient(theta)
        fd_grad = finite_difference(
            func=gpr.marginal_likelihood, x0=theta, vectorised_arguments=True
        )
        assert abs(fd_grad / grad_lml - 1.0).max() < 1e-5


def test_loo_likelihood_gradient():
    points, values, errors = testing_data()
    gpr = GpRegressor(x=points, y=values, y_err=errors)
    # randomly sample some points in the hyperparameter space to test
    rng = default_rng(137)
    n_samples = 20
    theta_vectors = rng.uniform(
        low=[-0.3, -1.5, 0.1, 0.1], high=[0.3, 0.5, 1.5, 1.5], size=[n_samples, 4]
    )
    # check the gradient at each point using finite-difference
    for theta in theta_vectors:
        _, grad_lml = gpr.loo_likelihood_gradient(theta)
        fd_grad = finite_difference(
            func=gpr.loo_likelihood, x0=theta, vectorised_arguments=True
        )
        assert abs(fd_grad / grad_lml - 1.0).max() < 1e-5


def test_gradient():
    rng = default_rng(42)
    N = 10
    S = 1.1
    x = linspace(0, 10, N)
    y = 0.3 * x + 0.02 * x**3 + 5.0 + rng.normal(size=N) * S
    err = zeros(N) + S

    gpr = GpRegressor(x, y, y_err=err)

    sample_x = linspace(0, 10, 120)
    delta = 1e-5
    grad, grad_sigma = gpr.gradient(sample_x)

    mu_pos, sig_pos = gpr(sample_x + delta)
    mu_neg, sig_neg = gpr(sample_x - delta)

    fd_grad = (mu_pos - mu_neg) / (2 * delta)
    grad_max_frac_error = abs(grad / fd_grad - 1.0).max()

    assert grad_max_frac_error < 1e-6


def test_spatial_derivatives():
    rng = default_rng(401)
    N = 10
    S = 1.1
    x = linspace(0, 10, N)
    y = 0.3 * x + 0.02 * x**3 + 5.0 + rng.normal(size=N) * S
    err = zeros(N) + S

    gpr = GpRegressor(x, y, y_err=err)

    sample_x = linspace(0, 10, 120)
    delta = 1e-5
    grad_mu, grad_var = gpr.spatial_derivatives(sample_x)

    mu_pos, sig_pos = gpr(sample_x + delta)
    mu_neg, sig_neg = gpr(sample_x - delta)

    fd_grad_mu = (mu_pos - mu_neg) / (2 * delta)
    fd_grad_var = (sig_pos**2 - sig_neg**2) / (2 * delta)

    mu_max_frac_error = abs(grad_mu / fd_grad_mu - 1.0).max()
    var_max_frac_error = abs(grad_var / fd_grad_var - 1.0).max()

    assert mu_max_frac_error < 1e-6
    assert var_max_frac_error < 1e-4


def test_optimizers():
    x, y, errors = testing_data()
    gpr = GpRegressor(x, y, y_err=errors, optimizer="bfgs", n_starts=6)
    gpr = GpRegressor(x, y, y_err=errors, optimizer="bfgs", n_processes=2)
    gpr = GpRegressor(x, y, y_err=errors, optimizer="diffev")


def test_input_consistency_checking():
    with pytest.raises(ValueError):
        GpRegressor(x=zeros(3), y=zeros(2))
    with pytest.raises(ValueError):
        GpRegressor(x=zeros([4, 3]), y=zeros(3))
    with pytest.raises(ValueError):
        GpRegressor(x=zeros([3, 1]), y=zeros([3, 2]))

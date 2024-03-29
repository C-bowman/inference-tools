from numpy import allclose, sqrt, ndarray, linspace, zeros, ones
from numpy.random import default_rng
from scipy.special import erfc
from inference.gp import SquaredExponential, RationalQuadratic, WhiteNoise
from inference.gp import GpLinearInverter
import pytest


def finite_difference(
    func: callable, x0: ndarray, delta=1e-4, vectorised_arguments=False
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


def normal_cdf(x, mu=0.0, sigma=1.0):
    z = -(x - mu) / (sqrt(2) * sigma)
    return 0.5 * erfc(z)


def lorentzian(x, A, w, c):
    z = (x - c) / w
    return A / (1 + z**2)


def build_test_data():
    # construct a test solution
    n_data, n_basis = 32, 64
    x = linspace(-1, 1, n_basis)
    data_axis = linspace(-1, 1, n_data)
    dx = 0.5 * (x[1] - x[0])
    solution = lorentzian(x, 1.0, 0.1, 0.0)
    solution += lorentzian(x, 0.8, 0.15, 0.3)
    solution += lorentzian(x, 0.3, 0.1, -0.45)

    # create a gaussian blur forward model matrix
    A = zeros([n_data, n_basis])
    blur_width = 0.075
    for k in range(n_basis):
        A[:, k] = normal_cdf(data_axis + dx, mu=x[k], sigma=blur_width)
        A[:, k] -= normal_cdf(data_axis - dx, mu=x[k], sigma=blur_width)

    # create some testing data using the forward model
    noise_sigma = 0.02
    rng = default_rng(123)
    y = A @ solution + rng.normal(size=n_data, scale=noise_sigma)
    y_err = zeros(n_data) + noise_sigma
    return x, y, y_err, A


@pytest.mark.parametrize(
    "cov_func",
    [
        SquaredExponential(),
        RationalQuadratic(),
        WhiteNoise(),
        RationalQuadratic() + SquaredExponential(),
    ],
)
def test_gp_linear_inverter(cov_func):
    x, y, y_err, A = build_test_data()

    # set up the inverter
    GLI = GpLinearInverter(
        model_matrix=A,
        y=y,
        y_err=y_err,
        parameter_spatial_positions=x.reshape([x.size, 1]),
        prior_covariance_function=cov_func,
    )

    # solve for the posterior mean and covariance
    theta_opt = GLI.optimize_hyperparameters(initial_guess=ones(GLI.n_hyperpars))
    mu, cov = GLI.calculate_posterior(theta_opt)
    mu_alt = GLI.calculate_posterior_mean(theta_opt)
    assert allclose(mu, mu_alt)

    # check that the forward prediction of the solution
    # matches the testing data
    chi_sqr = (((y - A @ mu) / y_err) ** 2).mean()
    assert chi_sqr <= 1.5


@pytest.mark.parametrize(
    "cov_func",
    [
        SquaredExponential(),
        RationalQuadratic(),
        WhiteNoise(),
        RationalQuadratic() + SquaredExponential(),
    ],
)
def test_gp_linear_inverter_lml_gradient(cov_func):
    x, y, y_err, A = build_test_data()

    GLI = GpLinearInverter(
        model_matrix=A,
        y=y,
        y_err=y_err,
        parameter_spatial_positions=x.reshape([x.size, 1]),
        prior_covariance_function=cov_func,
    )

    rng = default_rng(1)
    test_points = rng.uniform(low=0.1, high=1.0, size=(20, GLI.n_hyperpars))

    for theta in test_points:
        grad_fd = finite_difference(
            func=GLI.marginal_likelihood, x0=theta, vectorised_arguments=True
        )

        _, grad_analytic = GLI.marginal_likelihood_gradient(theta)
        abs_frac_error = abs(grad_fd / grad_analytic - 1.0).max()
        assert abs_frac_error < 1e-3

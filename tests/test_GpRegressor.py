from numpy import array, linspace, sin, zeros
from numpy.random import normal, seed
from inference.gp import GpRegressor


def test_marginal_likelihood_gradient():
    seed(1)
    N = 20
    S = 0.1
    x = linspace(0, 10, N)
    y = sin(x) + 3.0 + normal(size=N) * S
    errors = zeros(N) + S

    gpr = GpRegressor(x, y, y_err=errors)

    M = 2.5
    A = 0.1
    L = 0.6
    delta = 1e-5

    lml, grad_lml = gpr.marginal_likelihood_gradient(array([M, A, L]))

    M_pos = gpr.marginal_likelihood(array([M * (1 + delta), A, L]))
    M_neg = gpr.marginal_likelihood(array([M * (1 - delta), A, L]))

    A_pos = gpr.marginal_likelihood(array([M, A * (1 + delta), L]))
    A_neg = gpr.marginal_likelihood(array([M, A * (1 - delta), L]))

    L_pos = gpr.marginal_likelihood(array([M, A, L * (1 + delta)]))
    L_neg = gpr.marginal_likelihood(array([M, A, L * (1 - delta)]))

    fd_grad_M = (M_pos - M_neg) / (2 * M * delta)
    fd_grad_A = (A_pos - A_neg) / (2 * A * delta)
    fd_grad_L = (L_pos - L_neg) / (2 * L * delta)

    grad_M, grad_A, grad_L = grad_lml

    M_fractional_error = abs(fd_grad_M / grad_M - 1.0).max()
    A_fractional_error = abs(fd_grad_A / grad_A - 1.0).max()
    L_fractional_error = abs(fd_grad_L / grad_L - 1.0).max()

    assert M_fractional_error < 1e-6
    assert A_fractional_error < 1e-6
    assert L_fractional_error < 1e-6


def test_loo_likelihood_gradient():
    seed(1)
    N = 20
    S = 0.1
    x = linspace(0, 10, N)
    y = sin(x) + 3.0 + normal(size=N) * S
    errors = zeros(N) + S

    gpr = GpRegressor(x, y, y_err=errors)

    M = 2.5
    A = 0.1
    L = 0.6
    delta = 1e-5

    lml, grad_lml = gpr.loo_likelihood_gradient(array([M, A, L]))

    M_pos = gpr.loo_likelihood(array([M * (1 + delta), A, L]))
    M_neg = gpr.loo_likelihood(array([M * (1 - delta), A, L]))

    A_pos = gpr.loo_likelihood(array([M, A * (1 + delta), L]))
    A_neg = gpr.loo_likelihood(array([M, A * (1 - delta), L]))

    L_pos = gpr.loo_likelihood(array([M, A, L * (1 + delta)]))
    L_neg = gpr.loo_likelihood(array([M, A, L * (1 - delta)]))

    fd_grad_M = (M_pos - M_neg) / (2 * M * delta)
    fd_grad_A = (A_pos - A_neg) / (2 * A * delta)
    fd_grad_L = (L_pos - L_neg) / (2 * L * delta)

    grad_M, grad_A, grad_L = grad_lml

    M_fractional_error = abs(fd_grad_M / grad_M - 1.0).max()
    A_fractional_error = abs(fd_grad_A / grad_A - 1.0).max()
    L_fractional_error = abs(fd_grad_L / grad_L - 1.0).max()

    assert M_fractional_error < 1e-6
    assert A_fractional_error < 1e-6
    assert L_fractional_error < 1e-6


def test_gradient():
    seed(4)
    N = 10
    S = 1.1
    x = linspace(0, 10, N)
    y = 0.3 * x + 0.02 * x**3 + 5.0 + normal(size=N) * S
    err = zeros(N) + S

    gp = GpRegressor(x, y, y_err=err)

    sample_x = linspace(0, 10, 120)
    delta = 1e-4
    grad, grad_sigma = gp.gradient(sample_x)

    mu_pos, sig_pos = gp(sample_x + delta)
    mu_neg, sig_neg = gp(sample_x - delta)

    fd_grad = (mu_pos - mu_neg) / (2 * delta)
    grad_max_frac_error = (grad / fd_grad - 1.0).max()

    assert grad_max_frac_error < 1e-6


def test_spatial_derivatives():
    seed(4)
    N = 10
    S = 1.1
    x = linspace(0, 10, N)
    y = 0.3 * x + 0.02 * x**3 + 5.0 + normal(size=N) * S
    err = zeros(N) + S

    gp = GpRegressor(x, y, y_err=err)

    sample_x = linspace(0, 10, 120)
    delta = 1e-4
    grad_mu, grad_var = gp.spatial_derivatives(sample_x)

    mu_pos, sig_pos = gp(sample_x + delta)
    mu_neg, sig_neg = gp(sample_x - delta)

    fd_grad_mu = (mu_pos - mu_neg) / (2 * delta)
    fd_grad_var = (sig_pos**2 - sig_neg**2) / (2 * delta)

    mu_max_frac_error = (grad_mu / fd_grad_mu - 1.0).max()
    var_max_frac_error = (grad_var / fd_grad_var - 1.0).max()

    assert mu_max_frac_error < 1e-6
    assert var_max_frac_error < 1e-6


def test_optimizers():
    seed(1)
    N = 20
    S = 0.1
    x = linspace(0, 10, N)
    y = sin(x) + 3.0 + normal(size=N) * S
    errors = zeros(N) + S

    gpr = GpRegressor(x, y, y_err=errors, optimizer="bfgs")
    gpr = GpRegressor(x, y, y_err=errors, optimizer="bfgs", n_processes=2)
    gpr = GpRegressor(x, y, y_err=errors, optimizer="diffev")

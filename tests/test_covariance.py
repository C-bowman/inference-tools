from numpy import linspace, sin, eye
from numpy.random import normal, seed
from inference.gp_tools import SquaredExponential, RationalQuadratic


def test_SquaredExponential_covariance_and_gradients():
    seed(2)
    N = 20
    S = 0.1
    x = linspace(0, 10, N).reshape([N, 1])
    y = sin(x) + 0.1 + normal(size=N) * S

    cov = SquaredExponential()
    cov.pass_data(x, y)

    A = 0.1
    L = 0.6
    delta = 1e-5
    _, grads = cov.covariance_and_gradients([A, L])
    K_A_pos, _ = cov.covariance_and_gradients([A * (1 + delta), L])
    K_A_neg, _ = cov.covariance_and_gradients([A * (1 - delta), L])
    K_L_pos, _ = cov.covariance_and_gradients([A, L * (1 + delta)])
    K_L_neg, _ = cov.covariance_and_gradients([A, L * (1 - delta)])

    grad_A, grad_L = grads

    fd_grad_A = (K_A_pos - K_A_neg) / (2 * A * delta)
    fd_grad_L = (K_L_pos - K_L_neg) / (2 * L * delta)

    D = 1e-50 * eye(N)
    A_max_fractional_error = abs(fd_grad_A / grad_A - 1).max()
    L_max_fractional_error = abs((fd_grad_L + D) / (grad_L + D) - 1).max()

    assert A_max_fractional_error < 1e-6
    assert L_max_fractional_error < 1e-6


def test_RationalQuadratic_covariance_and_gradients():
    seed(2)
    N = 20
    S = 0.1
    x = linspace(0, 10, N).reshape([N, 1])
    y = sin(x) + 0.1 + normal(size=N) * S

    cov = RationalQuadratic()
    cov.pass_data(x, y)

    A = 0.1
    q = 1.2
    L = 0.6
    delta = 1e-5
    _, grads = cov.covariance_and_gradients([A, q, L])
    grad_A, grad_q, grad_L = grads

    K_A_pos, _ = cov.covariance_and_gradients([A * (1 + delta), q, L])
    K_A_neg, _ = cov.covariance_and_gradients([A * (1 - delta), q, L])
    K_q_pos, _ = cov.covariance_and_gradients([A, q * (1 + delta), L])
    K_q_neg, _ = cov.covariance_and_gradients([A, q * (1 - delta), L])
    K_L_pos, _ = cov.covariance_and_gradients([A, q, L * (1 + delta)])
    K_L_neg, _ = cov.covariance_and_gradients([A, q, L * (1 - delta)])

    fd_grad_A = (K_A_pos - K_A_neg) / (2 * A * delta)
    fd_grad_q = (K_q_pos - K_q_neg) / (2 * q * delta)
    fd_grad_L = (K_L_pos - K_L_neg) / (2 * L * delta)

    D = 1e-50 * eye(N)
    A_max_fractional_error = abs(fd_grad_A / grad_A - 1).max()
    q_max_fractional_error = abs((fd_grad_q + D) / (grad_q + D) - 1).max()
    L_max_fractional_error = abs((fd_grad_L + D) / (grad_L + D) - 1).max()

    assert A_max_fractional_error < 1e-6
    assert q_max_fractional_error < 1e-6
    assert L_max_fractional_error < 1e-6

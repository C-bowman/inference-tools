
import pytest
import unittest
from numpy import array, zeros, allclose
from inference.priors import GaussianPrior, ExponentialPrior, UniformPrior, JointPrior

def finite_difference(func=None, x0=None, delta=1e-5, vectorised_arguments=False):
    grad = zeros(x0.size)
    for i in range(x0.size):
        x1 = x0.copy()
        x2 = x0.copy()
        dx = x0[i]*delta

        x1[i] -= dx
        x2[i] += dx

        if vectorised_arguments:
            f1 = func(x1)
            f2 = func(x2)
        else:
            f1 = func(*x1)
            f2 = func(*x2)

        grad[i] = 0.5*(f2-f1)/dx
    return grad





class test_priors(unittest.TestCase):

    def test_GaussianPrior(self):
        # test combining multiple Gaussians
        P1 = GaussianPrior(mean=10, sigma=1, variable_indices=[0])
        P2 = GaussianPrior(mean=20, sigma=2, variable_indices=[1])
        P3 = GaussianPrior(mean=[30,40], sigma=[3,4], variable_indices=[2,3])
        combo = GaussianPrior.combine([P1, P2, P3])
        # check the combined distribution has the right values
        assert (combo.mean == array([10., 20., 30., 40.])).all()
        assert (combo.sigma == array([1., 2., 3., 4.])).all()
        assert all(a == b for a, b in zip(combo.variables, [0, 1, 2, 3]))
        # evaluate the prior at a test point
        test_point = array([9.,21.,34.,35.])
        log_prob = combo(test_point)
        # check the analytic gradient calculation against finite difference
        analytic_gradient = combo.gradient(test_point)
        numeric_gradient = finite_difference(func=combo, x0=test_point, vectorised_arguments=True)
        assert allclose(analytic_gradient, numeric_gradient)

    def test_ExponentialPrior(self):
        # test combining multiple exponentials
        P1 = ExponentialPrior(beta=10, variable_indices=[0])
        P2 = ExponentialPrior(beta=20, variable_indices=[1])
        P3 = ExponentialPrior(beta=[30,40], variable_indices=[2, 3])
        combo = ExponentialPrior.combine([P1, P2, P3])
        # check the combined distribution has the right values
        assert (combo.beta == array([10., 20., 30., 40.])).all()
        assert all(a == b for a, b in zip(combo.variables, [0, 1, 2, 3]))
        # evaluate the prior at a test point
        test_point = array([9., 21., 34., 35.])
        test_point_log_prob = combo(test_point)
        # check the analytic gradient calculation against finite difference
        analytic_gradient = combo.gradient(test_point)
        numeric_gradient = finite_difference(func=combo, x0=test_point, vectorised_arguments=True)
        assert allclose(analytic_gradient, numeric_gradient)

    def test_UniformPrior(self):
        # test combining multiple uniforms
        P1 = UniformPrior(lower=2, upper=4, variable_indices=[0])
        P2 = UniformPrior(lower=4, upper=8, variable_indices=[1])
        P3 = UniformPrior(lower=[8,16], upper=[16,32], variable_indices=[2, 3])
        combo = UniformPrior.combine([P1, P2, P3])
        # check the combined distribution has the right values
        assert (combo.lower == array([2., 4., 8., 16.])).all()
        assert (combo.upper == array([4., 8., 16., 32.])).all()
        assert all(a == b for a, b in zip(combo.variables, [0, 1, 2, 3]))
        # evaluate the prior at a test point
        test_point = array([3., 5., 15., 19.])
        test_point_log_prob = combo(test_point)
        # check the analytic gradient calculation against finite difference
        analytic_gradient = combo.gradient(test_point)
        numeric_gradient = finite_difference(func=combo, x0=test_point, vectorised_arguments=True)
        assert allclose(analytic_gradient, numeric_gradient)

    def test_JointPrior(self):
        P1 = ExponentialPrior(beta=10, variable_indices=[1])
        P2 = GaussianPrior(mean=20, sigma=2, variable_indices=[0])
        P3 = UniformPrior(lower=8, upper=16, variable_indices=[2])
        JP = JointPrior(components=[P1, P2, P3], n_variables=3)
        test_point = array([4., 23., 12.])
        test_point_log_prob = JP(test_point)
        analytic_gradient = JP.gradient(test_point)
        numeric_gradient = finite_difference(func=JP, x0=test_point, vectorised_arguments=True)
        assert allclose(analytic_gradient, numeric_gradient)
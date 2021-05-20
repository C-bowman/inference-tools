
import pytest
import unittest
from numpy import array, allclose, linspace, exp, sin, cos, zeros
from numpy.random import normal
from inference.likelihoods import GaussianLikelihood, CauchyLikelihood, LogisticLikelihood


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


class TestingModel(object):
    def __init__(self):
        self.x = linspace(0, 10, 51)
        self.N_data = self.x.size
        self.N_params = 3

    def forward(self, theta):
        A, k, f = theta
        return A*exp(-k*self.x)*sin(f*self.x)

    def gradient(self, theta):
        A, k, f = theta
        partials = zeros([self.N_data, self.N_params])
        exp_term = exp(-k*self.x)
        sin_term = sin(f*self.x)

        partials[:,0] = exp_term*sin_term
        partials[:,1] = -self.x*A*exp_term*sin_term
        partials[:,2] = self.x*A*exp_term*cos(f*self.x)
        return partials

    def generate_test_data(self, theta, error=1.):
        return self.forward(theta) + error*normal(size=self.N_data), zeros(self.N_data)+error






class test_likelihoods(unittest.TestCase):

    def test_GaussianLikelihood(self):
        model = TestingModel()
        y, sigma = model.generate_test_data([10., 0.2, 2.], error=1.5)

        GL = GaussianLikelihood(y_data=y, sigma=sigma, forward_model=model.forward, forward_model_gradient=model.gradient)

        test_point = array([12., 0.25, 1.4])
        test_likelihood = GL(test_point)
        analytic_gradient = GL.gradient(test_point)
        numeric_gradient = finite_difference(func=GL, x0=test_point, vectorised_arguments=True)

        assert allclose(analytic_gradient, numeric_gradient)

    def test_CauchyLikelihood(self):
        model = TestingModel()
        y, sigma = model.generate_test_data([10., 0.2, 2.], error=1.5)

        CL = CauchyLikelihood(y_data=y, gamma=sigma, forward_model=model.forward, forward_model_gradient=model.gradient)

        test_point = array([12., 0.25, 1.4])
        test_likelihood = CL(test_point)
        analytic_gradient = CL.gradient(test_point)
        numeric_gradient = finite_difference(func=CL, x0=test_point, vectorised_arguments=True)

        assert allclose(analytic_gradient, numeric_gradient)

    def test_LogisticLikelihood(self):
        model = TestingModel()
        y, sigma = model.generate_test_data([10., 0.2, 2.], error=1.5)

        LL = LogisticLikelihood(y_data=y, sigma=sigma, forward_model=model.forward, forward_model_gradient=model.gradient)

        test_point = array([12., 0.25, 1.4])
        test_likelihood = LL(test_point)
        analytic_gradient = LL.gradient(test_point)
        numeric_gradient = finite_difference(func=LL, x0=test_point, vectorised_arguments=True)

        assert allclose(analytic_gradient, numeric_gradient)

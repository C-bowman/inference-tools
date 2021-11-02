from numpy import array, allclose, linspace, exp, sin, cos, zeros, ones
from numpy.random import default_rng
from inference.likelihoods import (
    GaussianLikelihood,
    CauchyLikelihood,
    LogisticLikelihood,
)

import pytest


def finite_difference(func=None, x0=None, delta=1e-5, vectorised_arguments=False):
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


class ModelTesting(object):
    def __init__(self):
        self.x = linspace(0, 10, 51)
        self.N_data = self.x.size
        self.N_params = 3

    def forward(self, theta):
        A, k, f = theta
        return A * exp(-k * self.x) * sin(f * self.x)

    def jacobian(self, theta):
        A, k, f = theta
        partials = zeros([self.N_data, self.N_params])
        exp_term = exp(-k * self.x)
        sin_term = sin(f * self.x)

        partials[:, 0] = exp_term * sin_term
        partials[:, 1] = -self.x * A * exp_term * sin_term
        partials[:, 2] = self.x * A * exp_term * cos(f * self.x)
        return partials

    def generate_test_data(self, theta, error=1.0):
        return (
            self.forward(theta) + error * default_rng(1324).normal(size=self.N_data),
            zeros(self.N_data) + error,
        )


def test_GaussianLikelihood():
    model = ModelTesting()
    y, sigma = model.generate_test_data([10.0, 0.2, 2.0], error=1.5)

    GL = GaussianLikelihood(
        y_data=y,
        sigma=sigma,
        forward_model=model.forward,
        forward_model_jacobian=model.jacobian,
    )

    assert GL.gradient_available

    test_point = array([12.0, 0.25, 1.4])
    test_likelihood = GL(test_point)

    assert test_likelihood < 0.0

    analytic_gradient = GL.gradient(test_point)
    numeric_gradient = finite_difference(
        func=GL, x0=test_point, vectorised_arguments=True
    )

    assert allclose(analytic_gradient, numeric_gradient)


def test_GaussianLikelihood_needs_callable_forward_model():
    with pytest.raises(ValueError):
        GaussianLikelihood(y_data=zeros(1), sigma=ones(1), forward_model=None)


def test_GaussianLikelihood_needs_callable_forward_model_jacobian():
    with pytest.raises(ValueError):
        GaussianLikelihood(
            y_data=zeros(1),
            sigma=zeros(1),
            forward_model=lambda x: None,
            forward_model_jacobian=1,
        )


def test_GaussianLikelihood_gradient_raises_error_without_jacobian():
    likelihood = GaussianLikelihood(
        y_data=ones(1),
        sigma=ones(1),
        forward_model=lambda x: None,
        forward_model_jacobian=None,
    )

    assert not likelihood.gradient_available

    with pytest.raises(ValueError):
        likelihood.gradient(4)


def test_GaussianLikelihood_inconsistent_sizes():
    with pytest.raises(ValueError):
        GaussianLikelihood(y_data=ones(3), sigma=ones(1), forward_model=lambda: None)


def test_GaussianLikelihood_too_many_dims():
    with pytest.raises(ValueError):
        GaussianLikelihood(
            y_data=ones((2, 2)), sigma=ones(4), forward_model=lambda: None
        )

    with pytest.raises(ValueError):
        GaussianLikelihood(
            y_data=ones(4), sigma=ones((2, 2)), forward_model=lambda: None
        )


def test_GaussianLikelihood_bad_sigma():
    with pytest.raises(ValueError):
        GaussianLikelihood(y_data=ones(1), sigma=zeros(1), forward_model=lambda: None)


def test_CauchyLikelihood():
    model = ModelTesting()
    y, sigma = model.generate_test_data([10.0, 0.2, 2.0], error=1.5)

    CL = CauchyLikelihood(
        y_data=y,
        gamma=sigma,
        forward_model=model.forward,
        forward_model_jacobian=model.jacobian,
    )

    assert CL.gradient_available

    test_point = array([12.0, 0.25, 1.4])
    test_likelihood = CL(test_point)

    assert test_likelihood < 0.0

    analytic_gradient = CL.gradient(test_point)
    numeric_gradient = finite_difference(
        func=CL, x0=test_point, vectorised_arguments=True
    )

    assert allclose(analytic_gradient, numeric_gradient)


def test_CauchyLikelihood_needs_callable_forward_model():
    with pytest.raises(ValueError):
        CauchyLikelihood(y_data=zeros(1), gamma=ones(1), forward_model=None)


def test_CauchyLikelihood_needs_callable_forward_model_jacobian():
    with pytest.raises(ValueError):
        CauchyLikelihood(
            y_data=zeros(1),
            gamma=zeros(1),
            forward_model=lambda x: None,
            forward_model_jacobian=1,
        )


def test_CauchyLikelihood_gradient_raises_error_without_jacobian():
    likelihood = CauchyLikelihood(
        y_data=ones(1),
        gamma=ones(1),
        forward_model=lambda x: None,
        forward_model_jacobian=None,
    )

    assert not likelihood.gradient_available

    with pytest.raises(ValueError):
        likelihood.gradient(4)


def test_CauchyLikelihood_inconsistent_sizes():
    with pytest.raises(ValueError):
        CauchyLikelihood(y_data=ones(3), gamma=ones(1), forward_model=lambda: None)


def test_CauchyLikelihood_too_many_dims():
    with pytest.raises(ValueError):
        CauchyLikelihood(y_data=ones((2, 2)), gamma=ones(4), forward_model=lambda: None)
    with pytest.raises(ValueError):
        CauchyLikelihood(y_data=ones(4), gamma=ones((2, 2)), forward_model=lambda: None)


def test_CauchyLikelihood_bad_gamma():
    with pytest.raises(ValueError):
        CauchyLikelihood(y_data=ones(1), gamma=zeros(1), forward_model=lambda: None)


def test_LogisticLikelihood():
    model = ModelTesting()
    y, sigma = model.generate_test_data([10.0, 0.2, 2.0], error=1.5)

    LL = LogisticLikelihood(
        y_data=y,
        sigma=sigma,
        forward_model=model.forward,
        forward_model_jacobian=model.jacobian,
    )

    assert LL.gradient_available

    test_point = array([12.0, 0.25, 1.4])
    test_likelihood = LL(test_point)

    assert test_likelihood < 0.0

    analytic_gradient = LL.gradient(test_point)
    numeric_gradient = finite_difference(
        func=LL, x0=test_point, vectorised_arguments=True
    )

    assert allclose(analytic_gradient, numeric_gradient)


def test_LogisticLikelihood_needs_callable_forward_model():
    with pytest.raises(ValueError):
        LogisticLikelihood(y_data=zeros(1), sigma=ones(1), forward_model=None)


def test_LogisticLikelihood_needs_callable_forward_model_jacobian():
    with pytest.raises(ValueError):
        LogisticLikelihood(
            y_data=zeros(1),
            sigma=zeros(1),
            forward_model=lambda x: None,
            forward_model_jacobian=1,
        )


def test_LogisticLikelihood_gradient_raises_error_without_jacobian():
    likelihood = LogisticLikelihood(
        y_data=ones(1),
        sigma=ones(1),
        forward_model=lambda x: None,
        forward_model_jacobian=None,
    )

    assert not likelihood.gradient_available

    with pytest.raises(ValueError):
        likelihood.gradient(4)


def test_LogisticLikelihood_inconsistent_sizes():
    with pytest.raises(ValueError):
        LogisticLikelihood(y_data=ones(3), sigma=ones(1), forward_model=lambda: None)


def test_LogisticLikelihood_too_many_dims():
    with pytest.raises(ValueError):
        LogisticLikelihood(
            y_data=ones((2, 2)), sigma=ones(4), forward_model=lambda: None
        )
    with pytest.raises(ValueError):
        LogisticLikelihood(
            y_data=ones(4), sigma=ones((2, 2)), forward_model=lambda: None
        )


def test_LogisticLikelihood_bad_sigma():
    with pytest.raises(ValueError):
        LogisticLikelihood(y_data=ones(1), sigma=zeros(1), forward_model=lambda: None)

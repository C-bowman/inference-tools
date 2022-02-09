from numpy import array, zeros, allclose, isclose, linspace, diff, ones, exp, sqrt, pi
from inference.priors import (
    GaussianPrior,
    ExponentialPrior,
    UniformPrior,
    JointPrior,
    BasePrior,
)

import pytest

from hypothesis import assume, given, strategies as st


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


@given(variable_inds=st.integers())
def test_check_variables_scalar(variable_inds):
    result = BasePrior.check_variables(variable_inds, 1)
    assert result == [variable_inds]


@given(variable_inds=st.integers(), length=st.integers())
def test_check_variables_scalar_bad(variable_inds, length):
    assume(length != 1)
    with pytest.raises(ValueError):
        BasePrior.check_variables(variable_inds, length)


@given(variable_inds=st.lists(st.integers(), unique=True))
def test_check_variables_list(variable_inds):
    result = BasePrior.check_variables(variable_inds, len(variable_inds))
    assert result == variable_inds


@given(variable_inds=st.lists(st.integers(), unique=True), length=st.integers())
def test_check_variables_list_bad_length(variable_inds, length):
    assume(length != len(variable_inds))
    with pytest.raises(ValueError):
        BasePrior.check_variables(variable_inds, length)


def test_check_variables_list_not_int():
    with pytest.raises(TypeError):
        BasePrior.check_variables([1, 3, 2.0, 4], 4)


def test_check_variables_list_repeated():
    with pytest.raises(ValueError):
        BasePrior.check_variables([1, 3, 4, 3], 4)


def test_GaussianPrior():
    mean = 20
    sigma = 4
    P1 = GaussianPrior(mean=mean, sigma=sigma, variable_indices=[0])

    # Check some basic properties
    assert P1.mean == mean
    assert P1.sigma == sigma
    assert P1.bounds == [(None, None)]

    # Check that log probability is symmetric about mean
    log_probability = [P1(array([i])) for i in range(1, mean * 2)]
    assert log_probability[:mean] == log_probability[mean - 1 :][::-1]

    def analytic_probability(x, mu, sigma):
        return (1.0 / sqrt(2 * pi * sigma**2)) * exp(
            -((x - mu) ** 2) / (2 * sigma**2)
        )

    test_point = mean - 2 * sigma
    assert isclose(
        exp(P1(array([test_point]))), analytic_probability(test_point, mean, sigma)
    )
    assert isclose(exp(P1(array([-1]))), analytic_probability(-1, mean, sigma))

    # Check sampling produces expected distribution
    sample = array([P1.sample() for _ in range(1000)])
    assert isclose(sample.mean(), mean, rtol=0.2)
    assert isclose(sample.std(), sigma, rtol=0.2)


def test_GaussianPrior_combined():
    # test combining multiple Gaussians
    P1 = GaussianPrior(mean=10, sigma=1, variable_indices=[0])
    P2 = GaussianPrior(mean=20, sigma=2, variable_indices=[1])
    P3 = GaussianPrior(mean=[30, 40], sigma=[3, 4], variable_indices=[2, 3])
    combo = GaussianPrior.combine([P1, P2, P3])
    # check the combined distribution has the right values
    assert (combo.mean == array([10.0, 20.0, 30.0, 40.0])).all()
    assert (combo.sigma == array([1.0, 2.0, 3.0, 4.0])).all()
    assert all(a == b for a, b in zip(combo.variables, [0, 1, 2, 3]))
    assert combo.bounds == [(None, None), (None, None), (None, None), (None, None)]


def test_GaussianPrior_gradient():
    # test combining multiple Gaussians
    P1 = GaussianPrior(mean=10, sigma=1, variable_indices=[0])
    P2 = GaussianPrior(mean=20, sigma=2, variable_indices=[1])
    P3 = GaussianPrior(mean=[30, 40], sigma=[3, 4], variable_indices=[2, 3])
    combo = GaussianPrior.combine([P1, P2, P3])
    # evaluate the prior at a test point
    test_point = array([9.0, 21.0, 34.0, 35.0])
    # check the analytic gradient calculation against finite difference
    analytic_gradient = combo.gradient(test_point)
    numeric_gradient = finite_difference(
        func=combo, x0=test_point, vectorised_arguments=True
    )
    assert allclose(analytic_gradient, numeric_gradient)


def test_GaussianPrior_inconsistent_size():
    with pytest.raises(ValueError):
        GaussianPrior(mean=zeros(1), sigma=ones(2), variable_indices=[0])


def test_GaussianPrior_inconsistent_size_squeezed_dim():
    with pytest.raises(ValueError):
        GaussianPrior(mean=zeros((2, 1, 2)), sigma=ones(2), variable_indices=[0, 1])


def test_GaussianPrior_inconsistent_dimensions():
    with pytest.raises(ValueError):
        GaussianPrior(mean=zeros((2, 2)), sigma=ones(4), variable_indices=[0])

    with pytest.raises(ValueError):
        GaussianPrior(mean=zeros(4), sigma=ones((2, 2)), variable_indices=[0])


def test_GaussianPrior_bad_sigma():
    with pytest.raises(ValueError):
        GaussianPrior(mean=zeros(2), sigma=[1.0, 0.0], variable_indices=[0])


def test_ExponentialPrior():
    beta = 10
    P1 = ExponentialPrior(beta=beta, variable_indices=[0])

    # Check some basic properties
    assert P1.bounds == [(0.0, None)]

    # Check that log probability is monotonically decreasing
    log_probability = [P1(array([i])) for i in linspace(0, 10)]
    assert all(diff(log_probability) < 0)

    def analytic_probability(x):
        if x < 0:
            return 0
        return (1.0 / beta) * exp(-x / beta)

    test_point = 2.0
    assert isclose(exp(P1(array([test_point]))), analytic_probability(test_point))
    assert isclose(exp(P1(array([-1]))), analytic_probability(-1))
    assert isclose(exp(P1(array([0]))), analytic_probability(0))


def test_ExponentialPrior_combine():
    # test combining multiple exponentials
    P1 = ExponentialPrior(beta=10, variable_indices=[0])
    P2 = ExponentialPrior(beta=20, variable_indices=[1])
    P3 = ExponentialPrior(beta=[30, 40], variable_indices=[2, 3])
    combo = ExponentialPrior.combine([P1, P2, P3])
    # check the combined distribution has the right values
    assert (combo.beta == array([10.0, 20.0, 30.0, 40.0])).all()
    assert all(a == b for a, b in zip(combo.variables, [0, 1, 2, 3]))


def test_ExponentialPrior_gradient():
    # test combining multiple exponentials
    P1 = ExponentialPrior(beta=10, variable_indices=[0])
    P2 = ExponentialPrior(beta=20, variable_indices=[1])
    P3 = ExponentialPrior(beta=[30, 40], variable_indices=[2, 3])
    combo = ExponentialPrior.combine([P1, P2, P3])
    # evaluate the prior at a test point
    test_point = array([9.0, 21.0, 34.0, 35.0])
    # check the analytic gradient calculation against finite difference
    analytic_gradient = combo.gradient(test_point)
    numeric_gradient = finite_difference(
        func=combo, x0=test_point, vectorised_arguments=True
    )
    assert allclose(analytic_gradient, numeric_gradient)
    # Point entirely outside of range
    assert (combo.gradient(array([-1, -4, -0.1, -1e-4])) == zeros(4)).all()
    # At exactly zero
    assert (combo.gradient(zeros(4)) == -1.0 / combo.beta).all()


def test_ExponentialPrior_bad_beta():
    with pytest.raises(ValueError):
        ExponentialPrior(beta=-1, variable_indices=[0])

    with pytest.raises(ValueError):
        ExponentialPrior(beta=[3.0, 0.0], variable_indices=[0])


def test_ExponentialPrior_bad_beta_too_many_dims():
    with pytest.raises(ValueError):
        ExponentialPrior(beta=ones((2, 2)), variable_indices=[0])


def test_UniformPrior():
    lower = 2
    upper = 4
    P1 = UniformPrior(lower=lower, upper=upper, variable_indices=[0])

    # Check some basic properties
    assert P1.lower == lower
    assert P1.upper == upper
    assert P1.bounds == [(lower, upper)]

    # Check sampling produces expected distribution
    sample = array([P1.sample() for _ in range(100)])
    assert all(sample >= lower)
    assert all(sample <= upper)

    # Check that probability is constant within the range
    theta_in_range = linspace(lower, upper, endpoint=True)
    log_probability = [P1(array([i])) for i in theta_in_range]
    assert all(log_probability == log_probability[0])
    assert exp(log_probability[0]) == 1.0 / (upper - lower)

    # Check that the probability is very small outside the range
    theta_left_range = linspace(-15, lower, endpoint=False)
    log_probability_left = array([P1(array([i])) for i in theta_left_range])
    assert all(log_probability_left < -1e32)
    theta_right_range = linspace(upper + 1e-5, 15)
    log_probability_right = array([P1(array([i])) for i in theta_right_range])
    assert all(log_probability_right < -1e32)


def test_UniformPrior_combine():
    # test combining multiple uniforms
    P1 = UniformPrior(lower=2, upper=4, variable_indices=[0])
    P2 = UniformPrior(lower=4, upper=8, variable_indices=[1])
    P3 = UniformPrior(lower=[8, 16], upper=[16, 32], variable_indices=[2, 3])
    combo = UniformPrior.combine([P1, P2, P3])
    # check the combined distribution has the right values
    assert (combo.lower == array([2.0, 4.0, 8.0, 16.0])).all()
    assert (combo.upper == array([4.0, 8.0, 16.0, 32.0])).all()
    assert combo.bounds == [(2, 4), (4, 8), (8, 16), (16, 32)]
    assert combo.variables == [0, 1, 2, 3]

    # Check sampling produces expected distribution
    sample = array([combo.sample() for _ in range(100)])
    for index in combo.variables:
        assert all(sample[:, index] >= combo.lower[index])
        assert all(sample[:, index] <= combo.upper[index])


def test_UniformPrior_gradient():
    # test combining multiple uniforms
    P1 = UniformPrior(lower=2, upper=4, variable_indices=[0])
    P2 = UniformPrior(lower=4, upper=8, variable_indices=[1])
    P3 = UniformPrior(lower=[8, 16], upper=[16, 32], variable_indices=[2, 3])
    combo = UniformPrior.combine([P1, P2, P3])
    # evaluate the prior at a test point
    test_point = array([3.0, 5.0, 15.0, 19.0])
    # check the analytic gradient calculation against finite difference
    analytic_gradient = combo.gradient(test_point)
    numeric_gradient = finite_difference(
        func=combo, x0=test_point, vectorised_arguments=True
    )
    assert allclose(analytic_gradient, numeric_gradient)


def test_UniformPrior_inconsistent_size():
    with pytest.raises(ValueError):
        UniformPrior(lower=zeros(1), upper=ones(2), variable_indices=[0])


def test_UniformPrior_inconsistent_dimensions():
    with pytest.raises(ValueError):
        UniformPrior(lower=zeros((2, 2)), upper=ones(4), variable_indices=[0])

    with pytest.raises(ValueError):
        UniformPrior(lower=zeros(4), upper=ones((2, 2)), variable_indices=[0])


def test_UniformPrior_bad_upper():
    with pytest.raises(ValueError):
        UniformPrior(lower=zeros(2), upper=[1.0, 0.0], variable_indices=[0])


def test_JointPrior():
    P1 = ExponentialPrior(beta=10, variable_indices=[1])
    P2 = GaussianPrior(mean=20, sigma=2, variable_indices=[0])
    P3 = UniformPrior(lower=8, upper=16, variable_indices=[2])
    components = [P1, P2, P3]
    JP = JointPrior(components=components, n_variables=3)

    sample = array([JP.sample() for _ in range(1000)])
    assert sample.shape[1] == len(components)
    assert isclose(sample[:, 0].mean(), P2.mean, rtol=0.2)
    assert isclose(sample[:, 0].std(), P2.sigma, rtol=0.2)
    for index, (lower, upper) in enumerate(JP.bounds):
        if lower is not None:
            assert all(sample[:, index] >= lower)
        if upper is not None:
            assert all(sample[:, index] <= upper)


def test_JointPrior_gradient():
    P1 = ExponentialPrior(beta=10, variable_indices=[1])
    P2 = GaussianPrior(mean=20, sigma=2, variable_indices=[0])
    P3 = UniformPrior(lower=8, upper=16, variable_indices=[2])
    JP = JointPrior(components=[P1, P2, P3], n_variables=3)
    test_point = array([4.0, 23.0, 12.0])
    analytic_gradient = JP.gradient(test_point)
    numeric_gradient = finite_difference(
        func=JP, x0=test_point, vectorised_arguments=True
    )
    assert allclose(analytic_gradient, numeric_gradient)


def test_JointPrior_bad_indices():
    P1 = ExponentialPrior(beta=10, variable_indices=[3])
    P2 = GaussianPrior(mean=20, sigma=2, variable_indices=[0])
    P3 = UniformPrior(lower=8, upper=16, variable_indices=[2])
    with pytest.raises(ValueError):
        JointPrior(components=[P1, P2, P3], n_variables=3)


def test_JointPrior_repeat_variable():
    P1 = ExponentialPrior(beta=10, variable_indices=[0])
    P2 = GaussianPrior(mean=20, sigma=2, variable_indices=[0])
    P3 = UniformPrior(lower=8, upper=16, variable_indices=[2])
    with pytest.raises(ValueError):
        JointPrior(components=[P1, P2, P3], n_variables=3)

from inference.pdf.diffusion import DiffusionKDE
from inference.pdf.hdi import sample_hdi
from inference.pdf.unimodal import UnimodalPdf
from inference.pdf.kde import GaussianKDE, BinaryTree, unique_index_groups
from inference.pdf import DensityEstimator

from dataclasses import dataclass
from numpy.random import default_rng
from numpy import array, ndarray, arange, linspace, concatenate, zeros
from numpy import isclose, allclose
from scipy.stats import norm, exponnorm
from scipy.integrate import simpson

import pytest
from hypothesis import given, strategies as st


@dataclass
class DensityTestCase:
    samples: ndarray
    fraction: float
    interval: tuple[float, float]
    mean: float
    variance: float
    skewness: float
    kurtosis: float
    x_axis: ndarray
    pdf_values: ndarray

    @classmethod
    def normal(cls, n_samples=20000):
        rng = default_rng(13)
        mu, sigma = 5.0, 2.0
        samples = rng.normal(loc=mu, scale=sigma, size=n_samples)
        x_axis = linspace(mu - 5 * sigma, mu + 5 * sigma, 1000)
        return cls(
            samples=samples,
            fraction=0.68269,
            interval=(mu - sigma, mu + sigma),
            mean=mu,
            variance=sigma**2,
            skewness=0.0,
            kurtosis=0.0,
            x_axis=x_axis,
            pdf_values=norm.pdf(x_axis, loc=mu, scale=sigma),
        )

    @classmethod
    def expgauss(cls, n_samples=20000):
        rng = default_rng(7)
        mu, sigma, lmbda = 5.0, 2.0, 0.25
        samples = rng.normal(loc=mu, scale=sigma, size=n_samples) + rng.exponential(
            scale=1.0 / lmbda, size=n_samples
        )
        v = 1 / (sigma * lmbda) ** 2
        K = 1.0 / (sigma * lmbda)
        x_axis = linspace(mu - 5 * sigma, mu + 3.0 / lmbda + 5 * sigma, 1000)
        return cls(
            samples=samples,
            fraction=0.68269,
            interval=(4.047, 11.252),
            mean=mu + 1.0 / lmbda,
            variance=sigma**2 + lmbda**-2,
            skewness=2.0 * (1 + 1 / v) ** -1.5,
            kurtosis=3 * (1 + 2 * v + 3 * v**2) / (1 + v) ** 2 - 3,
            x_axis=x_axis,
            pdf_values=exponnorm.pdf(x_axis, K, loc=mu, scale=sigma),
        )


@pytest.mark.parametrize("estimator_cls", [GaussianKDE, DiffusionKDE, UnimodalPdf])
def test_estimator_density(estimator_cls: DensityEstimator):
    testcase = DensityTestCase.expgauss()
    pdf = estimator_cls(testcase.samples)
    density_estimate = pdf(testcase.x_axis)
    abs_err = simpson(abs(testcase.pdf_values - density_estimate), x=testcase.x_axis)
    assert abs_err < 0.03


@pytest.mark.parametrize(
    "estimator_cls, n_samples, rtol, atol",
    [
        (GaussianKDE, 20000, 0.1, 0.0),
        (DiffusionKDE, 20000, 0.1, 0.0),
        (UnimodalPdf, 5000, 0.2, 0.0),
    ],
)
def test_estimator_moments(
    estimator_cls: DensityEstimator, n_samples: int, rtol: float, atol: float
):
    testcase = DensityTestCase.expgauss(n_samples=n_samples)
    pdf = estimator_cls(testcase.samples)
    mu, variance, skew, kurt = pdf.moments()

    assert isclose(mu, testcase.mean, rtol=rtol, atol=atol)
    assert isclose(variance, testcase.variance, rtol=rtol, atol=atol)
    assert isclose(skew, testcase.skewness, rtol=rtol, atol=atol)
    assert isclose(kurt, testcase.kurtosis, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "testcase",
    [DensityTestCase.normal(), DensityTestCase.expgauss()],
)
def test_gaussian_kde_interval(testcase):
    pdf = GaussianKDE(testcase.samples)
    left, right = pdf.interval(fraction=testcase.fraction)
    left_target, right_target = testcase.interval
    tolerance = (right_target - left_target) * 0.05
    assert isclose(left, left_target, rtol=0.0, atol=tolerance)
    assert isclose(right, right_target, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize(
    "testcase",
    [DensityTestCase.normal(n_samples=5000), DensityTestCase.expgauss(n_samples=5000)],
)
def test_unimodal_pdf_interval(testcase):
    pdf = UnimodalPdf(testcase.samples)
    left, right = pdf.interval(fraction=testcase.fraction)
    left_target, right_target = testcase.interval
    tolerance = (right_target - left_target) * 0.05
    assert isclose(left, left_target, rtol=0.0, atol=tolerance)
    assert isclose(right, right_target, rtol=0.0, atol=tolerance)


def test_gaussian_kde_plotting():
    N = 20000
    expected_mu = 5.0
    expected_sigma = 2.0

    sample = default_rng(1324).normal(expected_mu, expected_sigma, size=N)
    pdf = GaussianKDE(sample)
    min_value, max_value = pdf.interval(0.99)

    fig, ax = pdf.plot_summary(show=False, label="test label")
    assert ax[0].get_xlabel() == "test label"
    left, right, bottom, top = ax[0].axis()
    assert left <= min_value
    assert right >= max_value
    assert bottom <= 0.0
    assert top >= pdf(expected_mu)


def test_sample_hdi_gaussian():
    N = 20000
    expected_mu = 5.0
    expected_sigma = 3.0
    rng = default_rng(1324)

    # test for a single sample
    sample = rng.normal(expected_mu, expected_sigma, size=N)
    left, right = sample_hdi(sample, fraction=0.9545)

    tolerance = 0.2
    assert isclose(
        left, expected_mu - 2 * expected_sigma, rtol=tolerance, atol=tolerance
    )
    assert isclose(
        right, expected_mu + 2 * expected_sigma, rtol=tolerance, atol=tolerance
    )

    # test for a multiple samples
    sample = rng.normal(expected_mu, expected_sigma, size=[N, 3])
    intervals = sample_hdi(sample, fraction=0.9545)
    assert allclose(
        intervals[0, :],
        expected_mu - 2 * expected_sigma,
        rtol=tolerance,
        atol=tolerance,
    )
    assert allclose(
        intervals[1, :],
        expected_mu + 2 * expected_sigma,
        rtol=tolerance,
        atol=tolerance,
    )


@given(st.floats(min_value=1.0e-4, max_value=1, exclude_min=True, exclude_max=True))
def test_sample_hdi_linear(fraction):
    N = 20000
    sample = linspace(0, 1, N)

    left, right = sample_hdi(sample, fraction=fraction)

    assert left < right
    assert isclose(right - left, fraction, rtol=1e-2, atol=1e-2)

    inverse_sample = 1 - linspace(0, 1, N)

    left, right = sample_hdi(inverse_sample, fraction=fraction)

    assert left < right
    assert isclose(right - left, fraction, rtol=1e-2, atol=1e-2)


def test_sample_hdi_invalid_fractions():
    # Create some samples from the exponentially-modified Gaussian distribution
    sample = default_rng(1324).normal(size=3000)
    with pytest.raises(ValueError):
        sample_hdi(sample, fraction=2.0)
    with pytest.raises(ValueError):
        sample_hdi(sample, fraction=-0.1)


def test_sample_hdi_invalid_shapes():
    rng = default_rng(1324)
    sample_3D = rng.normal(size=[1000, 2, 2])
    with pytest.raises(ValueError):
        sample_hdi(sample_3D, fraction=0.65)

    sample_0D = array(0.0)
    with pytest.raises(ValueError):
        sample_hdi(sample_0D, fraction=0.65)

    sample_len1 = rng.normal(size=[1, 5])
    with pytest.raises(ValueError):
        sample_hdi(sample_len1, fraction=0.65)


def test_binary_tree():
    limit_left, limit_right = [-1.0, 1.0]
    tree = BinaryTree(2, (limit_left, limit_right))
    vals = array([-10.0 - 1.0, -0.9, 0.0, 0.4, 10.0])
    region_inds, groups = tree.region_groups(vals)
    assert (region_inds == array([0, 1, 2, 3])).all()


@pytest.mark.parametrize(
    "values",
    [
        default_rng(1).integers(low=0, high=6, size=124),
        default_rng(2).random(size=64),
        zeros(5, dtype=int) + 1,
        array([5.0]),
    ],
)
def test_unique_index_groups(values):
    uniques, groups = unique_index_groups(values)

    for u, g in zip(uniques, groups):
        assert (u == values[g]).all()

    k = concatenate(groups)
    k.sort()
    assert (k == arange(k.size)).all()

from inference.pdf.hdi import sample_hdi
from inference.pdf.unimodal import UnimodalPdf
from inference.pdf.kde import GaussianKDE, BinaryTree, unique_index_groups

from dataclasses import dataclass
from numpy.random import default_rng
from numpy import array, ndarray, arange, isclose, linspace, concatenate, zeros

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

    @classmethod
    def normal(cls, n_samples=20000):
        rng = default_rng(13)
        mu, sigma = 5.0, 2.0
        samples = rng.normal(loc=mu, scale=sigma, size=n_samples)
        return cls(
            samples=samples,
            fraction=0.68269,
            interval=(mu - sigma, mu + sigma),
            mean=mu,
            variance=sigma**2,
            skewness=0.0,
            kurtosis=0.0,
        )

    @classmethod
    def expgauss(cls, n_samples=20000):
        rng = default_rng(7)
        mu, sigma, lmbda = 5.0, 2.0, 0.25
        samples = rng.normal(loc=mu, scale=sigma, size=n_samples) + rng.exponential(
            scale=1.0 / lmbda, size=n_samples
        )
        v = 1 / (sigma * lmbda) ** 2
        return cls(
            samples=samples,
            fraction=0.68269,
            interval=(4.047, 11.252),
            mean=mu + 1.0 / lmbda,
            variance=sigma**2 + lmbda**-2,
            skewness=2.0 * (1 + 1 / v) ** -1.5,
            kurtosis=3 * (1 + 2 * v + 3 * v**2) / (1 + v) ** 2 - 3,
        )


def test_gaussian_kde_moments():
    testcase = DensityTestCase.expgauss()
    pdf = GaussianKDE(testcase.samples)
    mu, variance, skew, kurt = pdf.moments()

    tolerance = 0.1
    assert isclose(mu, testcase.mean, rtol=tolerance, atol=0.0)
    assert isclose(variance, testcase.variance, rtol=tolerance, atol=0.0)
    assert isclose(skew, testcase.skewness, rtol=tolerance, atol=0.0)
    assert isclose(kurt, testcase.kurtosis, rtol=tolerance, atol=0.0)


def test_unimodal_pdf_moments():
    testcase = DensityTestCase.expgauss(n_samples=5000)
    pdf = UnimodalPdf(testcase.samples)
    mu, variance, skew, kurt = pdf.moments()

    assert isclose(mu, testcase.mean, rtol=0.1, atol=0.0)
    assert isclose(variance, testcase.variance, rtol=0.1, atol=0.0)
    assert isclose(skew, testcase.skewness, rtol=0.1, atol=0.0)
    assert isclose(kurt, testcase.kurtosis, rtol=0.2, atol=0.0)


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


def test_sample_hdi_95():
    N = 20000
    expected_mu = 5.0
    expected_sigma = 2.0
    sample = default_rng(1324).normal(expected_mu, expected_sigma, size=N)

    left, right = sample_hdi(sample, fraction=0.95)

    tolerance = 0.2
    assert isclose(
        left, expected_mu - (expected_sigma**2), rtol=tolerance, atol=tolerance
    )
    assert isclose(
        right, expected_mu + (expected_sigma**2), rtol=tolerance, atol=tolerance
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


def test_sample_hdi_double():
    N = 20000
    half = linspace(0, 1, N)
    sample = concatenate((half, 1 - half))
    fraction = 1e-4

    (left_a, right_a), (left_b, right_b) = sample_hdi(
        sample, fraction=fraction, allow_double=True
    )
    assert left_a <= right_a
    assert left_b <= right_b
    assert isclose(right_a - left_a, fraction, rtol=1e-2, atol=1e-2)
    assert isclose(right_b - left_b, fraction, rtol=1e-2, atol=1e-2)


def test_sample_hdi_invalid_fractions():
    # Create some samples from the exponentially-modified Gaussian distribution
    sample = default_rng(1324).normal(size=3000)
    with pytest.raises(ValueError):
        sample_hdi(sample, fraction=2.0)
    with pytest.raises(ValueError):
        sample_hdi(sample, fraction=-0.1)
    with pytest.raises(ValueError):
        sample_hdi([1], fraction=0.95)
    with pytest.raises(ValueError):
        sample_hdi(1, fraction=0.95)


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

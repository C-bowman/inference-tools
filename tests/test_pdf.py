from inference.pdf import GaussianKDE, UnimodalPdf, sample_hdi, BinaryTree
from numpy.random import default_rng

from numpy import isclose, linspace, concatenate

import pytest
from hypothesis import given, strategies as st


def test_gaussian_kde_moments():
    N = 20000
    expected_mu = 5.0
    expected_sigma = 2.0

    sample = default_rng(1324).normal(expected_mu, expected_sigma, size=N)
    pdf = GaussianKDE(sample)
    mu, variance, skew, kurt = pdf.moments()

    tolerance = 0.2
    assert isclose(pdf.mode, expected_mu, rtol=tolerance, atol=tolerance)
    assert isclose(mu, expected_mu, rtol=tolerance, atol=tolerance)
    assert isclose(variance, expected_sigma**2, rtol=tolerance, atol=tolerance)
    assert isclose(skew, 0.0, rtol=tolerance, atol=tolerance)
    assert isclose(kurt, 0.0, rtol=tolerance, atol=tolerance)


def test_gaussian_kde_interval():
    N = 20000
    expected_mu = 5.0
    expected_sigma = 2.0

    sample = default_rng(1324).normal(expected_mu, expected_sigma, size=N)
    pdf = GaussianKDE(sample)
    left, right = pdf.interval()

    tolerance = 0.2
    assert isclose(
        left, expected_mu - (expected_sigma**2), rtol=tolerance, atol=tolerance
    )
    assert isclose(
        right, expected_mu + (expected_sigma**2), rtol=tolerance, atol=tolerance
    )


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


def test_unimodal_pdf_moments():
    N = 5000
    # Create some samples from the exponentially-modified Gaussian distribution
    rng = default_rng(1234)
    sample = rng.normal(size=N) + rng.exponential(scale=3.0, size=N)
    pdf = UnimodalPdf(sample)
    mu, variance, skew, kurt = pdf.moments()
    tolerance = 0.2
    assert isclose(pdf.mode, 1.0, rtol=tolerance, atol=tolerance)
    assert isclose(mu, 3.0, rtol=tolerance, atol=tolerance)
    assert isclose(variance, 10.0, rtol=tolerance, atol=tolerance)
    assert isclose(skew, 7.5, rtol=tolerance, atol=tolerance)
    assert isclose(kurt, 3.5, rtol=1.0, atol=1.0)


def test_unimodal_pdf_interval():
    N = 5000
    # Create some samples from the exponentially-modified Gaussian distribution
    rng = default_rng(1234)
    sample = rng.normal(size=N) + rng.exponential(scale=3.0, size=N)
    pdf = UnimodalPdf(sample)
    left, right = pdf.interval()

    tolerance = 0.2
    assert isclose(left, -1.75, rtol=tolerance, atol=tolerance)
    assert isclose(right, 9.5, rtol=tolerance, atol=tolerance)


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


@given(st.floats())
def test_binary_tree(value):
    limit_left, limit_right = [-1, 1]

    tree = BinaryTree(4, [limit_left, limit_right])
    left, right, _ = tree.lookup(value)

    if limit_left <= value <= limit_right:
        assert left <= value <= right
    elif value < limit_left:
        assert value < left < right
    elif value > limit_right:
        assert value > right > left

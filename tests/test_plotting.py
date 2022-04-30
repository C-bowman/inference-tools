from numpy import linspace, zeros, subtract, exp, array
from numpy.random import default_rng
from inference.plotting import matrix_plot, trace_plot, hdi_plot, transition_matrix_plot
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt

import pytest


@pytest.fixture
def gp_samples():
    N = 5
    x = linspace(1, N, N)
    mean = zeros(N)
    covariance = exp(-0.1 * subtract.outer(x, x) ** 2)

    samples = default_rng(1234).multivariate_normal(mean, covariance, size=100)
    return [samples[:, i] for i in range(N)]


def test_matrix_plot(gp_samples):
    n = len(gp_samples)
    labels = [f"test {i}" for i in range(n)]

    fig = matrix_plot(gp_samples, labels=labels, show=False)
    expected_plots = n**2 - n * (n - 1) / 2
    assert len(fig.get_axes()) == expected_plots


def test_matrix_plot_input_parsing(gp_samples):
    n = len(gp_samples)

    labels = [f"test {i}" for i in range(n + 1)]
    with pytest.raises(ValueError):
        matrix_plot(gp_samples, labels=labels, show=False)

    ref_vals = [i for i in range(n + 1)]
    with pytest.raises(ValueError):
        matrix_plot(gp_samples, reference=ref_vals, show=False)

    with pytest.raises(ValueError):
        matrix_plot(gp_samples, hdi_fractions=[0.95, 1.05], show=False)

    with pytest.raises(ValueError):
        matrix_plot(gp_samples, hdi_fractions=0.5, show=False)


def test_trace_plot():
    N = 11
    x = linspace(1, N, N)
    mean = zeros(N)
    covariance = exp(-0.1 * subtract.outer(x, x) ** 2)

    samples = default_rng(1234).multivariate_normal(mean, covariance, size=100)
    samples = [samples[:, i] for i in range(N)]
    labels = ["test {}".format(i) for i in range(len(samples))]

    fig = trace_plot(samples, labels=labels, show=False)

    assert len(fig.get_axes()) == N


def test_hdi_plot():
    N = 10
    start = 0
    end = 12
    x_fits = linspace(start, end, N)
    curves = array([default_rng(1324).normal(size=N) for _ in range(N)])
    intervals = [0.5, 0.65, 0.95]

    ax = hdi_plot(x_fits, curves, intervals)

    # Not much to check here, so check the viewing portion is sensible
    # and we've plotted the same number of PolyCollections as
    # requested intervals -- this could fail if the implementation
    # changes!
    number_of_plotted_intervals = len(
        [child for child in ax.get_children() if isinstance(child, PolyCollection)]
    )

    assert len(intervals) == number_of_plotted_intervals

    left, right, bottom, top = ax.axis()
    assert left <= start
    assert right >= end
    assert bottom <= curves.min()
    assert top >= curves.max()


def test_hdi_plot_bad_intervals():
    intervals = [0.5, 0.65, 1.2, 0.95]

    with pytest.raises(ValueError):
        hdi_plot(zeros(5), zeros(5), intervals)


def test_hdi_plot_bad_dimensions():
    N = 10
    start = 0
    end = 12
    x_fits = linspace(start, end, N)
    curves = array([default_rng(1324).normal(size=N + 1) for _ in range(N + 1)])

    with pytest.raises(ValueError):
        hdi_plot(x_fits, curves)


def test_transition_matrix_plot():
    N = 5
    matrix = default_rng(1324).random((N, N))

    ax = transition_matrix_plot(matrix=matrix)

    # Check that every square has some percentile text in it.
    # Not a great test, but does check we've plotted something!
    def filter_percent_text(child):
        if not isinstance(child, plt.Text):
            return False
        return "%" in child.get_text()

    percentage_texts = len(
        [child for child in ax.get_children() if filter_percent_text(child)]
    )

    assert percentage_texts == N**2


def test_transition_matrix_plot_upper_triangle():
    N = 5
    matrix = default_rng(1324).random((N, N))

    ax = transition_matrix_plot(
        matrix=matrix, exclude_diagonal=True, upper_triangular=True
    )

    # Check that every square has some percentile text in it.
    # Not a great test, but does check we've plotted something!
    def filter_percent_text(child):
        if not isinstance(child, plt.Text):
            return False
        return "%" in child.get_text()

    percentage_texts = len(
        [child for child in ax.get_children() if filter_percent_text(child)]
    )

    assert percentage_texts == sum(range(N))


def test_transition_matrix_plot_bad_shapes():
    # Wrong number of dimensions
    with pytest.raises(ValueError):
        transition_matrix_plot(matrix=zeros((2, 2, 2)))
    # Not square
    with pytest.raises(ValueError):
        transition_matrix_plot(matrix=zeros((2, 3)))
    # Too small
    with pytest.raises(ValueError):
        transition_matrix_plot(matrix=zeros((1, 1)))

from numpy import array, sqrt, linspace, ones
from numpy.random import default_rng
import pytest


def rosenbrock(t):
    # This is a modified form of the rosenbrock function, which
    # is commonly used to test optimisation algorithms
    X, Y = t
    X2 = X**2
    b = 15  # correlation strength parameter
    v = 3  # variance of the gaussian term
    return -X2 - b * (Y - X2) ** 2 - 0.5 * (X2 + Y**2) / v


class ToroidalGaussian:
    def __init__(self):
        self.R0 = 1.0  # torus major radius
        self.ar = 10.0  # torus aspect ratio
        self.inv_w2 = (self.ar / self.R0) ** 2

    def __call__(self, theta):
        x, y, z = theta
        r_sqr = z**2 + (sqrt(x**2 + y**2) - self.R0) ** 2
        return -0.5 * r_sqr * self.inv_w2

    def gradient(self, theta):
        x, y, z = theta
        R = sqrt(x**2 + y**2)
        K = 1 - self.R0 / R
        g = array([K * x, K * y, z])
        return -g * self.inv_w2


class LinePosterior:
    """
    This is a simple posterior for straight-line fitting
    with gaussian errors.
    """

    def __init__(self, x=None, y=None, err=None):
        self.x = x
        self.y = y
        self.err = err

    def __call__(self, theta):
        m, c = theta
        fwd = m * self.x + c
        ln_P = -0.5 * sum(((self.y - fwd) / self.err) ** 2)
        return ln_P


@pytest.fixture
def line_posterior():
    N = 25
    x = linspace(-2, 5, N)
    m = 0.5
    c = 0.05
    sigma = 0.3
    y = m * x + c + default_rng(1324).normal(size=N) * sigma
    return LinePosterior(x=x, y=y, err=ones(N) * sigma)


def sliced_length(length, start, step):
    return (length - start - 1) // step + 1

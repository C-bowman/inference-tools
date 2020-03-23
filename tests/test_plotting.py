
import pytest
import unittest

from numpy import linspace, zeros, subtract, exp
from numpy.random import multivariate_normal
from inference.plotting import matrix_plot, trace_plot


class test_plotting_functions(unittest.TestCase):

    def test_matrix_plot(self):
        N = 5
        x = linspace(1, N, N)
        mean = zeros(N)
        covariance = exp(-0.1*subtract.outer(x,x)**2)

        samples = multivariate_normal(mean, covariance, size=10000)
        samples = [samples[:,i] for i in range(N)]
        labels = [ 'test {}'.format(i) for i in range(len(samples)) ]

        matrix_plot(samples, labels=labels, show=False)

    def test_trace_plot(self):
        N = 11
        x = linspace(1, N, N)
        mean = zeros(N)
        covariance = exp(-0.1*subtract.outer(x,x)**2)

        samples = multivariate_normal(mean, covariance, size=10000)
        samples = [samples[:,i] for i in range(N)]
        labels = [ 'test {}'.format(i) for i in range(len(samples)) ]

        trace_plot(samples, labels=labels, show=False)


if __name__ == '__main__':

    unittest.main()

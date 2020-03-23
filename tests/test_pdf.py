
import pytest
import unittest

from inference.pdf_tools import GaussianKDE, UnimodalPdf, sample_hdi
from numpy.random import normal, exponential

from numpy import linspace, zeros


class test_pdf_tools(unittest.TestCase):

    def test_gaussian_kde(self):
        # generate a test sample
        N = 15000
        sample = zeros(N)
        sample[:N//3] = normal(size=N//3)
        sample[N//3:] = normal(size=2*(N//3)) + 10
        # construct estimators
        pdf = GaussianKDE(sample)
        pdf_crossval = GaussianKDE(sample, cross_validation=True)
        # evaluate the estimate on an axis
        x = linspace(-5, 15, 200)
        p = pdf(x)
        # test methods and attributes
        mode = pdf.mode
        hdi_95 = pdf.interval(frac=0.95)
        mu, var, skew, kurt = pdf.moments()
        pdf.plot_summary(show=False)

    def test_unimodal_pdf(self):
        # Create some samples from the exponentially-modified Gaussian distribution
        sample = normal(size=3000) + exponential(scale=3., size=3000)
        # create an instance of the density estimator
        pdf = UnimodalPdf(sample)
        # evaluate the estimate on an axis
        x = linspace(-5, 15, 1000)
        p = pdf(x)
        # test methods and attributes
        mode = pdf.mode
        hdi_95 = pdf.interval(frac=0.95)
        mu, var, skew, kurt = pdf.moments()
        pdf.plot_summary(show=False)

    def test_sample_hdi(self):
        # Create some samples from the exponentially-modified Gaussian distribution
        sample = normal(size=3000) + exponential(scale=3., size=3000)
        interval = sample_hdi(sample, fraction=0.95)


if __name__ == '__main__':

    unittest.main()

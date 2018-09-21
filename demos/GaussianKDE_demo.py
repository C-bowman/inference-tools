from numpy import linspace, zeros
from numpy.random import normal

from inference.pdf_tools import GaussianKDE

"""
Code to demonstrate the use of the GaussianKDE class.
"""

# first generate a test sample
N = 150000
sample = zeros(N)
sample[:N//3] = normal(size=N//3)*0.5 + 1.8
sample[N//3:] = normal(size=2*(N//3))*0.5 + 3.5

# GaussianKDE takes an array of sample values as its only argument
pdf = GaussianKDE(sample)

# much like the UnimodalPdf class, GaussianKDE returns a density estimator object
# which can be called as a function to return an estimate of the PDF at a set of
# points:
x = linspace(0, 6, 1000)
p = pdf(x)

# GaussianKDE is fast even for large samples, as it uses a binary tree search to
# match any given spatial location with a slice of the sample array which contains
# all samples that have a non-negligible contribution to the density estimate.

# We could plot (x, P) manually, but for convenience the plot_summary
# method will generate a plot automatically as well as summary statistics:
pdf.plot_summary()

# The summary statistics can be accessed via properties or methods:
# the location of the mode is a property
mode = pdf.mode

# The highest-density interval for any fraction of total probability
# can is returned by the interval() method
hdi_95 = pdf.interval(frac = 0.95)

# the mean, variance, skewness and excess kurtosis are returned
# by the moments() method:
mu, var, skew, kurt = pdf.moments()

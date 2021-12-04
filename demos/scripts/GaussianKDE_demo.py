from numpy import linspace, zeros, exp, sqrt, pi
from numpy.random import normal
import matplotlib.pyplot as plt
from inference.pdf import GaussianKDE

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

# By default, GaussianKDE uses a simple but easy to compute estimate of the
# bandwidth (the standard deviation of each Gaussian kernel). However, when
# estimating strongly non-normal distributions, this simple approach will
# over-estimate required bandwidth.

# In these cases, the cross-validation bandwidth selector can be used to
# obtain better results, but with higher computational cost.

# to demonstrate, lets create a new sample:
N = 30000
sample = zeros(N)
sample[:N//3] = normal(size=N//3)
sample[N//3:] = normal(size=2*(N//3)) + 10

# now construct estimators using the simple and cross-validation estimators
pdf_simple = GaussianKDE(sample)
pdf_crossval = GaussianKDE(sample, cross_validation = True)

# now build an axis on which to evaluate the estimates
x = linspace(-4,14,500)

# for comparison also compute the real distribution
exact = (exp(-0.5*x**2)/3 + 2*exp(-0.5*(x-10)**2)/3)/sqrt(2*pi)

# plot everything together
plt.plot(x, pdf_simple(x), label = 'simple')
plt.plot(x, pdf_crossval(x), label = 'cross-validation')
plt.plot(x, exact, label = 'exact')

plt.ylabel('probability density')
plt.xlabel('x')

plt.grid()
plt.legend()
plt.show()

from numpy import linspace, log
from numpy.random import normal

from inference.pdf import UnimodalPdf

"""
Code to demonstrate the use of the UnimodalPdf class.
"""


# first create a sample from a skewed distribution.
samp = log((normal(size = 25000)+4)**2)

# now we create the object 'pdf' as an instance of the UnimodalPdf class
pdf = UnimodalPdf(samp)  # the only argument is the list/array of samples

# A UnimodalPdf object can be treated like a function that returns
# the estimate of the PDF. For example, if we create an axis:
x = linspace(-1, 6, 1000)
# to get the estimate we call the object as follows
P = pdf(x)

# We could plot (x, P) manually, but for convenience the plot_summary
# method will generate a plot automatically as well as summary statistics:
pdf.plot_summary()

# The summary statistics can be accessed via properties or methods:
# the location of the mode is a property
mode = pdf.mode

# The highest-density interval for any fraction of total probability
# is returned by the interval() method
hdi_95 = pdf.interval(frac = 0.95)

# the mean, variance, skewness and excess kurtosis are returned
# by the moments() method:
mu, var, skew, kurt = pdf.moments()

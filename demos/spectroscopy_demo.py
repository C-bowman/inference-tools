
import matplotlib.pyplot as plt
from numpy import linspace, sqrt, pi, array
from numpy.random import normal

from inference.mcmc import GibbsChain

"""
This code uses a toy problem of fitting peaks to simulated spectroscopy data in
order to demonstrate how to construct a posterior distribution which may be used 
with the samplers in inference.mcmc.
"""


# it is usually very convenient to construct posteriors as classes with
# a __call__ method - this allows all relevant data to be stored within
# the object, while still being able to be called as a function.

# here we define the posterior class for this toy fitting problem
class SpectroPosterior(object):

    def __init__(self, wavelength, intensity, errors):
        """
        The __init__ should be used to load all data into the object
        which is necessary for calculation of the log-posterior
        probability which will be returned by the __call__ method.

        Typically this includes experimental measurements with their
        associated uncertainties, and any relevant physical constants.
        """
        # store experimental data
        self.x = wavelength
        self.y = intensity
        self.sigma = errors

        # Central wavelengths of the lines are known constants:
        self.c1 = 422.
        self.c2 = 428.

    def __call__(self, theta):
        """
        When the posterior object is called as a function, this method is invoked.
        All inference_tools samplers expect a posterior function which maps some set
        of model parameters 'theta' to a log-probability density.

        The posterior need only be defined up to a constant of proportionality, as
        mcmc samplers only require fractional differences in densities to function.
        This means the log-posterior can be written as the sum of the log-likelihood
        and log-prior functions:
        """
        return self.likelihood(theta) + self.prior(theta)

    def prior(self, theta):
        """
        This is a place-holder function which serves as an example of how a prior can be
        incorporated into the posterior class. Priors which are either uniform between
        2 chosen values, or simply non-negative should be enforced using options available
        in the Parameter classes.
        """
        return 0.

    def likelihood(self, theta):
        """
        In this example we assume that the errors on our data
        are Gaussian, such that the log-likelihood takes the
        form given below:
        """
        return -0.5*sum( ((self.y - self.forward_model(self.x, theta)) / self.sigma)**2 )

    def forward_model(self, x, theta):
        """
        The forward model must make a prediction of the experimental
        data we would expect to measure given a specific state of the
        system, which is specified by the model parameters theta.

        For the sake of simplicity, in this example our forward model
        is just a physics model - however in an applied case both a
        physics model and an instrument model are needed ( for example
        to account for the effect of an instrument function)
        """
        # unpack the model parameters
        A1, w1, A2, w2, b0, b1 = theta

        # evaluate each term of the model
        peak_1 = (A1 / (pi*w1)) / (1 + ((x - self.c1)/w1)**2)
        peak_2 = (A2 / (pi*w2)) / (1 + ((x - self.c2)/w2)**2)
        d = (b1-b0)/(max(x) - min(x))
        background = d*x + (b0 - d*min(x))

        # return the prediction of the data
        return peak_1 + peak_2 + background






# Create some simulated data
N = 40
x_data = linspace(410, 440, N)
P = SpectroPosterior(x_data, None, None)
theta = [1400, 2, 600, 1.5, 55, 35]
y_data = P.forward_model(x_data, theta)
errors = sqrt(y_data+1)
y_data += normal(size=N)*errors




# plot the simulated data we're going to use
plt.plot( x_data, y_data, '.-')
plt.title('synthetic spectroscopy data')
plt.xlabel('wavelength (nm)')
plt.ylabel('intensity')
plt.grid()
plt.show()




# create the posterior object
posterior = SpectroPosterior(x_data, y_data, errors)

# create the markov chain object
chain = GibbsChain( posterior = posterior, start = [1000, 1, 1000, 1, 30, 30] )

# In this problem, every parameter is a physical quantity that must be
# non-negative. This constraint can be easily enforced setting the
# Parameter.non_negative flag to True for each parameter object stored
# in chain.params
for p in chain.params:
    p.non_negative = True

# generate a sample by advancing the chain
chain.advance(25000)

# we can check the status of the chain using the plot_diagnostics method
chain.plot_diagnostics()

# based on the diagnostics we can choose to set a global burn and thin
# value, which is used (unless otherwise specified) by all methods which
# access the samples
chain.burn = 5000
chain.thin = 2

# we can get a quick overview of the posterior using the matrix_plot
# functionality of chain objects, which plots all possible 1D & 2D
# marginal distributions of the full parameter set (or a chosen sub-set).
chain.matrix_plot()

# We can easily estimate 1D marginal distributions for any parameter
# using the 'marginalise' method:
pdf_1 = chain.marginalise(1, unimodal = True)
pdf_2 = chain.marginalise(3, unimodal = True)

# marginalise returns a density estimator object, which can be called
# as a function to return the value of the pdf at any point.
# Make an axis on which to evaluate the PDFs:
ax = linspace(0.5, 3., 1000)
plt.plot(ax, pdf_1(ax), label = 'width #1 marginal', lw = 2)
plt.plot(ax, pdf_2(ax), label = 'width #2 marginal', lw = 2)
plt.title('Peak width 1D marginal distributions')
plt.xlabel('parameter value')
plt.ylabel('probability density')
plt.legend()
plt.grid()
plt.show()






# what if instead we wanted a PDF for the ratio of the two widths?
# get the sample for each width
width_1 = chain.get_parameter(1)
width_2 = chain.get_parameter(3)

# make a new set of samples for the ratio
widths_ratio = [i/j for i,j in zip(width_1, width_2)]

# Use one of the density estimator objects from pdf_tools to get the PDF
from inference.pdf_tools import UnimodalPdf
pdf = UnimodalPdf(widths_ratio)

# plot the PDF
ax = linspace(0,3,300)
plt.plot( ax, pdf(ax), lw = 2)
plt.title('Peak widths ratio distribution')
plt.xlabel('widths ratio')
plt.ylabel('probability density')
plt.grid()
plt.show()






# You may also want to assess the level of uncertainty in the model.
# This can be done very easily by passing each sample through the forward-model
# and observing the distribution of model expressions that result.

# However rather than taking the entire sample, it is better to take a sub-sample
# which corresponds to some credible interval. For example, the 95% credible interval
# sub sample can be generated by taking the 95% of samples which have the highest
# associated probabilities.

# GibbsChain has a convenience method for this called get_interval():
interval_sample, interval_probs = chain.get_interval(samples = 1500) # by default the interval is 95%, but any fraction can be used.

# generate an axis on which to evaluate the model
M = 500
x_fits = linspace(410, 440, M)

# now evaluate the model for each sample
models = []
for theta in interval_sample:
    curve = posterior.forward_model(x_fits, theta)
    models.append(curve)
models = array(models)

# calculate the 95% envelope
upper_bound = models.max(axis = 0)
lower_bound = models.min(axis = 0)

# also want to evaluate the most probable model using the mode:
mode = posterior.forward_model(x_fits, chain.mode())

# construct the plot
plt.figure(figsize = (10,7.5))
plt.plot(x_fits, mode, c = 'C2', lw = 2, label = 'mode')
plt.plot(x_fits, lower_bound, ls = 'dashed', c = 'red', lw = 2, label = '95% envelope')
plt.plot(x_fits, upper_bound, ls = 'dashed', c = 'red', lw = 2)
plt.plot( x_data, y_data, 'D', c = 'blue', markeredgecolor = 'black', markersize = 4, label = 'data')
plt.title('Forward model 95% interval')
plt.xlabel('wavelength (nm)')
plt.ylabel('intensity')
plt.legend()
plt.grid()
plt.show()
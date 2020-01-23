
import matplotlib.pyplot as plt
from numpy import linspace, sqrt, pi, array
from numpy.random import normal

from inference.mcmc import PcaChain

"""
This code uses a toy problem of fitting peaks to simulated spectroscopy data in
order to demonstrate how to construct a posterior distribution which may be used 
with the samplers in inference.mcmc
"""

# its usually convenient to construct posteriors as classes with
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
        return -0.5*( ((self.y - self.forward_model(self.x, theta)) / self.sigma)**2 ).sum()

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
        A1, w1, A2, w2, bg = theta
        # evaluate the peaks
        peak_1 = A1 / ((1 + ((x - self.c1)/w1)**2)*(pi*w1))
        peak_2 = A2 / ((1 + ((x - self.c2)/w2)**2)*(pi*w2))
        # return the prediction of the data
        return peak_1 + peak_2 + bg





# Create some simulated data
N = 40
x_data = linspace(410, 440, N)
P = SpectroPosterior(x_data, None, None)
theta = [1000, 2, 400, 1.5, 35]
y_data = P.forward_model(x_data, theta)
errors = sqrt(y_data+1)+2
y_data += normal(size=N)*errors



# plot the simulated data we're going to use
plt.errorbar( x_data, y_data, errors, marker = 'D', ls = 'none', markersize = 4)
plt.plot( x_data, y_data, alpha = 0.5, c = 'C0', ls = 'dashed')
plt.title('synthetic spectroscopy data')
plt.xlabel('wavelength (nm)')
plt.ylabel('intensity')
plt.grid()
plt.show()



# create the posterior object
posterior = SpectroPosterior(x_data, y_data, errors)

# create the markov chain object
chain = PcaChain( posterior = posterior, start = [600, 1, 600, 1, 15] )

# generate a sample by advancing the chain
chain.advance(50000)

# we can check the status of the chain using the plot_diagnostics method
chain.plot_diagnostics()

# We can automatically set sensible burn and thin values for the sample
chain.autoselect_burn_and_thin()

# we can get a quick overview of the posterior using the matrix_plot
# functionality of chain objects, which plots all possible 1D & 2D
# marginal distributions of the full parameter set (or a chosen sub-set).
chain.matrix_plot()

# We can easily estimate 1D marginal distributions for any parameter
# using the get_marginal method:
w1_pdf = chain.get_marginal(1, unimodal = True)
w2_pdf = chain.get_marginal(3, unimodal = True)

# get_marginal returns a density estimator object, which can be called
# as a function to return the value of the pdf at any point.
# Make an axis on which to evaluate the PDFs:
ax = linspace(0.2, 4., 1000)
plt.plot(ax, w1_pdf(ax), label = 'width #1 marginal', lw = 2)
plt.plot(ax, w2_pdf(ax), label = 'width #2 marginal', lw = 2)
plt.xlabel('peak width')
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
pdf.plot_summary()



# You may also want to assess the level of uncertainty in the model predictions.
# This can be done easily by passing each sample through the forward-model
# and observing the distribution of model expressions that result.

# generate an axis on which to evaluate the model
M = 500
x_fits = linspace(400, 450, M)
# get the sample
sample = chain.get_sample()
# pass each through the forward model
curves = array([ posterior.forward_model(x_fits, theta) for theta in sample])

# we can use the sample_hdi function from the pdf_tools module to produce highest-density
# intervals for each point where the model is evaluated:
from inference.pdf_tools import sample_hdi
hdi_1sigma = array([sample_hdi(c, 0.68, force_single = True) for c in curves.T])
hdi_2sigma = array([sample_hdi(c, 0.95, force_single = True) for c in curves.T])

# construct the plot
plt.figure(figsize = (8,5))
# plot the 1 and 2-sigma highest-density intervals
plt.fill_between(x_fits, hdi_2sigma[:,0], hdi_2sigma[:,1], color = 'red', alpha = 0.10, label = '2-sigma HDI')
plt.fill_between(x_fits, hdi_1sigma[:,0], hdi_1sigma[:,1], color = 'red', alpha = 0.20, label = '1-sigma HDI')
# plot the MAP estimate
MAP = posterior.forward_model(x_fits, chain.mode())
plt.plot(x_fits, MAP, c = 'red', lw = 2, ls = 'dashed', label = 'MAP estimate')
# plot the data
plt.plot( x_data, y_data, 'D', c = 'blue', markeredgecolor = 'black', markersize = 5, label = 'data')
# configure the plot
plt.xlabel('wavelength (nm)')
plt.ylabel('intensity')
plt.xlim([410, 440])
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
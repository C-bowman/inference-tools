
import matplotlib.pyplot as plt
from numpy import linspace, sqrt, pi, array
from numpy.random import normal

from inference.mcmc import PcaChain
from inference.likelihoods import GaussianLikelihood

"""
This code uses a toy problem of fitting peaks to simulated spectroscopy data in
order to demonstrate how to construct a posterior distribution which may be used 
with the samplers in inference.mcmc
"""

# its usually convenient to construct posteriors as classes with
# a __call__ method - this allows all relevant data to be stored within
# the object, while still being able to be called as a function.

# here we define the posterior class for this toy fitting problem
class SpectroscopyModel(object):
    def __init__(self, wavelength):
        """
        The __init__ should be used to load all data into the object
        which is necessary for calculation of the log-posterior
        probability which will be returned by the __call__ method.

        Typically this includes experimental measurements with their
        associated uncertainties, and any relevant physical constants.
        """
        # store experimental data
        self.x = wavelength

        # Central wavelengths of the lines are known constants:
        self.c1 = 422.
        self.c2 = 428.

    def __call__(self, theta):
        return self.forward_model(self.x, theta)

    def forward_model(self, x, theta):
        """
        The forward model must make a prediction of the experimental data we would expect to measure
        given a specific state of the system, which is specified by the model parameters theta.
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
model = SpectroscopyModel(x_data)
theta = [1000, 2, 400, 1.5, 35]
y_data = model(theta)
errors = sqrt(y_data+1)+2
y_data += normal(size=N)*errors



# plot the simulated data we're going to use
plt.errorbar( x_data, y_data, errors, marker='D', ls='none', markersize=4)
plt.plot( x_data, y_data, alpha=0.5, c='C0', ls='dashed')
plt.title('synthetic spectroscopy data')
plt.xlabel('wavelength (nm)')
plt.ylabel('intensity')
plt.grid()
plt.show()



# create the posterior object
posterior = GaussianLikelihood(y_data=y_data, sigma=errors, forward_model=model)

# create the markov chain object
chain = PcaChain(posterior=posterior, start=[600, 1, 600, 1, 15])

# generate a sample by advancing the chain
chain.advance(20000)

# we can check the status of the chain using the plot_diagnostics method
chain.plot_diagnostics()

# We can automatically set sensible burn and thin values for the sample
chain.autoselect_burn()
chain.autoselect_thin()

# we can get a quick overview of the posterior using the matrix_plot
# functionality of chain objects, which plots all possible 1D & 2D
# marginal distributions of the full parameter set (or a chosen sub-set).
chain.matrix_plot()

# We can easily estimate 1D marginal distributions for any parameter
# using the get_marginal method:
w1_pdf = chain.get_marginal(1, unimodal=True)
w2_pdf = chain.get_marginal(3, unimodal=True)

# get_marginal returns a density estimator object, which can be called
# as a function to return the value of the pdf at any point.
# Make an axis on which to evaluate the PDFs:
ax = linspace(0.2, 4., 1000)
plt.plot(ax, w1_pdf(ax), label='width #1 marginal', lw=2)
plt.plot(ax, w2_pdf(ax), label='width #2 marginal', lw=2)
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
from inference.pdf import UnimodalPdf
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
curves = array([model.forward_model(x_fits, theta) for theta in sample])

plt.figure(figsize = (8,5))

# We can use the hdi_plot function from the plotting module to plot
# highest-density intervals for each point where the model is evaluated:
from inference.plotting import hdi_plot
hdi_plot(x_fits, curves)

# build the rest of the plot
plt.plot( x_data, y_data, 'D', c='red', label='data')
plt.xlabel('wavelength (nm)')
plt.ylabel('intensity')
plt.xlim([410, 440])
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
from numpy import array, exp, linspace, sqrt, pi
import matplotlib.pyplot as plt

# Suppose we have the following dataset, which we believe is described by a
# Gaussian peak plus a constant background. Our goal in this example is to
# infer the area of the Gaussian.

x_data = array([
    0.00, 0.80, 1.60, 2.40, 3.20, 4.00, 4.80, 5.60,
    6.40, 7.20, 8.00, 8.80, 9.60, 10.4, 11.2, 12.0
])

y_data = array([
    2.473, 1.329, 2.370, 1.135, 5.861, 7.045, 9.942, 7.335,
    3.329, 5.348, 1.462, 2.476, 3.096, 0.784, 3.342, 1.877
])

y_error = array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
])

plt.errorbar(
    x_data,
    y_data,
    yerr=y_error,
    ls="dashed",
    marker="D",
    c="red",
    markerfacecolor="none",
)
plt.ylabel("y")
plt.xlabel("x")
plt.grid()
plt.show()

# The first step is to implement our model. For simple models like this one
# this can be done using just a function, but as models become more complex
# it becomes useful to build them as classes.


class PeakModel:
    def __init__(self, x_data):
        """
        The __init__ should be used to pass in any data which is required
        by the model to produce predictions of the y-data values.
        """
        self.x = x_data

    def __call__(self, theta):
        return self.forward_model(self.x, theta)

    @staticmethod
    def forward_model(x, theta):
        """
        The forward model must make a prediction of the experimental data we would expect to measure
        given a specific set model parameters 'theta'.
        """
        # unpack the model parameters
        area, width, center, background = theta
        # return the prediction of the data
        z = (x - center) / width
        gaussian = exp(-0.5 * z**2) / (sqrt(2 * pi) * width)
        return area * gaussian + background


# inference-tools has a variety of Likelihood classes which allow you to easily construct a
# likelihood function given the measured data and your forward-model.
from inference.likelihoods import GaussianLikelihood

likelihood = GaussianLikelihood(
    y_data=y_data, sigma=y_error, forward_model=PeakModel(x_data)
)

# Instances of the likelihood classes can be called as functions, and return the log-likelihood
# when passed a vector of model parameters:
initial_guess = array([10.0, 2.0, 5.0, 2.0])
guess_log_likelihood = likelihood(initial_guess)
print(guess_log_likelihood)

# We could at this stage pair the likelihood object with an optimiser in order to obtain
# the maximum-likelihood estimate of the parameters. In this example however, we want to
# construct the posterior distribution for the model parameters, and that means we need
# a prior.

# The inference.priors module contains classes which allow for easy construction of
# prior distributions across all model parameters.
from inference.priors import ExponentialPrior, UniformPrior, JointPrior

# If we want different model parameters to have different prior distributions, as in this
# case where we give three variables an exponential prior and one a uniform prior, we first
# construct each type of prior separately:
prior_components = [
    ExponentialPrior(beta=[50.0, 20.0, 20.0], variable_indices=[0, 1, 3]),
    UniformPrior(lower=0.0, upper=12.0, variable_indices=[2]),
]
# Now we use the JointPrior class to combine the various components into a single prior
# distribution which covers all the model parameters.
prior = JointPrior(components=prior_components, n_variables=4)

# As with the likelihood, prior objects can also be called as function to return a
# log-probability value when passed a vector of model parameters. We can also draw
# samples from the prior directly using the sample() method:
prior_sample = prior.sample()
print(prior_sample)

# The likelihood and prior can be easily combined into a posterior distribution
# using the Posterior class:
from inference.posterior import Posterior
posterior = Posterior(likelihood=likelihood, prior=prior)

# Now we have constructed a posterior distribution, we can sample from it
# using Markov-chain Monte-Carlo (MCMC).

# The inference.mcmc module contains implementations of various MCMC sampling algorithms.
# Here we import the PcaChain class and use it to create a Markov-chain object:
from inference.mcmc import PcaChain
chain = PcaChain(posterior=posterior, start=initial_guess)

# We generate samples by advancing the chain by a chosen number of steps using the advance method:
chain.advance(25000)

# we can check the status of the chain using the plot_diagnostics method:
chain.plot_diagnostics()

# The burn-in (how many samples from the start of the chain are discarded)
# can be specified as an argument to methods which act on the samples:
burn = 5000

# we can get a quick overview of the posterior using the matrix_plot method
# of chain objects, which plots all possible 1D & 2D marginal distributions
# of the full parameter set (or a chosen sub-set).
chain.matrix_plot(labels=["area", "width", "center", "background"], burn=burn)

# We can easily estimate 1D marginal distributions for any parameter
# using the get_marginal method:
area_pdf = chain.get_marginal(0, burn=burn)
area_pdf.plot_summary(label="Gaussian area")


# We can assess the level of uncertainty in the model predictions by passing each sample
# through the forward-model and observing the distribution of model expressions that result:

# generate an axis on which to evaluate the model
x_fits = linspace(-2, 14, 500)
# get the sample
sample = chain.get_sample(burn=burn)
# pass each through the forward model
curves = array([PeakModel.forward_model(x_fits, theta) for theta in sample])

# We could plot the predictions for each sample all on a single graph, but this is
# often cluttered and difficult to interpret.

# A better option is to use the hdi_plot function from the plotting module to plot
# highest-density intervals for each point where the model is evaluated:
from inference.plotting import hdi_plot

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
hdi_plot(x_fits, curves, intervals=[0.68, 0.95], axis=ax)

# plot the MAP estimate (the sample with the single highest posterior probability)
MAP_prediction = PeakModel.forward_model(x_fits, chain.mode())
ax.plot(x_fits, MAP_prediction, ls="dashed", lw=3, c="C0", label="MAP estimate")
# build the rest of the plot
ax.errorbar(
    x_data,
    y_data,
    yerr=y_error,
    linestyle="none",
    c="red",
    label="data",
    marker="o",
    markerfacecolor="none",
    markeredgewidth=1.5,
    markersize=8,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim([-0.5, 12.5])
ax.legend()
ax.grid()
plt.tight_layout()
plt.show()


from numpy import array, exp, linspace, sqrt, pi
import matplotlib.pyplot as plt

# Suppose we have the following dataset, which we believe is described by a
# Gaussian peak plus a constant background. Our goal in this example is to
# infer the area of the Gaussian.

x_data = [1.0, 1.6, 2.2, 2.8, 3.4, 4.0, 4.6, 5.2,
          5.8, 6.4, 7.0, 7.6, 8.2, 8.8, 9.4, 10.]

y_data = [2.473, 1.329, 2.370, 1.135, 5.861, 7.045, 9.942, 7.335,
          3.329, 5.348, 1.462, 2.476, 3.096, 0.784, 3.342, 1.877]

y_error = [1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1.]

plt.errorbar(x_data, y_data, yerr=y_error, ls='dashed', marker='D', c='red', markerfacecolor='none')
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()

# The first step is to implement our model. For simple models like this one
# this can be done using just a function, but as models become more complex
# it is becomes useful to build them as classes.


class PeakModel(object):
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
        gaussian = exp(-0.5*z**2)/(sqrt(2*pi)*width)
        return area*gaussian + background


# Inference-tools has a variety of Likelihood classes which allow you to easily construct a
# likelihood function given the measured data and your forward-model.
from inference.likelihoods import GaussianLikelihood
likelihood = GaussianLikelihood(y_data=y_data, sigma=y_error, forward_model=PeakModel(x_data))

# Instances of the likelihood classes can be called as functions, and return the log-likelihood
# when passed a vector of model parameters:
initial_guess = array([10., 2., 5., 2.])
guess_log_likelihood = likelihood(initial_guess)
print(guess_log_likelihood)

# We could at this stage pair the likelihood object with an optimiser in order to obtain
# the maximum-likelihood estimate of the parameters. In this example however, we want to
# construct the posterior distribution for the model parameters, and that means we need
# a prior.

# The inference.priors module contains classes which allow for easy construction of
# prior distributions across all model parameters.
from inference.priors import ExponentialPrior, UniformPrior, JointPrior

# First we need to make an ordered list of strings which identify all our model parameters.
model_variables = ['area', 'width', 'center', 'background']

# If we want different model parameters to have different prior distributions, as in this
# case where we give three variables an exponential prior and one a uniform prior, we first
# construct each type of prior separately:
prior_components = [
    ExponentialPrior(beta=[20., 20., 20.], variables=['area', 'width', 'background']),
    UniformPrior(lower=1., upper=10., variables=['center'])
]
# Now we use the JointPrior class to combine the various components into a single prior
# distribution which covers all the model parameters.
prior = JointPrior(model_variables=model_variables, components=prior_components)

# As with the likelihood, prior objects can also be called as function to return a
# log-probability value when passed a vector of model parameters. We can also draw
# samples from the prior directly using the sample() method:
prior_sample = prior.sample()
print(prior_sample)

class Posterior(object):
    def __init__(self, likelihood=None, prior=None):
        self.likelihood = likelihood
        self.prior = prior

    def __call__(self, theta):
        return self.likelihood(theta) + self.prior(theta)
posterior = Posterior(likelihood=likelihood, prior=prior)





from inference.mcmc import GibbsChain
chain = GibbsChain(posterior=posterior, start=[5., 5., 5., 5.])
chain.advance(25000)

chain.burn = 5000
chain.thin = 2

chain.plot_diagnostics()

chain.matrix_plot(labels=model_variables)

# generate an axis on which to evaluate the model
x_fits = linspace(1, 10, 500)
# get the sample
sample = chain.get_sample()
# pass each through the forward model
curves = array([PeakModel.forward_model(x_fits, theta) for theta in sample])

plt.figure(figsize=(8, 5))

# We can use the hdi_plot function from the plotting module to plot
# highest-density intervals for each point where the model is evaluated:
from inference.plotting import hdi_plot
hdi_plot(x_fits, curves, intervals=[0.68, 0.95])

# plot the MAP estimate (the sample with the single highest posterior probability)
plt.plot(x_fits, PeakModel.forward_model(x_fits, chain.mode()), ls='dashed', lw=3, c='C0', label='MAP estimate')
# build the rest of the plot
plt.errorbar(x_data, y_data, yerr=y_error, linestyle='none', c='red', label='data',
             marker='D', markerfacecolor='none', markeredgewidth=1.5, markersize=6)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()












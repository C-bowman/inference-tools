from numpy import linspace, array, concatenate, exp
from numpy.random import normal, seed
import matplotlib.pyplot as plt
from functools import partial
from inference.mcmc import HamiltonianChain
from inference.likelihoods import GaussianLikelihood
from inference.priors import GaussianPrior, ExponentialPrior, JointPrior
from inference.posterior import Posterior


def logistic(z):
    return 1.0 / (1.0 + exp(-z))


def forward_model(x, theta):
    h, w, c, b = theta
    z = (x - c) / w
    return h * logistic(z) + b


seed(3)
x = concatenate([linspace(0.3, 3, 6), linspace(5.0, 9.7, 5)])
start = array([4.0, 0.5, 5.0, 2.0])
y = forward_model(x, start)
sigma = y * 0.1 + 0.25
y += normal(size=y.size, scale=sigma)

likelihood = GaussianLikelihood(
    y_data=y,
    sigma=sigma,
    forward_model=partial(forward_model, x)
)

prior = JointPrior(
    components=[
        ExponentialPrior(beta=20., variable_indices=[0]),
        ExponentialPrior(beta=2.0, variable_indices=[1]),
        GaussianPrior(mean=5.0, sigma=5.0, variable_indices=[2]),
        GaussianPrior(mean=0., sigma=20., variable_indices=[3]),
    ],
    n_variables=4
)

posterior = Posterior(likelihood=likelihood, prior=prior)

bounds = [
    array([0., 0., 0., -5.]),
    array([15., 20., 10., 10.]),
]

chain = HamiltonianChain(
    posterior=posterior,
    start=start,
    bounds=bounds
)
chain.steps = 20
chain.advance(100000)
chain.burn = 200

chain.plot_diagnostics()
chain.trace_plot()

x_fits = linspace(0, 10, 100)
sample = array(chain.theta)
# pass each through the forward model
curves = array([forward_model(x_fits, theta) for theta in sample])

# We can use the hdi_plot function from the plotting module to plot
# highest-density intervals for each point where the model is evaluated:
from inference.plotting import hdi_plot

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)

hdi_plot(x_fits, curves, axis=ax, colormap="Greens")
ax.plot(
    x_fits,
    curves.mean(axis=0),
    ls="dashed",
    lw=3,
    color="darkgreen",
    label="predictive mean",
)
ax.errorbar(
    x,
    y,
    yerr=sigma,
    c="black",
    markeredgecolor="black",
    markeredgewidth=2,
    markerfacecolor="none",
    elinewidth=2,
    marker="o",
    ls="none",
    markersize=10,
    label="data",
)
ax.set_ylim([1.0, 7.])
ax.set_xlim([0, 10])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid()
ax.legend(loc=2, fontsize=13)
plt.tight_layout()
plt.savefig("gallery_hdi.png")
plt.show()

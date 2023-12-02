import matplotlib.pyplot as plt
from numpy import array, exp, linspace

from inference.mcmc import GibbsChain


def rosenbrock(t):
    # This is a modified form of the rosenbrock function, which
    # is commonly used to test optimisation algorithms
    X, Y = t
    X2 = X**2
    b = 15  # correlation strength parameter
    v = 3  # variance of the gaussian term
    return -X2 - b * (Y - X2) ** 2 - 0.5 * (X2 + Y**2) / v


"""
# Gibbs sampling example

In order to use the GibbsChain sampler from the mcmc module, we must
provide a log-posterior function to sample from, a point in the parameter
space to start the chain, and an initial guess for the proposal width
for each parameter.

In this example a modified version of the Rosenbrock function (shown
above) is used as the log-posterior.
"""

# The maximum of the rosenbrock function is [0, 0] - here we intentionally
# start the chain far from the mode.
start_location = array([2.0, -4.0])

# Here we make our initial guess for the proposal widths intentionally
# poor, to demonstrate that gibbs sampling allows each proposal width
# to be adjusted individually toward an optimal value.
width_guesses = array([5.0, 0.05])

# create the chain object
chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)

# advance the chain 150k steps
chain.advance(150000)

# the samples for the n'th parameter can be accessed through the
# get_parameter(n) method. We could use this to plot the path of
# the chain through the 2D parameter space:

p = chain.get_probabilities()  # color the points by their probability value
point_colors = exp(p - p.max())
plt.scatter(
    chain.get_parameter(0), chain.get_parameter(1), c=point_colors, marker="."
)
plt.xlabel("parameter 1")
plt.ylabel("parameter 2")
plt.grid()
plt.tight_layout()
plt.show()


# We can see from this plot that in order to take a representative sample,
# some early portion of the chain must be removed. This is referred to as
# the 'burn-in' period. This period allows the chain to both find the high
# density areas, and adjust the proposal widths to their optimal values.

# The plot_diagnostics() method can help us decide what size of burn-in to use:
chain.plot_diagnostics()

# Occasionally samples are also 'thinned' by a factor of n (where only every
# n'th sample is used) in order to reduce the size of the data set for
# storage, or to produce uncorrelated samples.

# based on the diagnostics we can choose burn and thin values,
# which can be passed as arguments to methods which act on the samples
burn = 2000
thin = 5

# After discarding burn-in, what we have left should be a representative
# sample drawn from the posterior. Repeating the previous plot as a
# scatter-plot shows the sample:
p = chain.get_probabilities(burn=burn, thin=thin)  # color the points by their probability value
plt.scatter(
    chain.get_parameter(index=0, burn=burn, thin=thin),
    chain.get_parameter(index=1, burn=burn, thin=thin),
    c=exp(p - p.max()),
    marker="."
)
plt.xlabel("parameter 1")
plt.ylabel("parameter 2")
plt.grid()
plt.tight_layout()
plt.show()


# We can easily estimate 1D marginal distributions for any parameter
# using the 'get_marginal' method:
pdf_1 = chain.get_marginal(0, burn=burn, thin=thin, unimodal=True)
pdf_2 = chain.get_marginal(1, burn=burn, thin=thin, unimodal=True)

# get_marginal returns a density estimator object, which can be called
# as a function to return the value of the pdf at any point.
# Make an axis on which to evaluate the PDFs:
ax = linspace(-3, 4, 500)

# plot the results
plt.plot(ax, pdf_1(ax), label="param #1 marginal", lw=2)
plt.plot(ax, pdf_2(ax), label="param #2 marginal", lw=2)

plt.xlabel("parameter value")
plt.ylabel("probability density")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# chain objects can be saved in their entirety as a single .npz file using
# the save() method, and then re-built using the load() class method, so
# that to save the chain you may write:

#       chain.save('chain_data.npz')

# and to re-build a chain object at it was before you would write

#       chain = GibbsChain.load('chain_data.npz')

# This allows you to advance a chain, store it, then re-load it at a later
# time to analyse the chain data, or re-start the chain should you decide
# more samples are required.

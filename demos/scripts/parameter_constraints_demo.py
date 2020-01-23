
from inference.mcmc import GibbsChain

from numpy.random import normal
from numpy import linspace, ones

import matplotlib.pyplot as plt

"""
Here we use a 2-parameter straight-line fitting problem as an example
of how boundary constraints can be applied directly to parameters in 
models without having to add these constraints explicitly as part of
a prior distribution.

Setting parameter boundaries in this way is advantageous because the
constraints are being imposed at a lower level, specifically the step
proposal values, meaning all proposed steps automatically satisfy the
prior constraints.

If instead we were to impose these constraints explicitly in a prior 
function, then all proposed steps which violated the prior constraints
would result in a 'failed' step attempt. These additional failures result
in a reduction in proposal distribution widths, and less efficient sampling.
"""


class LinePosterior(object):
    """
    This is a simple posterior for straight-line fitting
    with gaussian errors.
    """
    def __init__(self, x = None, y = None, err = None):
        self.x = x
        self.y = y
        self.err = err

    def __call__(self, theta):
        m, c = theta
        fwd = m*self.x + c
        ln_P = -0.5 * sum( ((self.y - fwd) / self.err)**2 )
        return ln_P


# create some synthetic data
N = 25
x = linspace(-2, 5, N)
m = 0.5; c = 0.05; sigma = 0.3
y = m*x + c + normal(size=N)*sigma

# plot the synthetic data and underlying line
plt.plot(x, m*x + c)
plt.plot(x, y, '.')
plt.grid()
plt.show()

# create an instance of the posterior class
posterior = LinePosterior(x = x, y = y, err = ones(N)*sigma)

# pass the posterior to the MCMC sampler
chain = GibbsChain(posterior = posterior, start = [0.5, 0.1])

# Now suppose we know the offset parameter must be non-negative.
# This constraint can be imposed by passing the index of the
# parameter to the set_non_negative method as follows:
chain.set_non_negative(1)

# For the purposes of this demo, let's assume we also know that
# the gradient must exist in the range [0.45, 0.55].
# The gradient can be constrained to values between chosen boundaries
# by passing the parameter index and the boundary values to the
# set_boundaries method as follows:
chain.set_boundaries(0, [0.45, 0.55])

# Advance the chain
chain.advance(50000)
chain.burn = 5000

# Use the matrix plot functionality to check the constraints are working
chain.matrix_plot()
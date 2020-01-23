
from numpy import linspace
from numpy.random import normal
import matplotlib.pyplot as plt

from inference.mcmc import PcaChain, GibbsChain

class CorrelatedLinePosterior(object):
    def __init__(self):
        # make some straight-line data with very low noise
        N = 100
        self.sigma = 0.06
        self.x = linspace(0,5,N)
        self.y = 2*self.x - 1 + normal(size=N)*self.sigma

    def __call__(self, theta):
        c, a, b = theta
        prediction = (a-b)*self.x + c
        likelihood = -0.5*(((self.y - prediction)/self.sigma)**2).sum()
        prior = -0.5*a**2 -0.5*b**2
        return likelihood + prior

"""
# PcaChain demo
 
The PcaChain sampler uses 'principal component analysis' (PCA) to improve
the performance of Gibbs sampling in cases where strong linear correlation
exists between two or more variables in a problem.

For an N-parameter problem, PcaChain produces a new sample by making N
sequential 1D Metropolis-Hastings steps in the direction of each of the
N eigenvectors of the NxN covariance matrix.

As an initial guess the covariance matrix is taken to be diagonal, which
results in standard gibbs sampling for the first samples in the chain.
Subsequently, the covariance matrix periodically updated with an estimate
derived from the sample itself, and the eigenvectors are re-calculated.
"""

# create our posterior with two highly-correlated parameters
posterior = CorrelatedLinePosterior()

# create a PcaChain, and also a GibbsChain for comparison
pca = PcaChain(posterior=posterior, start = [-1,1,-1])
gibbs = GibbsChain(posterior=posterior, start = [-1,1,-1])

# advance both chains for the same amount of samples
pca.advance(50000)
gibbs.advance(50000)

# get an estimate of the marginal distribution of one of the correlated parameters
pca_pdf = pca.get_marginal(2, burn = 5000)
gibbs_pdf = gibbs.get_marginal(2, burn = 5000)

# over-plot the marginal estimates to compare the performance
marginal_axis = linspace(-4,2,500)
plt.plot(marginal_axis, pca_pdf(marginal_axis), lw = 2, label = 'PcaChain estimate')
plt.plot(marginal_axis, gibbs_pdf(marginal_axis), lw = 2, label = 'GibbsChain estimate')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# The result shows that the GibbsChain has failed to properly estimate the marginal
# distribution, as the strong correlations cause it to explore the parameter space
# very inefficiently.

# For the same number of samples however, the PcaChain can accurately reproduce the
# marginal distribution.

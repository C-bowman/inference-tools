from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy import sqrt, exp, array
from inference.mcmc import HamiltonianChain

"""
# Hamiltonian sampling example

Hamiltonian Monte-Carlo (HMC) is a MCMC algorithm which is able to
efficiently sample from complex PDFs which present difficulty for
other algorithms, such as those which strong non-linear correlations.

However, this requires not only the log-posterior probability but also
its gradient in order to function. In cases where this gradient can be
calculated analytically HMC can be very effective.

The implementation of HMC shown here as HamiltonianChain is somewhat
naive, and should at some point be replaced with a more advanced
self-tuning version, such as the NUTS algorithm.
"""


# define a non-linearly correlated posterior distribution
class ToroidalGaussian(object):
    def __init__(self):
        self.R0 = 1. # torus major radius
        self.ar = 10. # torus aspect ratio
        self.w2 = (self.R0/self.ar)**2

    def __call__(self, theta):
        x, y, z = theta
        r = sqrt(z**2 + (sqrt(x**2 + y**2) - self.R0)**2)
        return -0.5*r**2 / self.w2

    def gradient(self, theta):
        x, y, z = theta
        R = sqrt(x**2 + y**2)
        K = 1 - self.R0/R
        g = array([K*x, K*y, z])
        return -g/self.w2


# create an instance of our posterior class
posterior = ToroidalGaussian()

# create the chain object
chain = HamiltonianChain(posterior = posterior, grad = posterior.gradient, start = [1,0.1,0.1])

# advance the chain to generate the sample
chain.advance(6000)

# choose how many samples will be thrown away from the start
# of the chain as 'burn-in'
chain.burn = 2000

chain.matrix_plot(filename = 'hmc_matrix_plot.png')








# extract sample and probability data from the chain
probs = chain.get_probabilities()
colors = exp(probs - max(probs))
xs, ys, zs = [ chain.get_parameter(i) for i in [0,1,2] ]


import plotly.graph_objects as go
from plotly import offline

fig = go.Figure(data=[go.Scatter3d(
    x=xs,
    y=ys,
    z=zs,
    mode='markers',
    marker=dict( size=5, color=colors, colorscale='Viridis', opacity=0.6)
)])

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0)) # tight layout
offline.plot(fig, filename='hmc_scatterplot.html')




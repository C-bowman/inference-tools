
import pytest
import unittest

from numpy import array, sqrt
from inference.mcmc import GibbsChain, HamiltonianChain


def rosenbrock(t):
    # This is a modified form of the rosenbrock function, which
    # is commonly used to test optimisation algorithms
    X, Y = t
    X2 = X**2
    b = 15  # correlation strength parameter
    v = 3   # variance of the gaussian term
    return -X2 - b*(Y - X2)**2 - 0.5*(X2 + Y**2)/v

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




class test_mcmc_samplers(unittest.TestCase):

    def test_gibbs_chain(self):
        start_location = array([2., -4.])
        width_guesses = array([5., 0.05])

        chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
        chain.advance(50000)

        p = chain.get_probabilities()
        chain.plot_diagnostics(show=False)

        chain.autoselect_burn()
        chain.autoselect_thin()

    def test_hamiltonian_chain(self):
        # create an instance of our posterior class
        posterior = ToroidalGaussian()
        chain = HamiltonianChain(posterior=posterior, grad=posterior.gradient, start=[1, 0.1, 0.1])
        chain.advance(3000)




if __name__ == '__main__':

    unittest.main()
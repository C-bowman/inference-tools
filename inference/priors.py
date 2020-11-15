
from inference.likelihoods import JointDistribution
from numpy import array, log, pi, zeros


class BasePrior(object):
    def __add__(self, other):
        if type(other) is JointDistribution:
            new = JointDistribution(other.components)
            new.components.append(self)
        else:
            new = JointDistribution([self,other])
        return new

    @staticmethod
    def check_parameters(parameters):
        if parameters is None:
            return slice(0, None)

        elif type(parameters) is int:
            return [parameters]

        elif type(parameters) is list and all(type(i) is int for i in parameters):
            return parameters

        else:
            raise ValueError('If specified, the "parameters" argument must be an integer or list of integers')




class GaussianPrior(BasePrior):
    def __init__(self, mean, sigma, parameters=None):

        self.mean = array(mean).squeeze()
        self.sigma = array(sigma).squeeze()
        self.n_params = self.mean.size

        if self.mean.size != self.sigma.size:
            raise ValueError('mean and sigma arguments must have the same number of elements')

        if self.mean.ndim > 1 or self.sigma.ndim > 1:
            raise ValueError('mean and sigma arguments must have either 0 or 1 dimensions')

        self.parameters = self.check_parameters(parameters)

        # pre-calculate some quantities as an optimisation
        self.inv_sigma = 1./self.sigma
        self.inv_sigma_sqr = self.inv_sigma**2
        self.normalisation = -log(self.sigma).sum() - 0.5*log(2*pi)*self.n_params

    def __call__(self, theta):
        z = (self.mean-theta)*self.inv_sigma
        return -0.5*(z**2).sum() + self.normalisation

    def gradient(self, theta):
        grad = zeros(len(theta))
        grad[self.parameters] = (self.mean-theta[self.parameters])*self.inv_sigma_sqr
        return grad




class ExponentialPrior(BasePrior):
    def __init__(self, beta, parameters=None):

        self.beta = array(beta).squeeze()
        self.n_params = self.beta.size

        if self.beta.ndim > 1:
            raise ValueError('beta argument must have either 0 or 1 dimensions')

        self.parameters = self.check_parameters(parameters)

        # pre-calculate some quantities as an optimisation
        self.lam = 1./self.beta
        self.normalisation = log(self.lam).sum()

    def __call__(self, theta):
        return -(self.lam*theta[self.parameters]).sum() + self.normalisation

    def gradient(self, theta):
        grad = zeros(len(theta))
        grad[self.parameters] = -self.lam
        return grad



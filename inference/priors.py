
from numpy import array, log, pi, zeros
from numpy.random import normal, exponential, uniform




class JointPrior(object):
    def __init__(self, components):
        self.components = components

        if not all( isinstance(c, BasePrior) for c in self.components):
            raise TypeError(
                """
                All objects contained in the 'components' argument must be instances
                of a subclass of BasePrior (e.g. GaussianPrior, UniformPrior)
                """
            )

    def __call__(self, theta):
        return sum( c(theta) for c in self.components )

    def gradient(self, theta):
        grad = zeros(theta.size)
        for c in self.components:
            grad[c.parameters] = c.gradient(theta)
        return grad

    def __add__(self, other):
        if isinstance(other, BasePrior):
            new = JointPrior(self.components)
            new.components.append(other)
            return new
        elif isinstance(other, JointPrior):
            new = JointPrior(self.components)
            new.components.extend(other.components)
            return new
        else:
            raise TypeError(
                """
                JointPrior.__add__ was given an argument of invalid type. Valid arguments are
                either an instance of JointPrior, or an instance of a subclass of BasePrior
                (e.g. GaussianPrior).
                """
            )




class BasePrior(object):
    def __add__(self, other):
        if isinstance(other, BasePrior):
            return JointPrior([self,other])
        elif isinstance(other, JointPrior):
            new = JointPrior(other.components)
            new.components.append(self)
            return new
        else:
            raise TypeError(
                """
                BasePrior.__add__ was given an argument of invalid type. Valid arguments are
                either an instance of JointPrior, or an instance of a subclass of BasePrior
                (e.g. GaussianPrior).
                """
            )

    @staticmethod
    def check_parameters(parameters, n_params):
        if parameters is None:
            return slice(0, None)

        elif type(parameters) is int:
            if n_params == 1:
                return [parameters]
            else:
                raise ValueError(
                    """
                    The total number of parameters specified via the 'parameters' argument does not match
                    the
                    """
                )

        elif type(parameters) is list and all(type(i) is int for i in parameters):
            return parameters

        else:
            raise TypeError('If specified, the "parameters" argument must be an integer or list of integers')




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
        z = (self.mean-theta[self.parameters])*self.inv_sigma
        return -0.5*(z**2).sum() + self.normalisation

    def gradient(self, theta):
        grad = zeros(len(theta))
        grad[self.parameters] = (self.mean-theta[self.parameters])*self.inv_sigma_sqr
        return grad

    def sample(self):
        return normal(loc=self.mean, scale=self.sigma, size=self.n_params)




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

    def sample(self):
        return exponential(scale=self.beta, size=self.n_params)




class UniformPrior(BasePrior):
    def __init__(self, lower, upper, parameters=None):

        self.lower = array(lower).squeeze()
        self.upper = array(upper).squeeze()
        self.n_params = self.lower.size

        if self.lower.size != self.upper.size:
            raise ValueError("""'lower' and 'upper' arguments must have the same number of elements""")

        if self.lower.ndim > 1 or self.upper.ndim > 1:
            raise ValueError("""'lower' and 'upper' arguments must have either 0 or 1 dimensions""")

        if (self.upper <= self.lower).any():
            raise ValueError("""All values in 'lower' must be less than the corresponding values in 'upper'""")

        self.parameters = self.check_parameters(parameters)

        # pre-calculate some quantities as an optimisation
        self.normalisation = -log(self.upper-self.lower).sum()

    def __call__(self, theta):
        inside = (self.lower <= theta[self.parameters]) & (theta[self.parameters] <= self.upper)
        if inside.all():
            return self.normalisation
        else:
            return -1e100

    def gradient(self, theta):
        return zeros(len(theta))

    def sample(self):
        return uniform(low=self.lower, high=self.upper, size=self.n_params)
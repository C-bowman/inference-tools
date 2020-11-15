
from numpy import array, log, exp, pi, sqrt



class JointDistribution(object):
    def __init__(self, components):
        self.components = components

    def __call__(self, theta):
        return sum( c(theta) for c in self.components )

    def gradient(self, theta):
        return sum( c.gradient(theta) for c in self.components )

    def __add__(self, other):
        new = JointDistribution(self.components)
        if type(other) is JointDistribution:
            new.components.extend(other.components)
        else:
            new.components.append(other)
        return new




class BaseLikelihood(object):
    def __add__(self, other):
        if type(other) is JointDistribution:
            new = JointDistribution(other.components)
            new.components.append(self)
        else:
            new = JointDistribution([self, other])
        return new




class GaussianLikelihood(BaseLikelihood):
    """
    :param y_data: \
        The measured data as a 1D array.

    :param sigma: \
        The standard deviations corresponding to each element in y_data as a 1D array.

    :param callable forward_model: \
        
    """
    def __init__(self, y_data, sigma, forward_model):

        if not hasattr(forward_model, '__call__'):
            raise AttributeError('Given forward_model object must be callable')

        self.y = array(y_data).squeeze()
        self.sigma = array(sigma).squeeze()
        self.model = forward_model

        if self.y.size != self.sigma.size:
            raise ValueError('y_data and sigma arguments must have the same number of elements')

        if self.y.ndim > 1 or self.sigma.ndim > 1:
            raise ValueError('y_data and sigma arguments must have either 0 or 1 dimensions')

        if (self.sigma <= 0).any():
            raise ValueError('All values in sigma argument must be greater than zero')

        # pre-calculate some quantities as an optimisation
        self.n_data = self.y.size
        self.inv_sigma = 1./self.sigma
        self.inv_sigma_sqr = self.inv_sigma**2
        self.normalisation = -log(self.sigma).sum() - 0.5*log(2*pi)*self.n_data

    def __call__(self, theta):
        prediction = self.model(theta)
        z = (self.y-prediction)*self.inv_sigma
        return -0.5*(z**2).sum() + self.normalisation

    def gradient(self, theta):
        prediction = self.model(theta)
        dF_dt = self.model.gradient(theta)
        dL_dF = (self.y-prediction)*self.inv_sigma_sqr
        return dF_dt.dot(dL_dF)




class CauchyLikelihood(BaseLikelihood):
    """
    :param y_data: \
        The measured data as a 1D array.

    :param gamma: \
        The uncertainties corresponding to each element in y_data as a 1D array.

    :param callable forward_model: \

    """

    def __init__(self, y_data, gamma, forward_model):

        if not hasattr(forward_model, '__call__'):
            raise AttributeError('Given forward_model object must be callable')

        self.y = array(y_data).squeeze()
        self.gamma = array(gamma).squeeze()
        self.model = forward_model

        if self.y.size != self.gamma.size:
            raise ValueError('y_data and gamma arguments must have the same number of elements')

        if self.y.ndim > 1 or self.gamma.ndim > 1:
            raise ValueError('y_data and gamma arguments must have either 0 or 1 dimensions')

        if (self.gamma <= 0).any():
            raise ValueError('All values in gamma argument must be greater than zero')

        # pre-calculate some quantities as an optimisation
        self.n_data = self.y.size
        self.inv_gamma = 1. / self.gamma
        self.normalisation = -log(pi*self.gamma).sum()

    def __call__(self, theta):
        prediction = self.model(theta)
        z = (self.y - prediction)*self.inv_gamma
        return -log(1 + z**2).sum() + self.normalisation

    def gradient(self, theta):
        prediction = self.model(theta)
        dF_dt = self.model.gradient(theta)
        z = (self.y - prediction)*self.inv_gamma
        dL_dF = -2*self.inv_gamma*z / (1 + z**2)
        return dF_dt.dot(dL_dF)



def ln_logistic(x, sigma=2, c=0):
    s = sigma*sqrt(3)/pi
    z = (x-c)/s
    return z - 2*log(1 + exp(z)) - log(s)



class LogisticLikelihood(BaseLikelihood):
    """
    :param y_data: \
        The measured data as a 1D array.

    :param sigma: \
        The uncertainties corresponding to each element in y_data as a 1D array.

    :param callable forward_model: \

    """
    def __init__(self, y_data, sigma, forward_model):

        if not hasattr(forward_model, '__call__'):
            raise AttributeError('Given forward_model object must be callable')

        self.y = array(y_data).squeeze()
        self.sigma = array(sigma).squeeze()
        self.model = forward_model

        if self.y.size != self.sigma.size:
            raise ValueError('y_data and sigma arguments must have the same number of elements')

        if self.y.ndim > 1 or self.sigma.ndim > 1:
            raise ValueError('y_data and sigma arguments must have either 0 or 1 dimensions')

        if (self.sigma <= 0).any():
            raise ValueError('All values in sigma argument must be greater than zero')

        # pre-calculate some quantities as an optimisation
        self.n_data = self.y.size
        self.scale = self.sigma*(sqrt(3)/pi)
        self.inv_scale = 1. / self.scale
        self.normalisation = -log(self.scale).sum()

    def __call__(self, theta):
        prediction = self.model(theta)
        z = (self.y - prediction)*self.inv_scale
        return z.sum() - 2*log(1 + exp(z)).sum() + self.normalisation

    def gradient(self, theta):
        prediction = self.model(theta)
        dF_dt = self.model.gradient(theta)
        z = (self.y - prediction)*self.inv_scale
        dL_dF = (2 / (1 + exp(-z)) - 1)*self.inv_scale
        return dF_dt.dot(dL_dF)
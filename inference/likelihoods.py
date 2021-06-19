
"""
.. moduleauthor:: Chris Bowman <chris.bowman.physics@gmail.com>
"""

from numpy import array, log, exp, pi, sqrt


class GaussianLikelihood(object):
    """
    A class for constructing a Gaussian likelihood function.

    :param y_data: \
        The measured data as a 1D array.

    :param sigma: \
        The standard deviations corresponding to each element in ``y_data`` as a 1D array.

    :param callable forward_model: \
        A callable which returns a prediction of the ``y_data`` values when passed an
        array of model parameter values.

    :keyword callable forward_model_jacobian: \
        A callable which returns the Jacobian of the forward-model when passed an array of model
        parameter values. The Jacobian is a 2D array containing the derivative of the predictions
        of each y_data value with respect to each model parameter, such that element ``(i,j)`` of the
        Jacobian corresponds to the derivative of the ``i``'th y_data value prediction with respect to
        the ``j``'th model parameter.
    """
    def __init__(self, y_data, sigma, forward_model, forward_model_jacobian=None):

        if not hasattr(forward_model, '__call__'):
            raise AttributeError('Given forward_model object must be callable')

        if forward_model_jacobian is None:
            self.model_jacobian = jacobian_not_given
            self.gradient_available = False
        elif hasattr(forward_model_jacobian, '__call__'):
            self.model_jacobian = forward_model_jacobian
            self.gradient_available = True
        else:
            raise AttributeError('Given forward_model_jacobian object must be callable')

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
        """
        Returns the log-likelihood value for the given set of model parameters.

        :param theta: \
            The model parameters as a 1D numpy array.

        :returns: \
            The log-likelihood value.
        """
        prediction = self.model(theta)
        z = (self.y-prediction)*self.inv_sigma
        return -0.5*(z**2).sum() + self.normalisation

    def gradient(self, theta):
        """
        Returns the gradient of the log-likelihood with respect to model parameters.

        Using this method requires that the ``forward_model_jacobian`` keyword argument
        was specified when the instance of ``GaussianLikelihood`` was created.

        :param theta: \
            The model parameters as a 1D numpy array.

        :returns: \
            The gradient of the log-likelihood as a 1D numpy array.
        """
        prediction = self.model(theta)
        dF_dt = self.model_jacobian(theta)
        dL_dF = (self.y-prediction)*self.inv_sigma_sqr
        return dL_dF.dot(dF_dt)






class CauchyLikelihood(object):
    """
    A class for constructing a Cauchy likelihood function.

    :param y_data: \
        The measured data as a 1D array.

    :param gamma: \
        The uncertainties corresponding to each element in ``y_data`` as a 1D array.

    :param callable forward_model: \
        A callable which returns a prediction of the ``y_data`` values when passed an
        array of model parameter values.

    :keyword callable forward_model_jacobian: \
        A callable which returns the Jacobian of the forward-model when passed an array of model
        parameter values. The Jacobian is a 2D array containing the derivative of the predictions
        of each y_data value with respect to each model parameter, such that element ``(i,j)`` of the
        Jacobian corresponds to the derivative of the ``i``'th y_data value prediction with respect to
        the ``j``'th model parameter.
    """
    def __init__(self, y_data, gamma, forward_model, forward_model_jacobian=None):

        if not hasattr(forward_model, '__call__'):
            raise AttributeError('Given forward_model object must be callable')

        if forward_model_jacobian is None:
            self.model_jacobian = jacobian_not_given
            self.gradient_available = False
        elif hasattr(forward_model_jacobian, '__call__'):
            self.model_jacobian = forward_model_jacobian
            self.gradient_available = True
        else:
            raise AttributeError('Given forward_model_jacobian object must be callable')

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
        """
        Returns the log-likelihood value for the given set of model parameters.

        :param theta: \
            The model parameters as a 1D numpy array.

        :returns: \
            The log-likelihood value.
        """
        prediction = self.model(theta)
        z = (self.y - prediction)*self.inv_gamma
        return -log(1 + z**2).sum() + self.normalisation

    def gradient(self, theta):
        """
        Returns the gradient of the log-likelihood with respect to model parameters.

        Using this method requires that the ``forward_model_jacobian`` keyword argument
        was specified when the instance of ``CauchyLikelihood`` was created.

        :param theta: \
            The model parameters as a 1D numpy array.

        :returns: \
            The gradient of the log-likelihood as a 1D numpy array.
        """
        prediction = self.model(theta)
        dF_dt = self.model_jacobian(theta)
        z = (self.y - prediction)*self.inv_gamma
        dL_dF = 2*self.inv_gamma*z / (1 + z**2)
        return dL_dF.dot(dF_dt)






class LogisticLikelihood(object):
    """
    A class for constructing a Logistic likelihood function.

    :param y_data: \
        The measured data as a 1D array.

    :param sigma: \
        The uncertainties corresponding to each element in ``y_data`` as a 1D array.

    :param callable forward_model: \
        A callable which returns a prediction of the ``y_data`` values when passed an
        array of model parameter values.

    :keyword callable forward_model_jacobian: \
        A callable which returns the Jacobian of the forward-model when passed an array of model
        parameter values. The Jacobian is a 2D array containing the derivative of the predictions
        of each y_data value with respect to each model parameter, such that element ``(i,j)`` of the
        Jacobian corresponds to the derivative of the ``i``'th y_data value prediction with respect to
        the ``j``'th model parameter.
    """
    def __init__(self, y_data, sigma, forward_model, forward_model_jacobian=None):

        if not hasattr(forward_model, '__call__'):
            raise AttributeError('Given forward_model object must be callable')

        if forward_model_jacobian is None:
            self.model_jacobian = jacobian_not_given
            self.gradient_available = False
        elif hasattr(forward_model_jacobian, '__call__'):
            self.model_jacobian = forward_model_jacobian
            self.gradient_available = True
        else:
            raise AttributeError('Given forward_model_jacobian object must be callable')

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
        """
        Returns the log-likelihood value for the given set of model parameters.

        :param theta: \
            The model parameters as a 1D numpy array.

        :returns: \
            The log-likelihood value.
        """
        prediction = self.model(theta)
        z = (self.y - prediction)*self.inv_scale
        return z.sum() - 2*log(1 + exp(z)).sum() + self.normalisation

    def gradient(self, theta):
        """
        Returns the gradient of the log-likelihood with respect to model parameters.

        Using this method requires that the ``forward_model_jacobian`` keyword argument
        was specified when the instance of ``LogisticLikelihood`` was created.

        :param theta: \
            The model parameters as a 1D numpy array.

        :returns: \
            The gradient of the log-likelihood as a 1D numpy array.
        """
        prediction = self.model(theta)
        dF_dt = self.model_jacobian(theta)
        z = (self.y - prediction)*self.inv_scale
        dL_dF = (2 / (1 + exp(-z)) - 1)*self.inv_scale
        return dL_dF.dot(dF_dt)






def jacobian_not_given(*args):
    raise ValueError(
        """
        The gradient() method of a likelihood class instance was called, however
        the forward_model_jacobian keyword argument was not specified when instance 
        was created.
        """
    )
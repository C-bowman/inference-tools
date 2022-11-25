"""
.. moduleauthor:: Chris Bowman <chris.bowman.physics@gmail.com>
"""
from abc import ABC, abstractmethod
from numpy import array, log, exp, pi, sqrt


class Likelihood(ABC):
    """
    Base class for likelihood functors

    :param y_data: \
        The measured data as a 1D array.

    :param uncertainties: \
        The standard deviations or uncertainties corresponding to each
        element in ``y_data`` as a 1D array.

    :param uncertainties_name: \
        The name of the standard_deviation or uncertainties attribute

    :param callable forward_model: \
        A callable which returns a prediction of the ``y_data`` values when passed an
        array of model parameter values.

    :keyword callable forward_model_jacobian: \
        A callable which returns the Jacobian of the forward-model when passed an array
        of model parameter values. The Jacobian is a 2D array containing the derivative
        of the predictions of each ``y_data`` value with respect to each model parameter,
        such that element ``(i,j)`` of the Jacobian corresponds to the derivative of the
        ``i``'th ``y_data`` value prediction with respect to the ``j``'th model parameter.
    """

    def __init__(
        self,
        y_data,
        uncertainties,
        uncertainties_name,
        forward_model,
        forward_model_jacobian=None,
    ):
        if not callable(forward_model):
            raise ValueError("Given forward_model object must be callable")

        if forward_model_jacobian is None:
            self.model_jacobian = jacobian_not_given
            self.gradient_available = False
        elif callable(forward_model_jacobian):
            self.model_jacobian = forward_model_jacobian
            self.gradient_available = True
        else:
            raise ValueError("Given forward_model_jacobian object must be callable")

        self.y = array(y_data).squeeze()
        _uncertainties = array(uncertainties).squeeze()
        setattr(self, uncertainties_name, _uncertainties)
        self.model = forward_model

        if self.y.size != _uncertainties.size:
            raise ValueError(
                f"y_data and {uncertainties_name} arguments must have the same number of elements"
            )

        if self.y.ndim > 1 or _uncertainties.ndim > 1:
            raise ValueError(
                f"y_data and {uncertainties_name} arguments must have either 0 or 1 dimensions"
            )

        if (_uncertainties <= 0).any():
            raise ValueError(
                f"All values in {uncertainties_name} argument must be greater than zero"
            )

    @abstractmethod
    def _log_likelihood(self, predictions):
        pass

    @abstractmethod
    def _log_likelihood_gradient(self, predictions, predictions_jacobian):
        pass

    def __call__(self, theta):
        """
        Returns the log-likelihood value for the given set of model parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The log-likelihood value.
        """
        return self._log_likelihood(predictions=self.model(theta))

    def gradient(self, theta):
        """
        Returns the gradient of the log-likelihood with respect to model parameters.

        Using this method requires that the ``forward_model_jacobian`` keyword argument
        was specified when the instance of the class was created.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The gradient of the log-likelihood as a 1D ``numpy.ndarray``.
        """
        return self._log_likelihood_gradient(
            predictions=self.model(theta),
            predictions_jacobian=self.model_jacobian(theta),
        )

    def cost(self, theta):
        return -self.__call__(theta)

    def cost_gradient(self, theta):
        return -self.gradient(theta)


class GaussianLikelihood(Likelihood):
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
        A callable which returns the Jacobian of the forward-model when passed an array
        of model parameter values. The Jacobian is a 2D array containing the derivative
        of the predictions of each ``y_data`` value with respect to each model parameter,
        such that element ``(i,j)`` of the Jacobian corresponds to the derivative of the
        ``i``'th ``y_data`` value prediction with respect to the ``j``'th model parameter.
    """

    def __init__(self, y_data, sigma, forward_model, forward_model_jacobian=None):

        super().__init__(y_data, sigma, "sigma", forward_model, forward_model_jacobian)

        # pre-calculate some quantities as an optimisation
        self.n_data = self.y.size
        self.inv_sigma = 1.0 / self.sigma
        self.inv_sigma_sqr = self.inv_sigma**2
        self.normalisation = -log(self.sigma).sum() - 0.5 * log(2 * pi) * self.n_data

    def _log_likelihood(self, predictions):
        z = (self.y - predictions) * self.inv_sigma
        return -0.5 * (z**2).sum() + self.normalisation

    def _log_likelihood_gradient(self, predictions, predictions_jacobian):
        dL_dF = (self.y - predictions) * self.inv_sigma_sqr
        return dL_dF @ predictions_jacobian


class CauchyLikelihood(Likelihood):
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
        A callable which returns the Jacobian of the forward-model when passed an array
        of model parameter values. The Jacobian is a 2D array containing the derivative
        of the predictions of each ``y_data`` value with respect to each model parameter,
        such that element ``(i,j)`` of the Jacobian corresponds to the derivative of the
        ``i``'th ``y_data`` value prediction with respect to the ``j``'th model parameter.
    """

    def __init__(self, y_data, gamma, forward_model, forward_model_jacobian=None):

        super().__init__(y_data, gamma, "gamma", forward_model, forward_model_jacobian)

        # pre-calculate some quantities as an optimisation
        self.n_data = self.y.size
        self.inv_gamma = 1.0 / self.gamma
        self.normalisation = -log(pi * self.gamma).sum()

    def _log_likelihood(self, predictions):
        z = (self.y - predictions) * self.inv_gamma
        return -log(1 + z**2).sum() + self.normalisation

    def _log_likelihood_gradient(self, predictions, predictions_jacobian):
        z = (self.y - predictions) * self.inv_gamma
        dL_dF = 2 * self.inv_gamma * z / (1 + z**2)
        return dL_dF @ predictions_jacobian


class LogisticLikelihood(Likelihood):
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
        A callable which returns the Jacobian of the forward-model when passed an array
        of model parameter values. The Jacobian is a 2D array containing the derivative
        of the predictions of each ``y_data`` value with respect to each model parameter,
        such that element ``(i,j)`` of the Jacobian corresponds to the derivative of the
        ``i``'th ``y_data`` value prediction with respect to the ``j``'th model parameter.
    """

    def __init__(self, y_data, sigma, forward_model, forward_model_jacobian=None):

        super().__init__(y_data, sigma, "sigma", forward_model, forward_model_jacobian)

        # pre-calculate some quantities as an optimisation
        self.n_data = self.y.size
        self.scale = self.sigma * (sqrt(3) / pi)
        self.inv_scale = 1.0 / self.scale
        self.normalisation = -log(self.scale).sum()

    def _log_likelihood(self, predictions):
        z = (self.y - predictions) * self.inv_scale
        return z.sum() - 2 * log(1 + exp(z)).sum() + self.normalisation

    def _log_likelihood_gradient(self, predictions, predictions_jacobian):
        z = (self.y - predictions) * self.inv_scale
        dL_dF = (2 / (1 + exp(-z)) - 1) * self.inv_scale
        return dL_dF @ predictions_jacobian


def jacobian_not_given(*args):
    raise ValueError(
        """
        The gradient() method of a likelihood class instance was called, however
        the forward_model_jacobian keyword argument was not specified when instance 
        was created.
        """
    )

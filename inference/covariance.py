
from abc import ABC, abstractmethod
from numpy import abs, exp, eye, log


class CovarianceFunction(ABC):
    """
    Abstract base class for covariance functions.
    """
    @abstractmethod
    def pass_data(self, x, y):
        pass

    @abstractmethod
    def __call__(self, u, v, theta):
        pass

    @abstractmethod
    def build_covariance(self, theta):
        pass

    @abstractmethod
    def covariance_and_gradients(self, theta):
        pass


class SquaredExponential(CovarianceFunction):
    r"""
    ``SquaredExponential`` is a covariance-function class which can be passed to
    ``GpRegressor`` via the ``kernel`` keyword argument. It uses the 'squared-exponential'
    covariance function given by:

    .. math::

       K(\underline{u}, \underline{v}) = A^2 \exp \left( -\frac{1}{2} \sum_{i=1}^{n} \left(\frac{u_i - v_i}{l_i}\right)^2 \right)

    The hyper-parameter vector :math:`\underline{\theta}` used by ``SquaredExponential``
    to define the above function is structured as follows:

    .. math::

       \underline{\theta} = [ \ln{A}, \ln{l_1}, \ldots, \ln{l_n}]

    :param hyperpar_bounds: \
        By default, ``SquaredExponential`` will automatically set sensible lower and
        upper bounds on the value of the hyperparameters based on the available data.
        However, this keyword allows the bounds to be specified manually as a list of
        length-2 tuples giving the lower/upper bounds for each parameter.
    """

    def __init__(self, hyperpar_bounds=None):
        if hyperpar_bounds is None:
            self.bounds = None
        else:
            self.bounds = hyperpar_bounds

    def pass_data(self, x, y):
        """
        Pre-calculates hyperparameter-independent part of the data covariance
        matrix as an optimisation, and sets bounds on hyperparameter values.
        """
        # distributed outer subtraction using broadcasting
        dx = x[:, None, :] - x[None, :, :]
        self.distances = -0.5 * dx ** 2
        # small values added to the diagonal for stability
        self.epsilon = 1e-12 * eye(dx.shape[0])

        # construct sensible bounds on the hyperparameter values
        if self.bounds is None:
            s = log(y.std())
            self.bounds = [(s - 4, s + 4)]
            for i in range(x.shape[1]):
                lwr = log(abs(dx[:, :, i]).mean()) - 4
                upr = log(dx[:, :, i].max()) + 2
                self.bounds.append((lwr, upr))
        self.n_params = len(self.bounds)

    def __call__(self, u, v, theta):
        a = exp(theta[0])
        L = exp(theta[1:])
        D = -0.5 * (u[:, None, :] - v[None, :, :]) ** 2
        C = exp((D / L[None, None, :] ** 2).sum(axis=2))
        return (a ** 2) * C

    def build_covariance(self, theta):
        """
        Optimized version of self.matrix() specifically for the data
        covariance matrix where the vectors v1 & v2 are both self.x.
        """
        a = exp(theta[0])
        L = exp(theta[1:])
        C = exp((self.distances / L[None, None, :] ** 2).sum(axis=2)) + self.epsilon
        return (a ** 2) * C

    def gradient_terms(self, v, x, theta):
        """
        Calculates the covariance-function specific parts of
        the expression for the predictive mean and covariance
        of the gradient of the GP estimate.
        """
        a = exp(theta[0])
        L = exp(theta[1:])
        A = (x - v[None, :]) / L[None, :] ** 2
        return A.T, (a / L) ** 2

    def covariance_and_gradients(self, theta):
        a = exp(theta[0])
        L = exp(theta[1:])
        C = exp((self.distances / L[None, None, :] ** 2).sum(axis=2)) + self.epsilon
        K = (a ** 2) * C
        grads = [2.0 * K]
        for i, k in enumerate(L):
            grads.append((-2.0 / k ** 2) * self.distances[:, :, i] * K)
        return K, grads

    def get_bounds(self):
        return self.bounds


class RationalQuadratic(CovarianceFunction):
    r"""
    ``RationalQuadratic`` is a covariance-function class which can be passed to
    ``GpRegressor`` via the ``kernel`` keyword argument. It uses the 'rational quadratic'
    covariance function given by:

    .. math::

       K(\underline{u}, \underline{v}) = A^2 \left( 1 + \frac{1}{2\alpha} \sum_{i=1}^{n} \left(\frac{u_i - v_i}{l_i}\right)^2 \right)^{-\alpha}

    The hyper-parameter vector :math:`\underline{\theta}` used by ``RationalQuadratic``
    to define the above function is structured as follows:

    .. math::

       \underline{\theta} = [ \ln{A}, \ln{\alpha}, \ln{l_1}, \ldots, \ln{l_n}]

    :param hyperpar_bounds: \
        By default, ``RationalQuadratic`` will automatically set sensible lower and
        upper bounds on the value of the hyperparameters based on the available data.
        However, this keyword allows the bounds to be specified manually as a list of
        length-2 tuples giving the lower/upper bounds for each parameter.
    """

    def __init__(self, hyperpar_bounds=None):
        if hyperpar_bounds is None:
            self.bounds = None
        else:
            self.bounds = hyperpar_bounds

    def pass_data(self, x, y):
        """
        Pre-calculates hyperparameter-independent part of the data covariance
        matrix as an optimisation, and sets bounds on hyperparameter values.
        """
        # distributed outer subtraction using broadcasting
        dx = x[:, None, :] - x[None, :, :]
        self.distances = 0.5 * dx ** 2
        # small values added to the diagonal for stability
        self.epsilon = 1e-12 * eye(dx.shape[0])

        # construct sensible bounds on the hyperparameter values
        if self.bounds is None:
            s = log(y.std())
            self.bounds = [(s - 4, s + 4), (-2, 6)]
            for i in range(x.shape[1]):
                lwr = log(abs(dx[:, :, i]).mean()) - 4
                upr = log(dx[:, :, i].max()) + 2
                self.bounds.append((lwr, upr))

    def __call__(self, u, v, theta):
        a = exp(theta[0])
        k = exp(theta[1])
        L = exp(theta[2:])
        D = 0.5 * (u[:, None, :] - v[None, :, :]) ** 2
        Z = (D / L[None, None, :] ** 2).sum(axis=2)
        return (a ** 2) * (1 + Z / k) ** (-k)

    def build_covariance(self, theta):
        a = exp(theta[0])
        k = exp(theta[1])
        L = exp(theta[2:])
        Z = (self.distances / L[None, None, :] ** 2).sum(axis=2)
        return (a ** 2) * ((1 + Z / k) ** (-k) + self.epsilon)

    def gradient_terms(self, v, x, theta):
        """
        Calculates the covariance-function specific parts of
        the expression for the predictive mean and covariance
        of the gradient of the GP estimate.
        """
        raise ValueError(
            """
            Gradient calculations are not yet available for the
            RationalQuadratic covariance function.
            """
        )

    def covariance_and_gradients(self, theta):
        a = exp(theta[0])
        q = exp(theta[1])
        L = exp(theta[2:])
        Z = (self.distances / L[None, None, :] ** 2).sum(axis=2)

        F = 1 + Z / q
        ln_F = log(F)
        C = exp(-q * ln_F) + self.epsilon

        K = (a ** 2) * C
        grads = [2.0 * K, -K * (ln_F * q - Z / F)]
        G = 2 * K / F
        for i, l in enumerate(L):
            grads.append(G * (self.distances[:, :, i] / l ** 2))
        return K, grads

    def get_bounds(self):
        return self.bounds
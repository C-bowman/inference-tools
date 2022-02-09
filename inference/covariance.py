from abc import ABC, abstractmethod
from numpy import abs, exp, eye, log, zeros


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

    def __add__(self, other):
        K1 = self.components if isinstance(self, CompositeCovariance) else [self]
        K2 = other.components if isinstance(other, CompositeCovariance) else [other]
        return CompositeCovariance([*K1, *K2])

    def gradient_terms(self, v, x, theta):
        raise NotImplementedError(
            f"""
            Gradient calculations are not yet available for the
            {type(self)} covariance function.
            """
        )


class CompositeCovariance(CovarianceFunction):
    def __init__(self, covariance_components):
        self.components = covariance_components

    def pass_data(self, x, y):
        [comp.pass_data(x, y) for comp in self.components]
        # here we'd need to build slices to pass parameters
        # to each component, and construct a top-level bounds list
        self.slices = []
        for comp in self.components:
            L = comp.n_params
            if len(self.slices) == 0:
                self.slices.append(slice(0, L))
            else:
                last = self.slices[-1].stop
                self.slices.append(slice(last, last + L))

        self.bounds = []
        [self.bounds.extend(comp.bounds) for comp in self.components]

        self.hyperpar_labels = []
        for i, comp in enumerate(self.components):
            labels = [f"K{i+1}_{s}" for s in comp.hyperpar_labels]
            self.hyperpar_labels.extend(labels)
        self.n_params = len(self.hyperpar_labels)

    def __call__(self, u, v, theta):
        return sum(
            comp(u, v, theta[slc]) for comp, slc in zip(self.components, self.slices)
        )

    def build_covariance(self, theta):
        return sum(
            comp.build_covariance(theta[slc])
            for comp, slc in zip(self.components, self.slices)
        )

    def covariance_and_gradients(self, theta):
        results = [
            comp.covariance_and_gradients(theta[slc])
            for comp, slc in zip(self.components, self.slices)
        ]
        K = sum(r[0] for r in results)
        gradients = []
        [gradients.extend(r[1]) for r in results]
        return K, gradients


class WhiteNoise(CovarianceFunction):
    r"""
    ``WhiteNoise`` is a covariance-function class which models the presence of
    independent identically-distributed Gaussian (i.e. white) noise on the input data.
    The covariance can be expressed as:

    .. math::

       K(x_i, x_j) = \delta_{ij} \sigma_{n}^{2}

    where :math:`\delta_{ij}` is the Kronecker delta and  :math:`\sigma_{n}` is the
    Gaussian noise standard-deviation. The natural log of the noise-level
    :math:`\ln{\sigma_{n}}` is the only hyperparameter.

    ``WhiteNoise`` should be used as part of a 'composite' covariance function, as it
    doesn't model the underlying structure of the data by itself. Composite covariance
    functions can be constructed by addition, for example:

    .. code-block:: python

       from inference.gp import SquaredExponential, WhiteNoise
       composite_kernel = SquaredExponential() + WhiteNoise()

    :param hyperpar_bounds: \
        By default, ``WhiteNoise`` will automatically set sensible lower and
        upper bounds on the value of the log-noise-level based on the available data.
        However, this keyword allows the bounds to be specified manually as a length-2
        tuple giving the lower/upper bound.
    """

    def __init__(self, hyperpar_bounds=None):
        self.bounds = hyperpar_bounds

    def pass_data(self, x, y):
        """
        Pre-calculates hyperparameter-independent part of the data covariance
        matrix as an optimisation, and sets bounds on hyperparameter values.
        """
        self.I = eye(y.size)

        # construct sensible bounds on the hyperparameter values
        if self.bounds is None:
            s = log(y.ptp())
            self.bounds = [(s - 8, s + 2)]

        self.n_params = 1
        self.hyperpar_labels = ["log_noise_level"]

    def __call__(self, u, v, theta):
        return zeros([u.size, v.size])

    def build_covariance(self, theta):
        """
        Optimized version of self.matrix() specifically for the data
        covariance matrix where the vectors v1 & v2 are both self.x.
        """
        sigma_sq = exp(2 * theta[0])
        return sigma_sq * self.I

    def covariance_and_gradients(self, theta):
        sigma_sq = exp(2 * theta[0])
        K = sigma_sq * self.I
        grads = [2.0 * K]
        return K, grads

    def get_bounds(self):
        return self.bounds


class SquaredExponential(CovarianceFunction):
    r"""
    ``SquaredExponential`` is a covariance-function class which can be passed to
    ``GpRegressor`` via the ``kernel`` keyword argument. It uses the 'squared-exponential'
    covariance function given by:

    .. math::

       K(\underline{u}, \underline{v}) = A^2 \exp \left( -\frac{1}{2} \sum_{i=1}^{n} \left(\frac{u_i - v_i}{l_i}\right)^2 \right)

    The hyperparameter vector :math:`\underline{\theta}` used by ``SquaredExponential``
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
        self.bounds = hyperpar_bounds

    def pass_data(self, x, y):
        """
        Pre-calculates hyperparameter-independent part of the data covariance
        matrix as an optimisation, and sets bounds on hyperparameter values.
        """
        # distributed outer subtraction using broadcasting
        dx = x[:, None, :] - x[None, :, :]
        self.distances = -0.5 * dx**2
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
        self.n_params = x.shape[1] + 1
        self.hyperpar_labels = ["log_amplitude"]
        self.hyperpar_labels.extend([f"log_scale_{i}" for i in range(x.shape[1])])

    def __call__(self, u, v, theta):
        a = exp(theta[0])
        L = exp(theta[1:])
        D = -0.5 * (u[:, None, :] - v[None, :, :]) ** 2
        C = exp((D / L[None, None, :] ** 2).sum(axis=2))
        return (a**2) * C

    def build_covariance(self, theta):
        """
        Optimized version of self.matrix() specifically for the data
        covariance matrix where the vectors v1 & v2 are both self.x.
        """
        a = exp(theta[0])
        L = exp(theta[1:])
        C = exp((self.distances / L[None, None, :] ** 2).sum(axis=2)) + self.epsilon
        return (a**2) * C

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
        K = (a**2) * C
        grads = [2.0 * K]
        for i, k in enumerate(L):
            grads.append((-2.0 / k**2) * self.distances[:, :, i] * K)
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
        self.bounds = hyperpar_bounds

    def pass_data(self, x, y):
        """
        Pre-calculates hyperparameter-independent part of the data covariance
        matrix as an optimisation, and sets bounds on hyperparameter values.
        """
        # distributed outer subtraction using broadcasting
        dx = x[:, None, :] - x[None, :, :]
        self.distances = 0.5 * dx**2
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
        self.n_params = x.shape[1] + 2
        self.hyperpar_labels = ["log_amplitude", "log_alpha"]
        self.hyperpar_labels.extend([f"log_scale_{i}" for i in range(x.shape[1])])

    def __call__(self, u, v, theta):
        a = exp(theta[0])
        k = exp(theta[1])
        L = exp(theta[2:])
        D = 0.5 * (u[:, None, :] - v[None, :, :]) ** 2
        Z = (D / L[None, None, :] ** 2).sum(axis=2)
        return (a**2) * (1 + Z / k) ** (-k)

    def build_covariance(self, theta):
        a = exp(theta[0])
        k = exp(theta[1])
        L = exp(theta[2:])
        Z = (self.distances / L[None, None, :] ** 2).sum(axis=2)
        return (a**2) * ((1 + Z / k) ** (-k) + self.epsilon)

    def covariance_and_gradients(self, theta):
        a = exp(theta[0])
        q = exp(theta[1])
        L = exp(theta[2:])
        Z = (self.distances / L[None, None, :] ** 2).sum(axis=2)

        F = 1 + Z / q
        ln_F = log(F)
        C = exp(-q * ln_F) + self.epsilon

        K = (a**2) * C
        grads = [2.0 * K, -K * (ln_F * q - Z / F)]
        G = 2 * K / F
        for i, l in enumerate(L):
            grads.append(G * (self.distances[:, :, i] / l**2))
        return K, grads

    def get_bounds(self):
        return self.bounds

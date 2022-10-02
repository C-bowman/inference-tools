from abc import ABC, abstractmethod
from inspect import isclass
from numpy import abs, exp, eye, log, zeros, ndarray


class CovarianceFunction(ABC):
    """
    Abstract base class for covariance functions.
    """

    @abstractmethod
    def pass_spatial_data(self, x):
        pass

    @abstractmethod
    def estimate_hyperpar_bounds(self, y):
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
        self.bounds = None

    def pass_spatial_data(self, x):
        """
        Pre-calculates hyperparameter-independent part of the data covariance
        matrix as an optimisation.
        """
        [comp.pass_spatial_data(x) for comp in self.components]
        # Create slices to address the parameters of each component
        self.slices = slice_builder([c.n_params for c in self.components])
        # combine hyperparameter labels for each component
        self.hyperpar_labels = []
        for i, comp in enumerate(self.components):
            labels = [f"K{i+1}: {s}" for s in comp.hyperpar_labels]
            self.hyperpar_labels.extend(labels)
        # check for consistency of length of bounds, labels
        self.n_params = sum(c.n_params for c in self.components)
        assert self.n_params == len(self.hyperpar_labels)

    def estimate_hyperpar_bounds(self, y):
        """
        Estimates bounds on the hyper-parameters to be
        used during optimisation.
        """
        # estimate bounds for components where they were not specified
        for comp in self.components:
            if comp.bounds is None:
                comp.estimate_hyperpar_bounds(y)
        # combine parameter bounds for each component
        self.bounds = []
        [self.bounds.extend(comp.bounds) for comp in self.components]
        assert self.n_params == len(self.bounds)

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

    :param hyperpar_bounds:
        By default, ``WhiteNoise`` will automatically set sensible lower and
        upper bounds on the value of the log-noise-level based on the available data.
        However, this keyword allows the bounds to be specified manually as a length-2
        tuple giving the lower/upper bound.
    """

    def __init__(self, hyperpar_bounds=None):
        self.bounds = hyperpar_bounds
        self.n_params = 1
        self.hyperpar_labels = ["WhiteNoise log-sigma"]

    def pass_spatial_data(self, x: ndarray):
        """
        Pre-calculates hyperparameter-independent part of the data covariance
        matrix as an optimisation.
        """
        self.I = eye(x.shape[0])

    def estimate_hyperpar_bounds(self, y: ndarray):
        """
        Estimates bounds on the hyper-parameters to be
        used during optimisation.
        """
        # construct sensible bounds on the hyperparameter values
        s = log(y.ptp())
        self.bounds = [(s - 8, s + 2)]

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

    :param hyperpar_bounds:
        By default, ``SquaredExponential`` will automatically set sensible lower and
        upper bounds on the value of the hyperparameters based on the available data.
        However, this keyword allows the bounds to be specified manually as a list of
        length-2 tuples giving the lower/upper bounds for each parameter.
    """

    def __init__(self, hyperpar_bounds=None):
        self.bounds = hyperpar_bounds
        self.n_params: int
        self.dx: ndarray
        self.distances: ndarray
        self.hyperpar_labels: list

    def pass_spatial_data(self, x: ndarray):
        """
        Pre-calculates hyperparameter-independent part of the data covariance
        matrix as an optimisation.
        """
        # distributed outer subtraction using broadcasting
        self.dx = x[:, None, :] - x[None, :, :]
        self.distances = -0.5 * self.dx**2
        # small values added to the diagonal for stability
        self.epsilon = 1e-12 * eye(self.dx.shape[0])
        self.n_params = x.shape[1] + 1
        self.hyperpar_labels = ["SqrExp log-amplitude"]
        self.hyperpar_labels.extend(
            [f"SqrExp log-scale {i}" for i in range(x.shape[1])]
        )

    def estimate_hyperpar_bounds(self, y: ndarray):
        """
        Estimates bounds on the hyper-parameters to be
        used during optimisation.
        """
        s = log(y.std())
        self.bounds = [(s - 4, s + 4)]
        for i in range(self.dx.shape[2]):
            lwr = log(abs(self.dx[:, :, i]).mean()) - 4
            upr = log(self.dx[:, :, i].max()) + 2
            self.bounds.append((lwr, upr))

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

    :param hyperpar_bounds:
        By default, ``RationalQuadratic`` will automatically set sensible lower and
        upper bounds on the value of the hyperparameters based on the available data.
        However, this keyword allows the bounds to be specified manually as a list of
        length-2 tuples giving the lower/upper bounds for each parameter.
    """

    def __init__(self, hyperpar_bounds=None):
        self.bounds = hyperpar_bounds

    def pass_spatial_data(self, x: ndarray):
        """
        Pre-calculates hyperparameter-independent part of the data covariance
        matrix as an optimisation.
        """
        # distributed outer subtraction using broadcasting
        self.dx = x[:, None, :] - x[None, :, :]
        self.distances = 0.5 * self.dx**2
        # small values added to the diagonal for stability
        self.epsilon = 1e-12 * eye(self.dx.shape[0])
        self.n_params = x.shape[1] + 2
        self.hyperpar_labels = ["RQ log-amplitude", "RQ log-alpha"]
        self.hyperpar_labels.extend([f"RQ log-scale {i}" for i in range(x.shape[1])])

    def estimate_hyperpar_bounds(self, y: ndarray):
        """
        Estimates bounds on the hyper-parameters to be
        used during optimisation.
        """
        s = log(y.std())
        self.bounds = [(s - 4, s + 4), (-2, 6)]
        for i in range(self.dx.shape[2]):
            lwr = log(abs(self.dx[:, :, i]).mean()) - 4
            upr = log(self.dx[:, :, i].max()) + 2
            self.bounds.append((lwr, upr))

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


class ChangePoint(CovarianceFunction):
    r"""
    ``ChangePoint`` is a covariance function which divides the input space into two
    regions (at some point along a chosen input dimension), allowing each of the two
    regions to be modelled using a separate covariance function.

    This is useful in cases where properties of the data (e.g. the scale-lengths
    over which the data vary) change significantly over the input dimension which is
    used to divide the space.

    The change-point kernel :math:`K_{\mathrm{cp}}` is a weighted-sum of the two
    input kernels :math:`K_{1}, \, K_{2}` which model each of the two regions:

    .. math::

       K_{\mathrm{cp}}(u, v) = K_{1}(u, v) (1 - f(u))(1 - f(v)) + K_{2}(u, v) f(u) f(v)

    where the weighting :math:`f(x)` is the logistic function

    .. math::

       f(x) = \frac{1}{1 + e^{-(x - x_0) / w}}

    and :math:`x_0, \, w` are the location and width of the change-point respectively.
    :math:`x_0` and :math:`w` are hyperparameters which are determined automatically
    (alongside the hyperparameters for :math:`K_{1}, \, K_{2}`).

    :param K1:
        The covariance kernel which applies to the 'low' side of the change-point.

    :param K2:
        The covariance kernel which applies to the 'high' side of the change-point.

    :param int axis:
        The spatial axis over which the transition between the two kernels occurs.

    :param location_bounds:
        The bounds for the change-point location hyperparameter
        :math:`x_0` as a tuple of the form ``(lower_bound, upper_bound)``.

    :param width_bounds:
        The bounds for the change-point width hyperparameter :math:`w` as a tuple
        of the form ``(lower_bound, upper_bound)``.
    """

    def __init__(
        self,
        K1=SquaredExponential,
        K2=SquaredExponential,
        axis=0,
        location_bounds=None,
        width_bounds=None,
    ):
        self.cov1 = K1() if isclass(K1) else K1
        self.cov2 = K2() if isclass(K2) else K2
        self.axis = axis
        self.location_bounds = check_bounds(location_bounds)
        self.width_bounds = check_bounds(width_bounds)
        self.hyperpar_labels = []
        self.bounds = None

    def pass_spatial_data(self, x: ndarray):
        self.cov1.pass_spatial_data(x)
        self.cov2.pass_spatial_data(x)
        # Create slices to address the parameters of each component
        param_counts = [self.cov1.n_params, self.cov2.n_params, 2]
        self.n_params = sum(param_counts)
        self.K1_slc, self.K2_slc, self.CP_slc = slice_builder(param_counts)
        # combine hyperparameter labels for K1, K2 and the change-point
        label_groups = [
            [f"ChngPnt K1: {lab}" for lab in self.cov1.hyperpar_labels],
            [f"ChngPnt K2: {lab}" for lab in self.cov2.hyperpar_labels],
            ["ChngPnt location", "ChngPnt width"],
        ]
        [self.hyperpar_labels.extend(L) for L in label_groups]

        # store x-data from the dimension of the change-point
        self.x_cp = x[:, self.axis]
        assert self.n_params == len(self.hyperpar_labels)

    def estimate_hyperpar_bounds(self, y: ndarray):
        xr = self.x_cp.min(), self.x_cp.max()
        dx = xr[1] - xr[0]
        # combine parameter bounds for K1, K2 and the change-point
        if self.cov1.bounds is None:
            self.cov1.estimate_hyperpar_bounds(y)
        if self.cov2.bounds is None:
            self.cov2.estimate_hyperpar_bounds(y)
        self.bounds = []
        self.bounds.extend(self.cov1.bounds)
        self.bounds.extend(self.cov2.bounds)
        self.bounds.extend(
            [
                xr if self.location_bounds is None else self.location_bounds,
                (5e-3 * dx, 0.5 * dx)
                if self.width_bounds is None
                else self.width_bounds,
            ]
        )
        # check for consistency of length of bounds
        assert self.n_params == len(self.bounds)

    def __call__(self, u, v, theta):
        K1 = self.cov1(u, v, theta[self.K1_slc])
        K2 = self.cov2(u, v, theta[self.K2_slc])
        w_u = self.logistic(u[:, self.axis], theta[self.CP_slc])
        w_v = self.logistic(v[:, self.axis], theta[self.CP_slc])
        w1 = (1 - w_u)[:, None] * (1 - w_v)[None, :]
        w2 = w_u[:, None] * w_v[None, :]
        return K1 * w1 + K2 * w2

    def build_covariance(self, theta):
        K1 = self.cov1.build_covariance(theta[self.K1_slc])
        K2 = self.cov2.build_covariance(theta[self.K2_slc])
        w = self.logistic(self.x_cp, theta[self.CP_slc])
        w1 = (1 - w)[:, None] * (1 - w)[None, :]
        w2 = w[:, None] * w[None, :]
        return K1 * w1 + K2 * w2

    def covariance_and_gradients(self, theta):
        K1, K1_grads = self.cov1.covariance_and_gradients(theta[self.K1_slc])
        K2, K2_grads = self.cov2.covariance_and_gradients(theta[self.K2_slc])
        w, w_grads = self.logistic_and_gradient(self.x_cp, theta[self.CP_slc])
        w1 = (1 - w)[:, None] * (1 - w)[None, :]
        w2 = w[:, None] * w[None, :]
        K = K1 * w1 + K2 * w2
        gradients = [c * w1 for c in K1_grads]
        gradients.extend([c * w2 for c in K2_grads])
        for g in w_grads:
            A = -g[:, None] * (1 - w)[None, :]
            B = g[:, None] * w[None, :]
            gradients.append(K1 * (A + A.T) + K2 * (B + B.T))
        return K, gradients

    @staticmethod
    def logistic(x, theta):
        z = (x - theta[0]) / theta[1]
        return 1.0 / (1.0 + exp(-z))

    @staticmethod
    def logistic_and_gradient(x, theta):
        z = (x - theta[0]) / theta[1]
        f = 1.0 / (1.0 + exp(-z))
        dfdc = -f * (1 - f) / theta[1]
        return f, [dfdc, dfdc * z]


def slice_builder(lengths):
    slices = [slice(0, lengths[0])]
    for L in lengths[1:]:
        last = slices[-1].stop
        slices.append(slice(last, last + L))
    return slices


def check_bounds(bounds):
    if bounds is not None:
        assert type(bounds) in [list, tuple, ndarray]
        assert len(bounds) == 2
        assert bounds[1] > bounds[0]
    return bounds

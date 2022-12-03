from abc import ABC, abstractmethod
from collections.abc import Sequence
from inspect import isclass
from itertools import chain
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

    def __call__(self, u: ndarray, v: ndarray, theta):
        return zeros([u.shape[0], v.shape[0]])

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
    ``ChangePoint`` is a covariance function which divides the input space into multiple
    regions (at various points along a chosen input dimension), allowing each of the
    regions to be modelled using a separate covariance function. The boundaries which
    define the extent of each region are referred to as 'change-points'. The locations
    of the change-points, and the width over which the transition between regions occurs
    are hyperparameters determined from the data.

    This is useful in cases where properties of the data (e.g. the scale-lengths
    over which the data vary) change significantly over the input dimension which is
    used to divide the space.

    The change-point kernel :math:`K_{\mathrm{cp}}` is a weighted-sum of the
    input kernels :math:`K_{1}, \, K_{2}, \dots , K_{n}` which model each of the
    :math:`n` regions:

    .. math::
       K_{\mathrm{cp}}(u, v) = K_1 a_1 + \left(\sum_{i=2}^{n-1} K_i a_{i+1} b_{i}\right) + K_n b_n

    where

    .. math::
       a_{i}(u, v) = (1 - f_i (u)) (1 - f_i (v)), \quad b_{i}(u, v) = f_i (u) f_i (v)

    and :math:`f_i` is the logistic weighting function associated with the :math:`i`'th
    change-point:

    .. math::
       f_i(x) = \frac{1}{1 + e^{-(x - c_i) / w_i}}

    and :math:`c_i, \, w_i` are the location and width of the :math:`i`'th change-point
    respectively. The :math:`c_i` and :math:`w_i` are hyperparameters which are determined
    automatically (alongside the hyperparameters for the kernels in each region).

    :param kernels:
        A tuple of the kernel objects to be used ``(K1, K2, K3, ...)``

    :param int axis:
        The spatial axis over which the transitions between kernels occur.

    :param location_bounds:
        The bounds for the change-point location hyperparameters :math:`c_i` as a tuple
        of the form ``((lower_bound_0, upper_bound_0),(lower_bound_1, upper_bound_1),...)``.
        There should always be :math:`n-1` pairs of bounds where :math:`n` is the number
        of kernels specified.

    :param width_bounds:
        The bounds for the change-point width hyperparameters :math:`w_i` as a tuple of
        the form ``((lower_bound_0, upper_bound_0),(lower_bound_1, upper_bound_1),...)``.
        There should always be :math:`n-1` pairs of bounds where :math:`n` is the number
        of kernels specified.
    """

    def __init__(
        self,
        kernels: Sequence,
        axis: int = 0,
        location_bounds: Sequence = None,
        width_bounds: Sequence = None,
    ):
        # check that all the kernels are valid
        self.cov = [
            K() if isclass(K) and issubclass(K, CovarianceFunction) else K
            for K in kernels
        ]
        for K in self.cov:
            if not isinstance(K, CovarianceFunction):
                raise TypeError(
                    """
                    [ ChangePoint error ]
                    >> Each of the specified covariance kernels must be an instance of
                    >> a class which inherits from the 'CovarianceFunction' abstract
                    >> base-class.
                    """
                )

        self.n_kernels = len(kernels)

        if location_bounds is not None:
            if len(location_bounds) != self.n_kernels - 1:
                raise ValueError(
                    """
                    [ ChangePoint error ]
                    >> The length of 'location_bounds' must be one less than the number of kernels
                    """
                )
            self.location_bounds = [check_bounds(lb) for lb in location_bounds]
        else:
            self.location_bounds = None

        if width_bounds is not None:
            if len(width_bounds) != self.n_kernels - 1:
                raise ValueError(
                    """
                    [ ChangePoint error ]
                    >> The length of 'width_bounds' must be one less than the number of kernels
                    """
                )
            self.width_bounds = [check_bounds(wb) for wb in width_bounds]
        else:
            self.width_bounds = None

        self.axis = axis
        self.hyperpar_labels = []
        self.bounds = None

    def pass_spatial_data(self, x: ndarray):
        [K.pass_spatial_data(x) for K in self.cov]
        # Create slices to address the parameters of each component
        param_counts = [K.n_params for K in self.cov]
        param_counts.extend([2] * (self.n_kernels - 1))

        self.n_params = sum(param_counts)
        slices = slice_builder(param_counts)
        self.cov_slc = slices[: self.n_kernels]
        self.cp_slc = slices[self.n_kernels :]

        # combine hyperparameter labels for Kernels and the change-point
        label_groups = []
        for i, K in enumerate(self.cov):
            label_groups.append([f"ChngPnt K{i}: {lab}" for lab in K.hyperpar_labels])

        for i in range(self.n_kernels - 1):
            label_groups.append([f"ChngPnt{i} location", f"ChngPnt{i} width"])

        [self.hyperpar_labels.extend(L) for L in label_groups]

        # store x-data from the dimension of the change-point
        self.x_cp = x[:, self.axis]
        assert self.n_params == len(self.hyperpar_labels)

    def estimate_hyperpar_bounds(self, y: ndarray):
        xr = self.x_cp.min(), self.x_cp.max()
        dx = xr[1] - xr[0]
        # combine parameter bounds for K1, K2 and the change-point
        self.bounds = []
        for cov in self.cov:
            cov.estimate_hyperpar_bounds(y)
            self.bounds.extend(cov.bounds)

        if self.location_bounds is None:
            self.location_bounds = [xr] * (self.n_kernels - 1)

        if self.width_bounds is None:
            self.width_bounds = [(5e-3 * dx, 0.5 * dx)] * (self.n_kernels - 1)

        # interleave the location / width bounds using chain and zip
        cp_bounds = chain.from_iterable(zip(self.location_bounds, self.width_bounds))
        self.bounds.extend([b for b in cp_bounds])

        # check for consistency of length of bounds
        assert self.n_params == len(self.bounds)

    def __call__(self, u: ndarray, v: ndarray, theta):
        kernel_coeffs = [1.0]
        for slc in self.cp_slc:
            w_u = self.logistic(u[:, self.axis], theta[slc])
            w_v = self.logistic(v[:, self.axis], theta[slc])

            w1 = (1 - w_u)[:, None] * (1 - w_v)[None, :]
            w2 = w_u[:, None] * w_v[None, :]

            kernel_coeffs[-1] *= w1
            kernel_coeffs.append(w2)

        return sum(
            self.cov[i](u, v, theta[self.cov_slc[i]]) * kernel_coeffs[i]
            for i in range(self.n_kernels)
        )

    def build_covariance(self, theta):
        kernel_coeffs = [1.0]
        for slc in self.cp_slc:
            w = self.logistic(self.x_cp, theta[slc])
            w1 = (1 - w)[:, None] * (1 - w)[None, :]
            w2 = w[:, None] * w[None, :]

            kernel_coeffs[-1] *= w1
            kernel_coeffs.append(w2)

        return sum(
            self.cov[i].build_covariance(theta[self.cov_slc[i]]) * kernel_coeffs[i]
            for i in range(self.n_kernels)
        )

    def covariance_and_gradients(self, theta):
        K_vals = []
        K_grads = []
        for i in range(self.n_kernels):
            K, dK = self.cov[i].covariance_and_gradients(theta[self.cov_slc[i]])
            K_vals.append(K)
            K_grads.append(dK)

        kernel_coeffs = [1.0]
        w_vals = []
        w_grads = []
        for slc in self.cp_slc:
            w, dw = self.logistic_and_gradient(self.x_cp, theta[slc])
            w1 = (1 - w)[:, None] * (1 - w)[None, :]
            w2 = w[:, None] * w[None, :]
            kernel_coeffs[-1] *= w1
            kernel_coeffs.append(w2)
            w_grads.append(dw)
            w_vals.append(w)

        covar = sum(K_vals[i] * kernel_coeffs[i] for i in range(self.n_kernels))

        gradients = []
        for i in range(self.n_kernels):
            gradients.extend([dK * kernel_coeffs[i] for dK in K_grads[i]])

        for i in range(self.n_kernels - 1):
            w = w_vals[i]
            for dw in w_grads[i]:
                A = -dw[:, None] * (1 - w)[None, :]
                B = dw[:, None] * w[None, :]
                gradients.append(K_vals[i] * (A + A.T) + K_vals[i + 1] * (B + B.T))
        return covar, gradients

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

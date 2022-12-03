"""
.. moduleauthor:: Chris Bowman <chris.bowman.physics@gmail.com>
"""

from numpy import append, diagonal, arange, diag
from numpy import sum as npsum
from numpy.linalg import cholesky, LinAlgError
from scipy.linalg import solve, solve_triangular
from scipy.optimize import minimize, differential_evolution, fmin_l_bfgs_b
from multiprocessing import Pool
from warnings import warn
from copy import copy

import matplotlib.pyplot as plt

from inference.covariance import *
from inference.mean import *
from inference.acquisition import *


class GpRegressor:
    """
    A class for performing Gaussian-process regression in one or more dimensions.

    Gaussian-process regression (GPR) is a non-parametric regression technique
    which can fit arbitrarily spaced data in any number of dimensions. A unique
    feature of GPR is its ability to account for uncertainties on the data
    (which must be assumed to be Gaussian) and propagate that uncertainty to the
    regression estimate by modelling the regression estimate itself as a multivariate
    normal distribution.

    :param x: \
        The x-data points as a 2D ``numpy.ndarray`` with shape (number of points,
        number of dimensions). Alternatively, a list of array-like objects can be
        given, which will be converted to a ``ndarray`` internally.

    :param y: \
        The y-data values as a 1D ``numpy.ndarray``.

    :param y_err: \
        The error on the y-data values supplied as a 1D ``numpy.ndarray``.
        This technique explicitly assumes that errors are Gaussian, so the supplied
        error values represent normal distribution standard deviations. If this
        argument is not specified the errors are taken to be small but non-zero.

    :param y_cov: \
        A covariance matrix representing the uncertainties on the y-data values.
        This is an alternative to the 'y_err' keyword argument, allowing the
        y-data covariance matrix to be specified directly.

    :param hyperpars: \
        An array specifying the hyper-parameter values to be used by the
        covariance function class, which by default is ``SquaredExponential``.
        See the documentation for the relevant covariance function class for
        a description of the required hyper-parameters. Generally this argument
        should be left unspecified, in which case the hyper-parameters will be
        selected automatically.

    :param class kernel: \
        The covariance function class which will be used to model the data. The
        covariance function classes can be imported from the ``gp`` module and
        then passed to ``GpRegressor`` using this keyword argument.

    :param bool cross_val: \
        If set to ``True``, leave-one-out cross-validation is used to select the
        hyper-parameters in place of the marginal likelihood.

    :param str optimizer: \
        Selects the method used to optimize the hyper-parameter values. The available
        options are "bfgs" for ``scipy.optimize.fmin_l_bfgs_b`` or "diffev" for
        ``scipy.optimize.differential_evolution``.

    :param int n_processes: \
        Sets the number of processes used in optimizing the hyper-parameter values.
        Multiple processes are only used when the optimizer keyword is set to "bfgs".

    :param int n_starts: \
        Sets the number of randomly-selected starting positions from which the BFGS
        algorithm is launched during hyper-parameter optimization. If unspecified,
        the number of starting positions is determined based on the total number
        of hyper-parameters.
    """

    def __init__(
        self,
        x: ndarray,
        y: ndarray,
        y_err: ndarray = None,
        y_cov: ndarray = None,
        hyperpars: ndarray = None,
        kernel: CovarianceFunction = SquaredExponential,
        mean: MeanFunction = ConstantMean,
        cross_val: bool = False,
        optimizer: str = "bfgs",
        n_processes: int = 1,
        n_starts: int = None,
    ):

        # store the data
        self.x = x if isinstance(x, ndarray) else array(x)
        self.y = y if isinstance(y, ndarray) else array(y)
        self.y = self.y.squeeze()

        if self.y.ndim != 1:
            raise ValueError(
                f"""\n
                [ GpRegressor error ]
                >> 'y' argument must be a 1D array, but instead has shape {self.y.shape}
                """
            )

        # determine the number of data points and spatial dimensions
        self.n_points = self.y.size
        if self.x.ndim == 2:
            self.n_dimensions = self.x.shape[1]
        elif self.x.ndim <= 1:
            self.n_dimensions = 1
            self.x = self.x.reshape([self.x.size, self.n_dimensions])
        else:
            raise ValueError(
                f"""\n
                [ GpRegressor Error ]
                >> 'x' argument must be a 2D array, but instead has
                >> {self.x.ndim} dimensions and shape {self.x.shape}.
                """
            )

        if self.x.shape[0] != self.n_points:
            raise ValueError(
                f"""\n
                [ GpRegressor Error ]
                >> The first dimension of the 'x' array must be equal in size
                >> to the 'y' array.
                >> 'x' has shape {self.x.shape}, but 'y' has size {self.y.size}.
                """
            )

        # build data errors covariance matrix
        self.sig = self.check_error_data(y_err, y_cov)

        # create an instance of the covariance function if only the class was passed
        self.cov = kernel() if isclass(kernel) else kernel

        # create an instance of the mean function if only the class was passed
        self.mean = mean() if isclass(mean) else mean

        # pass the data to the mean and covariance functions for pre-calculations
        self.cov.pass_spatial_data(self.x)
        self.mean.pass_spatial_data(self.x)
        # if bounds on the hyper-parameters were not given, estimate them
        if self.cov.bounds is None:
            self.cov.estimate_hyperpar_bounds(self.y)
        if self.mean.bounds is None:
            self.mean.estimate_hyperpar_bounds(self.y)
        # collect the bounds on all the hyper-parameters
        self.hp_bounds = copy(self.mean.bounds)
        self.hp_bounds.extend(copy(self.cov.bounds))
        # build slices to address the different parameter sets
        self.n_hyperpars = len(self.hp_bounds)
        self.mean_slice = slice(0, self.mean.n_params)
        self.cov_slice = slice(self.mean.n_params, self.n_hyperpars)
        # collect the hyper-parameter labels from the mean and covariance
        self.hyperpar_labels = [*self.mean.hyperpar_labels, *self.cov.hyperpar_labels]

        if cross_val:
            self.model_selector = self.loo_likelihood
            self.model_selector_gradient = self.loo_likelihood_gradient
        else:
            self.model_selector = self.marginal_likelihood
            self.model_selector_gradient = self.marginal_likelihood_gradient

        # if hyper-parameters are not specified, run an optimizer to select them
        if hyperpars is None:
            if optimizer not in ["bfgs", "diffev"]:
                optimizer = "bfgs"
                warn(
                    """
                    An invalid option was passed to the 'optimizer' keyword argument.
                    The default option 'bfgs' was used instead.
                    Valid options are 'bfgs' and 'diffev'.
                    """
                )

            if optimizer == "diffev":
                hyperpars = self.differential_evo()
            else:
                hyperpars = self.multistart_bfgs(
                    n_processes=n_processes, starts=n_starts
                )

        # build the covariance matrix
        self.set_hyperparameters(hyperpars)

    def __call__(self, points):
        """
        Calculate the mean and standard deviation of the regression estimate at a series
        of specified spatial points.

        :param points: \
            The points at which the mean and standard deviation of the regression
            estimate is to be calculated, given as a 2D ``numpy.ndarray`` with shape
            (number of points, number of dimensions). Alternatively, a list of array-like
            objects can be given, which will be converted to a ``ndarray`` internally.

        :return: \
            Two 1D arrays, the first containing the means and the second containing the
            standard deviations.
        """

        mu_q = []
        errs = []
        p = self.process_points(points)

        for q in p[:, None, :]:
            K_qx = self.cov(q, self.x, self.cov_hyperpars)
            K_qq = self.cov(q, q, self.cov_hyperpars)

            mu_q.append(dot(K_qx, self.alpha)[0] + self.mean(q, self.mean_hyperpars))
            v = solve_triangular(self.L, K_qx.T, lower=True)
            errs.append(K_qq[0, 0] - npsum(v**2))

        return array(mu_q), sqrt(abs(array(errs)))

    def set_hyperparameters(self, hyperpars):
        """
        Update the hyper-parameter values of the model.

        :param hyperpars: \
            An array containing the hyper-parameter values to be used.
        """
        # check to make sure the right number of hyper-parameters were given
        if len(hyperpars) != self.n_hyperpars:
            raise ValueError(
                f"""\n
                [ GpRegressor error ]
                >> An incorrect number of hyper-parameter values were passed via the 
                >> 'hyperpars' keyword argument:
                >> There are {self.n_hyperpars} hyper-parameters but {len(hyperpars)} values were given.
                """
            )

        self.hyperpars = hyperpars
        self.mean_hyperpars = self.hyperpars[self.mean_slice]
        self.cov_hyperpars = self.hyperpars[self.cov_slice]
        self.K_xx = self.cov.build_covariance(self.cov_hyperpars) + self.sig
        self.mu = self.mean.build_mean(self.mean_hyperpars)
        self.L = cholesky(self.K_xx)
        self.alpha = solve_triangular(
            self.L.T, solve_triangular(self.L, self.y - self.mu, lower=True)
        )

    def check_error_data(self, y_err, y_cov):
        if y_cov is not None:
            # if y_cov is given as a list or tuple, attempt conversion to an array
            if any([type(y_cov) is t for t in [list, tuple]]):
                y_err = array(y_cov).squeeze()
            elif type(y_cov) is not ndarray:
                # else if it isn't already an array raise an error
                raise TypeError(
                    f"""\n
                    [ GpRegressor error ]
                    >> The 'y_cov' keyword argument should be given as a numpy array:
                    >> Expected type {ndarray} but type {type(y_cov)} was given.
                    """
                )

            # now check to make sure the given error array is a valid size
            if y_cov.shape != (self.n_points, self.n_points):
                raise ValueError(
                    """\n
                    [ GpRegressor error ]
                    >> The 'y_cov' keyword argument was passed an array with an incorrect
                    >> shape. 'y_cov' must be a 2D array of shape (N,N), where 'N' is the
                    >> number of given y-data values.
                    """
                )

            # check to make sure the given matrix is symmetric
            if not (y_cov == y_cov.T).all():
                raise ValueError(
                    """\n
                    [ GpRegressor error ]
                    >> The covariance matrix passed to the 'y_cov' keyword argument
                    >> is not symmetric.
                    """
                )

            # raise a warning if both keywords have been specified
            if y_err is not None:
                warn(
                    """\n
                    [ GpRegressor warning ]
                    >> Only one of the 'y_err' and 'y_cov' keyword arguments should 
                    >> be specified. Only the input to 'y_cov' will be used - the
                    >> input to 'y_err' will be ignored.
                    """
                )

            return y_cov

        elif y_err is not None:
            # if y_err is given as a list or tuple, attempt conversion to an array
            if any([type(y_err) is t for t in [list, tuple]]):
                y_err = array(y_err).squeeze()
            elif type(y_err) is not ndarray:
                # else if it isn't already an array raise an error
                raise TypeError(
                    f"""\n
                    [ GpRegressor error ]
                    >> The 'y_err' keyword argument should be given as a numpy array:
                    >> Expected type {ndarray} but type {type(y_err)} was given.
                    """
                )

            # now check to make sure the given error array is a valid size
            if y_err.shape != (self.n_points,):
                raise ValueError(
                    """\n
                    [ GpRegressor error ]
                    >> The 'y_err' keyword argument was passed an array with an
                    >> incorrect shape. 'y_err' must be a 1D array of length 'N',
                    >> where 'N' is the number of given y-data values.
                    """
                )

            return diag(y_err**2)
        else:
            return zeros([self.n_points, self.n_points])

    def process_points(self, points):
        x = points if isinstance(points, ndarray) else array(points)

        if x.ndim <= 1 and self.n_dimensions == 1:
            x = x.reshape([x.size, 1])
        elif x.ndim == 1 and x.size == self.n_dimensions:
            x = x.reshape([1, x.size])
        elif x.ndim > 2:
            raise ValueError(
                f"""\n
                [ GpRegressor error ]
                >> 'points' argument must be a 2D array, but given array
                >> has {x.ndim} dimensions and shape {x.shape}.
                """
            )

        if x.shape[1] != self.n_dimensions:
            raise ValueError(
                f"""\n
                [ GpRegressor error ]
                >> The second dimension of the 'points' array must have size
                >> equal to the number of dimensions of the input data.
                >> The input data have {self.n_dimensions} dimensions but 'points' has shape {x.shape}.
                """
            )
        return x

    def gradient(self, points):
        """
        Calculate the mean and covariance of the gradient of the regression estimate
        with respect to the spatial coordinates at a series of specified points.

        :param points: \
            The points at which the mean vector and covariance matrix of the
            gradient of the regression estimate are to be calculated, given as a 2D
            ``numpy.ndarray`` with shape (number of points, number of dimensions).
            Alternatively, a list of array-like objects can be given, which will be
            converted to a ``ndarray`` internally.

        :return means, covariances: \
            Two arrays containing the means and covariances of each given spatial point.
            If the number of spatial dimensions ``N`` is greater than 1, then the
            covariances array is a set of 2D covariance matrices, having shape
            ``(M,N,N)`` where ``M`` is the given number of spatial points.
        """
        mu_q = []
        vars = []
        p = self.process_points(points)
        for pnt in p[:, None, :]:
            K_qx = self.cov(pnt, self.x, self.cov_hyperpars)
            A, R = self.cov.gradient_terms(pnt[0, :], self.x, self.cov_hyperpars)

            B = (K_qx * self.alpha).T
            Q = solve_triangular(self.L, (A * K_qx).T, lower=True)

            # calculate the mean and covariance
            mean = dot(A, B)
            covariance = R - Q.T.dot(Q)

            # store the results for the current point
            mu_q.append(mean)
            vars.append(covariance)
        return array(mu_q).squeeze(), array(vars).squeeze()

    def spatial_derivatives(self, points):
        """
        Calculate the spatial derivatives (i.e. the gradient) of the predictive mean
        and variance of the GP estimate. These quantities are useful in the analytic
        calculation of the spatial derivatives of acquisition functions like the
        expected improvement.

        :param points: \
            The points at which gradient of the predictive mean and variance are to be
            calculated, given as a 2D ``numpy.ndarray`` with shape (number of points,
            number of dimensions). Alternatively, a list of array-like objects can be
            given, which will be converted to a ``ndarray`` internally.

        :return mean_gradients, variance_gradients: \
            Two arrays containing the gradient vectors of the mean and variance at each
            given spatial point.
        """
        mu_gradients = []
        var_gradients = []
        p = self.process_points(points)
        for pnt in p[:, None, :]:
            K_qx = self.cov(pnt, self.x, self.cov_hyperpars)
            A, _ = self.cov.gradient_terms(pnt[0, :], self.x, self.cov_hyperpars)
            B = (K_qx * self.alpha).T
            Q = solve_triangular(self.L.T, solve_triangular(self.L, K_qx.T, lower=True))

            # calculate the mean and covariance
            dmu_dx = dot(A, B)
            dV_dx = -2 * (A * K_qx[None, :]).dot(Q)

            # store the results for the current point
            mu_gradients.append(dmu_dx)
            var_gradients.append(dV_dx)
        return array(mu_gradients).squeeze(), array(var_gradients).squeeze()

    def build_posterior(self, points):
        """
        Generates the full mean vector and covariance matrix for the Gaussian-process
        posterior distribution at a set of specified points.

        :param points: \
            The points for which the mean vector and covariance matrix are to be
            calculated, given as a 2D ``numpy.ndarray`` with shape (number of points,
            number of dimensions). Alternatively, a list of array-like objects can be
            given, which will be converted to a ``ndarray`` internally.

        :return: \
            The mean vector as a 1D array, followed by the covariance matrix as a 2D array.
        """
        v = self.process_points(points)
        K_qx = self.cov(v, self.x, self.cov_hyperpars)
        K_qq = self.cov(v, v, self.cov_hyperpars)
        mu = dot(K_qx, self.alpha) + array(
            [self.mean(p, self.mean_hyperpars) for p in v]
        )
        sigma = K_qq - dot(
            K_qx,
            solve_triangular(self.L.T, solve_triangular(self.L, K_qx.T, lower=True)),
        )
        return mu, sigma

    def loo_predictions(self):
        """
        Calculates the 'leave-one out' (LOO) predictions for the data, where each data
        point is removed from the training set and then has its value predicted using
        the remaining data.

        This implementation is based on equation (5.12) from Rasmussen & Williams.
        """
        # Use the Cholesky decomposition of the covariance to find its inverse
        iK = solve_triangular(self.L, eye(self.n_points), lower=True)
        iK = iK.T @ iK
        var = 1.0 / diag(iK)

        mu = self.y - self.alpha * var
        sigma = sqrt(var)
        return mu, sigma

    def loo_likelihood(self, theta):
        """
        Calculates the 'leave-one out' (LOO) log-likelihood.

        This implementation is based on equations (5.10, 5.11, 5.12) from
        Rasmussen & Williams.
        """
        K_xx = self.cov.build_covariance(theta[self.cov_slice]) + self.sig
        mu = self.mean.build_mean(theta[self.mean_slice])
        try:
            # Use the Cholesky decomposition of the covariance to find its inverse
            L = cholesky(K_xx)
            iK = solve_triangular(L, eye(L.shape[0]), lower=True)
            iK = iK.T @ iK
            alpha = iK.dot(self.y - mu)
            var = 1.0 / diag(iK)
            return -0.5 * (var * alpha**2 + log(var)).sum()
        except LinAlgError:
            warn("Cholesky decomposition failure in loo_likelihood")
            return -1e50

    def loo_likelihood_gradient(self, theta):
        """
        Calculates the 'leave-one out' (LOO) log-likelihood, as well as its
        gradient with respect to the hyper-parameters.

        This implementation is based on equations (5.10, 5.11, 5.12, 5.13, 5.14)
        from Rasmussen & Williams.
        """
        K_xx, grad_K = self.cov.covariance_and_gradients(theta[self.cov_slice])
        K_xx += self.sig
        mu, grad_mu = self.mean.mean_and_gradients(theta[self.mean_slice])
        # use the Cholesky decomp to get the covariance inverse
        L = cholesky(K_xx)
        iK = solve_triangular(L, eye(L.shape[0]), lower=True)
        iK = iK.T @ iK
        alpha = iK.dot(self.y - mu)
        var = 1.0 / diag(iK)
        LOO = -0.5 * (var * alpha**2 + log(var)).sum()

        cov_gradients = []
        c1 = alpha * var
        c2 = 0.5 * var * (1 + var * alpha**2)
        for dK in grad_K:
            Z = iK.dot(dK)
            g = (c1 * Z.dot(alpha) - c2 * diag(Z.dot(iK))).sum()
            cov_gradients.append(g)

        mean_gradients = []
        for dmu in grad_mu:
            Z = iK.dot(dmu)
            g = (c1 * Z).sum()
            mean_gradients.append(g)

        grad = zeros(self.n_hyperpars)
        grad[self.cov_slice] = array(cov_gradients)
        grad[self.mean_slice] = array(mean_gradients)

        return LOO, grad

    def marginal_likelihood(self, theta):
        """
        returns the log-marginal likelihood for the supplied hyper-parameter values.

        This implementation is based on equation (5.8) from Rasmussen & Williams.
        """
        K_xx = self.cov.build_covariance(theta[self.cov_slice]) + self.sig
        mu = self.mean.build_mean(theta[self.mean_slice])
        try:  # protection against singular matrix error crash
            L = cholesky(K_xx)
            v = solve_triangular(L, self.y - mu, lower=True)
            return -0.5 * (v @ v) - log(diagonal(L)).sum()
        except LinAlgError:
            warn("Cholesky decomposition failure in marginal_likelihood")
            return -1e50

    def marginal_likelihood_gradient(self, theta):
        """
        returns the log-marginal likelihood and its gradient with respect
        to the hyperparameters.

        This implementation is based on equations (5.8, 5.9) from Rasmussen & Williams.
        """
        K_xx, grad_K = self.cov.covariance_and_gradients(theta[self.cov_slice])
        K_xx += self.sig
        mu, grad_mu = self.mean.mean_and_gradients(theta[self.mean_slice])
        # get the cholesky decomposition
        L = cholesky(K_xx)
        iK = solve_triangular(L, eye(L.shape[0]), lower=True)
        iK = iK.T @ iK
        # calculate the log-marginal likelihood
        alpha = iK.dot(self.y - mu)
        LML = -0.5 * dot((self.y - mu).T, alpha) - log(diagonal(L)).sum()
        # calculate the mean parameter gradients
        grad = zeros(self.n_hyperpars)
        grad[self.mean_slice] = array([(alpha * dmu).sum() for dmu in grad_mu])
        # calculate the covariance parameter gradients
        Q = alpha[:, None] * alpha[None, :] - iK
        grad[self.cov_slice] = array([0.5 * (Q * dK.T).sum() for dK in grad_K])
        return LML, grad

    def differential_evo(self):
        # optimise the hyper-parameters
        opt_result = differential_evolution(
            func=lambda x: -self.model_selector(x), bounds=self.hp_bounds
        )
        return opt_result.x

    def bfgs_cost_func(self, theta):
        y, grad_y = self.model_selector_gradient(theta)
        return -y, -grad_y

    def launch_bfgs(self, x0):
        return fmin_l_bfgs_b(
            func=self.bfgs_cost_func, x0=x0, approx_grad=False, bounds=self.hp_bounds
        )

    def multistart_bfgs(self, starts: int = None, n_processes: int = 1):
        if starts is None:
            starts = int(2 * sqrt(len(self.hp_bounds))) + 1
        # starting positions guesses by random sampling + one in the centre of the hypercube
        lwr, upr = [array([k[i] for k in self.hp_bounds]) for i in [0, 1]]
        starting_positions = [
            lwr + (upr - lwr) * random(size=len(self.hp_bounds))
            for _ in range(starts - 1)
        ]
        starting_positions.append(0.5 * (lwr + upr))

        # run BFGS for each starting position
        if n_processes == 1:
            results = [self.launch_bfgs(x0) for x0 in starting_positions]
        else:
            workers = Pool(n_processes)
            results = workers.map(self.launch_bfgs, starting_positions)

        # extract best solution
        solution = sorted(results, key=lambda x: x[1])[0][0]
        return solution

    def __str__(self):
        L_max = max([len(lb) for lb in self.hyperpar_labels])
        strings = ["\n>>> [ GpRegressor hyperparameters ]\n"]
        for lb, val in zip(self.hyperpar_labels, self.hyperpars):
            strings.append(" " * (L_max - len(lb)) + f"{lb} = {val:.4}\n")
        return "".join(strings)


class MarginalisedGpRegressor:
    def __init__(
        self,
        x,
        y,
        y_err=None,
        hyperparameter_samples=None,
        kernel=SquaredExponential,
        cross_val=False,
    ):
        self.gps = [
            GpRegressor(
                x, y, y_err=y_err, kernel=kernel, cross_val=cross_val, hyperpars=theta
            )
            for theta in hyperparameter_samples
        ]
        self.n = len(self.gps)

    def __call__(self, points):
        results = [gp(points) for gp in self.gps]
        means, sigmas = [array([v[i] for v in results]) for i in [0, 1]]
        return means.mean(axis=0), sigmas.mean(axis=0)

    def spatial_derivatives(self, points):
        results = [gp.spatial_derivatives(points) for gp in self.gps]
        grad_mu, grad_var = [array([v[i] for v in results]) for i in [0, 1]]
        return grad_mu.mean(axis=0), grad_var.mean(axis=0)

    def gradient(self, points):
        results = [gp.gradient(points) for gp in self.gps]
        grad_mu, grad_var = [array([v[i] for v in results]) for i in [0, 1]]
        return grad_mu.mean(axis=0), grad_var.mean(axis=0)


class GpOptimiser:
    """
    A class for performing Gaussian-process optimisation in one or more dimensions.

    GpOptimiser extends the functionality of GpRegressor to perform Gaussian-process
    optimisation, often also referred to as 'Bayesian optimisation'. This technique
    is suited to problems for which a single evaluation of the function being explored
    is expensive, such that the total number of function evaluations must be made as
    small as possible.

    In order to construct the Gaussian-process regression estimate which is used to
    search for the global maximum, on initialisation GpOptimiser must be provided with
    at least two evaluations of the function which is to be maximised.

    :param x: \
        The x-data points as a 2D ``numpy.ndarray`` with shape (number of points,
        number of dimensions). Alternatively, a list of array-like objects can be
        given, which will be converted to a ``ndarray`` internally.

    :param y: \
        The y-data values as a 1D ``numpy.ndarray``.

    :param bounds: \
        An iterable containing tuples which specify the upper and lower bounds
        for the optimisation in each dimension in the format (lower_bound, upper_bound).

    :param y_err: \
        The error on the y-data values supplied as a 1D array.
        This technique explicitly assumes that errors are Gaussian, so the supplied
        error values represent normal distribution standard deviations. If this
        argument is not specified the errors are taken to be small but non-zero.

    :param hyperpars: \
        An array specifying the hyper-parameter values to be used by the
        covariance function class, which by default is ``SquaredExponential``.
        See the documentation for the relevant covariance function class for
        a description of the required hyper-parameters. Generally this argument
        should be left unspecified, in which case the hyper-parameters will be
        selected automatically.

    :param class kernel: \
        The covariance-function class which will be used to model the data. The
        covariance-function classes can be imported from the ``gp`` module and
        then passed to ``GpOptimiser`` using this keyword argument.

    :param class mean: \
        The mean-function class which will be used to model the data. The
        mean-function classes can be imported from the ``gp`` module and
        then passed to ``GpOptimiser`` using this keyword argument.

    :param bool cross_val: \
        If set to ``True``, leave-one-out cross-validation is used to select the
        hyper-parameters in place of the marginal likelihood.

    :param class acquisition: \
        The acquisition-function class which is used to select new points at which
        the objective function is evaluated. The acquisition-function classes can be
        imported from the ``gp`` module and then passed as arguments - see their
        documentation for the list of available acquisition functions. If left unspecified,
        the ``ExpectedImprovement`` acquisition function is used by default.

    :param str optimizer: \
        Selects the optimisation method used for selecting hyper-parameter values and proposed
        evaluations. Available options are "bfgs" for ``scipy.optimize.fmin_l_bfgs_b`` or
        "diffev" for ``scipy.optimize.differential_evolution``.

    :param int n_processes: \
        Sets the number of processes used when selecting hyper-parameters or proposed evaluations.
        Multiple processes are only used when the optimizer keyword is set to "bfgs".
    """

    def __init__(
        self,
        x: ndarray,
        y: ndarray,
        bounds: Sequence,
        y_err: ndarray = None,
        hyperpars: ndarray = None,
        kernel: CovarianceFunction = SquaredExponential,
        mean: MeanFunction = ConstantMean,
        cross_val: bool = False,
        acquisition=ExpectedImprovement,
        optimizer: str = "bfgs",
        n_processes: int = 1,
    ):
        self.x = x if isinstance(x, ndarray) else array(x)
        if self.x.ndim == 1:
            self.x.resize([self.x.size, 1])
        self.y = y if isinstance(y, ndarray) else array(y)
        self.y_err = y_err if isinstance(y_err, (ndarray, type(None))) else array(y_err)

        self.bounds = bounds
        self.kernel = kernel
        self.mean = mean
        self.cross_val = cross_val
        self.n_processes = n_processes
        self.optimizer = optimizer

        self.gp = GpRegressor(
            x=x,
            y=y,
            y_err=y_err,
            hyperpars=hyperpars,
            kernel=kernel,
            mean=mean,
            cross_val=cross_val,
            optimizer=self.optimizer,
            n_processes=self.n_processes,
        )

        # if the class has been passed instead of an instance, create an instance
        self.acquisition = acquisition() if isclass(acquisition) else acquisition
        self.acquisition.update_gp(self.gp)

        # create storage for tracking
        self.acquisition_max_history = []
        self.convergence_metric_history = []
        self.iteration_history = []

    def __call__(self, x):
        return self.gp(x)

    def add_evaluation(self, new_x: ndarray, new_y: ndarray, new_y_err: ndarray = None):
        """
        Add the latest evaluation to the data set and re-build the
        Gaussian process so a new proposed evaluation can be made.

        :param new_x: location of the new evaluation
        :param new_y: function value of the new evaluation
        :param new_y_err: Error of the new evaluation.
        """
        new_x = new_x if isinstance(new_x, ndarray) else array(new_x)
        if new_x.shape != (1, self.x.shape[1]):
            new_x.resize((1, self.x.shape[1]))
        new_y = new_y if isinstance(new_y, ndarray) else array(new_y)
        good_type = isinstance(new_y_err, (ndarray, type(None)))
        new_y_err = new_y_err if good_type else array(new_y_err)

        # store the acquisition function value of the new point
        self.acquisition_max_history.append(self.acquisition(new_x))
        self.convergence_metric_history.append(
            self.acquisition.convergence_metric(new_x)
        )
        self.iteration_history.append(self.y.size + 1)

        # update the data arrays
        self.x = append(self.x, new_x, axis=0)
        self.y = append(self.y, new_y)

        if self.y_err is not None:
            if new_y_err is not None:
                self.y_err = append(self.y_err, new_y_err)
            else:
                raise ValueError(
                    "y_err must be specified for new evaluations if y_err was specified during __init__"
                )

        # re-train the GP
        self.gp = GpRegressor(
            x=self.x,
            y=self.y,
            y_err=self.y_err,
            kernel=self.kernel,
            mean=self.mean,
            cross_val=self.cross_val,
            optimizer=self.optimizer,
            n_processes=self.n_processes,
        )
        self.mu_max = self.y.max()

        # update the acquisition function info
        self.acquisition.update_gp(self.gp)

    def diff_evo(self):
        opt_result = differential_evolution(
            self.acquisition.opt_func, self.bounds, popsize=30
        )
        solution = opt_result.x
        funcval = opt_result.fun
        if hasattr(funcval, "__len__"):
            funcval = funcval[0]
        return solution, funcval

    def launch_bfgs(self, x0):
        return fmin_l_bfgs_b(
            self.acquisition.opt_func_gradient,
            x0,
            approx_grad=False,
            bounds=self.bounds,
            pgtol=1e-10,
        )

    def multistart_bfgs(self):
        starting_positions = self.acquisition.starting_positions(self.bounds)
        # run BFGS for each starting position
        if self.n_processes == 1:
            results = [self.launch_bfgs(x0) for x0 in starting_positions]
        else:
            workers = Pool(self.n_processes)
            results = workers.map(self.launch_bfgs, starting_positions)
        # extract best solution
        best_result = sorted(results, key=lambda x: float(x[1]))[0]
        solution = best_result[0]
        funcval = float(best_result[1])
        return solution, funcval

    def propose_evaluation(self, optimizer=None):
        """
        Request a proposed location for the next evaluation. This proposal is
        selected by maximising the chosen acquisition function.

        :param str optimizer: \
            Selects the optimization method used for selecting the proposed evaluation.
            Available options are "bfgs" for ``scipy.optimize.fmin_l_bfgs_b`` or
            "diffev" for ``scipy.optimize.differential_evolution``. This keyword allows
            the user to override the choice of optimizer given when ``GpOptimiser`` was
            initialised.

        :return: location of the next proposed evaluation.
        """
        opt = optimizer if optimizer is not None else self.optimizer
        if opt == "bfgs":
            # find the evaluation point which maximises the acquisition function
            proposed_ev, max_acq = self.multistart_bfgs()
        else:
            proposed_ev, max_acq = self.diff_evo()
        # if the problem is 1D, but the result is returned as a length-1 array,
        # extract the result from the array
        if hasattr(proposed_ev, "__len__") and len(proposed_ev) == 1:
            proposed_ev = proposed_ev[0]
        return proposed_ev

    def plot_results(self, filename=None, show_plot=True):
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121)
        maxvals = maximum.accumulate(self.y)
        pad = maxvals.ptp() * 0.1
        iterations = arange(len(self.y)) + 1
        ax1.plot(iterations, maxvals, c="red", alpha=0.6, label="max observed value")
        ax1.plot(iterations, self.y, ".", label="function evaluations", markersize=10)
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("function value")
        ax1.set_ylim([maxvals.min() - pad, maxvals.max() + pad])
        ax1.legend(loc=4)
        ax1.grid()

        ax2 = fig.add_subplot(122)
        ax2.plot(
            self.iteration_history, self.convergence_metric_history, c="C0", alpha=0.35
        )
        ax2.plot(
            self.iteration_history,
            self.convergence_metric_history,
            ".",
            c="C0",
            label=self.acquisition.convergence_description,
            markersize=10,
        )
        ax2.set_yscale("log")
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("acquisition function value")
        ax2.set_xlim([0, None])
        ax2.set_title("Convergence summary")
        ax2.legend()
        ax2.grid()

        fig.tight_layout()

        if filename is not None:
            plt.savefig(filename)
        if show_plot:
            plt.show()
        else:
            plt.close()


class GpLinearInverter:
    """
    A class for performing Gaussian-process linear inversion.

    In the case where both the likelihood and prior distributions are multivariate
    normal, and the forward-model is linear, the posterior distribution is also
    multivariate normal, and the posterior mean and covariance can be calculated
    directly from the likelihood and prior mean and covariance.

    If each of the model parameters can be associated with a position in some
    space (which is often the case in tomography and deconvolution problems)
    and we expect their values to be correlated within that space, we can
    model this behavior using a gaussian-process prior distribution.

    :param y: \
        The y-data values as a 1D ``numpy.ndarray``.

    :param y_err: \
        The error on the y-data values supplied as a 1D ``numpy.ndarray``.
        These values are used to construct the likelihood covariance, which
        is assumed to be diagonal.

    :param model_matrix: \
        The linear forward-model as a 2D ``numpy.ndarray``. The product of
        this model matrix with a vector of model parameter values should
        yield a prediction of the y-data values.

    :param parameter_spatial_positions: \
        A 2D ``numpy.ndarray`` specifying the 'positions' of the model parameters
        in some space over which their values are expected to be correlated.

    :param class prior_covariance_function: \
        A covariance function class which will be used to generate the prior
        covariance. Covariance function classes can be found in the ``gp`` module,
        or custom covariance functions can be written using the ``CovarianceFunction``
        abstract base class.

    :param class prior_mean_function: \
        A mean function class which will be used to generate the prior
        mean. Mean function classes can be found in the ``gp`` module, or custom
        mean functions can be written using the ``MeanFunction`` abstract base class.
    """

    def __init__(
        self,
        y: ndarray,
        y_err: ndarray,
        model_matrix: ndarray,
        parameter_spatial_positions: ndarray,
        prior_covariance_function: CovarianceFunction = SquaredExponential,
        prior_mean_function: MeanFunction = ConstantMean,
    ):
        if model_matrix.ndim != 2:
            raise ValueError(
                """
                [ GpLinearInverter error ]
                >> 'model_matrix' argument must be a 2D numpy.ndarray
                """
            )

        if y.ndim != y_err.ndim != 1 or y.size != y_err.size:
            raise ValueError(
                """
                [ GpLinearInverter error ]
                >> 'y' and 'y_err' arguments must be 1D numpy.ndarray
                >> of equal size.
                """
            )

        if model_matrix.shape[0] != y.size:
            raise ValueError(
                f"""
                [ GpLinearInverter error ]
                >> The size of the first dimension of 'model_matrix' must
                >> equal the size of 'y', however they have shapes
                >> {model_matrix.shape}, {y.shape}
                >> respectively.
                """
            )

        if parameter_spatial_positions.ndim != 2:
            raise ValueError(
                """
                [ GpLinearInverter error ]
                >> 'parameter_spatial_positions' must be a 2D numpy.ndarray, with the
                >> size of first dimension being equal to the number of model parameters
                >> and the size of the second dimension being equal to the number of
                >> spatial dimensions.
                """
            )

        if model_matrix.shape[1] != parameter_spatial_positions.shape[0]:
            raise ValueError(
                f"""
                [ GpLinearInverter error ]
                >> The size of the second dimension of 'model_matrix' must be equal
                >> to the size of the first dimension of 'parameter_spatial_positions',
                >> however they have shapes
                >> {model_matrix.shape}, {parameter_spatial_positions.shape}
                >> respectively.
                """
            )

        self.A = model_matrix
        self.y = y

        self.cov = prior_covariance_function
        self.cov = self.cov() if isclass(self.cov) else self.cov
        self.cov.pass_spatial_data(parameter_spatial_positions)

        self.mean = prior_mean_function
        self.mean = self.mean() if isclass(self.mean) else self.mean
        self.mean.pass_spatial_data(parameter_spatial_positions)

        self.n_hyperpars = self.mean.n_params + self.cov.n_params
        self.mean_slice = slice(0, self.mean.n_params)
        self.cov_slice = slice(self.mean.n_params, self.n_hyperpars)
        self.hyperpar_labels = [*self.mean.hyperpar_labels, *self.cov.hyperpar_labels]

        self.sigma = diag(y_err**2)
        self.inv_sigma = diag(y_err**-2)
        self.I = eye(self.A.shape[1])

    def calculate_posterior(self, theta: ndarray):
        """
        Calculate the posterior mean and covariance for the given
        hyper-parameter values.

        :param theta: \
            The hyper-parameter values as a 1D ``numpy.ndarray``.

        :return: \
            The posterior mean and covariance.
        """
        K = self.cov.build_covariance(theta[self.cov_slice])
        prior_mean = self.mean.build_mean(theta[self.mean_slice])
        W = self.A.T @ self.inv_sigma @ self.A
        u = self.A.T @ (self.inv_sigma @ (self.y - self.A @ prior_mean))
        posterior_cov = solve(self.I + K @ W, K)
        posterior_mean = posterior_cov @ u + prior_mean
        return posterior_mean, posterior_cov

    def calculate_posterior_mean(self, theta: ndarray):
        """
        Calculate the posterior mean for the given
        hyper-parameter values.

        :param theta: \
            The hyper-parameter values as a 1D ``numpy.ndarray``.

        :return: \
            The posterior mean and covariance.
        """
        K = self.cov.build_covariance(theta[self.cov_slice])
        prior_mean = self.mean.build_mean(theta[self.mean_slice])
        u = self.A.T @ (self.inv_sigma @ (self.y - self.A @ prior_mean))
        W = self.A.T @ self.inv_sigma @ self.A
        return solve(self.I + K @ W, K @ u) + prior_mean

    def marginal_likelihood(self, theta: ndarray):
        """
        Calculate the log-marginal likelihood for the given hyper-parameter values.

        :param theta: \
            The hyper-parameter values as a 1D ``numpy.ndarray``.

        :return: \
            The log-marginal likelihood value.
        """
        K = self.cov.build_covariance(theta[self.cov_slice])
        prior_mean = self.mean.build_mean(theta[self.mean_slice])
        L = cholesky(self.A @ K @ self.A.T + self.sigma)
        v = solve_triangular(L, self.y - self.A @ prior_mean, lower=True)
        return -0.5 * (v @ v) - log(diagonal(L)).sum()

    def optimize_hyperparameters(self, initial_guess: ndarray) -> ndarray:
        """
        Finds the hyper-parameter values which maximise the
        marginal-likelihood.

        :param initial_guess: \
            An initial guess for the hyper-parameter values as
            a 1D ``numpy.ndarray``.

        :return: \
            The hyper-parameters which maximise the marginal-likelihood
            as a 1D ``numpy.ndarray``.
        """
        if initial_guess.size != self.n_hyperpars:
            raise ValueError(
                f"""
                [ GpLinearInverter error ]
                >> There are a total of {self.n_hyperpars} hyper-parameters,
                >> but {initial_guess.size} values were given in 'initial_guess'.
                """
            )
        OptResult = minimize(
            fun=lambda t: -self.marginal_likelihood(t),
            x0=initial_guess,
            method="Nelder-Mead",
        )
        return OptResult.x

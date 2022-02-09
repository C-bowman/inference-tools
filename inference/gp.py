"""
.. moduleauthor:: Chris Bowman <chris.bowman.physics@gmail.com>
"""

from numpy import diagonal, arange, diag
from numpy import sum as npsum
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.optimize import differential_evolution, fmin_l_bfgs_b
from multiprocessing import Pool
from warnings import warn
from copy import copy
from inspect import isclass

import matplotlib.pyplot as plt

from inference.covariance import *
from inference.mean import *
from inference.acquisition import *


class GpRegressor(object):
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
        The y-data values as a 1D array.

    :param y_err: \
        The error on the y-data values supplied as a 1D array.
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
    """

    def __init__(
        self,
        x,
        y,
        y_err=None,
        y_cov=None,
        hyperpars=None,
        kernel=SquaredExponential,
        mean=ConstantMean,
        cross_val=False,
        optimizer="bfgs",
        n_processes=1,
    ):

        # store the data
        self.x = array(x)
        self.y = array(y).squeeze()

        # determine the number of data points and spatial dimensions
        self.N_points = self.y.shape[0]
        if len(self.x.shape) == 1:
            self.N_dimensions = 1
            self.x = self.x.reshape([self.x.shape[0], self.N_dimensions])
        else:
            self.N_dimensions = self.x.shape[1]

        if self.x.shape[0] != self.N_points:
            raise ValueError(
                "The given number of x-data points does not match the number of y-data values"
            )

        # build data errors covariance matrix
        self.sig = self.check_error_data(y_err, y_cov)

        # create an instance of the covariance function if only the class was passed
        self.cov = kernel() if isclass(kernel) else kernel

        # create an instance of the mean function if only the class was passed
        self.mean = mean() if isclass(mean) else mean

        # pass the data to the mean and covariance functions for pre-calculations
        self.cov.pass_data(self.x, self.y)
        self.mean.pass_data(self.x, self.y)
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
                hyperpars = self.multistart_bfgs(n_processes=n_processes)

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
                f"""
                [ GpRegressor error ]
                An incorrect number of hyper-parameters were passed via the 'hyperpars' keyword argument:
                There are {self.n_hyperpars} hyper-parameters but {len(hyperpars)} were given.
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
                    f"""
                    [ GpRegressor error ]
                    The 'y_cov' keyword argument should be given as a numpy array:
                    Expected type {ndarray} but type {type(y_cov)} was given.
                    """
                )

            # now check to make sure the given error array is a valid size
            if y_cov.shape != (self.N_points, self.N_points):
                raise ValueError(
                    """
                    [ GpRegressor error ]
                    The 'y_cov' keyword argument was passed an array with an incorrect shape.
                    'y_cov' must be a 2D array of shape (N,N), where 'N' is the number of given 
                    y-data values.
                    """
                )

            # check to make sure the given matrix is symmetric
            if not (y_cov == y_cov.T).all():
                raise ValueError(
                    """
                    [ GpRegressor error ]
                    The covariance matrix passed to the 'y_cov' keyword argument is not symmetric.
                    """
                )

            # raise a warning if both keywords have been specified
            if y_err is not None:
                warn(
                    """
                    [ GpRegressor warning ]
                    Only one of the 'y_err' and 'y_cov' keyword arguments should be specified.
                    Only the input to 'y_cov' will be used, the input to 'y_err' will be ignored.
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
                    f"""
                    [ GpRegressor error ]
                    The 'y_err' keyword argument should be given as a numpy array:
                    Expected type {ndarray} but type {type(y_err)} was given.
                    """
                )

            # now check to make sure the given error array is a valid size
            if y_err.shape != (self.N_points,):
                raise ValueError(
                    """
                    [ GpRegressor error ]
                    The 'y_err' keyword argument was passed an array with an incorrect shape.
                    'y_err' must be a 1D array of length N, where 'N' is the number of given 
                    y-data values.
                    """
                )

            return diag(y_err**2)
        else:
            return zeros([self.N_points, self.N_points])

    def process_points(self, points):
        if type(points) is ndarray:
            x = points
        else:
            x = array(points)

        m = len(x.shape)
        if self.N_dimensions == 1:
            if m == 0:  # the case when given a float
                x = x.reshape([1, 1])
            elif m == 1:
                x = x.reshape([x.shape[0], 1])
            elif m == 2 and x.shape[1] != 1:
                raise ValueError(
                    "given spatial points have an incorrect number of dimensions"
                )
        else:
            if m == 0:
                raise ValueError(
                    "given spatial points have an incorrect number of dimensions"
                )
            elif m == 1 and x.shape[0] == self.N_dimensions:
                x = x.reshape([1, self.N_dimensions])
            elif m == 2 and x.shape[1] != self.N_dimensions:
                raise ValueError(
                    "given spatial points have an incorrect number of dimensions"
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
        for q in p[:, None, :]:
            K_qx = self.cov(q, self.x, self.cov_hyperpars)
            A, R = self.cov.gradient_terms(q[0, :], self.x, self.cov_hyperpars)

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
        for q in p[:, None, :]:
            K_qx = self.cov(q, self.x, self.cov_hyperpars)
            A, _ = self.cov.gradient_terms(q[0, :], self.x, self.cov_hyperpars)
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
        I = eye(len(self.x))
        iK = solve_triangular(self.L, eye(self.L.shape[0]), lower=True)
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
        try:
            K_xx = self.cov.build_covariance(theta[self.cov_slice]) + self.sig
            mu = self.mean.build_mean(theta[self.mean_slice])

            # Use the Cholesky decomposition of the covariance to find its inverse
            L = cholesky(K_xx)
            iK = solve_triangular(L, eye(L.shape[0]), lower=True)
            iK = iK.T @ iK
            alpha = iK.dot(self.y - mu)
            var = 1.0 / diag(iK)
            return -0.5 * (var * alpha**2 + log(var)).sum()
        except:
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
        # use the cholesky decomp to get the covariance inverse
        L = cholesky(K_xx)
        iK = solve_triangular(L, eye(L.shape[0]), lower=True)
        iK = iK.T @ iK
        # Use the Cholesky decomposition of the covariance to find its inverse
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
            alpha = solve_triangular(L.T, solve_triangular(L, self.y - mu, lower=True))
            return -0.5 * dot((self.y - mu).T, alpha) - log(diagonal(L)).sum()
        except:
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
            lambda x: -self.model_selector(x), self.hp_bounds
        )
        return opt_result.x

    def bfgs_cost_func(self, theta):
        y, grad_y = self.model_selector_gradient(theta)
        return -y, -grad_y

    def launch_bfgs(self, x0):
        return fmin_l_bfgs_b(
            self.bfgs_cost_func, x0, approx_grad=False, bounds=self.hp_bounds
        )

    def multistart_bfgs(self, starts=None, n_processes=1):
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


class MarginalisedGpRegressor(object):
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


class GpOptimiser(object):
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

    :param list x: \
        The spatial coordinates of the y-data values. For the 1-dimensional case,
        this should be a list or array of floats. For greater than 1 dimension,
        a list of coordinate arrays or tuples should be given.

    :param list y: \
        The y-data values as a list or array of floats.

    :keyword bounds: \
        An iterable containing tuples which specify the upper and lower bounds
        for the optimisation in each dimension in the format (lower_bound, upper_bound).

    :keyword y_err: \
        The error on the y-data values supplied as a list or array of floats.
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
        Selects the optimization method used for selecting hyper-parameter values and proposed
        evaluations. Available options are "bfgs" for ``scipy.optimize.fmin_l_bfgs_b`` or
        "diffev" for ``scipy.optimize.differential_evolution``.

    :param int n_processes: \
        Sets the number of processes used when selecting hyper-parameters or proposed evaluations.
        Multiple processes are only used when the optimizer keyword is set to "bfgs".
    """

    def __init__(
        self,
        x,
        y,
        bounds,
        y_err=None,
        hyperpars=None,
        kernel=SquaredExponential,
        mean=ConstantMean,
        cross_val=False,
        acquisition=ExpectedImprovement,
        optimizer="bfgs",
        n_processes=1,
    ):
        self.x = list(x)
        self.y = list(y)
        self.y_err = y_err

        if y_err is not None:
            self.y_err = list(self.y_err)

        self.bounds = bounds
        self.kernel = kernel
        self.mean = mean
        self.cross_val = cross_val
        self.n_processes = n_processes
        self.optimizer = optimizer

        self.gp = GpRegressor(
            x,
            y,
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

    def add_evaluation(self, new_x, new_y, new_y_err=None):
        """
        Add the latest evaluation to the data set and re-build the
        Gaussian process so a new proposed evaluation can be made.

        :param new_x: location of the new evaluation
        :param new_y: function value of the new evaluation
        :param new_y_err: Error of the new evaluation.
        """
        # store the acquisition function value of the new point
        self.acquisition_max_history.append(self.acquisition(new_x))
        self.convergence_metric_history.append(
            self.acquisition.convergence_metric(new_x)
        )
        self.iteration_history.append(len(self.y) + 1)

        # update the data arrays
        self.x.append(new_x)
        self.y.append(new_y)
        if self.y_err is not None:
            if new_y_err is not None:
                self.y_err.append(new_y_err)
            else:
                raise ValueError(
                    "y_err must be specified for new evaluations if y_err was specified during __init__"
                )

        # re-train the GP
        self.gp = GpRegressor(
            self.x,
            self.y,
            y_err=self.y_err,
            kernel=self.kernel,
            mean=self.mean,
            cross_val=self.cross_val,
            optimizer=self.optimizer,
            n_processes=self.n_processes,
        )
        self.mu_max = max(self.y)

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

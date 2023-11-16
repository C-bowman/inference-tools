from numpy import array, ndarray, diagonal, diag, dot, eye, log, zeros
from numpy.linalg import cholesky
from scipy.linalg import solve, solve_triangular
from scipy.optimize import minimize
from inspect import isclass

from inference.gp.covariance import CovarianceFunction, SquaredExponential
from inference.gp.mean import MeanFunction, ConstantMean


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
                """\n
                [ GpLinearInverter error ]
                >> 'model_matrix' argument must be a 2D numpy.ndarray
                """
            )

        if y.ndim != y_err.ndim != 1 or y.size != y_err.size:
            raise ValueError(
                """\n
                [ GpLinearInverter error ]
                >> 'y' and 'y_err' arguments must be 1D numpy.ndarray
                >> of equal size.
                """
            )

        if model_matrix.shape[0] != y.size:
            raise ValueError(
                f"""\n
                [ GpLinearInverter error ]
                >> The size of the first dimension of 'model_matrix' must
                >> equal the size of 'y', however they have shapes
                >> {model_matrix.shape}, {y.shape}
                >> respectively.
                """
            )

        if parameter_spatial_positions.ndim != 2:
            raise ValueError(
                """\n
                [ GpLinearInverter error ]
                >> 'parameter_spatial_positions' must be a 2D numpy.ndarray, with the
                >> size of first dimension being equal to the number of model parameters
                >> and the size of the second dimension being equal to the number of
                >> spatial dimensions.
                """
            )

        if model_matrix.shape[1] != parameter_spatial_positions.shape[0]:
            raise ValueError(
                f"""\n
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
        if self.cov.bounds is None:
            self.cov.bounds = [(None, None)] * self.cov.n_params

        self.mean = prior_mean_function
        self.mean = self.mean() if isclass(self.mean) else self.mean
        self.mean.pass_spatial_data(parameter_spatial_positions)
        if self.mean.bounds is None:
            self.mean.bounds = [(None, None)] * self.mean.n_params

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

    def calculate_posterior_mean(self, theta: ndarray) -> ndarray:
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

    def marginal_likelihood(self, theta: ndarray) -> float:
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

    def marginal_likelihood_gradient(self, theta: ndarray):
        """
        returns the log-marginal likelihood and its gradient with respect
        to the hyperparameters.
        """
        K, grad_K = self.cov.covariance_and_gradients(theta[self.cov_slice])
        J = self.A @ K @ self.A.T + self.sigma
        grad_J = [self.A @ dK @ self.A.T for dK in grad_K]

        mu, grad_mu = self.mean.mean_and_gradients(theta[self.mean_slice])
        f = self.A @ mu
        grad_f = [self.A @ du for du in grad_mu]

        # get the cholesky decomposition
        L = cholesky(J)
        # find inv(J) using inv(L)
        iJ = solve_triangular(L, eye(L.shape[0]), lower=True)
        iJ = iJ.T @ iJ
        # calculate the log-marginal likelihood
        alpha = iJ.dot(self.y - f)
        LML = -0.5 * dot((self.y - f).T, alpha) - log(diagonal(L)).sum()
        # calculate the mean parameter gradients
        grad = zeros(self.n_hyperpars)
        grad[self.mean_slice] = array([(alpha * df).sum() for df in grad_f])
        # calculate the covariance parameter gradients
        Q = alpha[:, None] * alpha[None, :] - iJ
        grad[self.cov_slice] = array([0.5 * (Q * dJ.T).sum() for dJ in grad_J])
        return LML, grad

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
                f"""\n
                [ GpLinearInverter error ]
                >> There are a total of {self.n_hyperpars} hyper-parameters,
                >> but {initial_guess.size} values were given in 'initial_guess'.
                """
            )

        hp_bounds = [*self.mean.bounds, *self.cov.bounds]

        OptResult = minimize(
            fun=lambda t: -self.marginal_likelihood(t),
            x0=initial_guess,
            method="Nelder-Mead",
            bounds=hp_bounds,
        )
        return OptResult.x

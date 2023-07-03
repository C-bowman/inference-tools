from inspect import isclass
from copy import copy
from numpy import exp, log
from numpy import array, eye, ndarray, tril, diag_indices_from, diagonal, zeros
from numpy.random import uniform, normal, random, multivariate_normal
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.optimize import fmin_l_bfgs_b

from inference.likelihoods import Likelihood
from inference.gp.covariance import CovarianceFunction, SquaredExponential
from inference.gp.mean import MeanFunction, ConstantMean


class LatentGPR:
    def __init__(
        self,
        x: ndarray,
        y: ndarray,
        likelihood: Likelihood,
        kernel: CovarianceFunction = SquaredExponential,
        mean: MeanFunction = ConstantMean,
    ):
        self.x = x
        self.y = y
        self.likelihood = likelihood
        self.n_points = self.x.shape[0]  # TODO - needs proper processing

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
        self.hp_bounds = [(None, None) for _ in range(self.y.size)]
        self.hp_bounds.extend(copy(self.mean.bounds))
        self.hp_bounds.extend(copy(self.cov.bounds))
        # build slices to address the different parameter sets
        self.n_hyperpars = len(self.hp_bounds)
        self.points_slc = slice(0, self.y.size)
        self.mean_slc = slice(self.y.size, self.y.size + self.mean.n_params)
        self.cov_slc = slice(self.y.size + self.mean.n_params, self.n_hyperpars)

    def log_posterior(self, theta):
        K_xx = self.cov.build_covariance(theta[self.cov_slc])
        mu = self.mean.build_mean(theta[self.mean_slc])
        # get the cholesky decomposition
        L = cholesky(K_xx)
        # get the point values
        v = theta[self.points_slc]
        f = mu + L @ v
        # calculate the log-marginal likelihood
        posterior = self.likelihood(f) - 0.5 * (v.T @ v) - log(diagonal(L)).sum()
        return posterior

    def log_posterior_and_gradient(self, theta):
        K_xx, grad_K = self.cov.covariance_and_gradients(theta[self.cov_slc])
        mu, grad_mu = self.mean.mean_and_gradients(theta[self.mean_slc])
        # get the cholesky decomposition
        L = cholesky(K_xx)
        # get the point values
        v = theta[self.points_slc]
        f = mu + L @ v
        # calculate the log-marginal likelihood
        posterior = self.likelihood(f) - 0.5 * (v.T @ v) - log(diagonal(L)).sum()
        # get the derivative of the likelihood w.r.t. the point values
        dL_df = self.likelihood.gradient(f)
        # calculate the mean parameter gradients
        grad = zeros(self.n_hyperpars)
        grad[self.points_slc] = (L.T @ dL_df) - v
        grad[self.mean_slc] = array([(dL_df * dmu).sum() for dmu in grad_mu])
        # calculate the covariance parameter gradients
        iL = solve_triangular(L, eye(L.shape[0]), lower=True)
        iK = iL.T @ iL

        grad[self.cov_slc] = array(
            [
                dL_df.dot(cholesky_derivative(L, iL, dK) @ v) - 0.5 * (iK * dK).sum()
                for dK in grad_K
            ]
        )
        return posterior, grad

    def cost_and_gradient(self, theta):
        p, dp = self.log_posterior_and_gradient(theta)
        return -p, -dp

    def generate_guesses(self, n_guesses):
        guesses = zeros([n_guesses, self.n_hyperpars])
        mu_lwr = array([v[0] for v in self.hp_bounds[self.mean_slc]])
        mu_upr = array([v[1] for v in self.hp_bounds[self.mean_slc]])
        cv_lwr = array([v[0] for v in self.hp_bounds[self.cov_slc]])
        cv_upr = array([v[1] for v in self.hp_bounds[self.cov_slc]])
        guesses[:, self.points_slc] = normal(size=(n_guesses, self.n_points))
        guesses[:, self.mean_slc] = uniform(low=mu_lwr, high=mu_upr)
        guesses[:, self.cov_slc] = uniform(low=cv_lwr, high=cv_upr)
        return guesses

    def locate_map(self):
        starts = self.generate_guesses(500)

        starts = sorted(starts, key=self.log_posterior)[-10:]
        results = [
            fmin_l_bfgs_b(
                func=self.cost_and_gradient,
                approx_grad=False,
                x0=s,
                bounds=self.hp_bounds,
            )
            for s in starts
        ]

        results = sorted(results, key=lambda x: x[1])
        return results[0][0]

    def get_points(self, theta):
        K_xx = self.cov.build_covariance(theta[self.cov_slc])
        mu = self.mean.build_mean(theta[self.mean_slc])
        # get the cholesky decomposition
        L = cholesky(K_xx)
        # get the point values
        v = theta[self.points_slc]
        return mu + L @ v

    def laplace_approx(self, theta):
        H = zeros([self.n_hyperpars, self.n_hyperpars])
        delta = 1e-4
        for i in range(self.n_hyperpars):
            t1 = theta.copy()
            t2 = theta.copy()
            dt = theta[i] * delta
            t1[i] -= dt
            t2[i] += dt

            _, g1 = self.log_posterior_and_gradient(t1)
            _, g2 = self.log_posterior_and_gradient(t2)
            H[:, i] = 0.5 * (g2 - g1) / dt

        H = -0.5 * (H + H.T)
        L = cholesky(H)
        iL = solve_triangular(L, eye(L.shape[0]), lower=True)
        return iL.T @ iL

    def laplace_sample(self, theta, n_samples):
        covar = self.laplace_approx(theta)
        p0 = self.log_posterior(theta)
        samples = []
        while len(samples) < n_samples:
            s = multivariate_normal(mean=theta, cov=covar)
            p = self.log_posterior(s)
            dt = s - theta
            q = p0 - 0.5 * dt.dot(covar.dot(dt))
            if exp(p - q) > random():
                samples.append(s)
        return samples


def tril_half_diag(A):
    Q = tril(A)
    Q[diag_indices_from(Q)] *= 0.5
    return Q


def cholesky_derivative(L: ndarray, iL: ndarray, dK: ndarray):
    """
    Calculates the derivative of the Cholesky decomposition of a
    covariance matrix.

    :param L: Cholesky factor of the covariance matrix.
    :param iL: Inverse Cholesky factor of the covariance matrix.
    :param dK: Derivative of the covariance matrix.
    :return: Derivative of the Cholesky factor.
    """
    return L @ tril_half_diag(iL @ dK @ iL.T)

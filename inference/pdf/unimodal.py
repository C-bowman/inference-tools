from itertools import product
from numpy import cos, pi, log, exp, mean, sqrt, tanh
from numpy import array, ndarray, linspace, zeros, atleast_1d
from scipy.integrate import simpson, quad
from scipy.optimize import minimize
from inference.pdf.base import DensityEstimator
from inference.pdf.hdi import sample_hdi


class UnimodalPdf(DensityEstimator):
    """
    Construct a UnimodalPdf object, which can be called as a function to
    return the estimated PDF of the given sample.

    The UnimodalPdf class is designed to robustly estimate univariate, unimodal probability
    distributions given a sample drawn from that distribution. This is a parametric method
    based on a heavily modified student-t distribution, which is extremely flexible.

    :param sample: \
        1D array of samples from which to estimate the probability distribution.
    """

    def __init__(self, sample: ndarray):
        self.sample = array(sample).flatten()
        self.n_samps = self.sample.size

        # chebyshev quadrature weights and axes
        self.sd = 0.2
        self.n_nodes = 128
        k = linspace(1, self.n_nodes, self.n_nodes)
        t = cos(0.5 * pi * ((2 * k - 1) / self.n_nodes))
        self.u = t / (1.0 - t**2)
        self.w = (pi / self.n_nodes) * (1 + t**2) / (self.sd * (1 - t**2) ** 1.5)

        # first minimise based on a slice of the sample, if it's large enough
        self.cutoff = 2000
        self.skip = max(self.n_samps // self.cutoff, 1)
        self.fitted_samples = self.sample[:: self.skip]

        # makes guesses based on sample moments
        guesses, self.bounds = self.generate_guesses_and_bounds()
        # sort the guesses by the lowest cost
        cost_func = lambda x: -self.posterior(x)
        guesses = sorted(guesses, key=cost_func)

        # minimise based on the best guess
        opt_method = "Nelder-Mead"
        self.min_result = minimize(
            fun=cost_func, x0=guesses[0], bounds=self.bounds, method=opt_method
        )
        self.MAP = self.min_result.x
        self.mode = self.MAP[0]

        # if we were using a reduced sample, use full sample
        if self.skip > 1:
            self.fitted_samples = self.sample
            self.min_result = minimize(
                fun=cost_func,
                x0=self.MAP,
                bounds=self.bounds,
                method=opt_method,
            )
            self.MAP = self.min_result.x
            self.mode = self.MAP[0]

        # normalising constant for the MAP estimate curve
        self.map_lognorm = log(self.norm(self.MAP))

        # set some bounds for the confidence limits calculation
        x0, s0, v, f, k, q = self.MAP
        self.upr_limit = x0 + s0 * (4 * exp(f) + 1)
        self.lwr_limit = x0 - s0 * (4 * exp(-f) + 1)

    def generate_guesses_and_bounds(self) -> tuple[list, list]:
        mu, sigma, skew = self.sample_moments(self.fitted_samples)
        lwr, upr = sample_hdi(sample=self.sample, fraction=0.5)

        bounds = [
            (lwr, upr),
            (sigma * 0.1, sigma * 10),
            (0.0, 5.0),
            (-3.0, 3.0),
            (1e-2, 20.0),
            (1.0, 6.0),
        ]
        x0 = [lwr * (1 - f) + upr * f for f in [0.3, 0.5, 0.7]]
        s0 = [sigma, sigma * 2]
        ln_v = [0.25, 2.0]
        f = [0.5 * skew, skew]
        k = [1.0, 4.0, 8.0]
        q = [2.0]

        return [array(i) for i in product(x0, s0, ln_v, f, k, q)], bounds

    @staticmethod
    def sample_moments(samples: ndarray) -> tuple[float, float, float]:
        mu = mean(samples)
        x2 = samples**2
        x3 = x2 * samples
        sig = sqrt(mean(x2) - mu**2)
        skew = (mean(x3) - 3 * mu * sig**2 - mu**3) / sig**3
        return mu, sig, skew

    def __call__(self, x: ndarray) -> ndarray:
        """
        Evaluate the PDF estimate at a set of given axis positions.

        :param x: axis location(s) at which to evaluate the estimate.
        :return: values of the PDF estimate at the specified locations.
        """
        return exp(self.log_pdf_model(x, self.MAP) - self.map_lognorm)

    def cdf(self, x: ndarray) -> ndarray:
        x = atleast_1d(x)
        sorter = x.argsort()
        inverse_sort = sorter.argsort()
        v = x[sorter]
        intervals = zeros(x.size)
        intervals[0] = (
            quad(self.__call__, self.lwr_limit, v[0])[0]
            if v[0] > self.lwr_limit
            else 0.0
        )
        for i in range(1, x.size):
            intervals[i] = quad(self.__call__, v[i - 1], v[i])[0]
        integral = intervals.cumsum()[inverse_sort]
        return integral if x.size > 1 else integral[0]

    def evaluate_model(self, x: ndarray, theta: ndarray) -> ndarray:
        return self.pdf_model(x, theta) / self.norm(theta)

    def posterior(self, theta: ndarray) -> float:
        normalisation = self.fitted_samples.size * log(self.norm(theta))
        return self.log_pdf_model(self.fitted_samples, theta).sum() - normalisation

    def norm(self, theta: ndarray) -> float:
        v = self.pdf_model(self.u, [0.0, self.sd, *theta[2:]])
        integral = (self.w * v).sum() * theta[1]
        return integral

    def pdf_model(self, x: ndarray, theta: ndarray) -> ndarray:
        return exp(self.log_pdf_model(x, theta))

    def log_pdf_model(self, x: ndarray, theta: ndarray) -> ndarray:
        x0, s0, ln_v, f, k, q = theta
        v = exp(ln_v)
        z0 = (x - x0) / s0
        z = z0 * exp(-f * tanh(z0 / k))

        log_prob = -(0.5 * (1 + v)) * log(1 + (abs(z) ** q) / v)
        return log_prob

    def moments(self) -> tuple[float, ...]:
        """
        Calculate the mean, variance skewness and excess kurtosis of the estimated PDF.

        :return: mean, variance, skewness, ex-kurtosis
        """
        s = self.MAP[1]
        f = self.MAP[3]

        lwr = self.mode - 5 * max(exp(-f), 1.0) * s
        upr = self.mode + 5 * max(exp(f), 1.0) * s
        x = linspace(lwr, upr, 1000)
        p = self(x)

        mu = simpson(p * x, x=x)
        var = simpson(p * (x - mu) ** 2, x=x)
        skw = simpson(p * (x - mu) ** 3, x=x) / var**1.5
        kur = (simpson(p * (x - mu) ** 4, x=x) / var**2) - 3.0
        return mu, var, skw, kur

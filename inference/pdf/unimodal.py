from itertools import product
from numpy import cos, pi, log, exp, mean, sqrt, tanh
from numpy import array, ndarray, linspace, zeros, atleast_1d
from scipy.integrate import simps, quad
from scipy.optimize import minimize
from inference.pdf.base import DensityEstimator


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
        self.n_samps = sample.size

        # chebyshev quadtrature weights and axes
        self.sd = 0.2
        self.n_nodes = 128
        k = linspace(1, self.n_nodes, self.n_nodes)
        t = cos(0.5 * pi * ((2 * k - 1) / self.n_nodes))
        self.u = t / (1.0 - t**2)
        self.w = (pi / self.n_nodes) * (1 + t**2) / (self.sd * (1 - t**2) ** 1.5)

        # first minimise based on a slice of the sample, if it's large enough
        self.cutoff = 2000
        self.skip = max(self.n_samps // self.cutoff, 1)

        self.x = self.sample[:: self.skip]
        self.n = len(self.x)

        # makes guesses based on sample moments
        guesses = self.generate_guesses()

        # sort the guesses by the lowest score
        minfunc = lambda x: -self.posterior(x)
        guesses = sorted(guesses, key=minfunc)

        # minimise based on the best guess
        self.min_result = minimize(minfunc, guesses[0], method="Nelder-Mead")
        self.MAP = self.min_result.x
        self.mode = self.MAP[0]

        # if we were using a reduced sample, use full sample
        if self.skip > 1:
            self.x = self.sample
            self.n = self.n_samps
            self.min_result = minimize(minfunc, self.MAP, method="Nelder-Mead")
            self.MAP = self.min_result.x
            self.mode = self.MAP[0]

        # normalising constant for the MAP estimate curve
        self.map_lognorm = log(self.norm(self.MAP))

        # set some bounds for the confidence limits calculation
        x0, s0, v, f, k, q = self.MAP
        self.upr_limit = x0 + s0 * (4 * exp(f) + 1)
        self.lwr_limit = x0 - s0 * (4 * exp(-f) + 1)

    def generate_guesses(self):
        mu, sigma, skew = self.sample_moments()

        x0 = [mu, mu - sigma * skew * 0.15, mu - sigma * skew * 0.3]
        v = [0, 5.0]
        s0 = [sigma, sigma * 2]
        f = [0.5 * skew, skew]
        k = [1.0, 4.0, 8.0]
        q = [2.0]

        return [array(i) for i in product(x0, s0, v, f, k, q)]

    def sample_moments(self):
        mu = mean(self.x)
        x2 = self.x**2
        x3 = x2 * self.x
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

    def posterior(self, paras):
        x0, s0, v, f, k, q = paras

        # prior checks
        if (s0 > 0) & (0 < k < 20) & (1 < q < 6):
            normalisation = self.n * log(self.norm(paras))
            return self.log_pdf_model(self.x, paras).sum() - normalisation
        else:
            return -1e50

    def norm(self, pvec):
        v = self.pdf_model(self.u, [0.0, self.sd, *pvec[2:]])
        integral = (self.w * v).sum() * pvec[1]
        return integral

    def pdf_model(self, x, pvec):
        return exp(self.log_pdf_model(x, pvec))

    def log_pdf_model(self, x, pvec):
        x0, s0, v, f, k, q = pvec
        v = exp(v) + 1
        z0 = (x - x0) / s0
        ds = exp(f * tanh(z0 / k))
        z = z0 / ds

        log_prob = -(0.5 * (1 + v)) * log(1 + (abs(z) ** q) / v)
        return log_prob

    def moments(self):
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

        mu = simps(p * x, x=x)
        var = simps(p * (x - mu) ** 2, x=x)
        skw = simps(p * (x - mu) ** 3, x=x) / var**1.5
        kur = (simps(p * (x - mu) ** 4, x=x) / var**2) - 3.0
        return mu, var, skw, kur

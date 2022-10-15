"""
.. moduleauthor:: Chris Bowman <chris.bowman.physics@gmail.com>
"""

from numpy import exp, log, mean, std, sqrt, tanh, cos, cov
from numpy import array, linspace, sort, searchsorted, pi, argmax, argsort, logaddexp
from numpy.random import random
from scipy.integrate import quad, simps
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from warnings import warn
from itertools import product
from functools import reduce
import matplotlib.pyplot as plt


class DensityEstimator:
    """
    Parent class for the 1D density estimation classes GaussianKDE and UnimodalPdf.
    """

    def __init__(self):
        self.lwr_limit = None
        self.upr_limit = None
        self.mode = None

    def __call__(self, x):
        return None

    def interval(self, frac=0.95):
        p_max = self(self.mode)
        p_conf = self.binary_search(
            self.interval_prob, frac, [0.0, p_max], uphill=False
        )
        return self.get_interval(p_conf)

    def get_interval(self, z):
        lwr = self.binary_search(self, z, [self.lwr_limit, self.mode], uphill=True)
        upr = self.binary_search(self, z, [self.mode, self.upr_limit], uphill=False)
        return lwr, upr

    def interval_prob(self, z):
        lwr, upr = self.get_interval(z)
        return quad(self, lwr, upr, limit=100)[0]

    def moments(self):
        pass

    def plot_summary(self, filename=None, show=True, label=None):
        """
        Plot the estimated PDF along with summary statistics.

        :keyword str filename: Filename to which the plot will be saved. If unspecified, the plot will not be saved.
        :keyword bool show: Boolean value indicating whether the plot should be displayed in a window. (Default is True)
        :keyword str label: The label to be used for the x-axis on the plot as a string.
        """

        def ensure_is_nested_list(var):
            if not isinstance(var[0], (list, tuple)):
                var = [var]
            return var

        sigma_1 = ensure_is_nested_list(self.interval(frac=0.68268))
        sigma_2 = ensure_is_nested_list(self.interval(frac=0.95449))
        sigma_3 = ensure_is_nested_list(self.interval(frac=0.9973))
        mu, var, skw, kur = self.moments()

        if type(self) is GaussianKDE:
            lwr = sigma_3[0][0] - 5 * self.h
            upr = sigma_3[0][1] + 5 * self.h
        else:
            s_min = sigma_3[0][0]
            s_max = sigma_3[-1][1]

            lwr = s_min - 0.1 * (s_max - s_min)
            upr = s_max + 0.1 * (s_max - s_min)

        axis = linspace(lwr, upr, 500)

        fig, ax = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(10, 6),
            gridspec_kw={"width_ratios": [2, 1]},
        )
        ax[0].plot(axis, self(axis), lw=1, c="C0")
        ax[0].fill_between(axis, self(axis), color="C0", alpha=0.1)
        ax[0].plot([self.mode, self.mode], [0.0, self(self.mode)], c="red", ls="dashed")

        ax[0].set_xlabel(label or "argument", fontsize=13)
        ax[0].set_ylabel("probability density", fontsize=13)
        ax[0].set_ylim([0.0, None])
        ax[0].grid()

        gap = 0.05
        h = 0.95
        x1 = 0.35
        x2 = 0.40

        def section_title(height, name):
            ax[1].text(0.0, height, name, horizontalalignment="left", fontweight="bold")
            return height - gap

        def write_quantity(height, name, value):
            ax[1].text(x1, height, f"{name}:", horizontalalignment="right")
            ax[1].text(x2, height, f"{value:.5G}", horizontalalignment="left")
            return height - gap

        h = section_title(h, "Basics")
        h = write_quantity(h, "Mode", self.mode)
        h = write_quantity(h, "Mean", mu)
        h = write_quantity(h, "Standard dev", sqrt(var))
        h -= gap

        h = section_title(h, "Highest-density intervals")

        def write_sigma(height, name, sigma):
            ax[1].text(x1, height, name, horizontalalignment="right")
            for itvl in sigma:
                ax[1].text(
                    x2,
                    height,
                    rf"{itvl[0]:.5G} $\rightarrow$ {itvl[1]:.5G}",
                    horizontalalignment="left",
                )
                height -= gap
            return height

        h = write_sigma(h, "1-sigma:", sigma_1)
        h = write_sigma(h, "2-sigma:", sigma_2)
        h = write_sigma(h, "3-sigma:", sigma_3)
        h -= gap

        h = section_title(h, "Higher moments")
        h = write_quantity(h, "Variance", var)
        h = write_quantity(h, "Skewness", skw)
        h = write_quantity(h, "Kurtosis", kur)

        ax[1].axis("off")

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()

        return fig, ax

    @staticmethod
    def binary_search(func, value, bounds, uphill=True):
        x_min, x_max = bounds
        x = (x_min + x_max) * 0.5

        converged = False
        while not converged:
            f = func(x)
            if f > value:
                if uphill:
                    x_max = x
                else:
                    x_min = x
            else:
                if uphill:
                    x_min = x
                else:
                    x_max = x

            x = (x_min + x_max) * 0.5
            if abs((x_max - x_min) / x) < 1e-3:
                converged = True

        # now linearly interpolate as a polish step
        f_max = func(x_max)
        f_min = func(x_min)
        df = f_max - f_min

        return x_min * ((f_max - value) / df) + x_max * ((value - f_min) / df)


class UnimodalPdf(DensityEstimator):
    """
    Construct a UnimodalPdf object, which can be called as a function to
    return the estimated PDF of the given sample.

    The UnimodalPdf class is designed to robustly estimate univariate, unimodal probability
    distributions given a sample drawn from that distribution. This is a parametric method
    based on an heavily modified student-t distribution, which is extremely flexible.

    :param sample: 1D array of samples from which to estimate the probability distribution
    """

    def __init__(self, sample):

        self.sample = array(sample)
        self.n_samps = len(sample)

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
        guesses = sorted(guesses, key=self.minfunc)

        # minimise based on the best guess
        self.min_result = minimize(self.minfunc, guesses[0], method="Nelder-Mead")
        self.MAP = self.min_result.x
        self.mode = self.MAP[0]

        # if we were using a reduced sample, use full sample
        if self.skip > 1:
            self.x = self.sample
            self.n = self.n_samps
            self.min_result = minimize(self.minfunc, self.MAP, method="Nelder-Mead")
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

    def __call__(self, x):
        """
        Evaluate the PDF estimate at a set of given axis positions.

        :param x: axis location(s) at which to evaluate the estimate.
        :return: values of the PDF estimate at the specified locations.
        """
        return exp(self.log_pdf_model(x, self.MAP) - self.map_lognorm)

    def posterior(self, paras):
        x0, s0, v, f, k, q = paras

        # prior checks
        if (s0 > 0) & (0 < k < 20) & (1 < q < 6):
            normalisation = self.n * log(self.norm(paras))
            return self.log_pdf_model(self.x, paras).sum() - normalisation
        else:
            return -1e50

    def minfunc(self, paras):
        return -self.posterior(paras)

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
        skw = simps(p * (x - mu) ** 3, x=x) / var * 1.5
        kur = (simps(p * (x - mu) ** 4, x=x) / var**2) - 3.0
        return mu, var, skw, kur


class GaussianKDE(DensityEstimator):
    """
    Construct a GaussianKDE object, which can be called as a function to
    return the estimated PDF of the given sample.

    GaussianKDE uses Gaussian kernel-density estimation to estimate the PDF
    associated with a given sample.

    :param sample: \
        1D array of samples from which to estimate the probability distribution

    :param float bandwidth: \
        Width of the Gaussian kernels used for the estimate. If not specified,
        an appropriate width is estimated based on sample data.

    :param bool cross_validation: \
        Indicate whether or not cross-validation should be used to estimate
        the bandwidth in place of the simple 'rule of thumb' estimate which
        is normally used.

    :param int max_cv_samples: \
        The maximum number of samples to be used when estimating the bandwidth
        via cross-validation. The computational cost scales roughly quadratically
        with the number of samples used, and can become prohibitive for samples of
        size in the tens of thousands and up. Instead, if the sample size is greater
        than *max_cv_samples*, the cross-validation is performed on a sub-sample of
        this size.
    """

    def __init__(
        self, sample, bandwidth=None, cross_validation=False, max_cv_samples=5000
    ):

        self.s = sort(array(sample).flatten())  # sorted array of the samples
        self.max_cvs = (
            max_cv_samples  # maximum number of samples to be used for cross-validation
        )

        if self.s.size < 3:
            raise ValueError(
                """
                [ GaussianKDE error ]
                Not enough samples were given to estimate the PDF.
                At least 3 samples are required.
                """
            )

        if bandwidth is None:
            self.h = self.simple_bandwidth_estimator()  # very simple bandwidth estimate
            if cross_validation:
                self.h = self.cross_validation_bandwidth_estimator(self.h)
        else:
            self.h = bandwidth

        # define some useful constants
        self.norm = 1.0 / (len(self.s) * sqrt(2 * pi) * self.h)
        self.cutoff = self.h * 4
        self.q = 1.0 / (sqrt(2) * self.h)
        self.lwr_limit = self.s[0] - self.cutoff * 0.5
        self.upr_limit = self.s[-1] + self.cutoff * 0.5

        # decide how many regions the axis should be divided into
        n = int(log((self.s[-1] - self.s[0]) / self.h) / log(2)) + 1

        # now generate midpoints of these regions
        mids = linspace(self.s[0], self.s[-1], 2**n + 1)
        mids = 0.5 * (mids[1:] + mids[:-1])

        # get the cutoff indices
        lwr_inds = searchsorted(self.s, mids - self.cutoff)
        upr_inds = searchsorted(self.s, mids + self.cutoff)
        slices = [slice(l, u) for l, u in zip(lwr_inds, upr_inds)]

        # now build a dict that maps midpoints to the slices
        self.slice_map = dict(zip(mids, slices))

        # build a binary tree which allows fast look-up of which
        # region contains a given value
        self.tree = BinaryTree(n, (self.s[0], self.s[-1]))

        #: The mode of the pdf, calculated automatically when an instance of GaussianKDE is created.
        self.mode = self.locate_mode()

    def __call__(self, x_vals):
        """
        Evaluate the PDF estimate at a set of given axis positions.

        :param x_vals: axis location(s) at which to evaluate the estimate.
        :return: values of the PDF estimate at the specified locations.
        """
        if hasattr(x_vals, "__iter__"):
            return [self.density(x) for x in x_vals]
        else:
            return self.density(x_vals)

    def density(self, x):
        # look-up the region
        region = self.tree.lookup(x)
        # look-up the cutting points
        slc = self.slice_map[region[2]]
        # evaluate the density estimate from the slice
        return self.norm * exp(-(((x - self.s[slc]) * self.q) ** 2)).sum()

    def simple_bandwidth_estimator(self):
        # A simple estimate which assumes the distribution close to a Gaussian
        return 1.06 * std(self.s) / (len(self.s) ** 0.2)

    def cross_validation_bandwidth_estimator(self, initial_h):
        """
        Selects the bandwidth by maximising a log-probability derived
        using a 'leave-one-out cross-validation' approach.
        """
        # first check if we need to sub-sample for computational cost reduction
        if len(self.s) > self.max_cvs:
            scrambler = argsort(random(size=len(self.s)))
            samples = (self.s[scrambler])[: self.max_cvs]
        else:
            samples = self.s

        # create a grid in log-bandwidth space and evaluate the log-prob across it
        dh = 0.5
        log_h = [initial_h + m * dh for m in (-2, -1, 0, 1, 2)]
        log_p = [self.cross_validation_logprob(samples, exp(h)) for h in log_h]

        # if the maximum log-probability is at the edge of the grid, extend it
        for i in range(5):
            # stop when the maximum is not at the edge
            max_ind = argmax(log_p)
            if 0 < max_ind < len(log_h) - 1:
                break

            if max_ind == 0:  # extend grid to lower bandwidths
                new_h = log_h[0] - dh
                new_lp = self.cross_validation_logprob(samples, exp(new_h))
                log_h.insert(0, new_h)
                log_p.insert(0, new_lp)

            else:  # extend grid to higher bandwidths
                new_h = log_h[-1] + dh
                new_lp = self.cross_validation_logprob(samples, exp(new_h))
                log_h.append(new_h)
                log_p.append(new_lp)

        # cost of evaluating the cross-validation is expensive, so we want to
        # minimise total evaluations. Here we assume the CV score has only one
        # maxima, and use recursive grid refinement to rapidly find it.
        for refine in range(6):
            max_ind = int(argmax(log_p))
            lwr_h = 0.5 * (log_h[max_ind - 1] + log_h[max_ind])
            upr_h = 0.5 * (log_h[max_ind] + log_h[max_ind + 1])

            lwr_lp = self.cross_validation_logprob(samples, exp(lwr_h))
            upr_lp = self.cross_validation_logprob(samples, exp(upr_h))

            log_h.insert(max_ind, lwr_h)
            log_p.insert(max_ind, lwr_lp)

            log_h.insert(max_ind + 2, upr_h)
            log_p.insert(max_ind + 2, upr_lp)

        h_estimate = exp(log_h[argmax(log_p)])
        return h_estimate

    def cross_validation_logprob(self, samples, width, c=0.99):
        """
        This function uses a 'leave-one-out cross-validation' (LOO-CV)
        approach to calculate a log-probability associated with the
        density estimate - the bandwidth can be selected by maximising
        this log-probability.
        """
        # evaluate the log-pdf estimate at each sample point
        log_pdf = self.log_evaluation(samples, samples, width)
        # remove the contribution at each sample due to itself
        d = log(c) - log(width * len(samples) * sqrt(2 * pi)) - log_pdf
        loo_adjustment = log(1 - exp(d))
        log_probs = log_pdf + loo_adjustment
        return log_probs.sum()  # sum to find the overall log-probability

    @staticmethod
    def log_kernel(x, c, h):
        z = (x - c) / h
        return -0.5 * z**2 - log(h)

    def log_evaluation(self, points, samples, width):
        # evaluate the log-pdf in a way which prevents underflow
        generator = (self.log_kernel(points, s, width) for s in samples)
        return reduce(logaddexp, generator) - log(len(samples) * sqrt(2 * pi))

    def locate_mode(self):
        # if there are enough samples, use the 20% HDI to bound the search for the mode
        if self.s.size > 50:
            lwr, upr = sample_hdi(self.s, 0.2)
        else:  # else just use the entire range of the samples
            lwr, upr = self.s[0], self.s[-1]

        result = minimize_scalar(
            lambda x: -self(x), bounds=[lwr, upr], method="bounded"
        )
        return result.x

    def moments(self):
        """
        Calculate the mean, variance skewness and excess kurtosis of the estimated PDF.

        :return: mean, variance, skewness, ex-kurtosis

        Note that these quantities are calculated directly from the estimated PDF, and
        not from the sample values.
        """
        N = 1000
        x = linspace(self.lwr_limit, self.upr_limit, N)
        p = self(x)

        mu = simps(p * x, x=x)
        var = simps(p * (x - mu) ** 2, x=x)
        skw = simps(p * (x - mu) ** 3, x=x) / var * 1.5
        kur = (simps(p * (x - mu) ** 4, x=x) / var**2) - 3.0
        return mu, var, skw, kur

    def interval(self, frac=0.95):
        """
        Calculate the highest-density interval(s) which contain a given fraction of total probability.

        :param float frac: Fraction of total probability contained by the desired interval(s).
        :return: A list of tuples which specify the intervals.
        """
        return sample_hdi(self.s, frac, allow_double=True)


class KDE2D:
    def __init__(self, x=None, y=None):

        self.x = array(x)
        self.y = array(y)
        # very simple bandwidth estimate
        s_x, s_y = self.estimate_bandwidth(self.x, self.y)
        self.q_x = 1.0 / (sqrt(2) * s_x)
        self.q_y = 1.0 / (sqrt(2) * s_y)
        self.norm = 1.0 / (len(self.x) * sqrt(2 * pi) * s_x * s_y)

    def __call__(self, x_vals, y_vals):
        if hasattr(x_vals, "__iter__") and hasattr(y_vals, "__iter__"):
            return [self.density(x, y) for x, y in zip(x_vals, y_vals)]
        else:
            return self.density(x_vals, y_vals)

    def density(self, x, y):
        z_x = ((self.x - x) * self.q_x) ** 2
        z_y = ((self.y - y) * self.q_y) ** 2
        return exp(-z_x - z_y).sum() * self.norm

    def estimate_bandwidth(self, x, y):
        S = cov(x, y)
        p = S[0, 1] / sqrt(S[0, 0] * S[1, 1])
        return 1.06 * sqrt(S.diagonal() * (1 - p**2)) / (len(x) ** 0.2)


class BinaryTree:
    """
    divides the range specified by limits into n = 2**layers equal regions,
    and builds a binary tree which allows fast look-up of which of region
    contains a given value.

    :param int layers: number of layers that make up the tree
    :param limits: tuple of the lower and upper bounds of the look-up region.
    """

    def __init__(self, layers, limits):
        self.n = layers
        self.lims = limits
        self.edges = linspace(limits[0], limits[1], 2**self.n + 1)

        self.p = [
            [a, b, 0.5 * (a + b)] for a, b in zip(self.edges[:-1], self.edges[1:])
        ]
        self.p.insert(0, self.p[0])
        self.p.append(self.p[-1])

    def lookup(self, val):
        return self.p[searchsorted(self.edges, val)]


def sample_hdi(sample, fraction, allow_double=False):
    """
    Estimate the highest-density interval(s) for a given sample.

    This function computes the shortest possible interval which contains a chosen
    fraction of the elements in the given sample.

    :param sample: \
        A sample for which the interval will be determined.

    :param float fraction: \
        The fraction of the total probability to be contained by the interval.

    :param bool allow_double: \
        When set to True, a double-interval is returned instead if one exists whose
        total length is meaningfully shorter than the optimal single interval.

    :return: \
        Tuple(s) specifying the lower and upper bounds of the highest-density interval(s).
    """

    # verify inputs are valid
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction parameter must be between 0 and 1")
    if not hasattr(sample, "__len__") or len(sample) < 2:
        raise ValueError("The sample must have at least 2 elements")

    s = array(sample)
    if len(s.shape) > 1:
        s = s.flatten()
    s = sort(s)
    n = len(s)
    L = int(fraction * n)

    # check that we have enough samples to estimate the HDI for the chosen fraction
    if n <= L:
        warn(
            "The number of samples is insufficient to estimate the interval for the given fraction"
        )
        return (s[0], s[-1])
    elif n - L < 20:
        warn(
            "len(sample)*(1 - fraction) is small - calculated interval may be inaccurate"
        )

    # find the optimal single HDI
    widths = s[L:] - s[: n - L]
    i = widths.argmin()
    r1, w1 = (s[i], s[i + L]), s[i + L] - s[i]

    if allow_double:
        # now get the best 2-interval solution
        minfunc = dbl_interval_length(sample, fraction)
        bounds = minfunc.get_bounds()
        de_result = differential_evolution(minfunc, bounds)
        I1, I2 = minfunc.return_intervals(de_result.x)
        w2 = (I2[1] - I2[0]) + (I1[1] - I1[0])

    # return the split interval if the width reduction is non-trivial:
    if allow_double and w2 < w1 * 0.99:
        return I1, I2
    else:
        return r1


class dbl_interval_length:
    def __init__(self, sample, fraction):
        self.sample = sort(sample)
        self.f = fraction
        self.N = len(sample)
        self.L = int(self.f * self.N)
        self.space = self.N - self.L
        self.max_length = self.sample[-1] - self.sample[0]

    def get_bounds(self):
        return [(0.0, 1.0), (0, self.space - 1), (0, self.space - 1)]

    def __call__(self, paras):
        f1 = paras[0]
        start = int(paras[1])
        gap = int(paras[2])

        if (start + gap) > self.space - 1:
            return self.max_length

        w1 = int(f1 * self.L)
        w2 = self.L - w1
        start_2 = start + w1 + gap

        I1 = self.sample[start + w1] - self.sample[start]
        I2 = self.sample[start_2 + w2] - self.sample[start_2]
        return I1 + I2

    def return_intervals(self, paras):
        f1 = paras[0]
        start = int(paras[1])
        gap = int(paras[2])

        w1 = int(f1 * self.L)
        w2 = self.L - w1
        start_2 = start + w1 + gap

        I1 = (self.sample[start], self.sample[start + w1])
        I2 = (self.sample[start_2], self.sample[start_2 + w2])
        return I1, I2

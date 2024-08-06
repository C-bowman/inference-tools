from functools import reduce
from numpy import arange, array, ndarray, atleast_1d, zeros
from numpy import sort, linspace, searchsorted, argsort, argmax, unique
from numpy import sqrt, pi, log, exp, std, logaddexp, cov
from numpy.random import random
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar
from scipy.special import erf
from inference.pdf.hdi import sample_hdi
from inference.pdf.base import DensityEstimator


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
        Indicate whether cross-validation should be used to estimate
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
        self,
        sample: ndarray,
        bandwidth: float = None,
        cross_validation: bool = False,
        max_cv_samples=5000,
    ):
        self.sample = sort(array(sample).flatten())  # sorted array of the samples
        # maximum number of samples to be used for cross-validation
        self.max_cvs = max_cv_samples

        if self.sample.size < 3:
            raise ValueError(
                """\n
                \r[ GaussianKDE error ]
                \r>> Not enough samples were given to estimate the PDF.
                \r>> At least 3 samples are required.
                """
            )

        if bandwidth is None:
            self.h = self.simple_bandwidth_estimator()  # very simple bandwidth estimate
            if cross_validation:
                self.h = self.cross_validation_bandwidth_estimator(self.h)
        else:
            self.h = bandwidth

        # define some useful constants
        self.norm = 1.0 / (len(self.sample) * sqrt(2 * pi) * self.h)
        self.cutoff = self.h * 4
        self.q = 1.0 / (sqrt(2) * self.h)
        self.lwr_limit = self.sample[0] - self.cutoff * 0.5
        self.upr_limit = self.sample[-1] + self.cutoff * 0.5

        # decide how many regions the axis should be divided into
        n = int(log((self.sample[-1] - self.sample[0]) / self.h) / log(2)) + 1

        # now generate midpoints of these regions
        mids = linspace(self.sample[0], self.sample[-1], 2**n + 1)
        mids = 0.5 * (mids[1:] + mids[:-1])

        # get the cutoff indices
        lwr_inds = searchsorted(self.sample, mids - self.cutoff)
        upr_inds = searchsorted(self.sample, mids + self.cutoff)
        self.slices = [slice(l, u) for l, u in zip(lwr_inds, upr_inds)]
        self.cdf_offsets = lwr_inds / self.sample.size

        # build a binary tree which allows fast look-up of which
        # region contains a given value
        self.tree = BinaryTree(n, (self.sample[0], self.sample[-1]))

        # The mode of the pdf, calculated automatically when an instance of GaussianKDE is created.
        self.mode = self.locate_mode()

    def __call__(self, x: ndarray) -> ndarray:
        """
        Evaluate the estimate of the probability distribution function (PDF)
        at the given parameter values.

        :param x: axis location(s) at which to evaluate the estimate.
        :return: values of the PDF estimate at the specified locations.
        """
        x = atleast_1d(x)
        pdf = zeros(x.size)
        # look-up the region
        regions, index_groups = self.tree.region_groups(x)
        # evaluate the density estimate from the slice
        for r, g in zip(regions, index_groups):
            dx = x[g, None] - self.sample[None, self.slices[r]]
            pdf[g] = exp(-((dx * self.q) ** 2)).sum(axis=1)
        pdf *= self.norm
        return pdf if pdf.size > 1 else pdf[0]

    def cdf(self, x: ndarray) -> ndarray:
        """
        Evaluate the estimate of the cumulative distribution function (CDF)
        at the given parameter values.

        :param x: axis location(s) at which to evaluate the estimate.
        :return: values of the PDF estimate at the specified locations.
        """
        x = atleast_1d(x)
        cdf = zeros(x.size)
        # look-up the region
        regions, index_groups = self.tree.region_groups(x)
        coeff = 0.5 / self.sample.size
        # evaluate the density estimate from the slice
        for r, g in zip(regions, index_groups):
            dx = x[g, None] - self.sample[None, self.slices[r]]
            k = 1 + erf(dx * self.q)
            cdf[g] = coeff * k.sum(axis=1) + self.cdf_offsets[r]
        return cdf if cdf.size > 1 else cdf[0]

    def simple_bandwidth_estimator(self):
        # A simple estimate which assumes the distribution close to a Gaussian
        return 1.06 * std(self.sample) / (self.sample.size**0.2)

    def cross_validation_bandwidth_estimator(self, initial_h: float) -> float:
        """
        Selects the bandwidth by maximising a log-probability derived
        using a 'leave-one-out cross-validation' approach.
        """
        # first check if we need to sub-sample for computational cost reduction
        if len(self.sample) > self.max_cvs:
            scrambler = argsort(random(size=len(self.sample)))
            samples = (self.sample[scrambler])[: self.max_cvs]
        else:
            samples = self.sample

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
        if self.sample.size > 50:
            lwr, upr = sample_hdi(self.sample, 0.2)
        else:  # else just use the entire range of the samples
            lwr, upr = self.sample[0], self.sample[-1]

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
        N = int(5 * (self.upr_limit - self.lwr_limit) / self.h)
        x = linspace(self.lwr_limit, self.upr_limit, N)
        p = self(x)

        mu = simpson(p * x, x=x)
        dx = x - mu
        I = p * dx**2
        var = simpson(I, x=x)
        I *= dx
        skw = simpson(I, x=x) / var**1.5
        I *= dx
        kur = (simpson(I, x=x) / var**2) - 3.0
        return mu, var, skw, kur


class KDE2D:
    def __init__(self, x: ndarray, y: ndarray):
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

    def __init__(self, layers: int, limits: tuple[float, float]):
        self.n = layers
        self.lims = limits
        self.edges = linspace(limits[0], limits[1], 2**self.n + 1)
        self.regions = arange(-1, self.edges.size)
        self.regions[0] = 0
        self.regions[-1] = self.edges.size - 2

    def region_groups(self, values: ndarray):
        """
        Finds the indices of the given 'values' array which correspond to each of the
        regions covered by the tree.
        """
        region_indices = self.regions[searchsorted(self.edges, values)]
        return unique_index_groups(region_indices)


def unique_index_groups(values: ndarray) -> tuple[ndarray, list[ndarray]]:
    """
    For the given 'values' array, generates a list of numpy arrays which
    contain the indices corresponding to the groupings of each of the unique values.
    """
    unique_values, inverse_inds, counts = unique(
        values, return_inverse=True, return_counts=True
    )
    # get ordered groups of the indices of the unique values
    unique_inds = inverse_inds.argsort()
    # use cumulative sum of the counts to get slicing positions for index groups
    cuts = zeros(counts.size + 1, dtype=int)
    cuts[1:] = counts.cumsum()
    # slice the indices into groups for each unique value
    groups = [unique_inds[i:j] for i, j in zip(cuts[:-1], cuts[1:])]
    return unique_values, groups

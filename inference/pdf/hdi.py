from _warnings import warn
from typing import Sequence

from numpy import ndarray, array, sort
from scipy.optimize import differential_evolution


def sample_hdi(sample: ndarray, fraction: float, allow_double=False):
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
        raise ValueError(
            f"""\n
            [ sample_hdi error ]
            >> The 'fraction' argument must be a float between 0 and 1,
            >> but the value given was {fraction}.
            """
        )

    if isinstance(sample, ndarray):
        s = sample.copy()
    elif isinstance(sample, Sequence):
        s = array(sample)
    else:
        raise ValueError(
            f"""\n
            [ sample_hdi error ]
            >> The 'sample' argument should be a numpy.ndarray or a
            >> Sequence which can be converted to an array, but
            >> instead has type {type(sample)}.
            """
        )

    if s.size < 2:
        raise ValueError(
            f"""\n
            [ sample_hdi error ]
            >> The given 'sample' array must contain at least 2 values.
            """
        )

    if s.ndim > 1:
        s = s.flatten()
    s.sort()
    n = s.size
    L = int(fraction * n)

    # check that we have enough samples to estimate the HDI for the chosen fraction
    if n <= L:
        warn(
            f"""\n
            [ sample_hdi warning ]
            >> The given number of samples is insufficient to estimate the interval
            >> for the given fraction.
            """
        )
        return s[0], s[-1]
    elif n - L < 20:
        warn(
            f"""\n
            [ sample_hdi warning ]
            >> len(sample)*(1 - fraction) is small - calculated interval may be inaccurate.
            """
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

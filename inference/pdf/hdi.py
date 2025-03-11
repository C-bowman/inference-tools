from _warnings import warn
from typing import Sequence
from numpy import ndarray, array, sort, zeros, take_along_axis, expand_dims


def sample_hdi(sample: ndarray, fraction: float) -> ndarray:
    """
    Estimate the highest-density interval(s) for a given sample.

    This function computes the shortest possible interval which contains a chosen
    fraction of the elements in the given sample.

    :param sample: \
        A sample for which the interval will be determined. If the sample is given
        as a 2D numpy array, the interval calculation will be distributed over the
        second dimension of the array, i.e. given a sample array of shape ``(m, n)``
        the highest-density intervals are returned as an array of shape ``(2, n)``.

    :param float fraction: \
        The fraction of the total probability to be contained by the interval.

    :return: \
        The lower and upper bounds of the highest-density interval(s) as a numpy array.
    """

    # verify inputs are valid
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            f"""\n
            \r[ sample_hdi error ]
            \r>> The 'fraction' argument must be a float between 0 and 1,
            \r>> but the value given was {fraction}.
            """
        )

    if isinstance(sample, ndarray):
        s = sample.copy()
    elif isinstance(sample, Sequence):
        s = array(sample)
    else:
        raise ValueError(
            f"""\n
            \r[ sample_hdi error ]
            \r>> The 'sample' argument should be a numpy.ndarray or a
            \r>> Sequence which can be converted to an array, but
            \r>> instead has type {type(sample)}.
            """
        )

    if s.ndim > 2 or s.ndim == 0:
        raise ValueError(
            f"""\n
            \r[ sample_hdi error ]
            \r>> The 'sample' argument should be a numpy.ndarray
            \r>> with either one or two dimensions, but the given
            \r>> array has dimensionality {s.ndim}.
            """
        )

    if s.ndim == 1:
        s.resize([s.size, 1])

    n_samples, n_intervals = s.shape
    L = int(fraction * n_samples)

    if n_samples < 2:
        raise ValueError(
            f"""\n
            \r[ sample_hdi error ]
            \r>> The first dimension of the given 'sample' array must 
            \r>> have have a length of at least 2.
            """
        )

    # check that we have enough samples to estimate the HDI for the chosen fraction
    if n_samples <= L:
        warn(
            f"""\n
            \r[ sample_hdi warning ]
            \r>> The given number of samples is insufficient to estimate the interval
            \r>> for the given fraction.
            """
        )

    elif n_samples - L < 20:
        warn(
            f"""\n
            \r[ sample_hdi warning ]
            \r>> n_samples * (1 - fraction) is small - calculated interval may be inaccurate.
            """
        )

    # check that we have enough samples to estimate the HDI for the chosen fraction
    s.sort(axis=0)
    hdi = zeros([2, n_intervals])
    if n_samples > L:
        # find the optimal single HDI
        widths = s[L:, :] - s[: n_samples - L, :]
        i = expand_dims(widths.argmin(axis=0), axis=0)
        hdi[0, :] = take_along_axis(s, i, 0).squeeze()
        hdi[1, :] = take_along_axis(s, i + L, 0).squeeze()
    else:
        hdi[0, :] = s[0, :]
        hdi[1, :] = s[-1, :]
    return hdi.squeeze()


class DoubleIntervalLength:
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

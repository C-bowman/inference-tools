from numpy import asarray, arange, append, ndarray
from numpy import cumsum, finfo, histogram, maximum, unique
from numpy import pi, prod, sqrt, exp, log2, ceil
from scipy.fft import dct, idct
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simpson

from inference.pdf.base import DensityEstimator


class DiffusionKDE(DensityEstimator):
    """
    Construct a DiffusionKDE object, which can be called as a function to
    return the estimated PDF of the given sample.

    DiffusionKDE uses the diffusion-based kernel density estimation method
    of Botev et al. (2010) to estimate the PDF associated with a given sample.

    :param sample: \
        1D array of samples from which to estimate the probability distribution.

    :param limits: \
        Lower and upper bounds of the interval on which the density estimate
        is constructed, given as a tuple ``(lower, upper)``. If not specified,
        the interval is set to ``(min - range/2, max + range/2)``.
    """

    def __init__(self, sample: ndarray, limits: tuple[float, float] = None):
        self.sample = asarray(sample, dtype=float).ravel()
        _, self._density, self._x_axis, cdf_values = self._kde(
            self.sample, limits=limits
        )

        self.pdf_spline = InterpolatedUnivariateSpline(
            self._x_axis, self._density, k=3, ext=1
        )

        self.cdf_spline = InterpolatedUnivariateSpline(
            self._x_axis, cdf_values, k=3, ext=3
        )

        max_ind = self._density.argmax()
        mode_bounds = (
            self._x_axis[max(max_ind - 2, 0)],
            self._x_axis[min(max_ind + 2, len(self._x_axis) - 1)],
        )
        self.mode = minimize_scalar(
            lambda x: -self.pdf_spline(x), bounds=mode_bounds, method="bounded"
        ).x

    def __call__(self, x: ndarray) -> ndarray:
        """
        Evaluate the estimate of the probability distribution function (PDF)
        at the given parameter values.

        :param x: \
            Axis location(s) at which to evaluate the estimate.

        :return: \
            Values of the PDF estimate at the specified locations.
        """
        return self.pdf_spline(x)

    def cdf(self, x: ndarray) -> ndarray:
        """
        Evaluate the estimate of the cumulative distribution function (CDF)
        at the given parameter values.

        :param x: \
            Axis location(s) at which to evaluate the estimate.

        :return: \
            Values of the CDF estimate at the specified locations.
        """
        return self.cdf_spline(x)

    def moments(self) -> tuple:
        """
        Calculate the mean, variance, skewness and excess kurtosis of the estimated PDF.

        :return: mean, variance, skewness, ex-kurtosis

        Note that these quantities are calculated directly from the estimated PDF, and
        not from the sample values.
        """

        mu = simpson(self._density * self._x_axis, x=self._x_axis)
        dx = self._x_axis - mu
        I = self._density * dx**2
        var = simpson(I, x=self._x_axis)
        I *= dx
        skw = simpson(I, x=self._x_axis) / var**1.5
        I *= dx
        kur = (simpson(I, x=self._x_axis) / var**2) - 3.0
        return mu, var, skw, kur

    @staticmethod
    def _kde(data: ndarray, n=2**14, limits: tuple[float, float] = None):
        """
        Reliable and extremely fast kernel density estimator for one-dimensional data.
        Gaussian kernel is assumed and the bandwidth is chosen automatically.

        :param data: \
            A vector of data from which the density estimate is constructed.

        :param int n: \
            The number of mesh points used in the uniform discretization. Must be
            a power of two; if not, it is rounded up to the next power of two.
            The default value is ``2**14``.

        :param limits: \
            Lower and upper bounds of the interval on which the density estimate
            is constructed, given as a tuple ``(lower, upper)``. If not specified,
            defaults to ``(min(data) - range/2, max(data) + range/2)``.

        :return: \
            A tuple ``(bandwidth, density, xmesh, cdf)`` where *bandwidth* is the
            optimal bandwidth, *density* is an array of PDF values at the grid
            points, *xmesh* is the grid, and *cdf* is an array of CDF values.

        Reference: Kernel density estimation via diffusion.
        Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
        Annals of Statistics, Volume 38, Number 5, pages 2916-2957.
        """
        data = asarray(data, dtype=float).ravel()

        # round up n to the next power of 2
        n = 2 ** int(ceil(log2(n)))

        if limits is None:
            data_min = data.min()
            data_max = data.max()
            padding = (data_max - data_min) * 0.5
            limits = (data_min - padding, data_max + padding)

        lwr_limit, upr_limit = limits

        # set up the grid over which the density estimate is computed
        R = upr_limit - lwr_limit
        dx = R / (n - 1)
        xmesh = lwr_limit + arange(0, R + dx, dx)[:n]
        n_unique = len(unique(data))

        # bin the data uniformly using the grid
        initial_data, _ = histogram(data, bins=append(xmesh, xmesh[-1] + dx))
        initial_data = initial_data.astype(float)
        initial_data = initial_data / initial_data.sum()

        # discrete cosine transform of initial data
        a = dct(initial_data, type=2)

        # now compute the optimal bandwidth^2 using the referenced method
        I = arange(1, n, dtype=float) ** 2
        a2 = (a[1:] / 2.0) ** 2

        # use root finding to solve the equation t = zeta * gamma^[5](t)
        t_star = DiffusionKDE._find_root(
            f=lambda t: DiffusionKDE._fixed_point(t, n_unique, I, a2), n_unique=n_unique
        )

        # smooth the discrete cosine transform of initial data using t_star
        a_t = a * exp(-arange(n, dtype=float) ** 2 * pi**2 * t_star / 2.0)

        # now apply the inverse discrete cosine transform
        density = n * idct(a_t, type=2) / R

        # take the rescaling of the data into account
        bandwidth = sqrt(t_star) * R

        # remove negatives due to round-off error
        density = maximum(density, finfo(float).eps)

        # cdf estimation
        f = 2 * pi**2 * (I * a2 * exp(-I * pi**2 * t_star)).sum()
        t_cdf = (sqrt(pi) * f * n_unique) ** (-2.0 / 3.0)
        a_cdf = a * exp(-arange(n, dtype=float) ** 2 * pi**2 * t_cdf / 2.0)
        cdf = cumsum(n * idct(a_cdf, type=2)) * (dx / R)

        return bandwidth, density, xmesh, cdf

    @staticmethod
    def _fixed_point(t, N, I, a2) -> float:
        """Implements the function t - zeta * gamma^[l](t)."""
        l = 7
        f = 2 * pi ** (2 * l) * (I**l * a2 * exp(-I * pi**2 * t)).sum()
        for s in range(l - 1, 1, -1):
            K0 = prod(arange(1, 2 * s, 2)) / sqrt(2 * pi)
            const = (1 + 0.5 ** (s + 0.5)) / 3.0
            time = (2 * const * K0 / N / f) ** (2.0 / (3 + 2 * s))
            f = 2 * pi ** (2 * s) * (I**s * a2 * exp(-I * pi**2 * time)).sum()
        return t - (2 * N * sqrt(pi) * f) ** (-2.0 / 5.0)

    @staticmethod
    def _find_root(f: callable, n_unique: int) -> float:
        """
        Find the smallest root of the function *f* on the interval ``[0, 0.1]``.

        Uses progressively wider search intervals with Brent's method, falling
        back to bounded minimisation of ``|f|`` if no sign change is found.

        :param f: \
            Scalar function whose root is sought. Must accept a single float.

        :param int n_unique: \
            Number of unique data points. Used to set the initial search
            interval — larger datasets permit a tighter initial bracket.

        :return: \
            The smallest root of *f*, or the point closest to a root.
        """
        # Clamp n_unique to [50, 1050] so the initial bracket stays in a reasonable range
        n_unique = max(50, min(1050, n_unique))

        # Initial upper bracket: for N=50 this is ~1e-12 (very tight),
        # for N=1050 it grows to ~0.01, reflecting that larger datasets
        # push the optimal bandwidth closer to zero.
        tol = 1e-12 + 1e-5 * (n_unique - 50)

        f_zero = f(0)
        while True:
            f_tol = f(tol)
            # Check for a sign change in [0, tol]
            if f_zero * f_tol < 0:
                return brentq(f=f, a=0, b=tol)
            # No sign change — double the search interval
            tol *= 2
            if tol >= 0.1:
                # Search interval has reached the maximum allowed width (0.1)
                # without finding a sign change, so fall back to minimising
                # |f(x)| over [0, 0.1] to find the point closest to a root.
                result = minimize_scalar(
                    lambda x: abs(f(x)), bounds=(0, 0.1), method="bounded"
                )
                return result.x

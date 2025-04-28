import sys
from time import time
from numpy import array, exp, sqrt, pi, log, ndarray, mean, argmax
from numpy.fft import rfft, irfft
from numpy import divmod as np_divmod
from scipy.special import ndtr as normal_cdf
from scipy.special import ndtri as inv_normal_cdf


class ChainProgressPrinter:
    def __init__(self, display: bool = True, leading_msg: str = None):
        self.lead = "" if leading_msg is None else leading_msg

        if not display:
            self.iterations_initial = self.__no_status
            self.iterations_progress = self.__no_status
            self.iterations_final = self.__no_status
            self.percent_progress = self.__no_status
            self.percent_final = self.__no_status
            self.countdown_progress = self.__no_status
            self.countdown_final = self.__no_status

    def iterations_initial(self, total_itr: int):
        sys.stdout.write("\n")
        sys.stdout.write(f"\r  {self.lead}   [ 0 / {total_itr} iterations completed ]")
        sys.stdout.flush()

    def iterations_progress(self, t_start: float, current_itr: int, total_itr: int):
        dt = time() - t_start
        eta = int(dt * (total_itr / (current_itr + 1) - 1))
        sys.stdout.write(
            f"\r  {self.lead}   [ {current_itr + 1} / {total_itr} iterations completed  |  ETA: {eta} sec ]"
        )
        sys.stdout.flush()

    def iterations_final(self, total_itr: int):
        sys.stdout.write(
            f"\r  {self.lead}   [ {total_itr} / {total_itr} iterations completed ]                  "
        )
        sys.stdout.flush()
        sys.stdout.write("\n")

    def percent_progress(self, t_start: float, current_itr: int, total_itr: int):
        dt = time() - t_start
        pct = int(100 * (current_itr + 1) / total_itr)
        eta = int(dt * (total_itr / (current_itr + 1) - 1))
        sys.stdout.write(
            f"\r  {self.lead}   [ {pct}% complete  |  ETA: {eta} sec ]    "
        )
        sys.stdout.flush()

    def percent_final(self, t_start: float, total_itr: int):
        t_elapsed = int(time() - t_start)
        mins, secs = divmod(t_elapsed, 60)
        hrs, mins = divmod(mins, 60)
        sys.stdout.write(
            f"\r  {self.lead}   [ complete - {total_itr} steps taken in {hrs}:{mins:02d}:{secs:02d} ]      "
        )
        sys.stdout.flush()
        sys.stdout.write("\n")

    def countdown_progress(self, t_end, steps_taken):
        seconds_remaining = int(t_end - time())
        mins, secs = divmod(seconds_remaining, 60)
        hrs, mins = divmod(mins, 60)
        sys.stdout.write(
            f"\r  {self.lead}   [ {steps_taken} steps taken, time remaining: {hrs}:{mins:02d}:{secs:02d} ]    "
        )
        sys.stdout.flush()

    def countdown_final(self, run_time, steps_taken):
        mins, secs = divmod(int(run_time), 60)
        hrs, mins = divmod(mins, 60)
        sys.stdout.write(
            f"\r  {self.lead}   [ complete - {steps_taken} steps taken in {hrs}:{mins:02d}:{secs:02d} ]      "
        )
        sys.stdout.flush()
        sys.stdout.write("\n")

    @staticmethod
    def __no_status(*args):
        pass


def effective_sample_size(x: ndarray) -> int:
    # get the autocorrelation
    f = irfft(abs(rfft(x - mean(x))) ** 2)
    # remove reflected 2nd half
    f = f[: len(f) // 2]
    # check that the first value is not negative
    if f[0] < 0.0:
        raise ValueError("First element of the autocorrelation is negative")
    # cut to first negative value
    f = f[: argmax(f < 0.0)]
    # sum and normalise
    thin_factor = f.sum() / f[0]
    return int(len(x) / thin_factor)


class Bounds:
    def __init__(self, lower: ndarray, upper: ndarray, error_source="Bounds"):
        self.lower = lower if isinstance(lower, ndarray) else array(lower).squeeze()
        self.upper = upper if isinstance(upper, ndarray) else array(upper).squeeze()

        if self.lower.ndim > 1 or self.upper.ndim > 1:
            raise ValueError(
                f"""\n
                \r[ {error_source} error ]
                \r>> Lower and upper bounds must be one-dimensional arrays, but
                \r>> instead have dimensions {self.lower.ndim} and {self.upper.ndim} respectively.
                """
            )

        if self.lower.size != self.upper.size:
            raise ValueError(
                f"""\n
                \r[ {error_source} error ]
                \r>> Lower and upper bounds must be arrays of equal size, but
                \r>> instead have sizes {self.lower.size} and {self.upper.size} respectively.
                """
            )

        if (self.lower >= self.upper).any():
            raise ValueError(
                f"""\n
                \r[ {error_source} error ]
                \r>> All given upper bounds must be larger than the corresponding lower bounds.
                """
            )

        self.width = self.upper - self.lower
        self.n_bounds = self.width.size

    def validate_start_point(self, start: ndarray, error_source="Bounds"):
        if self.n_bounds != start.size:
            raise ValueError(
                f"""\n
                \r[ {error_source} error ]
                \r>> The number of parameters ({start.size}) does not
                \r>> match the given number of bounds ({self.n_bounds}).
                """
            )

        if not self.inside(start):
            raise ValueError(
                f"""\n
                \r[ {error_source} error ]
                \r>> Starting location for the chain is outside specified bounds.
                """
            )

    def reflect(self, theta: ndarray) -> ndarray:
        q, rem = np_divmod(theta - self.lower, self.width)
        n = q % 2
        return self.lower + (1 - 2 * n) * rem + n * self.width

    def reflect_momenta(self, theta: ndarray) -> tuple[ndarray, ndarray]:
        q, rem = np_divmod(theta - self.lower, self.width)
        n = q % 2
        reflection = 1 - 2 * n
        return self.lower + reflection * rem + n * self.width, reflection

    def inside(self, theta: ndarray) -> bool:
        return ((theta >= self.lower) & (theta <= self.upper)).all()


class BoundingTransform:
    """
    Construct a transformed version of a given posterior distribution which, when
    sampled from without bounds, is equivalent to sampling from a bounded version
    of the posterior.

    :param callable posterior: \
        A function which takes the vector of model parameters as a ``numpy.ndarray``,
        and returns the posterior log-probability.

    :param bounds: \
        An instance of the ``inference.mcmc.Bounds`` class, or a sequence of two
        ``numpy.ndarray`` specifying the lower and upper bounds for the parameters
        in the form ``(lower_bounds, upper_bounds)``.

    :keyword callable gradient: \
        A function which returns the gradient of the log-posterior probability density
        with respect to the model parameters.
    """

    def __init__(
        self,
        posterior: callable,
        bounds: Bounds,
        gradient: callable = None,
    ):

        if not isinstance(bounds, Bounds):
            bounds = Bounds(
                lower=bounds[0], upper=bounds[1], error_source="BoundingTransform"
            )

        self.func = posterior
        self.grad_func = gradient
        self.width = bounds.upper - bounds.lower
        self.lwr = bounds.lower
        self.grad_norm = self.width / sqrt(2 * pi)
        self.norm = log(self.grad_norm).sum()

    def __call__(self, theta: ndarray) -> float:
        """
        Calculate the log-probability of the transformed distribution.

        :param theta: \
            Parameter values for the unbounded, transformed distribution as a
            ``numpy.ndarray``.

        :return: \
            log-probability of the transformed distribution for the given parameters.

        """
        scale_factor = 0.5 * (theta**2).sum() + self.norm
        params = self.lwr + self.width * normal_cdf(theta)
        return self.func(params) - scale_factor

    def gradient(self, theta: ndarray) -> ndarray:
        """
        Calculate the gradient of the log-probability of the transformed distribution
        with respect to its parameters.

        :param theta: \
            Parameter values for the unbounded, transformed distribution as a
            ``numpy.ndarray``.

        :return: \
            The gradient of the log-probability of the transformed distribution
            for the given parameters.

        """
        grad_scale = self.grad_norm * exp(-0.5 * theta**2)
        params = self.lwr + self.width * normal_cdf(theta)
        return self.grad_func(params) * grad_scale - theta

    def transform_to_posterior(self, unbounded_parameters: ndarray) -> ndarray:
        """
        Converts unbounded parameter values for the transformed distribution
        to bounded parameter values for the original posterior distribution.

        :param unbounded_parameters: \
            Unbounded parameter values for the transformed distribution as a
            ``numpy.ndarray``. To convert many samples at once, parameters
             can be passed as a 2D array of shape ``(n_samples, n_parameters)``.

        :return: \
            Corresponding bounded parameter values for the posterior distribution.
        """
        return self.lwr + self.width * normal_cdf(unbounded_parameters)

    def transform_to_unbounded(self, posterior_parameters: ndarray) -> ndarray:
        """
        Converts bounded parameter values for the original posterior distribution
        to unbounded parameter values for the transformed distribution.

        :param posterior_parameters: \
            Bounded parameter values for the posterior distribution as a
            ``numpy.ndarray``. To convert many samples at once, parameters
             can be passed as a 2D array of shape ``(n_samples, n_parameters)``.

        :return: \
            Corresponding unbounded parameter values for the transformed distribution.
        """
        z = (posterior_parameters - self.lwr) / self.width
        return inv_normal_cdf(z.clip(min=1e-5, max=1 - 1e-5))

from typing import Sequence
from time import time
from warnings import warn
import matplotlib.pyplot as plt

from numpy import array, ndarray, linspace, concatenate, savez, load
from numpy import sqrt, var, cov, diag, isfinite, triu, exp, log, median
from numpy.random import random, randint

from inference.mcmc.utilities import Bounds, ChainProgressPrinter
from inference.plotting import matrix_plot, trace_plot


class EnsembleSampler:
    """
    ``EnsembleSampler`` is an implementation of the affine-invariant ensemble sampler
    proposed by Goodman & Weare. This algorithm is based on an 'ensemble' of points
    in the parameter space referred to as 'walkers'. Proposed updates to the position
    of each walker are generated based on the positions of the other walkers in the
    ensemble in such a way that the performance of the algorithm is unaffected by
    affine-transformations of the parameter space.

    :param callable posterior: \
        A callable which takes a ``numpy.ndarray`` of the model parameters as
        its only argument, and returns the posterior log-probability.

    :param starting_positions: \
        The starting positions of each walker as a 2D ``numpy.ndarray`` with shape
        ``(n_walkers, n_parameters)``.

    :param float alpha: \
        Parameter controlling the width of the distribution of stretch-move
        jump distances. A larger value of ``alpha`` results in a larger
        average jump distance, and therefore a lower jump acceptance rate.
        ``alpha`` must be greater than 1, as the jump distance becomes
        zero when ``alpha = 1``.

    :param parameter_boundaries: \
        A list of length-2 tuples specifying the lower and upper bounds to be set on
        each parameter, in the form (lower, upper).

    :param bool display_progress: \
        If set as ``True``, a message is displayed during sampling
        showing the current progress and an estimated time until completion.
    """

    def __init__(
        self,
        posterior: callable,
        starting_positions: ndarray,
        alpha=2.0,
        parameter_boundaries: Sequence = None,
        display_progress=True,
    ):
        self.posterior = posterior

        if starting_positions is not None:
            # store core data
            self.theta = self.__validate_starting_positions(starting_positions)
            self.n_walkers, self.n_params = starting_positions.shape
            self.probs = array([self.posterior(t) for t in self.theta])

            # storage for diagnostic information
            self.n_iterations = 0
            self.total_proposals = [[] for _ in range(self.n_walkers)]
            self.failed_updates = []

        if parameter_boundaries is not None:
            if len(parameter_boundaries) == self.n_params:
                self.bounds = Bounds(
                    lower=array([k[0] for k in parameter_boundaries]),
                    upper=array([k[1] for k in parameter_boundaries]),
                    error_source="EnsembleSampler",
                )
                self.process_proposal = self.bounds.reflect
            else:
                raise ValueError(
                    """
                    [ EnsembleSampler error ]
                    >> The number of given lower/upper bounds pairs does not match
                    >> the number of model parameters.
                    """
                )
        else:
            self.process_proposal = self.pass_through
            self.bounds = None

        # proposal settings
        if not alpha > 1.0:
            raise ValueError(
                """
                [ EnsembleSampler error ]
                >> The given value of the 'alpha' parameter must be greater than 1.
                """
            )
        self.alpha = alpha
        # uniform sampling in 'x' where z = 0.5*x**2 yields the correct PDF for z
        self.x_lwr = sqrt(2.0 / self.alpha)
        self.x_width = sqrt(2.0 * self.alpha) - self.x_lwr

        self.max_attempts = 100
        self.sample = None
        self.sample_probs = None
        self.display_progress = display_progress
        self.ProgressPrinter = ChainProgressPrinter(
            display=self.display_progress, leading_msg="EnsembleSampler:"
        )

    @staticmethod
    def __validate_starting_positions(positions: ndarray):
        if not isinstance(positions, ndarray):
            raise ValueError(
                f"""
                [ EnsembleSampler error ]
                >> 'starting_positions' should be a numpy.ndarray, but instead has type:
                >> {type(positions)}
                """
            )

        theta = (
            positions.reshape([positions.size, 1]) if positions.ndim == 1 else positions
        )

        if theta.ndim != 2 or theta.shape[0] < (theta.shape[1] + 1):
            raise ValueError(
                f"""
                [ EnsembleSampler error ]
                >> 'starting_positions' should be a numpy.ndarray with shape
                >> (n_walkers, n_parameters), where n_walkers >= n_parameters + 1.
                >> Instead, the given array has shape {positions.shape}.
                """
            )

        if not isfinite(theta).all():
            raise ValueError(
                """
                [ EnsembleSampler error ]
                >> The given 'starting_positions' array contains at least one
                >> value which is non-finite.
                """
            )

        if theta.shape[1] == 1:
            # only need to check the variance for the one-parameter case
            if var(theta) == 0:
                raise ValueError(
                    """
                    [ EnsembleSampler error ]
                    >> The values given in 'starting_positions' have zero variance,
                    >> and therefore the walkers are unable to move.
                    """
                )
        else:
            covar = cov(theta.T)
            std_dev = sqrt(diag(covar))  # get the standard devs
            if (std_dev == 0).any():
                raise ValueError(
                    """
                    [ EnsembleSampler error ]
                    >> For one or more variables, The values given in 'starting_positions' 
                    >> have zero variance, and therefore the walkers are unable to move
                    >> in those variables.
                    """
                )
            # now check if any pairs of variables are approximately co-linear
            correlation = covar / (std_dev[:, None] * std_dev[None, :])
            if (abs(triu(correlation, k=1)) > 0.999).any():
                raise ValueError(
                    """
                    [ EnsembleSampler error ]
                    >> The values given in 'starting_positions' are approximately
                    >> co-linear for one or more pair of variables. This will
                    >> prevent the walkers from moving properly in those variables.
                    """
                )
        return theta

    def __proposal(self, i: int):
        # randomly select walker that isn't 'i'
        j = (randint(low=1, high=self.n_walkers) + i) % self.n_walkers
        # sample the stretch distance
        z = 0.5 * (self.x_lwr + self.x_width * random()) ** 2
        prop = self.process_proposal(
            self.theta[i, :] + z * (self.theta[j, :] - self.theta[i, :])
        )
        return prop, z

    def __advance_walker(self, i: int):
        for attempts in range(1, self.max_attempts + 1):
            Y, z = self.__proposal(i)
            p = self.posterior(Y)
            q = exp((self.n_params - 1) * log(z) + p - self.probs[i])
            if random() <= q:
                self.theta[i, :] = Y
                self.probs[i] = p
                self.total_proposals[i].append(attempts)
                break
        else:
            self.total_proposals[i].append(self.max_attempts)
            self.failed_updates[-1] += 1

    def __advance_all(self):
        self.failed_updates.append(0)
        for i in range(self.n_walkers):
            self.__advance_walker(i)
        self.n_iterations += 1

    def advance(self, iterations: int):
        """
        Advance the ensemble sampler a chosen number of iterations.

        :param int iterations: \
            The number of sets of walker positions which will be stored as samples.
            The total number of samples generated is therefore ``iterations`` times
            the number of walkers.
        """
        t_start = time()
        self.ProgressPrinter.iterations_initial(iterations)

        sample_arrays = [] if self.sample is None else [self.sample]
        prob_arrays = [] if self.sample_probs is None else [self.sample_probs]
        for k in range(iterations):
            self.__advance_all()
            sample_arrays.append(self.theta.copy())
            prob_arrays.append(self.probs.copy())

            # display the progress status message
            self.ProgressPrinter.iterations_progress(t_start, k, iterations)

        # display completion message
        self.ProgressPrinter.iterations_final(iterations)
        self.sample = concatenate(sample_arrays)
        self.sample_probs = concatenate(prob_arrays)

    @staticmethod
    def pass_through(prop):
        return prop

    def plot_diagnostics(self):
        """
        Plot the acceptance rate and log-probability of each walker
        as a function of the number of iterations.
        """
        x = linspace(1, self.n_iterations, self.n_iterations)
        rates = x / array(self.total_proposals).cumsum(axis=1)
        avg_rate = rates.mean(axis=0)

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121)
        alpha = max(0.01, min(1, 20.0 / float(self.n_walkers)))
        for i in range(self.n_walkers):
            ax1.plot(x, rates[i, :], lw=0.5, c="C0", alpha=alpha)
        ax1.plot(x, avg_rate, lw=2, c="red", label="mean rate of all walkers")
        ax1.set_ylim([0, 1])
        ax1.grid()
        ax1.legend()
        ax1.set_title("walker acceptance rates")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("average acceptance rate per walker")

        del rates, avg_rate

        itr_probs = self.sample_probs.reshape([self.n_iterations, self.n_walkers])
        lowest_prob = itr_probs[self.n_iterations // 2 :, :].min()

        ax2 = fig.add_subplot(122)
        ax2.plot(x, itr_probs, marker=".", ls="none", c="C0", alpha=0.05)
        ax2.plot(
            x,
            median(itr_probs, axis=1),
            c="red",
            lw=2,
            label="median walker log-probability",
        )
        ax2.set_ylim([lowest_prob, self.sample_probs.max() * 1.1 - 0.1 * lowest_prob])
        ax2.grid()
        ax2.legend()
        ax2.set_title("walker log-probabilities")
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("walker log-probability")

        plt.tight_layout()
        plt.show()

    def matrix_plot(self, params=None, burn=0, thin=1, **kwargs):
        """
        Construct a 'matrix plot' of the parameters (or a subset) which displays
        all 1D and 2D marginal distributions. See the documentation of
        ``inference.plotting.matrix_plot`` for a description of other allowed
        keyword arguments.

        :param params: \
            A list of integers specifying the indices of parameters which are to
            be plotted.

        :param int burn: \
            Sets the number of samples from the beginning of the chain which are
            ignored when generating the plot.

        :param int thin: \
            Sets the factor by which the sample is 'thinned' before generating the
            plot. If ``thin`` is set to some integer value *m*, then only every
            *m*'th sample is used, and the remainder are ignored.
        """
        if self.sample is not None:
            params = range(self.n_params) if params is None else params
            samples = [self.sample[burn::thin, p] for p in params]
            matrix_plot(samples=samples, **kwargs)
        else:
            warn(
                """
                [ EnsembleSampler warning ]
                >> Cannot generate matrix plot as no samples have been produced.
                """
            )

    def trace_plot(self, params=None, burn=0, thin=1, **kwargs):
        """
        Construct a 'trace plot' of the parameters (or a subset) which displays
        the value of the parameters as a function of step number in the chain.
        See the documentation of ``inference.plotting.trace_plot`` for a description
        of other allowed keyword arguments.

        :param params: \
            A list of integers specifying the indices of parameters which are to
            be plotted.

        :param int burn: \
            Sets the number of samples from the beginning of the chain which are
            ignored when generating the plot.

        :param int thin: \
            Sets the factor by which the sample is 'thinned' before generating the
            plot. If ``thin`` is set to some integer value *m*, then only every
            *m*'th sample is used, and the remainder are ignored.
        """
        if self.sample is not None:
            params = range(self.n_params) if params is None else params
            samples = [self.sample[burn::thin, p] for p in params]
            trace_plot(samples=samples, **kwargs)
        else:
            warn(
                """
                [ EnsembleSampler warning ]
                >> Cannot generate trace plot as no samples have been produced.
                """
            )

    def mode(self) -> ndarray:
        """
        Return the sample with the current highest posterior probability.

        :return: \
            The model parameters corresponding to the highest observed
            posterior probability as a ``numpy.ndarray``.
        """
        return self.sample[self.sample_probs.argmax(), :]

    def get_parameter(self, index: int, burn=0, thin=1) -> ndarray:
        """
        Return sample values for a chosen parameter.

        :param int index: \
            Index of the parameter for which samples are to be returned.

        :param int burn: \
            Number of steps from the start of the chain which are ignored.

        :param int thin: \
            Sets the factor by which the sample is 'thinned' before returning
            the parameter values. If ``thin`` is set to some integer value *m*,
            then only every *m*'th sample is used, and the remainder are ignored.

        :return: \
            Samples for the chosen parameter as a ``numpy.ndarray``.
        """
        return self.sample[burn::thin, index]

    def get_probabilities(self, burn=0, thin=1) -> ndarray:
        """
        Return the log-probability values for each step in the chain.

        :param int burn: \
            Number of steps from the start of the chain which are ignored.

        :param int thin: \
            Sets the factor by which the sample is 'thinned' before returning
            corresponding log-probabilities. If ``thin`` is set to some integer
            value *m*, then only every *m*'th sample is used, and the remainder
            are ignored.

        :return: \
            Log-probability values as a ``numpy.ndarray``.
        """
        return self.sample_probs[burn::thin]

    def get_sample(self, burn=0, thin=1) -> ndarray:
        """
        Return the sample as a 2D ``numpy.ndarray``.

        :param int burn: \
            Number of steps from the start of the chain which are ignored.

        :param int thin: \
            Sets the factor by which the sample is 'thinned' before being returned.
            If ``thin`` is set to some integer value *m*, then only every *m*'th
            sample is used, and the remainder are ignored.

        :return: \
            The sample as a ``numpy.ndarray`` of shape ``(n_samples, n_parameters)``.
        """
        return self.sample[burn::thin, :]

    def save(self, filename):
        D = {
            "theta": self.theta,
            "n_params": self.n_params,
            "n_walkers": self.n_walkers,
            "probs": self.probs,
            "n_iterations": self.n_iterations,
            "total_proposals": array(self.total_proposals),
            "alpha": self.alpha,
            "max_attempts": self.max_attempts,
            "display_progress": self.display_progress,
        }

        if self.bounds is not None:
            D["lower"] = self.bounds.lower
            D["upper"] = self.bounds.upper

        if self.sample is not None:
            D["sample"] = self.sample
            D["sample_probs"] = self.sample_probs

        savez(filename, **D)

    @classmethod
    def load(cls, filename: str, posterior=None):
        D = load(filename)

        if "lower" in D and "upper" in D:
            bounds = [(l, u) for l, u in zip(D["lower"], D["upper"])]
        else:
            bounds = None

        sampler = cls(
            posterior=posterior,
            starting_positions=None,
            parameter_boundaries=bounds,
            alpha=D["alpha"],
            display_progress=bool(D["display_progress"]),
        )

        sampler.theta = D["theta"]
        sampler.n_params = int(D["n_params"])
        sampler.n_walkers = int(D["n_walkers"])
        sampler.probs = D["probs"]
        sampler.n_iterations = int(D["n_iterations"])
        sampler.total_proposals = [list(v) for v in D["total_proposals"]]
        sampler.max_attempts = int(D["max_attempts"])

        if "sample" in D:
            sampler.sample = D["sample"]
            sampler.sample_probs = D["sample_probs"]

        return sampler

from abc import ABC, abstractmethod
from copy import copy
from time import time
from numpy import ndarray
from numpy.random import permutation

from inference.pdf import GaussianKDE, UnimodalPdf, DensityEstimator
from inference.plotting import matrix_plot, trace_plot
from inference.mcmc.utilities import ChainProgressPrinter


class MarkovChain(ABC):
    chain_length: int
    n_parameters: int
    ProgressPrinter: ChainProgressPrinter

    @abstractmethod
    def get_parameter(self, index: int, burn: int = 1, thin: int = 1) -> ndarray:
        pass

    @abstractmethod
    def get_probabilities(self, burn: int = 1, thin: int = 1) -> ndarray:
        pass

    @abstractmethod
    def get_sample(self, burn: int = 1, thin: int = 1) -> ndarray:
        pass

    def advance(self, m: int):
        """
        Advances the chain by taking ``m`` new steps.

        :param int m: Number of steps the chain will advance.
        """
        k = 100  # divide chain steps into k groups to track progress
        t_start = time()
        for j in range(k):
            [self.take_step() for _ in range(m // k)]
            self.ProgressPrinter.percent_progress(t_start, j, k)

        # cleanup
        if m % k != 0:
            [self.take_step() for _ in range(m % k)]
        self.ProgressPrinter.percent_final(t_start, m)

    def run_for(self, minutes=0, hours=0, days=0):
        """
        Advances the chain for a chosen amount of computation time.

        :param int minutes: number of minutes for which to run the chain.
        :param int hours: number of hours for which to run the chain.
        :param int days: number of days for which to run the chain.
        """
        update_interval = 20  # small initial guess for the update interval
        start_length = copy(self.chain_length)

        # first find the runtime in seconds:
        run_time = ((days * 24.0 + hours) * 60.0 + minutes) * 60.0
        start_time = time()
        current_time = start_time
        end_time = start_time + run_time

        while current_time < end_time:
            for i in range(update_interval):
                self.take_step()
            # set the interval such that updates are roughly once per second
            steps_taken = self.chain_length - start_length
            current_time = time()
            update_interval = int(steps_taken / (current_time - start_time))
            self.ProgressPrinter.countdown_progress(end_time, steps_taken)
        self.ProgressPrinter.countdown_final(run_time, steps_taken)

    def get_marginal(
        self, index: int, burn: int = 1, thin: int = 1, unimodal=False
    ) -> DensityEstimator:
        """
        Estimate the 1D marginal distribution of a chosen parameter.

        :param int index: \
            Index of the parameter for which the marginal distribution is to be estimated.

        :param int burn: \
            Number of samples to discard from the start of the chain.

        :param int thin: \
            Rather than using every sample which is not discarded as part of the burn-in,
            every *m*'th sample is used for a specified integer *m*.
            
        :param bool unimodal: \
            Selects the type of density estimation to be used. The default value is False,
            which causes a GaussianKDE object to be returned. If however the marginal
            distribution being estimated is known to be unimodal, setting ``unimodal = True``
            will result in the ``UnimodalPdf`` class being used to estimate the density.

        :return: \
            An instance of a ``DensityEstimator`` from the ``inference.pdf`` module which
            represents the 1D marginal distribution of the chosen parameter. If the
            ``unimodal`` argument is ``False`` a ``GaussianKDE`` instance is returned,
            else if ``True`` a ``UnimodalPdf`` instance is returned.

        """
        if unimodal:
            return UnimodalPdf(self.get_parameter(index, burn=burn, thin=thin))
        else:
            return GaussianKDE(self.get_parameter(index, burn=burn, thin=thin))

    def get_interval(self, interval=0.95, burn: int = 1, thin: int = 1, samples=None):
        """
        Return the samples from the chain which lie inside a chosen highest-density interval.

        :param float interval: \
            Total probability of the desired interval. For example, if interval = 0.95, then
            the samples corresponding to the top 95% of posterior probability values are returned.

        :param int burn: \
            Number of samples to discard from the start of the chain.

        :param int thin: \
            Instead of returning every sample which is not discarded as part of the burn-in,
            every *m*'th sample is returned for a specified integer *m*.

        :param int samples: \
            The number of samples that should be returned from the requested interval. Note
            that specifying ``samples`` overrides the value of ``thin``.

        :return: \
            List containing sample points stored as tuples, and a corresponding list of
            log-probability values.
        """

        # get the sorting indices for the probabilities
        probs = self.get_probabilities(burn=burn)
        if samples is not None:
            thin = max(probs.size // samples, 1)

        sample = self.get_sample(burn=burn, thin=thin)
        probs = probs[::thin]

        sorter = probs.argsort()
        # sort the sample by probability
        sample = sample[sorter, :]
        probs = probs[sorter]
        # trim lowest-probability samples
        cutoff = int(probs.size * (1 - interval))
        sample = sample[cutoff:, :]
        probs = probs[cutoff:]

        if samples is not None:
            # we may need to trim some extra samples to meet the requested number,
            # but as they arranged in order of increasing probability, we must remove
            # elements at random in order not to introduce bias.
            n_trim = probs.size - samples
            if n_trim > 0:
                subsample = permutation(probs.size)[n_trim:].sort()
                sample = sample[subsample, :]
                probs = probs[subsample]

        return sample, probs

    def matrix_plot(
        self, params: list[int] = None, burn: int = 0, thin: int = 1, **kwargs
    ):
        """
        Construct a 'matrix plot' of the parameters (or a subset) which displays
        all 1D and 2D marginal distributions. See the documentation of
        ``inference.plotting.matrix_plot`` for a description of other allowed
        keyword arguments.

        :param params: \
            A list of integers specifying the indices of parameters which are to
            be plotted. If not specified, all parameters are plotted.

        :param int burn: \
            Sets the number of samples from the beginning of the chain which are
            ignored when generating the plot.

        :param int thin: \
            Sets the factor by which the sample is 'thinned' before generating the
            plot. If ``thin`` is set to some integer value *m*, then only every
            *m*'th sample is used, and the remainder are ignored.
        """
        self.__plot_checks(burn, thin, "matrix")
        params = params if params is not None else range(self.n_parameters)
        samples = [self.get_parameter(i, burn=burn, thin=thin) for i in params]
        matrix_plot(samples, **kwargs)

    def trace_plot(
        self, params: list[int] = None, burn: int = 0, thin: int = 1, **kwargs
    ):
        """
        Construct a 'trace plot' of the parameters (or a subset) which displays
        the value of the parameters as a function of step number in the chain.
        See the documentation of ``inference.plotting.trace_plot`` for a description
        of other allowed keyword arguments.

        :param params: \
            A list of integers specifying the indices of parameters which are to
            be plotted. If not specified, all parameters are plotted.

        :param int burn: \
            Sets the number of samples from the beginning of the chain which are
            ignored when generating the plot.

        :param int thin: \
            Sets the factor by which the sample is 'thinned' before generating the
            plot. If ``thin`` is set to some integer value *m*, then only every
            *m*'th sample is used, and the remainder are ignored.
        """
        self.__plot_checks(burn, thin, "trace")
        params = params if params is not None else range(self.n_parameters)
        samples = [self.get_parameter(i, burn=burn, thin=thin) for i in params]
        trace_plot(samples, **kwargs)

    def __plot_checks(self, burn: int, thin: int, plot_type: str):
        if self.chain_length < 2:
            raise ValueError(
                f"""\n
                \r[ {self.__class__.__name__} error ]
                \r>> Cannot generate the {plot_type} plot as no samples have
                \r>> been produced - current chain length is {self.chain_length}.
                """
            )

        reduced_length = max(self.chain_length - burn - 1, 0) // thin + 1
        if reduced_length < 2:
            raise ValueError(
                f"""\n
                \r[ {self.__class__.__name__} error ]
                \r>> The given values of 'burn' and 'thin' leave insufficient
                \r>> samples to generate the {plot_type} plot.
                \r>> Number of samples after burn / thin is {reduced_length}.
                """
            )

    @property
    def burn(self):
        self.__burn_thin_error()

    @burn.setter
    def burn(self, val):
        self.__burn_thin_error()

    @property
    def thin(self):
        self.__burn_thin_error()

    @thin.setter
    def thin(self, val):
        self.__burn_thin_error()

    def __burn_thin_error(self):
        raise AttributeError(
            f"""\n
            \r[ {self.__class__.__name__} error ]
            \r>> The 'burn' and 'thin' instance attributes of inference-tools
            \r>> mcmc samplers were removed in version 0.13.0. Burn and thin
            \r>> values should now be passed explicitly to any methods with
            \r>> 'burn' and 'thin' keyword arguments.
            """
        )

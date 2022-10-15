"""
.. moduleauthor:: Chris Bowman <chris.bowman.physics@gmail.com>
"""

import sys
from warnings import warn
from copy import copy, deepcopy
from multiprocessing import Process, Pipe, Event, Pool
from time import time
from random import choice

import matplotlib.pyplot as plt
from numpy import array, arange, float64, identity, linspace, zeros, concatenate
from numpy import exp, log, mean, sqrt, argmax, diff, dot
from numpy import cov, var, percentile, median, diag, triu
from numpy import isfinite, sort, argsort, savez, savez_compressed, load
from numpy.fft import rfft, irfft
from numpy.random import normal, random, shuffle, seed, randint
from scipy.linalg import eigh

from inference.pdf import UnimodalPdf, GaussianKDE
from inference.plotting import matrix_plot, trace_plot, transition_matrix_plot


class Parameter:
    """
    This class is used by the markov-chain samplers in this module
    to manage data specific to each model parameter which is being
    sampled.

    The class also adjusts the parameter's proposal distribution
    width automatically as the chain advances in order to ensure
    efficient sampling.
    """

    def __init__(self, value=None, sigma=None):
        self.samples = []  # list to store all samples for the parameter
        self.samples.append(value)  # add starting location as first sample
        self.sigma = sigma  # the width parameter for the proposal distribution

        # storage for proposal width adjustment algorithm
        self.avg = 0
        self.var = 0
        self.num = 0
        self.sigma_values = [copy(self.sigma)]  # sigma values after each assessment
        self.sigma_checks = [0.0]  # chain locations at which sigma was assessed
        self.try_count = 0  # counter variable tracking number of proposals
        self.last_update = 0  # chain location where sigma was last updated

        # settings for proposal width adjustment algorithm
        self.target_rate = 0.25  # default of 0.25 is optimal for MH sampling
        self.max_tries = 50  # maximum allowed tries before width is cut in half
        self.chk_int = 100  # interval of steps at which proposal widths are adjusted
        self.growth_factor = 1.75  # factor chk_int grows when width is adjusted
        self.adjust_rate = 0.25

        # properties
        self._non_negative = False
        self.bounded = False
        self.proposal = self.standard_proposal
        self.upper = 0.0
        self.lower = 0.0
        self.width = 0.0

    def set_boundaries(self, lower, upper):
        if lower < upper:
            self.upper = upper
            self.lower = lower
            self.width = upper - lower
            self.proposal = self.boundary_proposal
            self.bounded = True
        else:
            warn("Upper limit must be greater than lower limit")

    def remove_boundaries(self):
        self.proposal = self.standard_proposal
        self.bounded = False
        self.upper = 0.0
        self.lower = 0.0
        self.width = 0.0

    @property
    def non_negative(self):
        return self._non_negative

    @non_negative.setter
    def non_negative(self, value):
        if type(value) is bool:
            self._non_negative = value
            if self._non_negative is True:
                self.proposal = self.abs_proposal
            else:
                self.proposal = self.standard_proposal
        else:
            warn("non_negative must have a boolean value")

    def standard_proposal(self):
        # increment the try count
        self.try_count += 1
        # if tries climb too high, then cut sigma in half
        if self.try_count > self.max_tries:
            self.adjust_sigma(0.25)
        # return the proposed value
        return self.samples[-1] + self.sigma * normal()

    def abs_proposal(self):
        # increment the try count
        self.try_count += 1
        # if tries climb too high, then cut sigma in half
        if self.try_count > self.max_tries:
            self.adjust_sigma(0.25)
        # return the proposed value
        return abs(self.samples[-1] + self.sigma * normal())

    def boundary_proposal(self):
        # increment the try count
        self.try_count += 1
        # if tries climb too high, then cut sigma in half
        if self.try_count > self.max_tries:
            self.adjust_sigma(0.25)
        # generate the proposed value
        prop = self.samples[-1] + self.sigma * normal()

        # we now pass the proposal through a 'reflecting' function where
        # proposals falling outside the boundary are reflected inside
        d = prop - self.lower
        n = (d // self.width) % 2
        if n == 0:
            return self.lower + d % self.width
        else:
            return self.upper - d % self.width

    def submit_accept_prob(self, p):
        self.num += 1
        self.avg += p
        self.var += p * (1 - p)

        if self.num >= self.chk_int:
            self.update_epsilon()

    def update_epsilon(self):
        """
        looks at average tries over recent steps, and adjusts proposal
        widths self.sigma to bring the average towards self.target_tries.
        """
        # normal approximation of poisson binomial distribution
        mu = self.avg / self.num
        std = sqrt(self.var) / self.num

        # now check if the desired success rate is within 2-sigma
        if ~(mu - 2 * std < self.target_rate < mu + 2 * std):
            adj = (log(self.target_rate) / log(mu)) ** self.adjust_rate
            adj = min(adj, 3.0)
            adj = max(adj, 0.1)
            self.adjust_sigma(adj)
        else:  # increase the check interval
            self.chk_int = int((self.growth_factor * self.chk_int) * 0.1) * 10

    def adjust_sigma(self, ratio):
        self.sigma *= ratio
        self.sigma_values.append(copy(self.sigma))
        self.sigma_checks.append(len(self.samples))
        self.avg = 0
        self.var = 0
        self.num = 0

    def add_sample(self, s):
        self.samples.append(s)
        self.try_count = 0

    def get_items(self, param_id):
        i = f"param_{param_id}"
        items = {
            f"{i}samples": self.samples,
            f"{i}sigma": self.sigma,
            f"{i}avg": self.avg,
            f"{i}var": self.var,
            f"{i}num": self.num,
            f"{i}sigma_values": self.sigma_values,
            f"{i}sigma_checks": self.sigma_checks,
            f"{i}try_count": self.try_count,
            f"{i}last_update": self.last_update,
            f"{i}target_rate": self.target_rate,
            f"{i}max_tries": self.max_tries,
            f"{i}chk_int": self.chk_int,
            f"{i}growth_factor": self.growth_factor,
            f"{i}adjust_rate": self.adjust_rate,
            f"{i}_non_negative": self._non_negative,
            f"{i}bounded": self.bounded,
            f"{i}upper": self.upper,
            f"{i}lower": self.lower,
            f"{i}width": self.width,
        }
        return items

    def load_items(self, dictionary, param_id):
        i = "param_" + str(param_id)
        self.samples = list(dictionary[i + "samples"])
        self.sigma = float(dictionary[i + "sigma"])
        self.avg = float(dictionary[i + "avg"])
        self.var = float(dictionary[i + "var"])
        self.num = float(dictionary[i + "num"])
        self.sigma_values = list(dictionary[i + "sigma_values"])
        self.sigma_checks = list(dictionary[i + "sigma_checks"])
        self.try_count = int(dictionary[i + "try_count"])
        self.last_update = int(dictionary[i + "last_update"])
        self.target_rate = float(dictionary[i + "target_rate"])
        self.max_tries = int(dictionary[i + "max_tries"])
        self.chk_int = int(dictionary[i + "chk_int"])
        self.growth_factor = float(dictionary[i + "growth_factor"])
        self.adjust_rate = float(dictionary[i + "adjust_rate"])
        self._non_negative = bool(dictionary[i + "_non_negative"])
        self.bounded = bool(dictionary[i + "bounded"])
        self.upper = float(dictionary[i + "upper"])
        self.lower = float(dictionary[i + "lower"])
        self.width = float(dictionary[i + "width"])

        if self.bounded:
            self.proposal = self.boundary_proposal
        elif self._non_negative:
            self.proposal = self.abs_proposal
        else:
            self.proposal = self.standard_proposal


class MarkovChain:
    """
    Implementation of the metropolis-hastings algorithm using a multivariate-normal
    proposal distribution.

    :param func posterior: \
        A function which takes the vector of model parameters as a ``numpy.ndarray``,
        and returns the posterior log-probability.

    :param start: \
        Vector of model parameters which correspond to the parameter-space coordinates
        at which the chain will start.

    :param widths: \
        Vector of standard deviations which serve as initial guesses for the widths of
        the proposal distribution for each model parameter. If not specified, the
        starting widths will be approximated as 5% of the values in 'start'.

    :param bool display_progress: \
        If set as ``True``, a message is displayed during sampling
        showing the current progress and an estimated time until completion.
    """

    def __init__(
        self,
        posterior=None,
        start=None,
        widths=None,
        temperature=1.0,
        display_progress=True,
    ):

        if start is None:
            start = []

        self.inv_temp = 1.0 / temperature

        if posterior is not None:
            self.posterior = posterior

            # if widths are not specified, take 5% of the starting values (unless they're zero)
            if widths is None:
                widths = [v * 0.05 if v != 0 else 1.0 for v in start]

            # create a list of parameter objects
            self.params = [Parameter(value=v, sigma=s) for v, s in zip(start, widths)]

            # create storage
            self.n = 1  # tracks total length of the chain
            self.L = len(start)  # number of posterior parameters
            self.probs = []  # list of probabilities for all steps

            # add starting point as first step in chain
            if len(self.params) != 0:
                self.probs.append(self.posterior(self.get_last()) * self.inv_temp)

                # check posterior value of chain starting point is finite
                if not isfinite(self.probs[0]):
                    ValueError(
                        """
                        [ MarkovChain error ]
                        >> 'posterior' argument callable returns a non-finite value
                        >> for the starting position given to the 'start' argument.
                        """
                    )

            # add default burn and thin values
            self.burn = 1  # remove the starting position by default
            self.thin = 1  # no thinning by default

            self.display_progress = display_progress
            self.ProgressPrinter = ChainProgressPrinter(
                display=self.display_progress, leading_msg="advancing chain:"
            )

    def take_step(self):
        """
        Draws samples from the proposal distribution until one is
        found which satisfies the metropolis-hastings criteria.
        """
        while True:
            proposal = array([p.proposal() for p in self.params])
            pval = self.posterior(proposal) * self.inv_temp

            if pval > self.probs[-1]:
                break
            else:
                test = random()
                acceptance_prob = exp(pval - self.probs[-1])
                if test < acceptance_prob:
                    break

        for p, v in zip(self.params, proposal):
            p.add_sample(v)

        self.n += 1

    def advance(self, m):
        """
        Advances the chain by taking ``m`` new steps.

        :param int m: Number of steps the chain will advance.
        """
        k = 100  # divide chain steps into k groups to track progress
        t_start = time()
        for j in range(k):
            for i in range(m // k):
                self.take_step()
            self.ProgressPrinter.percent_progress(t_start, j, k)

        # cleanup
        if m % k != 0:
            for i in range(m % k):
                self.take_step()
        self.ProgressPrinter.percent_final(t_start, m)

    def run_for(self, minutes=0, hours=0, days=0):
        """
        Advances the chain for a chosen amount of computation time

        :param int minutes: number of minutes for which to run the chain.
        :param int hours: number of hours for which to run the chain.
        :param int days: number of days for which to run the chain.
        """
        update_interval = 20  # small initial guess for the update interval
        start_length = copy(self.n)

        # first find the runtime in seconds:
        run_time = ((days * 24.0 + hours) * 60.0 + minutes) * 60.0
        start_time = time()
        current_time = start_time
        end_time = start_time + run_time

        while current_time < end_time:
            for i in range(update_interval):
                self.take_step()
            # set the interval such that updates are roughly once per second
            steps_taken = self.n - start_length
            current_time = time()
            update_interval = int(steps_taken / (current_time - start_time))
            self.ProgressPrinter.countdown_progress(end_time, steps_taken)
        self.ProgressPrinter.countdown_final(run_time, steps_taken)

    def get_last(self):
        return array([p.samples[-1] for p in self.params], dtype=float64)

    def replace_last(self, theta):
        for p, t in zip(self.params, theta):
            p.samples[-1] = t

    def get_parameter(self, n, burn=None, thin=None):
        """
        Return sample values for a chosen parameter.

        :param int n: Index of the parameter for which samples are to be returned.

        :param int burn: \
            Number of samples to discard from the start of the chain. If not specified,
            the value of self.burn is used instead.

        :param int thin: \
            Instead of returning every sample which is not discarded as part of the burn-in,
            every *m*'th sample is returned for a specified integer *m*. If not specified,
            the value of self.thin is used instead.

        :return: List of samples for parameter *n*'th parameter.
        """
        burn = burn if burn is not None else self.burn
        thin = thin if thin is not None else self.thin
        return self.params[n].samples[burn::thin]

    def get_probabilities(self, burn=None, thin=None):
        """
        Return log-probability values for each step in the chain

        :param int burn: \
            Number of steps to discard from the start of the chain. If not specified, the
            value of self.burn is used instead.

        :param int thin: \
            Instead of returning every step which is not discarded as part of the burn-in,
            every *m*'th step is returned for a specified integer *m*. If not specified,
            the value of self.thin is used instead.

        :return: List of log-probability values for each step in the chain.
        """
        burn = burn if burn is not None else self.burn
        thin = thin if thin is not None else self.thin
        return self.probs[burn::thin]

    def get_sample(self, burn=None, thin=None):
        """
        Return the sample generated by the chain as a list of tuples

        :param int burn: \
            Number of samples to discard from the start of the chain. If not specified,
            the value of self.burn is used instead.

        :param int thin: \
            Instead of returning every sample which is not discarded as part of the burn-in,
            every *m*'th sample is returned for a specified integer *m*. If not specified,
            the value of self.thin is used instead.

        :return: List containing sample points stored as tuples.
        """
        burn = burn if burn is not None else self.burn
        thin = thin if thin is not None else self.thin
        return list(zip(*[p.samples[burn::thin] for p in self.params]))

    def get_interval(self, interval=0.95, burn=None, thin=None, samples=None):
        """
        Return the samples from the chain which lie inside a chosen highest-density interval.

        :param float interval: \
            Total probability of the desired interval. For example, if interval = 0.95, then
            the samples corresponding to the top 95% of posterior probability values are returned.

        :param int burn: \
            Number of samples to discard from the start of the chain. If not specified, the
            value of self.burn is used instead.

        :param int thin: \
            Instead of returning every sample which is not discarded as part of the burn-in,
            every *m*'th sample is returned for a specified integer *m*. If not specified,
            the value of self.thin is used instead.

        :param int samples: \
            The number of samples that should be returned from the requested interval. Note
            that specifying *samples* overrides the value of *thin*.

        :return: List containing sample points stored as tuples, and a corresponding list of
                 log-probability values
        """
        burn = burn if burn is not None else self.burn

        # get the sorting indices for the probabilities
        probs = array(self.probs[burn:])
        inds = probs.argsort()
        # sort the sample by probability
        arrays = [array(p.samples[burn:])[inds] for p in self.params]
        probs = probs[inds]
        # trim lowest-probability samples
        cutoff = int(len(probs) * (1 - interval))
        arrays = [a[cutoff:] for a in arrays]
        probs = probs[cutoff:]
        # if a specific number of samples is requested we override the thin value
        if samples is not None:
            thin = max(len(probs) // samples, 1)
        elif thin is None:
            thin = self.thin

        # thin the sample
        arrays = [a[::thin] for a in arrays]
        probs = probs[::thin]

        if samples is not None:
            # we may need to trim some extra samples to meet the requested number,
            # but as they arranged in order of increasing probability, we must remove
            # elements at random in order not to introduce bias.
            n_trim = len(probs) - samples
            if n_trim > 0:
                trim = sort(argsort(random(size=len(probs)))[n_trim:])
                arrays = [a[trim] for a in arrays]
                probs = probs[trim]

        return list(zip(*arrays)), probs

    def mode(self):
        """
        Return the sample with the current highest posterior probability.

        :return: List containing parameter values.
        """
        ind = argmax(self.probs)
        return [p.samples[ind] for p in self.params]

    def set_non_negative(self, parameter, flag=True):
        """
        Constrain a particular parameter to have non-negative values.

        :param int parameter: Index of the parameter which is to be set \
                              as non-negative.
        """
        self.params[parameter].non_negative = flag

    def set_boundaries(self, parameter, boundaries, remove=False):
        """
        Constrain the value of a particular parameter to specified boundaries.

        :param int parameter: Index of the parameter for which boundaries \
                              are to be set.

        :param boundaries: Tuple of boundaries in the format (lower_limit, upper_limit)
        """
        if remove:
            self.params[parameter].remove_boundaries()
        else:
            self.params[parameter].set_boundaries(*boundaries)

    def get_marginal(self, n, thin=None, burn=None, unimodal=False):
        """
        Estimate the 1D marginal distribution of a chosen parameter.

        :param int n: \
            Index of the parameter for which the marginal distribution is to be estimated.

        :param int burn: \
            Number of samples to discard from the start of the chain. If not specified,
            the value of self.burn is used instead.

        :param int thin: \
            Rather than using every sample which is not discarded as part of the burn-in,
            every *m*'th sample is used for a specified integer *m*. If not specified, the
            value of self.thin is used instead, which has a default value of 1.

        :param bool unimodal: \
            Selects the type of density estimation to be used. The default value is False,
            which causes a GaussianKDE object to be returned. If however the marginal
            distribution being estimated is known to be unimodal, setting `unimodal = True`
            will result in the UnimodalPdf class being used to estimate the density.

        Returns one of two 'density estimator' objects which can be
        called as functions to return the estimated PDF at any point.
        """
        burn = burn if burn is not None else self.burn
        thin = thin if thin is not None else self.thin

        if unimodal:
            return UnimodalPdf(self.get_parameter(n, burn=burn, thin=thin))
        else:
            return GaussianKDE(self.get_parameter(n, burn=burn, thin=thin))

    def plot_diagnostics(self, show=True, filename=None):
        """
        Plot diagnostic traces that give information on how the chain is progressing.

        Currently this method plots:

        - The posterior log-probability as a function of step number, which is useful
          for checking if the chain has reached a maximum. Any early parts of the chain
          where the probability is rising rapidly should be removed as burn-in.

        - The history of changes to the proposal widths for each parameter. Ideally, the
          proposal widths should converge, and the point in the chain where this occurs
          is often a good choice for the end of the burn-in. For highly-correlated pdfs,
          the proposal widths may never fully converge, but in these cases small fluctuations
          in the width values are acceptable.

        :param bool show: If set to True, the plot is displayed.

        :param str filename: \
            File path to which the diagnostics plot will be saved. If left unspecified the
            plot won't be saved.
        """
        burn = self.estimate_burn_in()
        param_ESS = [
            ESS(array(self.get_parameter(i, burn=burn))) for i in range(self.L)
        ]

        fig = plt.figure(figsize=(12, 9))

        # probability history plot
        ax1 = fig.add_subplot(221)
        step_ax = [i * 1e-3 for i in range(len(self.probs))]
        ax1.plot(step_ax, self.probs, marker=".", ls="none", markersize=3)
        ax1.set_xlabel("chain step number ($10^3$)", fontsize=12)
        ax1.set_ylabel("posterior log-probability", fontsize=12)
        ax1.set_title("Chain log-probability history")
        ylims = [
            min(self.probs[self.n // 2 :]),
            max(self.probs) * 1.1 - 0.1 * min(self.probs[self.n // 2 :]),
        ]
        plt.plot([burn * 1e-3, burn * 1e-3], ylims, c="red", ls="dashed", lw=2)
        ax1.set_ylim(ylims)
        ax1.grid()

        # proposal widths plot
        ax2 = fig.add_subplot(222)
        for p in self.params:
            y = array(p.sigma_values)
            x = array(p.sigma_checks[1:]) * 1e-3
            ax2.plot(x, 1e2 * diff(y) / y[:-1], marker="D", markersize=3)
        ax2.plot([0, self.n * 1e-3], [5, 5], ls="dashed", lw=2, color="black")
        ax2.plot([0, self.n * 1e-3], [-5, -5], ls="dashed", lw=2, color="black")
        ax2.set_xlabel("chain step number ($10^3$)", fontsize=12)
        ax2.set_ylabel("% change in proposal widths", fontsize=12)
        ax2.set_title("Parameter proposal widths adjustment summary")
        ax2.set_ylim([-50, 50])
        ax2.grid()

        # parameter ESS plot
        ax3 = fig.add_subplot(223)
        ax3.bar(range(self.L), param_ESS, color=["C0", "C1", "C2", "C3", "C4"])
        ax3.set_xlabel("parameter", fontsize=12)
        ax3.set_ylabel("effective sample size", fontsize=12)
        ax3.set_title("Parameter effective sample size estimate")
        ax3.set_xticks(range(self.L))

        # summary stats text plot
        ax4 = fig.add_subplot(224)
        gap = 0.1
        h = 0.85
        x1 = 0.5
        x2 = 0.55
        fntsiz = 14

        ax4.text(
            x1, h, "Estimated burn-in:", horizontalalignment="right", fontsize=fntsiz
        )
        ax4.text(
            x2, h, "{:.5G}".format(burn), horizontalalignment="left", fontsize=fntsiz
        )
        h -= gap
        ax4.text(x1, h, "Average ESS:", horizontalalignment="right", fontsize=fntsiz)
        ax4.text(
            x2,
            h,
            "{:.5G}".format(int(mean(param_ESS))),
            horizontalalignment="left",
            fontsize=fntsiz,
        )
        h -= gap
        ax4.text(x1, h, "Lowest ESS:", horizontalalignment="right", fontsize=fntsiz)
        ax4.text(
            x2,
            h,
            "{:.5G}".format(int(min(param_ESS))),
            horizontalalignment="left",
            fontsize=fntsiz,
        )
        ax4.axis("off")

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()
        else:
            fig.clear()
            plt.close(fig)

    def matrix_plot(self, params=None, thin=None, burn=None, **kwargs):
        """
        Construct a 'matrix plot' of the parameters (or a subset) which displays
        all 1D and 2D marginal distributions. See the documentation of
        inference.plotting.matrix_plot for a description of other allowed
        keyword arguments.

        :param params: \
            A list of integers specifying the indices of parameters which are to
            be plotted.

        :param int burn: \
            Number of samples to discard from the start of the chain. If not
            specified, the value of ``self.burn`` is used instead.

        :param int thin: \
            Rather than using every sample which is not discarded as part of the
            burn-in, every *m*'th sample is used for a specified integer *m*. If
            not specified, the value of ``self.thin`` is used instead, which has
            a default value of 1.
        """
        burn = burn if burn is not None else self.burn
        thin = thin if thin is not None else self.thin
        params = params if params is not None else range(self.L)
        samples = [self.get_parameter(i, burn=burn, thin=thin) for i in params]
        matrix_plot(samples, **kwargs)

    def trace_plot(self, params=None, thin=1, burn=0, **kwargs):
        """
        Construct a 'trace plot' of the parameters (or a subset) which displays
        the value of the parameters as a function of step number in the chain.
        See the documentation of inference.plotting.trace_plot for a description
        of other allowed keyword arguments.

        :param params: \
            A list of integers specifying the indices of parameters which are to
            be plotted.

        :param int burn: \
            Number of samples to discard from the start of the chain.

        :param int thin: \
            Rather than using every sample which is not discarded as part of the
            burn-in, every *m*'th sample is used for a specified integer *m*.
        """
        params = params if params is not None else range(self.L)
        samples = [self.get_parameter(i, burn=burn, thin=thin) for i in params]
        trace_plot(samples, **kwargs)

    def save(self, filename):
        """
        Save the entire state of the chain object as an .npz file.

        :param str filename: file path to which the chain will be saved.
        """
        # get the chain attributes
        items = {
            "n": self.n,
            "L": self.L,
            "probs": self.probs,
            "burn": self.burn,
            "thin": self.thin,
            "inv_temp": self.inv_temp,
            "display_progress": self.display_progress,
        }

        # get the parameter attributes
        for i, p in enumerate(self.params):
            items.update(p.get_items(param_id=i))

        # save as npz
        savez(filename, **items)

    @classmethod
    def load(cls, filename, posterior=None):
        """
        Load a chain object which has been previously saved using the save() method.

        :param str filename: \
            file path of the .npz file containing the chain object data.

        :param posterior: \
            The posterior which was sampled by the chain. This argument need only be
            specified if new samples are to be added to the chain.
        """
        # load the data and create a chain instance
        D = load(filename)
        chain = cls(posterior=posterior, display_progress=bool(D["display_progress"]))

        # re-build the chain's attributes
        chain.n = int(D["n"])
        chain.L = int(D["L"])
        chain.probs = list(D["probs"])
        chain.inv_temp = float(D["inv_temp"])
        chain.burn = int(D["burn"])
        chain.thin = int(D["thin"])

        # re-build all the parameter objects
        chain.params = []
        for i in range(chain.L):
            p = Parameter()
            p.load_items(dictionary=D, param_id=i)
            chain.params.append(p)

        return chain

    def estimate_burn_in(self):
        # first get an estimate based on when the chain first reaches
        # the top 1% of log-probabilities
        prob_estimate = argmax(self.probs > percentile(self.probs, 99))

        # now we find the point at which the proposal width for each parameter
        # starts to deviate significantly from the current value
        width_estimates = []
        for p in self.params:
            vals = abs((array(p.sigma_values)[::-1] / p.sigma) - 1.0)
            chks = array(p.sigma_checks)[::-1]
            first_true = chks[argmax(vals > 0.15)]
            width_estimates.append(first_true)

        width_estimate = mean(width_estimates)
        return int(max(prob_estimate, width_estimate))

    def autoselect_burn(self):
        self.burn = self.estimate_burn_in()
        msg = "[ burn-in set to {} | {:.1%} of total samples ]".format(
            self.burn, self.burn / self.n
        )
        print(msg)

    def autoselect_thin(self):
        param_ESS = [ESS(array(self.get_parameter(i, thin=1))) for i in range(self.L)]
        self.thin = int((self.n - self.burn) / min(param_ESS))
        if self.thin < 1:
            self.thin = 1
        elif (self.n - self.burn) / self.thin < 1:
            self.thin = 1
            warn("Thinning not performed as lowest ESS is below 1")
        elif (self.n - self.burn) / self.thin < 100:
            warn("Sample size after thinning is less than 100")

        thin_size = len(self.probs[self.burn :: self.thin])
        print(
            f"[ thinning factor set to {self.thin} | thinned sample size is {thin_size} ]"
        )

    def autoselect_burn_and_thin(self):
        self.autoselect_burn()
        self.autoselect_thin()


class GibbsChain(MarkovChain):
    """
    A class for sampling from distributions using Gibbs-sampling.

    In Gibbs sampling, each "step" in the chain consists of a series of 1D Metropolis-Hastings
    updates, one for each parameter, such that after each step all parameters have been adjusted.

    This allows Metropolis-Hastings update acceptance rate data to be collected independently for
    each parameter, thereby allowing the proposal width of each parameter to be tuned individually.

    :param func posterior: \
        A function which takes the vector of model parameters as a ``numpy.ndarray``,
        and returns the posterior log-probability.

    :param start: \
        Vector of model parameters which correspond to the parameter-space coordinates at which
        the chain will start.

    :param widths: \
        Vector of standard deviations which serve as initial guesses for the widths of the proposal
        distribution for each model parameter. If not specified, the starting widths will be
        approximated as 5% of the values in 'start'.
    """

    def __init__(self, *args, **kwargs):
        super(GibbsChain, self).__init__(*args, **kwargs)
        # we need to adjust the target acceptance rate to 50%
        # which is optimal for gibbs sampling:
        if hasattr(self, "params"):
            for p in self.params:
                p.target_rate = 0.5

    def take_step(self):
        """
        Take a 1D metropolis-hastings step for each parameter
        """
        p_old = self.probs[-1]
        prop = self.get_last()

        for i, p in enumerate(self.params):
            while True:
                prop[i] = p.proposal()
                p_new = self.posterior(prop) * self.inv_temp

                if p_new > p_old:
                    # automatically accept step if the probability goes up
                    p.submit_accept_prob(1.0)
                    break
                else:
                    # else calculate the acceptance probability and perform the test
                    acceptance_prob = exp(p_new - p_old)
                    p.submit_accept_prob(acceptance_prob)
                    if random() < acceptance_prob:
                        break

            p_old = deepcopy(p_new)  # NOTE - is deepcopy needed?

        for v, p in zip(prop, self.params):
            p.add_sample(v)

        self.probs.append(p_new)
        self.n += 1


class PcaChain(MarkovChain):
    """
    A class which performs Gibbs sampling over the eigenvectors of the covariance matrix.

    The PcaChain sampler uses 'principal component analysis' (PCA) to improve
    the performance of Gibbs sampling in cases where strong linear correlation
    exists between two or more variables in a problem.

    For an N-parameter problem, PcaChain produces a new sample by making N
    sequential 1D Metropolis-Hastings steps in the direction of each of the
    N eigenvectors of the NxN covariance matrix.

    As an initial guess the covariance matrix is taken to be diagonal, which results
    in standard gibbs sampling for the first samples in the chain. As the chain advances,
    the covariance matrix is periodically updated with an estimate derived from the sample
    itself, and the eigenvectors are re-calculated.

    :param func posterior: \
        A function which takes the vector of model parameters as a ``numpy.ndarray``,
        and returns the posterior log-probability.

    :param start: \
        Vector of model parameters which correspond to the parameter-space coordinates
        at which the chain will start.

    :param widths: \
        Vector of standard deviations which serve as initial guesses for the widths of
        the proposal distribution for each model parameter. If not specified, the starting
        widths will be approximated as 5% of the values in 'start'.

    :param parameter_boundaries: \
        A list of length-2 tuples specifying the lower and upper bounds to be set on each
        parameter, in the form (lower, upper).

    :param bool display_progress: \
        If set as ``True``, a message is displayed during sampling
        showing the current progress and an estimated time until completion.
    """

    def __init__(self, *args, parameter_boundaries=None, **kwargs):
        super(PcaChain, self).__init__(*args, **kwargs)
        # we need to adjust the target acceptance rate to 50%
        # which is optimal for gibbs sampling:
        if hasattr(self, "params"):
            for p in self.params:
                p.target_rate = 0.5

        self.directions = []
        if hasattr(self, "L"):
            for i in range(self.L):
                v = zeros(self.L)
                v[i] = 1.0
                self.directions.append(v)

        # PCA update settings
        self.dir_update_interval = 100
        self.dir_growth_factor = 1.5
        self.last_update = 0
        self.next_update = copy(self.dir_update_interval)

        # PCA convergence tracking
        self.angles_history = []
        self.update_history = []

        # Set-up for imposing boundaries if specified
        if parameter_boundaries is not None:
            if len(parameter_boundaries) == self.L:
                self.lower = array([k[0] for k in parameter_boundaries])
                self.upper = array([k[1] for k in parameter_boundaries])
                self.width = self.upper - self.lower
                self.process_proposal = self.impose_boundaries
            else:
                warn(
                    """
                    [ PcaChain warning ]
                    >> The number of given lower/upper bounds pairs does not match
                    >> the number of model parameters - bounds were not imposed.
                    """
                )
        else:
            self.process_proposal = self.pass_through

    def update_directions(self):
        # re-estimate the covariance and find its eigenvectors
        data = array([self.get_parameter(i)[self.last_update :] for i in range(self.L)])
        if hasattr(self, "covar"):
            nu = min(2 * self.dir_update_interval / self.last_update, 0.5)
            self.covar = self.covar * (1 - nu) + nu * cov(data)
        else:
            self.covar = cov(data)

        w, V = eigh(self.covar)

        # find the sine of the angle between the old and new eigenvectors to track convergence
        angles = [
            sqrt(1.0 - dot(V[:, i], self.directions[i]) ** 2) for i in range(self.L)
        ]
        self.angles_history.append(angles)
        self.update_history.append(copy(self.n))

        # store the new directions and plan the next update
        self.directions = [V[:, i] for i in range(self.L)]
        self.last_update = copy(self.n)
        self.dir_update_interval = int(
            self.dir_update_interval * self.dir_growth_factor
        )
        self.next_update = self.last_update + self.dir_update_interval

    def directions_diagnostics(self):
        for i in range(self.L):
            prods = [v[i] for v in self.angles_history]
            plt.plot(self.update_history, prods, ".-")
        plt.plot(
            [self.update_history[0], self.update_history[-1]],
            [1e-2, 1e-2],
            ls="dashed",
            c="black",
            lw=2,
        )
        plt.yscale("log")
        plt.ylim([1e-4, 1.0])
        plt.xlim([0, self.update_history[-1]])

        plt.ylabel(r"$|\sin{(\Delta \theta)}|$", fontsize=13)
        plt.xlabel(r"update step number", fontsize=13)

        plt.grid()
        plt.tight_layout()
        plt.show()

    def take_step(self):
        """
        Take a Metropolis-Hastings step along each principal component
        """
        p_old = self.probs[-1]
        theta0 = self.get_last()
        # loop over each eigenvector and take a step along each
        for v, p in zip(self.directions, self.params):
            while True:
                prop = theta0 + v * p.sigma * normal()
                prop = self.process_proposal(prop)
                p_new = self.posterior(prop) * self.inv_temp

                if p_new > p_old:
                    p.submit_accept_prob(1.0)
                    break
                else:
                    test = random()
                    acceptance_prob = exp(p_new - p_old)
                    p.submit_accept_prob(acceptance_prob)
                    if test < acceptance_prob:
                        break

            theta0 = copy(prop)
            p_old = copy(p_new)

        # add the new value for each parameter
        for v, p in zip(theta0, self.params):
            p.add_sample(v)

        self.probs.append(p_new)
        self.n += 1

        if self.n == self.next_update:
            self.update_directions()

    def save(self, filename):
        """
        Save the entire state of the chain object as an .npz file.

        :param str filename: file path to which the chain will be saved.
        """
        # get the chain attributes
        items = {
            "n": self.n,
            "L": self.L,
            "probs": self.probs,
            "burn": self.burn,
            "thin": self.thin,
            "inv_temp": self.inv_temp,
            "display_progress": self.display_progress,
            "dir_update_interval": self.dir_update_interval,
            "dir_growth_factor": self.dir_growth_factor,
            "last_update": self.last_update,
            "next_update": self.next_update,
            "angles_history": array(self.angles_history),
            "update_history": array(self.update_history),
            "directions": array(self.directions),
            "covar": self.covar,
        }

        # get the parameter attributes
        for i, p in enumerate(self.params):
            items.update(p.get_items(param_id=i))

        # save as npz
        savez(filename, **items)

    @classmethod
    def load(cls, filename, posterior=None):
        """
        Load a chain object which has been previously saved using the save() method.

        :param str filename: \
            file path of the .npz file containing the chain object data.

        :param posterior: \
            The posterior which was sampled by the chain. This argument need only be
            specified if new samples are to be added to the chain.
        """
        # load the data and create a chain instance
        D = load(filename)
        chain = cls(posterior=posterior, display_progress=bool(D["display_progress"]))

        # re-build the chain's attributes
        chain.n = int(D["n"])
        chain.L = int(D["L"])
        chain.probs = list(D["probs"])
        chain.burn = int(D["burn"])
        chain.thin = int(D["thin"])
        chain.inv_temp = float(D["inv_temp"])
        chain.dir_update_interval = int(D["dir_update_interval"])
        chain.dir_growth_factor = float(D["dir_growth_factor"])
        chain.last_update = int(D["last_update"])
        chain.next_update = int(D["next_update"])
        chain.angles_history = [
            D["angles_history"][i, :] for i in range(D["angles_history"].shape[0])
        ]
        chain.update_history = list(D["update_history"])
        chain.directions = [
            D["directions"][i, :] for i in range(D["directions"].shape[0])
        ]
        chain.covar = D["covar"]

        # re-build all the parameter objects
        chain.params = []
        for i in range(chain.L):
            p = Parameter()
            p.load_items(dictionary=D, param_id=i)
            chain.params.append(p)
        return chain

    def set_non_negative(self, *args, **kwargs):
        warn(
            """
             The set_non_negative method is not available for PcaChain:
             Limits on parameters should instead be set using
             the parameter_boundaries keyword argument.
             """
        )

    def set_boundaries(self, *args, **kwargs):
        warn(
            """
             The set_boundaries method is not available for PcaChain:
             Limits on parameters should instead be set using
             the parameter_boundaries keyword argument.
             """
        )

    def impose_boundaries(self, prop):
        d = prop - self.lower
        n = (d // self.width) % 2
        return self.lower + (1 - 2 * n) * (d % self.width) + n * self.width

    def pass_through(self, prop):
        return prop


class HamiltonianChain(MarkovChain):
    """
    Class for performing Hamiltonian Monte-Carlo sampling.

    Hamiltonian Monte-Carlo (HMC) is an MCMC algorithm where proposed steps are generated
    by integrating Hamilton’s equations, treating the negative posterior log-probability
    as a scalar potential. In order to do this, the algorithm requires the gradient of
    the log-posterior with respect to the model parameters. Assuming this gradient can be
    calculated efficiently, HMC deals  well with strongly correlated variables and scales
    favourably to higher-dimensionality problems.

    This implementation automatically selects an appropriate time-step for the Hamiltonian
    dynamics simulation, but currently does not dynamically select the number of time-steps
    per proposal, or appropriate inverse-mass values.

    :param func posterior: \
        A function which takes the vector of model parameters as a ``numpy.ndarray``,
        and returns the posterior log-probability.

    :param func grad: \
        A function which returns the gradient of the log-posterior probability density
        for a given set of model parameters theta. If this function is not given, the
        gradient will instead be estimated by finite difference.

    :param start: \
        Vector of model parameters which correspond to the parameter-space coordinates
        at which the chain will start.

    :param float epsilon: \
        Initial guess for the time-step of the Hamiltonian dynamics simulation.

    :param float temperature: \
        The temperature of the markov chain. This parameter is used for parallel
        tempering and should be otherwise left unspecified.

    :param bounds: \
        A list or tuple containing two numpy arrays which specify the upper and lower
        bounds for the parameters in the form (lower_bounds, upper_bounds).

    :param inv_mass: \
        A vector specifying the inverse-mass value to be used for each parameter. The
        inverse-mass is used to transform the momentum distribution in order to make
        the problem more isotropic. Ideally, the inverse-mass for each parameter should
        be set to the variance of the marginal distribution of that parameter.

    :param bool display_progress: \
        If set as ``True``, a message is displayed during sampling
        showing the current progress and an estimated time until completion.
    """

    def __init__(
        self,
        posterior=None,
        grad=None,
        start=None,
        epsilon=0.1,
        temperature=1,
        bounds=None,
        inv_mass=None,
        display_progress=True,
    ):

        self.posterior = posterior
        # if no gradient function is supplied, default to finite difference
        if grad is None:
            self.grad = self.finite_diff
        else:
            self.grad = grad

        # set either the bounded or unbounded leapfrog update
        if bounds is None:
            self.leapfrog = self.standard_leapfrog
            self.bounded = False
            self.lwr_bounds = None
            self.upr_bounds = None
            self.widths = None
        else:
            self.leapfrog = self.bounded_leapfrog
            self.bounded = True
            self.lwr_bounds = array(bounds[0])
            self.upr_bounds = array(bounds[1])
            if any((self.lwr_bounds > array(start)) | (self.upr_bounds < array(start))):
                raise ValueError(
                    """
                    [ HamiltonianChain error ]
                    >> Starting location for the chain is outside specified bounds.
                    """
                )
            self.widths = self.upr_bounds - self.lwr_bounds
            if not all(self.widths > 0):
                raise ValueError(
                    """
                    [ HamiltonianChain error ]
                    >> Specified upper bounds must be greater than lower bounds.
                    """
                )

        self.temperature = temperature
        self.inv_temp = 1.0 / temperature

        if start is not None:
            self.theta = [start]
            self.probs = [self.posterior(start) * self.inv_temp]
            self.leapfrog_steps = [0]
            self.L = len(start)
        self.n = 1

        # set the variance to 1 if none supplied
        if inv_mass is None:
            self.variance = 1.0
        else:
            self.variance = inv_mass

        self.ES = EpsilonSelector(epsilon)
        self.steps = 50
        self.burn = 1
        self.thin = 1

        self.display_progress = display_progress
        self.ProgressPrinter = ChainProgressPrinter(
            display=self.display_progress, leading_msg="advancing chain:"
        )

    def take_step(self):
        """
        Takes the next step in the HMC-chain
        """
        accept = False
        steps_taken = 0
        while not accept:
            r0 = normal(size=self.L) / sqrt(self.variance)
            t0 = self.theta[-1]
            H0 = 0.5 * dot(r0, r0 / self.variance) - self.probs[-1]

            r = copy(r0)
            t = copy(t0)
            g = self.grad(t) * self.inv_temp
            n_steps = int(self.steps * (1 + (random() - 0.5) * 0.2))

            t, r, g = self.run_leapfrog(t, r, g, n_steps)

            steps_taken += n_steps
            p = self.posterior(t) * self.inv_temp
            H = 0.5 * dot(r, r / self.variance) - p
            test = exp(H0 - H)

            if isfinite(test):
                self.ES.add_probability(min(test, 1))
            else:
                self.ES.add_probability(0.0)

            if test >= 1:
                accept = True
            else:
                q = random()
                if q <= test:
                    accept = True

        self.theta.append(t)
        self.probs.append(p)
        self.leapfrog_steps.append(steps_taken)
        self.n += 1

    def run_leapfrog(self, t, r, g, L):
        for i in range(L):
            t, r, g = self.leapfrog(t, r, g)
        return t, r, g

    def hamiltonian(self, t, r):
        return 0.5 * dot(r, r / self.variance) - self.posterior(t) * self.inv_temp

    def estimate_mass(self, burn=1, thin=1):
        self.variance = var(array(self.theta[burn::thin]), axis=0)

    def finite_diff(self, t):
        p = self.posterior(t) * self.inv_temp
        G = zeros(self.L)
        for i in range(self.L):
            delta = zeros(self.L) + 1
            delta[i] += 1e-5
            G[i] = (self.posterior(t * delta) * self.inv_temp - p) / (t[i] * 1e-5)
        return G

    def standard_leapfrog(self, t, r, g):
        r2 = r + (0.5 * self.ES.epsilon) * g
        t2 = t + self.ES.epsilon * r2 * self.variance

        g = self.grad(t2) * self.inv_temp
        r2 = r2 + (0.5 * self.ES.epsilon) * g
        return t2, r2, g

    def bounded_leapfrog(self, t, r, g):
        r2 = r + (0.5 * self.ES.epsilon) * g
        t2 = t + self.ES.epsilon * r2 * self.variance
        # check for values outside bounds
        lwr_diff = self.lwr_bounds - t2
        upr_diff = t2 - self.upr_bounds
        lwr_bools = lwr_diff > 0
        upr_bools = upr_diff > 0
        # calculate necessary adjustment
        lwr_adjust = lwr_bools * (lwr_diff + lwr_diff % (0.1 * self.widths))
        upr_adjust = upr_bools * (upr_diff + upr_diff % (0.1 * self.widths))
        t2 += lwr_adjust
        t2 -= upr_adjust

        # reverse momenta where necessary
        reflect = 1 - 2 * (lwr_bools | upr_bools)
        r2 *= reflect

        g = self.grad(t2) * self.inv_temp
        r2 = r2 + (0.5 * self.ES.epsilon) * g
        return t2, r2, g

    def get_last(self):
        return self.theta[-1]

    def replace_last(self, theta):
        self.theta[-1] = theta

    def get_parameter(self, n, burn=None, thin=None):
        """
        Return sample values for a chosen parameter.

        :param int n: \
            Index of the parameter for which samples are to be returned.

        :param int burn: \
            Number of samples to discard from the start of the chain. If not specified, the
            value of self.burn is used instead.

        :param int thin: \
            Instead of returning every sample which is not discarded as part of the burn-in,
            every *m*'th sample is returned for a specified integer *m*. If not specified,
            the value of self.thin is used instead.

        :return: \
            List of samples for parameter *n*'th parameter.
        """
        if burn is None:
            burn = self.burn
        if thin is None:
            thin = self.thin
        return [v[n] for v in self.theta[burn::thin]]

    def plot_diagnostics(self, show=True, filename=None, burn=None):
        """
        Plot diagnostic traces that give information on how the chain is progressing.

        Currently this method plots:

        - The posterior log-probability as a function of step number, which is useful
          for checking if the chain has reached a maximum. Any early parts of the chain
          where the probability is rising rapidly should be removed as burn-in.

        - The history of the simulation step-size epsilon as a function of number of
          total proposed jumps.

        - The estimated sample size (ESS) for every parameter, or in cases where the
          number of parameters is very large, a histogram of the ESS values.

        :param bool show: \
            If set to True, the plot is displayed.

        :param str filename: \
            File path to which the diagnostics plot will be saved. If left unspecified
            the plot won't be saved.
        """
        if burn is None:
            burn = self.estimate_burn_in()
        param_ESS = [
            ESS(array(self.get_parameter(i, burn=burn, thin=1))) for i in range(self.L)
        ]

        fig = plt.figure(figsize=(12, 9))

        # probability history plot
        ax1 = fig.add_subplot(221)
        # TODO - avoid making this axis but preserve figure form
        step_ax = [i * 1e-3 for i in range(len(self.probs))]
        ax1.plot(step_ax, self.probs, marker=".", ls="none", markersize=3)
        ax1.set_xlabel("chain step number ($10^3$)", fontsize=12)
        ax1.set_ylabel("posterior log-probability", fontsize=12)
        ax1.set_title("Chain log-probability history")
        ylims = [
            min(self.probs[self.n // 2 :]),
            max(self.probs) * 1.1 - 0.1 * min(self.probs[self.n // 2 :]),
        ]
        plt.plot([burn * 1e-3, burn * 1e-3], ylims, c="red", ls="dashed", lw=2)
        ax1.set_ylim(ylims)
        ax1.grid()

        # epsilon plot
        ax2 = fig.add_subplot(222)
        ax2.plot(array(self.ES.epsilon_checks) * 1e-3, self.ES.epsilon_values, ".-")
        ax2.set_xlabel("chain step number ($10^3$)", fontsize=12)
        ax2.set_ylabel("Leapfrog step-size", fontsize=12)
        ax2.set_title("Simulation time-step adjustment summary")
        ax2.grid()

        ax3 = fig.add_subplot(223)
        if self.L < 50:
            ax3.bar(range(self.L), param_ESS, color=["C0", "C1", "C2", "C3", "C4"])
            ax3.set_xlabel("parameter", fontsize=12)
            ax3.set_ylabel("effective sample size", fontsize=12)
            ax3.set_title("Parameter effective sample size estimate")
            ax3.set_xticks(range(self.L))
        else:
            ax3.hist(param_ESS, bins=20)
            ax3.set_xlabel("effective sample size", fontsize=12)
            ax3.set_ylabel("frequency", fontsize=12)
            ax3.set_title("Parameter effective sample size estimates")

        ax4 = fig.add_subplot(224)
        gap = 0.1
        h = 0.85
        x1 = 0.5
        x2 = 0.55
        fntsiz = 14

        ax4.text(
            x1, h, "Estimated burn-in:", horizontalalignment="right", fontsize=fntsiz
        )
        ax4.text(
            x2, h, "{:.5G}".format(burn), horizontalalignment="left", fontsize=fntsiz
        )
        h -= gap
        ax4.text(x1, h, "Average ESS:", horizontalalignment="right", fontsize=fntsiz)
        ax4.text(
            x2,
            h,
            "{:.5G}".format(int(mean(param_ESS))),
            horizontalalignment="left",
            fontsize=fntsiz,
        )
        h -= gap
        ax4.text(x1, h, "Lowest ESS:", horizontalalignment="right", fontsize=fntsiz)
        ax4.text(
            x2,
            h,
            "{:.5G}".format(int(min(param_ESS))),
            horizontalalignment="left",
            fontsize=fntsiz,
        )
        ax4.axis("off")

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()
        else:
            fig.clear()
            plt.close(fig)

    def get_sample(self, burn=None, thin=None):
        raise ValueError("This method is not available for HamiltonianChain")

    def get_interval(self, interval=None, burn=None, thin=None, samples=None):
        raise ValueError("This method is not available for HamiltonianChain")

    def mode(self):
        return self.theta[argmax(self.probs)]

    def estimate_burn_in(self):
        # first get an estimate based on when the chain first reaches
        # the top 1% of log-probabilities
        prob_estimate = argmax(self.probs > percentile(self.probs, 99))
        # now we find the point at which the proposal width for each parameter
        # starts to deviate significantly from the current value
        epsl = abs((array(self.ES.epsilon_values)[::-1] / self.ES.epsilon) - 1.0)
        chks = array(self.ES.epsilon_checks)[::-1]
        epsl_estimate = chks[argmax(epsl > 0.15)] * self.ES.accept_rate
        return int(min(max(prob_estimate, epsl_estimate), 0.9 * self.n))

    def save(self, filename, compressed=False):
        items = {
            "bounded": self.bounded,
            "lwr_bounds": self.lwr_bounds,
            "upr_bounds": self.upr_bounds,
            "widths": self.widths,
            "inv_mass": self.variance,
            "inv_temp": self.inv_temp,
            "theta": self.theta,
            "probs": self.probs,
            "leapfrog_steps": self.leapfrog_steps,
            "L": self.L,
            "n": self.n,
            "steps": self.steps,
            "burn": self.burn,
            "thin": self.thin,
            "display_progress": self.display_progress,
        }

        items.update(self.ES.get_items())

        # save as npz
        if compressed:
            savez_compressed(filename, **items)
        else:
            savez(filename, **items)

    @classmethod
    def load(cls, filename, posterior=None, grad=None):
        D = load(filename)
        chain = cls(
            posterior=posterior, grad=grad, display_progress=bool(D["display_progress"])
        )

        chain.bounded = bool(D["bounded"])
        chain.variance = array(D["inv_mass"])
        chain.inv_temp = float(D["inv_temp"])
        chain.temperature = 1.0 / chain.inv_temp
        chain.probs = list(D["probs"])
        chain.leapfrog_steps = list(D["leapfrog_steps"])
        chain.L = int(D["L"])
        chain.n = int(D["n"])
        chain.steps = int(D["steps"])
        chain.burn = int(D["burn"])
        chain.thin = int(D["thin"])
        chain.n = int(D["n"])

        t = D["theta"]
        chain.theta = [t[i, :] for i in range(t.shape[0])]

        if chain.bounded:
            chain.lwr_bounds = array(D["lwr_bounds"])
            chain.upr_bounds = array(D["upr_bounds"])
            chain.widths = array(D["widths"])

        # build the epsilon selector
        chain.ES.load_items(D)

        return chain


class EpsilonSelector:
    def __init__(self, epsilon):

        # storage
        self.epsilon = epsilon
        self.epsilon_values = [copy(epsilon)]  # sigma values after each assessment
        self.epsilon_checks = [0.0]  # chain locations at which sigma was assessed

        # tracking variables
        self.avg = 0
        self.var = 0
        self.num = 0

        # settings for epsilon adjustment algorithm
        self.accept_rate = 0.65
        self.chk_int = 15  # interval of steps at which proposal widths are adjusted
        self.growth_factor = 1.4  # growth factor for self.chk_int

    def add_probability(self, p):
        self.num += 1
        self.avg += p
        self.var += max(p * (1 - p), 0.03)

        if self.num >= self.chk_int:
            self.update_epsilon()

    def update_epsilon(self):
        """
        looks at average tries over recent steps, and adjusts proposal
        widths self.sigma to bring the average towards self.target_tries.
        """
        # normal approximation of poisson binomial distribution
        mu = self.avg / self.num
        std = sqrt(self.var) / self.num

        # now check if the desired success rate is within 2-sigma
        if ~(mu - 2 * std < self.accept_rate < mu + 2 * std):
            adj = (log(self.accept_rate) / log(mu)) ** 0.15
            adj = min(adj, 2.0)
            adj = max(adj, 0.5)
            self.adjust_epsilon(adj)
        else:  # increase the check interval
            self.chk_int = int((self.growth_factor * self.chk_int) * 0.1) * 10

    def adjust_epsilon(self, ratio):
        self.epsilon *= ratio
        self.epsilon_values.append(copy(self.epsilon))
        self.epsilon_checks.append(self.epsilon_checks[-1] + self.num)
        self.avg = 0
        self.var = 0
        self.num = 0

    def get_items(self):
        return self.__dict__

    def load_items(self, dictionary):
        self.epsilon = float(dictionary["epsilon"])
        self.epsilon_values = list(dictionary["epsilon_values"])
        self.epsilon_checks = list(dictionary["epsilon_checks"])
        self.avg = float(dictionary["avg"])
        self.var = float(dictionary["var"])
        self.num = float(dictionary["num"])
        self.accept_rate = float(dictionary["accept_rate"])
        self.chk_int = int(dictionary["chk_int"])
        self.growth_factor = float(dictionary["growth_factor"])


class ChainPool:
    def __init__(self, objects):
        self.chains = objects
        self.pool_size = len(self.chains)
        self.pool = Pool(self.pool_size)

    def advance(self, n):
        self.chains = self.pool.map(
            self.adv_func, [(n, chain) for chain in self.chains]
        )

    @staticmethod
    def adv_func(arg):
        n, chain = arg
        for _ in range(n):
            chain.take_step()
        return chain


def tempering_process(chain, connection, end, proc_seed):
    # used to ensure each process has a different random seed
    seed(proc_seed)
    # main loop
    while not end.is_set():
        # poll the pipe until there is something to read
        while not end.is_set():
            if connection.poll(timeout=0.05):
                D = connection.recv()
                break

        # if read loop was broken because of shutdown event
        # then break the main loop as well
        if end.is_set():
            break

        task = D["task"]

        # advance the chain
        if task == "advance":
            for _ in range(D["advance_count"]):
                chain.take_step()
            connection.send("advance_complete")  # send signal to confirm completion

        # return the current position of the chain
        elif task == "send_position":
            connection.send((chain.get_last(), chain.probs[-1]))

        # update the position of the chain
        elif task == "update_position":
            chain.replace_last(D["position"])
            chain.probs[-1] = D["probability"] * chain.inv_temp

        # return the local chain object
        elif task == "send_chain":
            connection.send(chain)


class ParallelTempering:
    """
    A class which enables 'parallel tempering', a sampling algorithm which
    advances multiple Markov-chains in parallel, each with a different
    'temperature', with a probability that the chains will exchange their
    positions during the advancement.

    The 'temperature' concept introduces a transformation to the distribution
    being sampled, such that a chain with temperature 'T' instead samples from
    the provided posterior distribution raised to the power 1/T.

    When T = 1, the original distribution is recovered, but choosing T > 1 has
    the effect of 'compressing' the distribution, such that any two points having
    different probability densities will have the difference between those densities
    reduced as the temperature is increased. This allows chains with higher
    temperatures to take much larger steps, and explore the distribution more
    quickly.

    Parallel tempering exploits this by advancing a collection of markov-chains at
    different temperatures, with at least one chain at T = 1 (i.e. sampling from
    the actual posterior distribution). At regular intervals, pairs of chains are
    selected at random and a metropolis-hastings test is performed to decide if
    the pair exchange their positions.

    The ability for the T = 1 chain to exchange positions with chains of higher
    temperatures allows it to make large jumps to other areas of the distribution
    which it may take a large number of steps to reach otherwise.

    This is particularly useful when sampling from highly-complex distributions
    which may have many separate maxima and/or strong correlations.

    :param chains: \
        A list of Markov-Chain objects (such as GibbsChain, PcaChain, HamiltonianChain)
        covering a range of different temperature levels. The list of chains should be
        sorted in order of increasing chain temperature.
    """

    def __init__(self, chains):
        self.shutdown_evt = Event()
        self.connections = []
        self.processes = []
        self.temperatures = [1.0 / chain.inv_temp for chain in chains]
        self.inv_temps = [chain.inv_temp for chain in chains]
        self.N_chains = len(chains)

        self.attempted_swaps = identity(self.N_chains)
        self.successful_swaps = zeros([self.N_chains, self.N_chains])

        if sorted(self.temperatures) != self.temperatures:
            warn(
                """
                The list of Markov-chain objects passed to ParallelTempering
                should be sorted in order of increasing chain temperature.
                """
            )

        # Spawn a separate process for each chain object
        for chn in chains:
            parent_ctn, child_ctn = Pipe()
            self.connections.append(parent_ctn)
            p = Process(
                target=tempering_process,
                args=(chn, child_ctn, self.shutdown_evt, randint(30000)),
            )
            self.processes.append(p)

        [p.start() for p in self.processes]

    def take_steps(self, n):
        """
        Advance all the chains *n* steps without performing any swaps.

        :param int n: The number of steps by which every chain is advanced.
        """
        # order the chains to advance n steps
        D = {"task": "advance", "advance_count": n}
        for pipe in self.connections:
            pipe.send(D)

        # block until all chains report successful advancement
        responses = [pipe.recv() == "advance_complete" for pipe in self.connections]
        if not all(responses):
            raise ValueError("Unexpected data received from pipe")

    def uniform_pairs(self):
        """
        Randomly pair up each chain, with uniform sampling across all possible pairings
        """
        proposed_swaps = arange(self.N_chains)
        shuffle(proposed_swaps)
        return [p for p in zip(proposed_swaps[::2], proposed_swaps[1::2])]

    def tight_pairs(self):
        """
        Randomly pair up each chain, with almost all paired chains being separated
        by either 1 or 2 temperature levels.
        """
        # first generate all possible pairings with a gap of 2 or less
        pairs = [(i, i + j) for i in range(self.N_chains - 1) for j in [1, 2]][:-1]
        sample = []
        # randomly sample from these pairings until no valid pairs remain
        while len(pairs) > 0:
            p = choice(pairs)
            pairs = [k for k in pairs if not any(j in k for j in p)]
            sample.append(p)
        # if there are still some pairs which haven't been paired, randomly pair the remaining ones
        remaining = len(sample) - self.N_chains // 2
        if remaining != 0:
            leftovers = [
                i for i in range(self.N_chains) if not any(i in p for p in sample)
            ]
            shuffle(leftovers)
            sample.extend(
                [
                    p if p[0] < p[1] else (p[1], p[0])
                    for p in zip(leftovers[::2], leftovers[1::2])
                ]
            )
        return sample

    def swap(self):
        """
        Randomly group all chains into pairs and propose a position swap between each pair.
        """
        # ask each process to report the current position of its chain
        D = {"task": "send_position"}
        [pipe.send(D) for pipe in self.connections]

        # receive the positions and probabilities
        data = [pipe.recv() for pipe in self.connections]
        positions = [k[0] for k in data]
        probabilities = [k[1] for k in data]

        # randomly pair up indices for all the processes
        proposed_swaps = self.tight_pairs()

        # perform MH tests to see if the swaps occur or not
        for pair in proposed_swaps:
            self.attempted_swaps[pair] += 1

        for i, j in proposed_swaps:
            dt = self.inv_temps[i] - self.inv_temps[j]
            pi = probabilities[i] / self.inv_temps[i]
            pj = probabilities[j] / self.inv_temps[j]
            dp = pi - pj

            if random() <= exp(-dt * dp):  # check if the swap is successful
                Di = {
                    "task": "update_position",
                    "position": positions[i],
                    "probability": pi,
                }

                Dj = {
                    "task": "update_position",
                    "position": positions[j],
                    "probability": pj,
                }

                self.connections[i].send(Dj)
                self.connections[j].send(Di)
                self.successful_swaps[i, j] += 1

    def advance(self, n, swap_interval=10):
        """
        Advances each chain by a total of *n* steps, performing swap attempts
        at intervals set by the *swap_interval* keyword.

        :param int n: The number of steps each chain will advance.
        :param int swap_interval: \
            The number of steps that are taken in each chain between swap attempts.
        """
        k = 50  # divide chain steps into k groups to track progress
        total_cycles = n // swap_interval
        if k < total_cycles:
            k = total_cycles
            cycles = 1
        else:
            cycles = total_cycles // k

        t_start = time()
        for j in range(k):
            for i in range(cycles):
                self.take_steps(swap_interval)
                self.swap()

            dt = time() - t_start

            # display the progress status message
            pct = str(int(100 * (j + 1) / k))
            eta = str(int(dt * (k / (j + 1) - 1)))
            sys.stdout.write(
                f"\r  [ Running ParallelTempering - {pct}% complete   ETA: {eta} sec ]    "
            )
            sys.stdout.flush()

        # run the remaining cycles
        if total_cycles % k != 0:
            for i in range(total_cycles % k):
                self.take_steps(swap_interval)
                self.swap()

        # run remaining steps
        if n % swap_interval != 0:
            self.take_steps(n % swap_interval)

        # print the completion message
        sys.stdout.write(
            "\r  [ Running ParallelTempering - complete! ]                    "
        )
        sys.stdout.flush()
        sys.stdout.write("\n")

    def run_for(self, minutes=0, hours=0, swap_interval=10):
        """
        Advances all chains for a chosen amount of computation time.

        :param float minutes: Number of minutes for which to advance the chains.
        :param float hours: Number of hours for which to advance the chains.
        :param int swap_interval: \
            The number of steps that are taken in each chain between swap attempts.
        """
        # first find the runtime in seconds:
        run_time = (hours * 60.0 + minutes) * 60.0
        start_time = time()
        end_time = start_time + run_time

        # estimate how long it takes to do one swap cycle
        t1 = time()
        self.take_steps(swap_interval)
        self.swap()
        t2 = time()

        # number of cycles chosen to give a print-out roughly every 2 seconds
        N = max(1, int(2.0 / (t2 - t1)))

        while time() < end_time:
            for i in range(N):
                self.take_steps(swap_interval)
                self.swap()

            # display the progress status message
            seconds_remaining = end_time - time()
            m, s = divmod(seconds_remaining, 60)
            h, m = divmod(m, 60)
            time_left = "%d:%02d:%02d" % (h, m, s)
            sys.stdout.write(
                f"\r  [ Running ParallelTempering - time remaining: {time_left} ]    "
            )
            sys.stdout.flush()

        # this is a little ugly...
        sys.stdout.write(
            "\r  [ Running ParallelTempering - complete! ]                    "
        )
        sys.stdout.flush()
        sys.stdout.write("\n")

    def swap_diagnostics(self):
        """
        Plot the acceptance rates of proposed position swaps between the
        different chains. This can be useful in selecting appropriate temperatures
        for the chains.
        """
        rate_matrix = self.successful_swaps / self.attempted_swaps.clip(min=1)

        pairs = [
            (i, i + j)
            for j in range(1, self.N_chains)
            for i in range(self.N_chains - j)
        ]
        total_swaps = zeros(self.N_chains)
        for i, j in pairs:
            total_swaps[i] += self.successful_swaps[i, j]
            total_swaps[j] += self.successful_swaps[i, j]

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        transition_matrix_plot(
            ax=ax1, matrix=rate_matrix, exclude_diagonal=True, upper_triangular=True
        )
        ax1.set_xlabel("chain number")
        ax1.set_ylabel("chain number")
        ax1.set_title("acceptance rate of chain position swaps")

        ax2 = fig.add_subplot(122)
        ax2.bar([i for i in range(1, self.N_chains + 1)], total_swaps)
        ax2.set_ylim([0, None])
        ax2.set_xlabel("chain number")
        ax2.set_ylabel("total successful position swaps")

        plt.tight_layout()
        plt.show()

    def return_chains(self):
        """
        Recover the chain held by each process and return them in a list.

        :return: A list containing the chain objects.
        """
        # order each process to return its locally stored chain object
        request = {"task": "send_chain"}
        for pipe in self.connections:
            pipe.send(request)

        # receive the chains and return them
        return [pipe.recv() for pipe in self.connections]

    def shutdown(self):
        """
        Trigger a shutdown event which tells the processes holding each of
        the chains to terminate.
        """
        self.shutdown_evt.set()
        [p.join() for p in self.processes]


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
        An iterable containing a series of parameter arrays corresponding to the starting
        positions of the 'walkers' which make up the ensemble.

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
        posterior,
        starting_positions=None,
        alpha=2.0,
        parameter_boundaries=None,
        display_progress=True,
    ):
        self.posterior = posterior

        if starting_positions is not None:
            # store core data
            self.n_params = len(starting_positions[0])
            self.n_walkers = len(starting_positions)
            self.theta = zeros([self.n_walkers, self.n_params])
            for i, v in enumerate(starting_positions):
                self.theta[i, :] = array(v)
            self.probs = array([self.posterior(t) for t in self.theta])
            self.__starting_positions_check()
            # storage for diagnostic information
            self.n_iterations = 0
            self.total_proposals = [[] for i in range(self.n_walkers)]
            self.failed_updates = []

        if parameter_boundaries is not None:
            if len(parameter_boundaries) == self.n_params:
                self.bounded = True
                self.lower = array([k[0] for k in parameter_boundaries])
                self.upper = array([k[1] for k in parameter_boundaries])
                self.width = self.upper - self.lower
                self.process_proposal = self.impose_boundaries
            else:
                warn(
                    """
                    [ EnsembleSampler warning ]
                    >> The number of given lower/upper bounds pairs does not match
                    >> the number of model parameters - bounds were not imposed.
                    """
                )
        else:
            self.process_proposal = self.pass_through
            self.bounded = False

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

    def __starting_positions_check(self):
        if self.n_params == 1:
            # only need to check the variance for the one-parameter case
            if var(self.theta) == 0:
                raise ValueError(
                    """
                    [ EnsembleSampler error ]
                    >> The values given in starting_positions have zero variance,
                    >> and therefore the walkers are unable to move.
                    """
                )
        else:
            covar = cov(self.theta.T)
            std_dev = sqrt(diag(covar))  # get the standard devs
            if (std_dev == 0).any():
                raise ValueError(
                    """
                    [ EnsembleSampler error ]
                    >> For one or more variables, The values given in starting_positions 
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
                    >> The values given in starting_positions are approximately
                    >> co-linear for one or more pair of variables. This will
                    >> prevent the walkers from moving properly in those variables.
                    """
                )

    def proposal(self, i):
        j = i  # randomly select walker
        while i == j:
            j = randint(self.n_walkers)
        # sample the stretch distance
        z = 0.5 * (self.x_lwr + self.x_width * random()) ** 2
        prop = self.process_proposal(
            self.theta[i, :] + z * (self.theta[j, :] - self.theta[i, :])
        )
        return prop, z

    def advance_walker(self, i):
        for attempts in range(1, self.max_attempts + 1):
            Y, z = self.proposal(i)
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

    def advance_all(self):
        self.failed_updates.append(0)
        for i in range(self.n_walkers):
            self.advance_walker(i)
        self.n_iterations += 1

    def advance(self, iterations):
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
            self.advance_all()
            sample_arrays.append(self.theta.copy())
            prob_arrays.append(self.probs.copy())

            # display the progress status message
            self.ProgressPrinter.iterations_progress(t_start, k, iterations)

        # display completion message
        self.ProgressPrinter.iterations_final(iterations)
        self.sample = concatenate(sample_arrays)
        self.sample_probs = concatenate(prob_arrays)

    def impose_boundaries(self, prop):
        d = prop - self.lower
        n = (d // self.width) % 2
        return self.lower + (1 - 2 * n) * (d % self.width) + n * self.width

    def pass_through(self, prop):
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

    def mode(self):
        """
        Return the sample with the current highest posterior probability.

        :return: \
            The model parameters corresponding to the highest observed
            posterior probability as a ``numpy.ndarray``.
        """
        return self.sample[self.sample_probs.argmax(), :]

    def get_parameter(self, index, burn=0, thin=1):
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

    def get_probabilities(self, burn=0, thin=1):
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

    def get_sample(self, burn=0, thin=1):
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
            "bounded": self.bounded,
            "alpha": self.alpha,
            "x_lwr": self.x_lwr,
            "x_width": self.x_width,
            "max_attempts": self.max_attempts,
            "display_progress": self.display_progress,
        }

        if self.bounded:
            D["lower"] = self.lower
            D["upper"] = self.upper
            D["width"] = self.width

        if self.sample is not None:
            D["sample"] = self.sample
            D["sample_probs"] = self.sample_probs

        savez(filename, **D)

    @classmethod
    def load(cls, filename, posterior=None):
        D = load(filename)
        sampler = cls(posterior=posterior, display_progress=bool(D["display_progress"]))

        sampler.theta = D["theta"]
        sampler.n_params = int(D["n_params"])
        sampler.n_walkers = int(D["n_walkers"])
        sampler.probs = D["probs"]
        sampler.n_iterations = int(D["n_iterations"])
        sampler.total_proposals = [list(v) for v in D["total_proposals"]]
        sampler.bounded = D["bounded"]
        sampler.alpha = D["alpha"]
        sampler.x_lwr = D["x_lwr"]
        sampler.x_width = D["x_width"]
        sampler.max_attempts = int(D["max_attempts"])

        if sampler.bounded:
            sampler.lower = D["lower"]
            sampler.upper = D["upper"]
            sampler.width = D["width"]
            sampler.process_proposal = sampler.impose_boundaries

        if "sample" in D:
            sampler.sample = D["sample"]
            sampler.sample_probs = D["sample_probs"]

        return sampler


class ChainProgressPrinter:
    def __init__(self, display=True, leading_msg=None):
        self.lead = "" if leading_msg is None else leading_msg

        if not display:
            self.iterations_initial = self.__no_status
            self.iterations_progress = self.__no_status
            self.iterations_final = self.__no_status
            self.percent_progress = self.__no_status
            self.percent_final = self.__no_status
            self.countdown_progress = self.__no_status
            self.countdown_final = self.__no_status

    def iterations_initial(self, total_itr):
        sys.stdout.write("\n")
        sys.stdout.write(f"\r  {self.lead}   [ 0 / {total_itr} iterations completed ]")
        sys.stdout.flush()

    def iterations_progress(self, t_start, current_itr, total_itr):
        dt = time() - t_start
        eta = int(dt * (total_itr / (current_itr + 1) - 1))
        sys.stdout.write(
            f"\r  {self.lead}   [ {current_itr + 1} / {total_itr} iterations completed  |  ETA: {eta} sec ]"
        )
        sys.stdout.flush()

    def iterations_final(self, total_itr):
        sys.stdout.write(
            f"\r  {self.lead}   [ {total_itr} / {total_itr} iterations completed ]                  "
        )
        sys.stdout.flush()
        sys.stdout.write("\n")

    def percent_progress(self, t_start, current_itr, total_itr):
        dt = time() - t_start
        pct = int(100 * (current_itr + 1) / total_itr)
        eta = int(dt * (total_itr / (current_itr + 1) - 1))
        sys.stdout.write(
            f"\r  {self.lead}   [ {pct}% complete  |  ETA: {eta} sec ]    "
        )
        sys.stdout.flush()

    def percent_final(self, t_start, total_itr):
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


def ESS(x):
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

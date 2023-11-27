from warnings import warn
from copy import copy, deepcopy

import matplotlib.pyplot as plt
from numpy import float64, ndarray
from numpy import array
from numpy import exp, log, mean, sqrt, argmax, diff
from numpy import percentile
from numpy import isfinite, savez, load

from numpy.random import normal, random
from inference.mcmc.utilities import ChainProgressPrinter, effective_sample_size
from inference.mcmc.base import MarkovChain


class Parameter:
    """
    This class is used by the markov-chain samplers in this module
    to manage data specific to each model parameter which is being
    sampled.

    The class also adjusts the parameter's proposal distribution
    width automatically as the chain advances in order to ensure
    efficient sampling.
    """

    def __init__(self, value: float, sigma: float):
        self.samples = [value]  # list to store all samples for the parameter
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

    def adjust_sigma(self, ratio: float):
        self.sigma *= ratio
        self.sigma_values.append(copy(self.sigma))
        self.sigma_checks.append(len(self.samples))
        self.avg = 0
        self.var = 0
        self.num = 0

    def add_sample(self, s):
        self.samples.append(s)
        self.try_count = 0

    def get_items(self, param_id: int) -> dict:
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

    @classmethod
    def load(cls, dictionary: dict, param_id: int):
        param = cls(1.0, 0.0)
        i = f"param_{param_id}"
        param.samples = list(dictionary[i + "samples"])
        param.sigma = float(dictionary[i + "sigma"])
        param.avg = float(dictionary[i + "avg"])
        param.var = float(dictionary[i + "var"])
        param.num = float(dictionary[i + "num"])
        param.sigma_values = list(dictionary[i + "sigma_values"])
        param.sigma_checks = list(dictionary[i + "sigma_checks"])
        param.try_count = int(dictionary[i + "try_count"])
        param.last_update = int(dictionary[i + "last_update"])
        param.target_rate = float(dictionary[i + "target_rate"])
        param.max_tries = int(dictionary[i + "max_tries"])
        param.chk_int = int(dictionary[i + "chk_int"])
        param.growth_factor = float(dictionary[i + "growth_factor"])
        param.adjust_rate = float(dictionary[i + "adjust_rate"])
        param._non_negative = bool(dictionary[i + "_non_negative"])
        param.bounded = bool(dictionary[i + "bounded"])
        param.upper = float(dictionary[i + "upper"])
        param.lower = float(dictionary[i + "lower"])
        param.width = float(dictionary[i + "width"])

        if param.bounded:
            param.proposal = param.boundary_proposal
        elif param._non_negative:
            param.proposal = param.abs_proposal
        else:
            param.proposal = param.standard_proposal
        return param


class MetropolisChain(MarkovChain):
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
        posterior: callable,
        start: ndarray,
        widths: ndarray = None,
        temperature: float = 1.0,
        display_progress: bool = True,
    ):
        self.inv_temp = 1.0 / temperature

        if posterior is not None:
            self.posterior = posterior

            # if widths are not specified, take 5% of the starting values (unless they're zero)
            if widths is None:
                widths = [v * 0.05 if v != 0 else 1.0 for v in start]

            # create a list of parameter objects
            self.params = [Parameter(value=v, sigma=s) for v, s in zip(start, widths)]

            # create storage
            self.chain_length = 1  # tracks total length of the chain
            self.n_parameters = len(start)  # number of posterior parameters
            self.probs = []  # list of probabilities for all steps

            # add starting point as first step in chain
            if len(self.params) != 0:
                self.probs.append(self.posterior(self.get_last()) * self.inv_temp)

                # check posterior value of chain starting point is finite
                if not isfinite(self.probs[0]):
                    ValueError(
                        """\n
                        \r[ MetropolisChain error ]
                        \r>> 'posterior' argument callable returns a non-finite value
                        \r>> for the starting position given to the 'start' argument.
                        """
                    )

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

        self.chain_length += 1

    def get_last(self):
        return array([p.samples[-1] for p in self.params], dtype=float64)

    def replace_last(self, theta):
        for p, t in zip(self.params, theta):
            p.samples[-1] = t

    def get_parameter(self, index: int, burn: int = 1, thin: int = 1) -> ndarray:
        """
        Return sample values for a chosen parameter.

        :param int index: \
            Index of the parameter for which samples are to be returned.

        :param int burn: \
            Number of samples to discard from the start of the chain.

        :param int thin: \
            Instead of returning every sample which is not discarded as part of the burn-in,
            every *m*'th sample is returned for a specified integer *m*.

        :return: \
            Samples for the parameter specified by ``index`` as a ``numpy.ndarray``.
        """
        return array(self.params[index].samples[burn::thin])

    def get_probabilities(self, burn: int = 1, thin: int = 1) -> ndarray:
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
        return array(self.probs[burn::thin])

    def get_sample(self, burn: int = 1, thin: int = 1) -> ndarray:
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
        return array([p.samples[burn::thin] for p in self.params]).T

    def mode(self) -> ndarray:
        """
        Return the sample with the current highest posterior probability.

        :return: Parameter values as a ``numpy.ndarray``.
        """
        ind = argmax(self.probs)
        return array([p.samples[ind] for p in self.params])

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
            effective_sample_size(array(self.get_parameter(i, burn=burn)))
            for i in range(self.n_parameters)
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
            min(self.probs[self.chain_length // 2 :]),
            max(self.probs) * 1.1 - 0.1 * min(self.probs[self.chain_length // 2 :]),
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
        ax2.plot(
            [0, self.chain_length * 1e-3], [5, 5], ls="dashed", lw=2, color="black"
        )
        ax2.plot(
            [0, self.chain_length * 1e-3], [-5, -5], ls="dashed", lw=2, color="black"
        )
        ax2.set_xlabel("chain step number ($10^3$)", fontsize=12)
        ax2.set_ylabel("% change in proposal widths", fontsize=12)
        ax2.set_title("Parameter proposal widths adjustment summary")
        ax2.set_ylim([-50, 50])
        ax2.grid()

        # parameter ESS plot
        ax3 = fig.add_subplot(223)
        ax3.bar(
            range(self.n_parameters), param_ESS, color=["C0", "C1", "C2", "C3", "C4"]
        )
        ax3.set_xlabel("parameter", fontsize=12)
        ax3.set_ylabel("effective sample size", fontsize=12)
        ax3.set_title("Parameter effective sample size estimate")
        ax3.set_xticks(range(self.n_parameters))

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

    def save(self, filename: str):
        """
        Save the entire state of the chain object as an .npz file.

        :param str filename: file path to which the chain will be saved.
        """
        # get the chain attributes
        items = {
            "chain_length": self.chain_length,
            "n_parameters": self.n_parameters,
            "probs": self.probs,
            "inv_temp": self.inv_temp,
            "display_progress": self.display_progress,
        }

        # get the parameter attributes
        for i, p in enumerate(self.params):
            items |= p.get_items(param_id=i)

        # save as npz
        savez(filename, **items)

    @classmethod
    def load(cls, filename: str, posterior=None):
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
        chain = cls(
            posterior=None,
            start=None,
            widths=None,
            display_progress=bool(D["display_progress"]),
        )

        # re-build the chain's attributes
        chain.posterior = posterior
        chain.chain_length = int(D["chain_length"])
        chain.n_parameters = int(D["n_parameters"])
        chain.probs = list(D["probs"])
        chain.inv_temp = float(D["inv_temp"])

        # re-build all the parameter objects
        chain.params = [
            Parameter.load(dictionary=D, param_id=i) for i in range(chain.n_parameters)
        ]
        return chain

    def estimate_burn_in(self) -> int:
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


class GibbsChain(MetropolisChain):
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
        self.chain_length += 1

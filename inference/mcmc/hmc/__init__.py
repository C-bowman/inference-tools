import matplotlib.pyplot as plt

from numpy import ndarray, float64
from numpy import array, savez, savez_compressed, load, zeros
from numpy import var, isfinite, exp, mean, argmax, percentile, cov
from numpy.random import default_rng

from inference.mcmc.utilities import Bounds, ChainProgressPrinter, effective_sample_size
from inference.mcmc.base import MarkovChain
from inference.mcmc.hmc.epsilon import EpsilonSelector
from inference.mcmc.hmc.mass import get_particle_mass


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

    :param start: \
        Vector of model parameters which correspond to the parameter-space coordinates
        at which the chain will start.

    :param func grad: \
        A function which returns the gradient of the log-posterior probability density
        for a given set of model parameters theta. If this function is not given, the
        gradient will instead be estimated by finite difference.

    :param float epsilon: \
        Initial guess for the time-step of the Hamiltonian dynamics simulation.

    :param float temperature: \
        The temperature of the markov chain. This parameter is used for parallel
        tempering and should be otherwise left unspecified.

    :param bounds: \
        An instance of the ``inference.mcmc.Bounds`` class, or a sequence of two
        ``numpy.ndarray`` specifying the lower and upper bounds for the parameters
        in the form ``(lower_bounds, upper_bounds)``.

    :param inverse_mass: \
        The inverse-mass can be given as either a vector or matrix, and is used to
        transform the momentum distribution so the chain can explore the posterior
        efficiently. If given as a vector, the inverse mass should have values which
        approximate the variance of the marginal distributions of each parameter.
        If given as a matrix, the inverse mass should be a valid covariance matrix
        for a multivariate normal distribution which approximates the posterior.

    :param bool display_progress: \
        If set as ``True``, a message is displayed during sampling
        showing the current progress and an estimated time until completion.
    """

    def __init__(
        self,
        posterior: callable,
        start: ndarray,
        grad: callable = None,
        epsilon: float = 0.1,
        temperature: float = 1.0,
        bounds: Bounds = None,
        inverse_mass: ndarray = None,
        display_progress=True,
    ):
        self.posterior = posterior
        self.rng = default_rng()
        # if no gradient function is supplied, default to finite difference
        self.grad = self.finite_diff if grad is None else grad

        self.temperature = temperature
        self.inv_temp = 1.0 / temperature

        if start is not None:
            start = start if isinstance(start, ndarray) else array(start)
            start = start if start.dtype is float64 else start.astype(float64)
            assert start.ndim == 1
            self._validate_posterior(posterior=posterior, start=start)
            self.theta = [start]
            self.probs = [self.posterior(start) * self.inv_temp]
            self.leapfrog_steps = [0]
            self.n_parameters = start.size
            self.mass = get_particle_mass(
                inverse_mass=inverse_mass if inverse_mass is not None else 1.0,
                n_parameters=self.n_parameters,
            )

        self.chain_length = 1

        # set either the bounded or unbounded leapfrog update
        if bounds is None:
            self.run_leapfrog = self.standard_leapfrog
            self.bounds = None
        else:
            self.run_leapfrog = self.bounded_leapfrog
            if isinstance(bounds, Bounds):
                self.bounds = bounds
            else:
                self.bounds = Bounds(
                    lower=bounds[0], upper=bounds[1], error_source="HamiltonianChain"
                )

            if start is not None:
                self.bounds.validate_start_point(start, error_source="HamiltonianChain")

        self.max_attempts = 200
        self.ES = EpsilonSelector(epsilon)
        self.steps = 50

        self.display_progress = display_progress
        self.ProgressPrinter = ChainProgressPrinter(
            display=self.display_progress, leading_msg="advancing chain:"
        )

    def take_step(self):
        """
        Takes the next step in the HMC-chain
        """
        steps_taken = 0
        for attempt in range(self.max_attempts):
            r0 = self.mass.sample_momentum(self.rng)
            t0 = self.theta[-1]
            H0 = self.kinetic_energy(r0) - self.probs[-1]

            n_steps = int(self.steps * (1 + (self.rng.random() - 0.5) * 0.2))
            t, r = self.run_leapfrog(t0.copy(), r0.copy(), n_steps)

            steps_taken += n_steps
            p = self.posterior(t) * self.inv_temp
            H = self.kinetic_energy(r) - p
            accept_prob = exp(H0 - H)

            self.ES.add_probability(
                min(accept_prob, 1) if isfinite(accept_prob) else 0.0
            )

            if (accept_prob >= 1) or (self.rng.random() <= accept_prob):
                break
        else:
            raise ValueError(
                f"""\n
                \r[ HamiltonianChain error ]
                \r>> Failed to take step within maximum allowed attempts of {self.max_attempts}
                """
            )

        self.theta.append(t)
        self.probs.append(p)
        self.leapfrog_steps.append(steps_taken)
        self.chain_length += 1

    def standard_leapfrog(
        self, t: ndarray, r: ndarray, n_steps: int
    ) -> tuple[ndarray, ndarray]:
        r_step = self.inv_temp * self.ES.epsilon
        r += (0.5 * r_step) * self.grad(t)

        for _ in range(n_steps - 1):
            t += self.ES.epsilon * self.mass.get_velocity(r)
            r += r_step * self.grad(t)

        t += self.ES.epsilon * self.mass.get_velocity(r)
        r += (0.5 * r_step) * self.grad(t)
        return t, r

    def bounded_leapfrog(
        self, t: ndarray, r: ndarray, n_steps: int
    ) -> tuple[ndarray, ndarray]:
        r_step = self.inv_temp * self.ES.epsilon
        r += (0.5 * r_step) * self.grad(t)

        for _ in range(n_steps - 1):
            t += self.ES.epsilon * self.mass.get_velocity(r)
            t, reflections = self.bounds.reflect_momenta(t)
            r *= reflections
            r += r_step * self.grad(t)

        t += self.ES.epsilon * self.mass.get_velocity(r)
        t, reflections = self.bounds.reflect_momenta(t)
        r *= reflections
        r += (0.5 * r_step) * self.grad(t)
        return t, r

    def hamiltonian(self, t: ndarray, r: ndarray) -> float:
        return 0.5 * (r @ self.mass.get_velocity(r)) - self.posterior(t) * self.inv_temp

    def kinetic_energy(self, r: ndarray) -> float:
        return 0.5 * (r @ self.mass.get_velocity(r))

    def estimate_mass(self, burn=1, thin=1, diagonal=True):
        if diagonal:
            inverse_mass = var(array(self.theta[burn::thin]), axis=0)
        else:
            inverse_mass = cov(array(self.theta[burn::thin]).T)
        self.mass = get_particle_mass(
            inverse_mass=inverse_mass, n_parameters=self.n_parameters
        )

    def finite_diff(self, t: ndarray) -> ndarray:
        p = self.posterior(t) * self.inv_temp
        G = zeros(self.n_parameters)
        for i in range(self.n_parameters):
            delta = zeros(self.n_parameters) + 1
            delta[i] += 1e-5
            G[i] = (self.posterior(t * delta) * self.inv_temp - p) / (t[i] * 1e-5)
        return G

    def get_last(self) -> ndarray:
        return self.theta[-1]

    def replace_last(self, theta: ndarray):
        self.theta[-1] = theta

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
        return array([v[index] for v in self.theta[burn::thin]]).squeeze()

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
            effective_sample_size(array(self.get_parameter(i, burn=burn, thin=1)))
            for i in range(self.n_parameters)
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
            min(self.probs[self.chain_length // 2 :]),
            max(self.probs) * 1.1 - 0.1 * min(self.probs[self.chain_length // 2 :]),
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
        ax2.set_yscale("log")
        ax2.grid()

        ax3 = fig.add_subplot(223)
        if self.n_parameters < 50:
            ax3.bar(
                range(self.n_parameters),
                param_ESS,
                color=["C0", "C1", "C2", "C3", "C4"],
            )
            ax3.set_xlabel("parameter", fontsize=12)
            ax3.set_ylabel("effective sample size", fontsize=12)
            ax3.set_title("Parameter effective sample size estimate")
            ax3.set_xticks(range(self.n_parameters))
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

    def get_sample(self, burn: int = 1, thin: int = 1):
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
        return array(self.theta[burn::thin])

    def mode(self) -> ndarray:
        return array(self.theta[argmax(self.probs)]).squeeze()

    def estimate_burn_in(self) -> int:
        # first get an estimate based on when the chain first reaches
        # the top 1% of log-probabilities
        prob_estimate = argmax(self.probs > percentile(self.probs, 99))
        # now we find the point at which the proposal width for each parameter
        # starts to deviate significantly from the current value
        epsl = abs((array(self.ES.epsilon_values)[::-1] / self.ES.epsilon) - 1.0)
        chks = array(self.ES.epsilon_checks)[::-1]
        epsl_estimate = chks[argmax(epsl > 0.15)] * self.ES.accept_rate
        return int(min(max(prob_estimate, epsl_estimate), 0.9 * self.chain_length))

    def save(self, filename, compressed=False):
        items = {
            "inv_mass": self.mass.inv_mass,
            "inv_temp": self.inv_temp,
            "theta": self.theta,
            "probs": self.probs,
            "leapfrog_steps": self.leapfrog_steps,
            "n_parameters": self.n_parameters,
            "chain_length": self.chain_length,
            "steps": self.steps,
            "display_progress": self.display_progress,
        }
        if self.bounds is not None:
            items.update(
                {"lower_bounds": self.bounds.lower, "upper_bounds": self.bounds.upper}
            )
        items.update(self.ES.get_items())

        # save as npz
        if compressed:
            savez_compressed(filename, **items)
        else:
            savez(filename, **items)

    @classmethod
    def load(cls, filename: str, posterior=None, grad=None):
        D = load(filename)

        if all(k in D for k in ["lower_bounds", "upper_bounds"]):
            bounds = Bounds(
                lower=D["lower_bounds"],
                upper=D["upper_bounds"],
                error_source="HamiltonianChain",
            )
        else:
            bounds = None

        chain = cls(
            posterior=posterior,
            start=None,
            grad=grad,
            bounds=bounds,
            inverse_mass=array(D["inv_mass"]),
            temperature=1.0 / float(D["inv_temp"]),
            display_progress=bool(D["display_progress"]),
        )

        chain.temperature = 1.0 / chain.inv_temp
        chain.probs = list(D["probs"])
        chain.leapfrog_steps = list(D["leapfrog_steps"])
        chain.n_parameters = int(D["n_parameters"])
        chain.chain_length = int(D["chain_length"])
        chain.steps = int(D["steps"])

        t = D["theta"]
        chain.theta = [t[i, :] for i in range(t.shape[0])]

        # build the epsilon selector
        chain.ES.load_items(D)
        return chain

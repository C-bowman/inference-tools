from copy import copy
from warnings import warn
import matplotlib.pyplot as plt

from numpy import array, savez, load, zeros
from numpy import sqrt, exp, dot, cov
from numpy.random import random, normal
from scipy.linalg import eigh

from inference.mcmc.markov import MarkovChain, Parameter


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
        if hasattr(self, "n_variables"):
            for i in range(self.n_variables):
                v = zeros(self.n_variables)
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
            if len(parameter_boundaries) == self.n_variables:
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
        data = array(
            [self.get_parameter(i)[self.last_update :] for i in range(self.n_variables)]
        )
        if hasattr(self, "covar"):
            nu = min(2 * self.dir_update_interval / self.last_update, 0.5)
            self.covar = self.covar * (1 - nu) + nu * cov(data)
        else:
            self.covar = cov(data)

        w, V = eigh(self.covar)

        # find the sine of the angle between the old and new eigenvectors to track convergence
        angles = [
            sqrt(1.0 - dot(V[:, i], self.directions[i]) ** 2)
            for i in range(self.n_variables)
        ]
        self.angles_history.append(angles)
        self.update_history.append(copy(self.chain_length))

        # store the new directions and plan the next update
        self.directions = [V[:, i] for i in range(self.n_variables)]
        self.last_update = copy(self.chain_length)
        self.dir_update_interval = int(
            self.dir_update_interval * self.dir_growth_factor
        )
        self.next_update = self.last_update + self.dir_update_interval

    def directions_diagnostics(self):
        for i in range(self.n_variables):
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
        self.chain_length += 1

        if self.chain_length == self.next_update:
            self.update_directions()

    def save(self, filename):
        """
        Save the entire state of the chain object as an .npz file.

        :param str filename: file path to which the chain will be saved.
        """
        # get the chain attributes
        items = {
            "chain_length": self.chain_length,
            "n_variables": self.n_variables,
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
        chain.chain_length = int(D["chain_length"])
        chain.n_variables = int(D["n_variables"])
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
        for i in range(chain.n_variables):
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

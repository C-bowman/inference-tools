"""
.. moduleauthor:: Chris Bowman <chris.bowman.physics@gmail.com>
"""


class Posterior:
    """
    Class for constructing a posterior distribution object for a given likelihood and prior.

    :param callable likelihood: \
        A callable which returns the log-likelihood probability when passed a vector of
        the model parameters.

    :param callable prior: \
        A callable which returns the log-prior probability when passed a vector of the
        model parameters.
    """

    def __init__(self, likelihood, prior):
        self.likelihood = likelihood
        self.prior = prior

    def __call__(self, theta):
        """
        Returns the log-posterior probability for the given set of model parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The log-posterior probability.
        """
        return self.likelihood(theta) + self.prior(theta)

    def gradient(self, theta):
        """
        Returns the gradient of the log-posterior with respect to model parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The gradient of the log-posterior as a 1D ``numpy.ndarray``.
        """
        return self.likelihood.gradient(theta) + self.prior.gradient(theta)

    def cost(self, theta):
        """
        Returns the 'cost', defined as the negative log-posterior probability, for the
        given set of model parameters. Minimising the value of the cost therefore
        maximises the log-posterior probability.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The negative log-posterior probability.
        """
        return -(self.likelihood(theta) + self.prior(theta))

    def cost_gradient(self, theta):
        """
        Returns the gradient of the negative log-posterior with respect to model parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The gradient of the negative log-posterior as a 1D ``numpy.ndarray``.
        """
        return -(self.likelihood.gradient(theta) + self.prior.gradient(theta))

    def generate_initial_guesses(self, n_guesses=1, prior_samples=100):
        """
        Generates initial guesses for optimisation or MCMC algorithms by drawing samples
        from the prior and returning a sub-set having the highest posterior log-probability.

        :param int n_guesses: \
            The number of initial guesses returned.

        :param int prior_samples: \
            The number of samples which will be drawn from the prior.

        :returns: \
            A list containing the initial guesses as 1D numpy arrays.
        """
        if type(n_guesses) is not int or type(prior_samples) is not int:
            raise TypeError("""'n_guesses' and 'prior_samples' must both be integers""")

        if n_guesses < 1 or prior_samples < 1:
            raise ValueError(
                """'n_guesses' and 'prior_samples' must both be greater than zero"""
            )

        if n_guesses > prior_samples:
            raise ValueError(
                """The value of 'n_guesses' must be less than that of 'prior_samples'"""
            )

        samples = sorted(
            [self.prior.sample() for _ in range(prior_samples)], key=self.cost
        )
        return samples[:n_guesses]

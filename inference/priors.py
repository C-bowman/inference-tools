"""
.. moduleauthor:: Chris Bowman <chris.bowman.physics@gmail.com>
"""
from typing import Union, Iterable

from numpy import array, log, pi, zeros, concatenate, float64, where
from numpy.random import normal, exponential, uniform
from itertools import chain


class JointPrior:
    """
    A class which combines multiple prior distribution objects into a single
    joint-prior distribution object.

    :param components: \
        A list of prior distribution objects (e.g. GaussianPrior, ExponentialPrior)
        which will be combined into a single joint-prior object.

    :param int n_variables: \
        The total number of model variables.
    """

    def __init__(self, components, n_variables):
        if not all(isinstance(c, BasePrior) for c in components):
            raise TypeError(
                """
                All objects contained in the 'components' argument must be instances
                of a subclass of BasePrior (e.g. GaussianPrior, UniformPrior)
                """
            )

        # Combine any prior components which are of the same type
        self.components = []
        for cls in [GaussianPrior, ExponentialPrior, UniformPrior]:
            L = [c for c in components if isinstance(c, cls)]
            if len(L) == 1:
                self.components.extend(L)
            elif len(L) > 1:
                self.components.append(cls.combine(L))

        # check that no variable appears more than once across all prior components
        self.prior_variables = []
        for var in chain(*[c.variables for c in self.components]):
            if var in self.prior_variables:
                raise ValueError(
                    f"Variable index '{var}' appears more than once in prior components"
                )
            self.prior_variables.append(var)

        if len(self.prior_variables) != n_variables:
            raise ValueError(
                f"""
                The total number of variables specified across the various prior
                components ({len(self.prior_variables)}) does not match the number specified in
                the 'n_variables' argument ({n_variables}).
                """
            )

        if not all(0 <= i < n_variables for i in self.prior_variables):
            raise ValueError(
                """
                All variable indices given to the prior components must have values
                in the range [0, n_variables-1].
                """
            )

        self.n_variables = n_variables

        all_bounds = chain(*[c.bounds for c in self.components])
        all_inds = chain(*[c.variables for c in self.components])
        both = sorted(
            [(b, i) for b, i in zip(all_bounds, all_inds)], key=lambda x: x[1]
        )
        self.bounds = [v[0] for v in both]

    def __call__(self, theta):
        """
        Returns the joint-prior log-probability value, calculated as the sum
        of the log-probabilities from each prior component for the provided
        set of model parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The prior log-probability value.
        """
        return sum(c(theta) for c in self.components)

    def gradient(self, theta):
        """
        Returns the gradient of the prior log-probability with respect to the model
        parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The gradient of the prior log-probability with respect to the model parameters.
        """
        grad = zeros(self.n_variables)
        for c in self.components:
            grad[c.variables] = c.gradient(theta)
        return grad

    def sample(self):
        """
        Draws a sample from the prior.

        :returns: \
            A single sample from the prior distribution as a 1D ``numpy.ndarray``.
        """
        sample = zeros(self.n_variables)
        for c in self.components:
            sample[c.variables] = c.sample()
        return sample


class BasePrior:
    @staticmethod
    def check_variables(variable_inds: Union[int, Iterable[int]], n_vars: int):
        if not isinstance(variable_inds, (int, Iterable)):
            raise TypeError("'variable_inds' must be an integer or list of integers")

        if isinstance(variable_inds, int):
            variable_inds = [variable_inds]

        if not all(isinstance(p, int) for p in variable_inds):
            raise TypeError("'variable_inds' must be an integer or list of integers")

        if n_vars != len(variable_inds):
            raise ValueError(
                """
                The total number of variables specified via the 'variable_indices' argument is
                inconsistent with the number specified by the other arguments.
                """
            )

        if len(variable_inds) != len(set(variable_inds)):
            raise ValueError(
                """
                All integers given via the 'variable_indices' must be unique.
                Two or more of the given integers are duplicates.
                """
            )

        return variable_inds


class GaussianPrior(BasePrior):
    """
    A class for generating a Gaussian prior for one or more of the model variables.

    :param mean: \
        A list specifying the means of the Gaussian priors on each of the variables specified
        in the ``variable_indices`` argument.

    :param sigma: \
        A list specifying the standard deviations of the Gaussian priors on each of the
        variables specified in the ``variable_indices`` argument.

    :param variable_indices: \
        A list of integers specifying the indices of the variables to which the prior will apply.
    """

    def __init__(self, mean, sigma, variable_indices):

        self.mean = array(mean, dtype=float64).squeeze()
        self.sigma = array(sigma, dtype=float64).squeeze()

        # if parameters were passed as floats, convert from 0D to 1D arrays
        if self.mean.ndim == 0:
            self.mean = self.mean.reshape([1])
        if self.sigma.ndim == 0:
            self.sigma = self.sigma.reshape([1])

        self.n_params = self.mean.size

        if self.mean.size != self.sigma.size:
            raise ValueError(
                "mean and sigma arguments must have the same number of elements"
            )

        if self.mean.ndim > 1 or self.sigma.ndim > 1:
            raise ValueError("mean and sigma arguments must be 1D arrays")

        if not (self.sigma > 0.0).all():
            raise ValueError('All values of "sigma" must be greater than zero')

        self.variables = self.check_variables(variable_indices, self.n_params)

        # pre-calculate some quantities as an optimisation
        self.inv_sigma = 1.0 / self.sigma
        self.inv_sigma_sqr = self.inv_sigma**2
        self.normalisation = -log(self.sigma).sum() - 0.5 * log(2 * pi) * self.n_params
        self.bounds = [(None, None)] * self.n_params

    def __call__(self, theta):
        """
        Returns the prior log-probability value for the provided set of model parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The prior log-probability value.
        """
        z = (self.mean - theta[self.variables]) * self.inv_sigma
        return -0.5 * (z**2).sum() + self.normalisation

    def gradient(self, theta):
        """
        Returns the gradient of the prior log-probability with respect to the model
        parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The gradient of the prior log-probability with respect to the model parameters.
        """
        return (self.mean - theta[self.variables]) * self.inv_sigma_sqr

    def sample(self):
        """
        Draws a sample from the prior.

        :returns: \
            A single sample from the prior distribution as a 1D ``numpy.ndarray``.
        """
        return normal(loc=self.mean, scale=self.sigma)

    @classmethod
    def combine(cls, priors):
        if not all(isinstance(p, cls) for p in priors):
            raise ValueError(f"All prior objects being combined must be of type {cls}")

        variables = []
        for p in priors:
            variables.extend(p.variables)

        means = concatenate([p.mean for p in priors])
        sigmas = concatenate([p.sigma for p in priors])

        return cls(mean=means, sigma=sigmas, variable_indices=variables)


class ExponentialPrior(BasePrior):
    """
    A class for generating an exponential prior for one or more of the model variables.

    :param beta: \
        A list specifying the 'beta' parameter value of the exponential priors on each of the
        variables specified in the ``variable_indices`` argument.

    :param variable_indices: \
        A list of integers specifying the indices of the variables to which the prior will apply.
    """

    def __init__(self, beta, variable_indices):

        self.beta = array(beta, dtype=float64).squeeze()
        if self.beta.ndim == 0:
            self.beta = self.beta.reshape([1])
        self.n_params = self.beta.size

        if self.beta.ndim > 1:
            raise ValueError("beta argument must be a 1D array")

        if not (self.beta > 0.0).all():
            raise ValueError('All values of "beta" must be greater than zero')

        self.variables = self.check_variables(variable_indices, self.n_params)

        # pre-calculate some quantities as an optimisation
        self.lam = 1.0 / self.beta
        self.normalisation = log(self.lam).sum()
        self.zeros = zeros(self.n_params)
        self.bounds = [(0.0, None)] * self.n_params

    def __call__(self, theta):
        """
        Returns the prior log-probability value for the provided set of model parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The prior log-probability value.
        """
        if (theta[self.variables] < 0.0).any():
            return -1e100
        return -(self.lam * theta[self.variables]).sum() + self.normalisation

    def gradient(self, theta):
        """
        Returns the gradient of the prior log-probability with respect to the model
        parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The gradient of the prior log-probability with respect to the model parameters.
        """
        return where(theta[self.variables] >= 0.0, -self.lam, self.zeros)

    def sample(self):
        """
        Draws a sample from the prior.

        :returns: \
            A single sample from the prior distribution as a 1D ``numpy.ndarray``.
        """
        return exponential(scale=self.beta)

    @classmethod
    def combine(cls, priors):
        if not all(isinstance(p, cls) for p in priors):
            raise ValueError(f"All prior objects being combined must be of type {cls}")

        variables = []
        for p in priors:
            variables.extend(p.variables)

        betas = concatenate([p.beta for p in priors])
        return cls(beta=betas, variable_indices=variables)


class UniformPrior(BasePrior):
    """
    A class for generating a uniform prior for one or more of the model variables.

    :param lower: \
        A list specifying the lower bound of the uniform priors on each of the variables
        specified in the ``variable_indices`` argument.

    :param upper: \
        A list specifying the upper bound of the uniform priors on each of the variables
        specified in the ``variable_indices`` argument.

    :param variable_indices: \
        A list of integers specifying the indices of the variables to which the prior will apply.
    """

    def __init__(self, lower, upper, variable_indices):
        self.lower = array(lower).squeeze()
        self.upper = array(upper).squeeze()

        # if parameters were passed as floats, convert from 0D to 1D arrays
        self.lower = self.lower.reshape([1]) if self.lower.ndim == 0 else self.lower
        self.upper = self.upper.reshape([1]) if self.upper.ndim == 0 else self.upper

        self.n_params = self.lower.size
        self.grad = zeros(self.n_params)

        if self.lower.size != self.upper.size:
            raise ValueError(
                """'lower' and 'upper' arguments must have the same number of elements"""
            )

        if self.lower.ndim > 1 or self.upper.ndim > 1:
            raise ValueError("'lower' and 'upper' arguments must be 1D arrays")

        if (self.upper <= self.lower).any():
            raise ValueError(
                "All values in 'lower' must be less than the corresponding values in 'upper'"
            )

        self.variables = self.check_variables(variable_indices, self.n_params)

        # pre-calculate some quantities as an optimisation
        self.normalisation = -log(self.upper - self.lower).sum()
        self.bounds = [(lo, up) for lo, up in zip(self.lower, self.upper)]

    def __call__(self, theta):
        """
        Returns the prior log-probability value for the provided set of model parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The prior log-probability value.
        """
        t = theta[self.variables]
        inside = (self.lower <= t) & (t <= self.upper)
        if inside.all():
            return self.normalisation
        return -1e100

    def gradient(self, theta):
        """
        Returns the gradient of the prior log-probability with respect to the model
        parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The gradient of the prior log-probability with respect to the model parameters.
        """
        return self.grad

    def sample(self):
        """
        Draws a sample from the prior.

        :returns: \
            A single sample from the prior distribution as a 1D ``numpy.ndarray``.
        """
        return uniform(low=self.lower, high=self.upper)

    @classmethod
    def combine(cls, priors):
        if not all(isinstance(p, cls) for p in priors):
            raise ValueError(f"All prior objects being combined must be of type {cls}")

        variables = []
        for p in priors:
            variables.extend(p.variables)

        lower = concatenate([p.lower for p in priors])
        upper = concatenate([p.upper for p in priors])

        return cls(lower=lower, upper=upper, variable_indices=variables)

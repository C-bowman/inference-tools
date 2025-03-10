"""
.. moduleauthor:: Chris Bowman <chris.bowman.physics@gmail.com>
"""

from abc import ABC, abstractmethod
from typing import Union, Iterable
from itertools import chain
from numpy import atleast_1d, log, pi, zeros, concatenate, where, ndarray, isfinite
from numpy.random import default_rng

rng = default_rng()


class BasePrior(ABC):
    variables: list[int]

    @staticmethod
    def validate_variable_indices(
        variable_inds: Union[int, Iterable[int]],
        n_parameters: int,
        class_name="BasePrior",
    ) -> list[int]:
        indices_type_error = TypeError(
            f"""\n
            \r[ {class_name} error ]
            \r>> 'variable_inds' argument of {class_name} must be
            \r>> given as an integer or list of integers
            """
        )

        if not isinstance(variable_inds, (int, Iterable)):
            raise indices_type_error

        if isinstance(variable_inds, int):
            variable_inds = [variable_inds]

        if not all(isinstance(p, int) for p in variable_inds):
            raise indices_type_error

        if not isinstance(variable_inds, list):
            variable_inds = list(variable_inds)

        if n_parameters != len(variable_inds):
            raise ValueError(
                f"""\n
                \r[ {class_name} error ]
                \r>> The total number of variables specified via the 'variable_indices' argument
                \r>> is inconsistent with the number specified by the other arguments.
                """
            )

        if len(variable_inds) != len(set(variable_inds)):
            raise ValueError(
                f"""\n
                \r[ {class_name} error ]
                \r>> All integers given via the 'variable_indices' must be unique.
                \r>> Two or more of the given integers are duplicates.
                """
            )

        return variable_inds

    @abstractmethod
    def __call__(self, theta: ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, theta: ndarray) -> ndarray:
        pass

    def cost(self, theta: ndarray) -> float:
        """
        Returns the 'cost', equal to the negative prior log-probability, for the
        provided set of model parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The negative prior log-probability value.
        """
        return -self(theta)

    def cost_gradient(self, theta: ndarray) -> ndarray:
        """
        Returns the gradient of the 'cost', equal to the negative prior log-probability,
        with respect to the model parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The gradient of the negative prior log-probability value.
        """
        return -self.gradient(theta)

    def sample(self) -> ndarray:
        """
        Draws a sample from the prior.

        :returns: \
            A single sample from the prior distribution as a 1D ``numpy.ndarray``.
        """
        raise NotImplementedError(
            f"""\n
            \r[ {self.__class__.__name__} error ]
            \r>> 'sample' is an optional method for classes inheriting from
            \r>> 'BasePrior', and has not been implemented for '{self.__class__.__name__}'.
            """
        )


class JointPrior(BasePrior):
    """
    A class which combines multiple prior distribution objects into a single
    joint-prior distribution object.

    :param components: \
        A list of prior distribution objects (e.g. GaussianPrior, ExponentialPrior)
        which will be combined into a single joint-prior object.

    :param int n_variables: \
        The total number of model variables.
    """

    def __init__(self, components: list[BasePrior], n_variables: int):
        if not all(isinstance(c, BasePrior) for c in components):
            raise TypeError(
                """\n
                \r[ JointPrior error ]
                \r>> The sequence of prior objects passed to the 'components' argument 
                \r>> of 'JointPrior' must be instances of a subclass of 'BasePrior'.
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
                    f"""\n
                    \r[ JointPrior error ]
                    \r>> Variable index '{var}' appears more than once in the prior
                    \r>> objects passed to the 'components' argument of 'JointPrior'.
                    """
                )
            self.prior_variables.append(var)

        if len(self.prior_variables) != n_variables:
            raise ValueError(
                f"""\n
                \r[ JointPrior error ]
                \r>> The total number of variables specified across the various prior
                \r>> components ({len(self.prior_variables)}) does not match the number
                \r>> specified in the 'n_variables' argument ({n_variables}).
                """
            )

        if not all(0 <= i < n_variables for i in self.prior_variables):
            raise ValueError(
                """\n
                \r[ JointPrior error ]
                \r>> All variable indices specified across the various prior
                \r>> objects passed to the 'components' argument of 'JointPrior'
                \r>> must have values in the range [0, n_variables - 1].
                """
            )

        self.n_variables = n_variables

        all_bounds = chain(*[c.bounds for c in self.components])
        all_inds = chain(*[c.variables for c in self.components])
        both = sorted(
            [(b, i) for b, i in zip(all_bounds, all_inds)], key=lambda x: x[1]
        )
        self.bounds = [v[0] for v in both]

    def __call__(self, theta: ndarray) -> float:
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

    def gradient(self, theta: ndarray) -> ndarray:
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

    def sample(self) -> ndarray:
        """
        Draws a sample from the prior.

        :returns: \
            A single sample from the prior distribution as a 1D ``numpy.ndarray``.
        """
        sample = zeros(self.n_variables)
        for c in self.components:
            sample[c.variables] = c.sample()
        return sample


class GaussianPrior(BasePrior):
    """
    A class for generating a Gaussian prior for one or more of the model variables.

    :param mean: \
        The means of the Gaussian priors on each of the variables specified
        in the ``variable_indices`` argument as a 1D ``numpy.ndarray``.

    :param sigma: \
        The standard deviations of the Gaussian priors on each of the variables
        specified in the ``variable_indices`` argument as a 1D ``numpy.ndarray``.

    :param variable_indices: \
        A list of integers specifying the indices of the variables to which the prior will apply.
    """

    def __init__(self, mean: ndarray, sigma: ndarray, variable_indices: list[int]):
        self.mean, self.sigma = validate_prior_parameters(
            class_name="GaussianPrior",
            params=[("mean", mean), ("sigma", sigma)],
            require_positive={"sigma"},
        )

        self.n_params = self.mean.size
        self.variables = self.validate_variable_indices(
            variable_inds=variable_indices,
            n_parameters=self.n_params,
            class_name="GaussianPrior",
        )

        # pre-calculate some quantities as an optimisation
        self.inv_sigma = 1.0 / self.sigma
        self.inv_sigma_sqr = self.inv_sigma**2
        self.normalisation = -log(self.sigma).sum() - 0.5 * log(2 * pi) * self.n_params
        self.bounds = [(None, None)] * self.n_params

    def __call__(self, theta: ndarray) -> float:
        """
        Returns the prior log-probability value for the provided set of model parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The prior log-probability value.
        """
        z = (self.mean - theta[self.variables]) * self.inv_sigma
        return -0.5 * (z**2).sum() + self.normalisation

    def gradient(self, theta: ndarray) -> ndarray:
        """
        Returns the gradient of the prior log-probability with respect to the model
        parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The gradient of the prior log-probability with respect to the model parameters.
        """
        return (self.mean - theta[self.variables]) * self.inv_sigma_sqr

    def sample(self) -> ndarray:
        """
        Draws a sample from the prior.

        :returns: \
            A single sample from the prior distribution as a 1D ``numpy.ndarray``.
        """
        return rng.normal(loc=self.mean, scale=self.sigma)

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
        The 'beta' parameter values of the exponential priors on each of the variables
        specified in the ``variable_indices`` argument as a 1D ``numpy.ndarray``.

    :param variable_indices: \
        A list of integers specifying the indices of the variables to which the prior will apply.
    """

    def __init__(self, beta: ndarray, variable_indices: list[int]):
        (self.beta,) = validate_prior_parameters(
            class_name="ExponentialPrior",
            params=[("beta", beta)],
            require_positive={"beta"},
        )

        self.n_params = self.beta.size
        self.variables = self.validate_variable_indices(
            variable_inds=variable_indices,
            n_parameters=self.n_params,
            class_name="ExponentialPrior",
        )

        # pre-calculate some quantities as an optimisation
        self.lam = 1.0 / self.beta
        self.normalisation = log(self.lam).sum()
        self.zeros = zeros(self.n_params)
        self.bounds = [(0.0, None)] * self.n_params

    def __call__(self, theta: ndarray) -> float:
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

    def gradient(self, theta: ndarray) -> ndarray:
        """
        Returns the gradient of the prior log-probability with respect to the model
        parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The gradient of the prior log-probability with respect to the model parameters.
        """
        return where(theta[self.variables] >= 0.0, -self.lam, self.zeros)

    def sample(self) -> ndarray:
        """
        Draws a sample from the prior.

        :returns: \
            A single sample from the prior distribution as a 1D ``numpy.ndarray``.
        """
        return rng.exponential(scale=self.beta)

    @classmethod
    def combine(cls, priors: list[BasePrior]):
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
        The lower bound of the uniform priors on each of the variables
        specified in the ``variable_indices`` argument as a 1D ``numpy.ndarray``.

    :param upper: \
        The upper bound of the uniform priors on each of the variables
        specified in the ``variable_indices`` argument as a 1D ``numpy.ndarray``.

    :param variable_indices: \
        A list of integers specifying the indices of the variables to which the prior will apply.
    """

    def __init__(self, lower: ndarray, upper: ndarray, variable_indices: list[int]):
        self.lower, self.upper = validate_prior_parameters(
            class_name="UniformPrior", params=[("lower", lower), ("upper", upper)]
        )

        self.n_params = self.lower.size
        self.grad = zeros(self.n_params)

        if (self.upper <= self.lower).any():
            raise ValueError(
                """\n
                \r[ UniformPrior error ]
                \r>> All values in 'lower' must be less than the corresponding values in 'upper'
                """
            )

        self.variables = self.validate_variable_indices(
            variable_inds=variable_indices,
            n_parameters=self.n_params,
            class_name="UniformPrior",
        )

        # pre-calculate some quantities as an optimisation
        self.normalisation = -log(self.upper - self.lower).sum()
        self.bounds = [(lo, up) for lo, up in zip(self.lower, self.upper)]

    def __call__(self, theta: ndarray) -> float:
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

    def gradient(self, theta: ndarray) -> ndarray:
        """
        Returns the gradient of the prior log-probability with respect to the model
        parameters.

        :param theta: \
            The model parameters as a 1D ``numpy.ndarray``.

        :returns: \
            The gradient of the prior log-probability with respect to the model parameters.
        """
        return self.grad

    def sample(self) -> ndarray:
        """
        Draws a sample from the prior.

        :returns: \
            A single sample from the prior distribution as a 1D ``numpy.ndarray``.
        """
        return rng.uniform(low=self.lower, high=self.upper)

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


def validate_prior_parameters(
    class_name: str, params: list[tuple], require_positive: set[str] = frozenset()
) -> list[ndarray]:
    validated_params = []
    for param_name, param in params:
        if attempt_array_conversion(param):
            param = atleast_1d(param).astype(float)

        if not isinstance(param, ndarray):
            raise TypeError(
                f"""\n
                \r[ {class_name} error ]
                \r>> Argument '{param_name}' should be an instance of a numpy.ndarray,
                \r>> but instead has type:
                \r>> {type(param)}
                """
            )

        if param.ndim != 1:
            raise ValueError(
                f"""\n
                \r[ {class_name} error ]
                \r>> Argument '{param_name}' should be a 1D numpy.ndarray, 
                \r>> but has {param.ndim} dimensions and shape {param.shape}.
                """
            )

        if not isfinite(param).all():
            raise ValueError(
                f"""\n
                \r[ {class_name} error ]
                \r>> Argument '{param_name}' contains non-finite values.
                """
            )

        if param_name in require_positive:
            if not (param > 0.0).all():
                raise ValueError(
                    f"""\n
                    \r[ {class_name} error ]
                    \r>> All values given in '{param_name}' must be greater than zero.
                    """
                )

        validated_params.append(param)

    # check all inputs are the same size by collecting their sizes in a set
    if len({param.size for param in validated_params}) != 1:
        raise ValueError(
            f"""\n
            \r[ {class_name} error ]
            \r>> Arguments
            \r>> {[param_name for param_name, _ in params]}
            \r>> must all be arrays of equal size, but instead have sizes
            \r>> {[param.size for param in validated_params]}
            \r>> respectively.
            """
        )

    return validated_params


def attempt_array_conversion(param) -> bool:
    # if input is a zero-dimensional array, we need to convert to 1D
    zero_dim_array = isinstance(param, ndarray) and param.ndim == 0
    # if the input is a float or an int, also convert to a 1D array
    valid_number = isinstance(param, (int, float))
    # if the input is a list or tuple containing only floats and ints, also convert
    valid_sequence = isinstance(param, (list, tuple)) and all(
        isinstance(v, (int, float)) for v in param
    )
    return zero_dim_array or valid_sequence or valid_number

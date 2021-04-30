
from numpy import array, log, pi, zeros, concatenate, float64, where
from numpy.random import normal, exponential, uniform
from itertools import chain

# individual prior components are given only the parameters they act on
# individual prior components shouldnt know anything about overall parameter set
# joint prior should know the full parameter set

class JointPrior(object):
    """
    A class which combines multiple prior distribution objects into a single
    joint-prior distribution object.

    :param model_variables: \
        An ordered list of strings identifying all parameters in the model. This list
        is used to determine which elements of a vector of model parameters need to be
        passed to the various prior distribution objects given in the 'components' list.
        For this reason the n'th string in the list *must* correspond to the n'th model
        parameter.

    :param components: \
        A list of prior distribution objects (e.g. GaussianPrior, ExponentialPrior)
        which will be combined into a single joint-prior object.
    """
    def __init__(self, model_variables, components):
        if not all(isinstance(c, BasePrior) for c in components):
            raise TypeError(
                """
                All objects contained in the 'components' argument must be instances
                of a subclass of BasePrior (e.g. GaussianPrior, UniformPrior)
                """
            )

        # Combine any prior components which are of the same type
        self.components = []
        for cls in [GaussianPrior, ExponentialPrior]:
            L = [c for c in components if isinstance(c, cls)]
            if len(L) == 1:
                self.components.extend(L)
            elif len(L) > 1:
                self.components.append(cls.combine(L))

        # check that no variable appears more than once across all prior components
        self.prior_variables = []
        for var in chain(*[c.variables for c in self.components]):
            if var not in self.prior_variables:
                self.prior_variables.append(var)
            else:
                raise ValueError("""Variable identifier string '{}' appears more than once in prior components""".format(var))

        # check that model_variables is a list of strings as expected
        if type(model_variables) is not list or not all( type(var) is str for var in model_variables):
            raise ValueError("""'model_variables' argument must be a list of strings""")

        # make sure model_variables contains no duplicates
        if len(model_variables) != len(set(model_variables)):
            raise ValueError("""All strings given in 'model_variables' must be unique - one or more of the strings given were duplicates""")

        # ensure that all strings given to the prior components are present in model_variables
        if not set(self.prior_variables).issubset(set(model_variables)):
            raise ValueError(
                """
                All variable identifier strings given to the various components of the prior must
                be a sub-set of those listed in the model_variables argument.
                """
            )

        self.model_variable_map = {var: i for i, var in enumerate(model_variables)}
        self.variable_indices = [[self.model_variable_map[v] for v in c.variables] for c in self.components]
        self.n_variables = len(model_variables)

    def __call__(self, theta):
        return sum(c(theta[i]) for c, i in zip(self.components, self.variable_indices))

    def gradient(self, theta):
        grad = zeros(self.n_variables)
        for c,i in zip(self.components, self.variable_indices):
            grad[i] = c.gradient(theta[i])
        return grad

    def sample(self):
        sample = zeros(self.n_variables)
        for c,i in zip(self.components, self.variable_indices):
            sample[i] = c.sample()
        return sample






class BasePrior(object):
    @staticmethod
    def check_variables(variables, n_vars):
        if type(variables) is str:
            if n_vars == 1:
                return [variables]
            else:
                raise ValueError(
                    """
                    The total number of variables specified via the 'variables' argument is inconsistent
                    with the number specified by the other arguments.
                    """
                )

        elif type(variables) is list and all(type(p) is str for p in variables):
            return variables

        else:
            raise TypeError('The "variables" argument must be an string or list of strings')






class GaussianPrior(BasePrior):
    """
    A class for generating a Gaussian prior for one or more of the model variables.

    :param mean: \
        A list specifying the means of the Gaussian priors on each variable.

    :param sigma: \
        A list specifying the standard deviations of the Gaussian priors on each variable.

    :param variables: \
        A list of strings specifying the names of the variables for which the priors are generated.
    """
    def __init__(self, mean, sigma, variables):

        self.mean = array(mean, dtype=float64).squeeze()
        self.sigma = array(sigma, dtype=float64).squeeze()

        # if parameters were passed as floats, convert from 0D to 1D arrays
        if self.mean.ndim == 0: self.mean = self.mean.reshape([1])
        if self.sigma.ndim == 0: self.sigma = self.sigma.reshape([1])

        self.n_params = self.mean.size

        if self.mean.size != self.sigma.size:
            raise ValueError('mean and sigma arguments must have the same number of elements')

        if self.mean.ndim > 1 or self.sigma.ndim > 1:
            raise ValueError('mean and sigma arguments must be 1D arrays')

        if not (self.sigma > 0.).all():
            raise ValueError('All values of "sigma" must be greater than zero')

        self.variables = self.check_variables(variables, self.n_params)

        # pre-calculate some quantities as an optimisation
        self.inv_sigma = 1./self.sigma
        self.inv_sigma_sqr = self.inv_sigma**2
        self.normalisation = -log(self.sigma).sum() - 0.5*log(2*pi)*self.n_params

    def __call__(self, theta):
        z = (self.mean-theta)*self.inv_sigma
        return -0.5*(z**2).sum() + self.normalisation

    def gradient(self, theta):
        return (self.mean-theta)*self.inv_sigma_sqr

    def sample(self):
        return normal(loc=self.mean, scale=self.sigma)

    @classmethod
    def combine(cls, priors):
        if not all(type(p) is cls for p in priors):
            raise ValueError(
                f"""
                All prior objects being combined must be of type {cls}
                """
            )

        variables = []
        for p in priors:
            variables.extend(p.variables)

        means = concatenate([p.mean for p in priors])
        sigmas = concatenate([p.sigma for p in priors])

        return cls(mean=means, sigma=sigmas, variables=variables)






class ExponentialPrior(BasePrior):
    """
    A class for generating an exponential prior for one or more of the model variables.

    :param beta: \
        A list specifying the 'beta' parameter value of the exponential priors on each variable.

    :param variables: \
        A list of strings specifying the names of the variables for which the priors are generated.
    """
    def __init__(self, beta, variables):

        self.beta = array(beta, dtype=float64).squeeze()
        if self.beta.ndim == 0: self.beta = self.beta.reshape([1])
        self.n_params = self.beta.size

        if self.beta.ndim > 1:
            raise ValueError('beta argument must be a 1D array')

        if not (self.beta > 0.).all():
            raise ValueError('All values of "beta" must be greater than zero')

        self.variables = self.check_variables(variables, self.n_params)

        # pre-calculate some quantities as an optimisation
        self.lam = 1./self.beta
        self.normalisation = log(self.lam).sum()
        self.zeros = zeros(self.n_params)

    def __call__(self, theta):
        if (theta < 0.).any():
            return -1e100
        else:
            return -(self.lam*theta).sum() + self.normalisation

    def gradient(self, theta):
        return where(theta >= 0., -self.lam, self.zeros)

    def sample(self):
        return exponential(scale=self.beta)

    @classmethod
    def combine(cls, priors):
        if not all(type(p) is cls for p in priors):
            raise ValueError(
                f"""
                All prior objects being combined must be of type {cls}
                """
            )

        variables = []
        for p in priors:
            variables.extend(p.variables)

        betas = concatenate([p.beta for p in priors])
        return cls(beta=betas, variables=variables)






class UniformPrior(BasePrior):
    """
    A class for generating an exponential prior for one or more of the model variables.

    :param lower: \
        A list specifying the lower bound of the uniform priors on each variable.

    :param upper: \
        A list specifying the upper bound of the uniform priors on each variable.

    :param variables: \
        A list of strings specifying the names of the variables for which the priors are generated.
    """
    def __init__(self, lower, upper, variables):
        self.lower = array(lower).squeeze()
        self.upper = array(upper).squeeze()

        # if parameters were passed as floats, convert from 0D to 1D arrays
        self.lower = self.lower.reshape([1]) if self.lower.ndim == 0 else self.lower
        self.upper = self.upper.reshape([1]) if self.upper.ndim == 0 else self.upper

        self.n_params = self.lower.size
        self.grad = zeros(self.n_params)

        if self.lower.size != self.upper.size:
            raise ValueError("""'lower' and 'upper' arguments must have the same number of elements""")

        if self.lower.ndim > 1 or self.upper.ndim > 1:
            raise ValueError("""'lower' and 'upper' arguments must be 1D arrays""")

        if (self.upper <= self.lower).any():
            raise ValueError("""All values in 'lower' must be less than the corresponding values in 'upper'""")

        self.variables = self.check_variables(variables, self.n_params)

        # pre-calculate some quantities as an optimisation
        self.normalisation = -log(self.upper-self.lower).sum()

    def __call__(self, theta):
        inside = (self.lower <= theta) & (theta <= self.upper)
        if inside.all():
            return self.normalisation
        else:
            return -1e100

    def gradient(self, theta):
        return self.grad

    def sample(self):
        return uniform(low=self.lower, high=self.upper)

    @classmethod
    def combine(cls, priors):
        if not all(type(p) is cls for p in priors):
            raise ValueError(
                f"""
                All prior objects being combined must be of type {cls}
                """
            )

        variables = []
        for p in priors:
            variables.extend(p.variables)

        lower = concatenate([p.lower for p in priors])
        upper = concatenate([p.upper for p in priors])

        return cls(lower=lower, upper=upper, variables=variables)

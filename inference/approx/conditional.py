from numpy import exp, sqrt, insert, searchsorted
from numpy import array, ndarray, linspace, zeros
from numpy.random import default_rng
from scipy.integrate import simps

rng = default_rng()


class Conditional:
    def __init__(self, posterior: callable, theta: ndarray, variable_index: int):
        self.posterior = posterior
        self.theta = theta
        self.variable_index = variable_index

    def __call__(self, x):
        t = self.theta.copy()
        t[self.variable_index] = x
        return self.posterior(t)


def binary_search(func: callable, target: float, limits: list, tol=0.05) -> float:
    x1, x2 = limits
    for i in range(10):
        x3 = 0.5 * (x1 + x2)
        y3 = func(x3)
        if y3 < target:
            x1 = x3
        else:
            x2 = x3
        if abs(y3 - target) < tol:
            break
    return x3


def trapezium_full(x, dh):
    b = dh - 1
    return (b + sqrt(b**2 + 4 * x * dh)) / (2 * dh)


def trapezium_near_zero(x, dh):
    return x + (1 - x) * x * dh


def trapezium_transform(x: ndarray, dh: ndarray) -> ndarray:
    """
    Transforms uniformly distributed random numbers in the interval [0, 1]
    to trapezium-distributed numbers in the interval [0, 1].
    
    :param x: \
        The random numbers to be transformed as a numpy ``ndarray``.
    
    :param dh: \
        The difference between the uniform distribution and the trapezium
        distribution at x = 1.
    """
    near_zero = abs(dh) < 1e-5
    if near_zero.any():
        stable = ~near_zero
        t = zeros(x.size)
        t[near_zero] = trapezium_near_zero(x[near_zero], dh[near_zero])
        t[stable] = trapezium_full(x[stable], dh[stable])
        return t
    else:
        return trapezium_full(x, dh)


def piecewise_linear_sample(x: ndarray, y: ndarray, n_samples: int) -> ndarray:
    dx = x[1:] - x[:-1]
    means = 0.5 * (y[1:] + y[:-1])
    delta = 0.5 * (y[1:] - y[:-1]) / means
    weights = means / dx
    weights /= weights.sum()
    # first sample indices of the trapeziums based on their total probability
    inds = rng.choice(weights.size, size=n_samples, p=weights)
    # now sample from trapezium distributions
    trapz = trapezium_transform(rng.random(size=n_samples), delta[inds]) * dx[inds]
    return x[inds] + trapz


def evaluate_conditional(func: callable, points: ndarray, grid_size=64):
    p = array([func(x) for x in points])
    x = points.copy()
    threshold = 8.0

    for i in range(8):
        ind = min(max(p.argmax(), 1), p.size - 2)
        x1, x2 = 0.5 * (x[ind - 1] + x[ind]), 0.5 * (x[ind + 1] + x[ind])
        p1, p2 = func(x1), func(x2)
        x = insert(x, [ind, ind + 1], [x1, x2])
        p = insert(p, [ind, ind + 1], [p1, p2])

    mode_ind = p.argmax()
    p_mode = p[mode_ind]
    x_mode = x[mode_ind]

    x_lwr = binary_search(func, p_mode - threshold, [x[0], x_mode])
    x_upr = binary_search(func, p_mode - threshold, [x[-1], x_mode])

    x_cond = linspace(x_lwr, x_upr, grid_size)
    p_cond = array([func(x) for x in x_cond])
    p_cond = exp(p_cond - p_mode)
    p_cond /= simps(p_cond, x=x_cond)
    return x_cond, p_cond


def get_conditionals(
    posterior: callable, bounds: list, conditioning_point: ndarray, grid_size: int = 64
):
    """
    Evaluates each of the 1D conditional distributions of the posterior around a given
    point in the parameter space. For each conditional, the space within the given
    bounds is searched to determine the range of values containing non-negligible
    probability, and the conditional is evaluated on a uniform grid covering that range.

    :param posterior: \
        A function which returns the posterior log-probability when given a
        numpy ``ndarray`` of the model parameters.

    :param bounds: \
        A list of length-2 tuples specifying the lower and upper bounds on
        each parameter, in the form ``(lower, upper)``.

    :param conditioning_point: \
        The point in the parameter space around which the conditional distributions are
        evaluated.

    :param grid_size: \
        The number of points used to evaluate each of the conditional distributions.

    :return samples: \
        The samples as a 2D numpy ``ndarray`` which has shape
        ``(n_samples, n_parameters)``.
    """
    conditional = Conditional(
        posterior=posterior, theta=conditioning_point, variable_index=0
    )

    n_params = conditioning_point.size
    n_search_points = 16

    axes = zeros([grid_size, n_params])
    prob = zeros([grid_size, n_params])
    for i in range(n_params):
        # switch to the conditional for the current variable
        conditional.variable_index = i
        # search using evenly spaced points plus the value from the mode
        search_points = linspace(*bounds[i], n_search_points)
        if (search_points != conditioning_point[i]).all():
            index = searchsorted(search_points, conditioning_point[i])
            search_points = insert(search_points, index, conditioning_point[i])

        # evaluate the conditional where its probability is non-negligible
        x_cond, p_cond = evaluate_conditional(
            func=conditional, points=search_points, grid_size=grid_size
        )

        axes[:, i] = x_cond
        prob[:, i] = p_cond
    return axes, prob


def conditional_sample(
    posterior: callable, bounds: list, conditioning_point: ndarray, n_samples: int
) -> ndarray:
    """
    Generates samples from each of the 1D conditional distributions of the posterior
    and combines them to approximate samples from the posterior itself. This can be
    a reasonable approximation if the posterior is close to conditionally independent,
    but is likely to underestimate uncertainties when correlations are present.

    :param posterior: \
        A function which returns the posterior log-probability when given a
        numpy ``ndarray`` of the model parameters.

    :param bounds: \
        A list of length-2 tuples specifying the lower and upper bounds on
        each parameter, in the form ``(lower, upper)``.

    :param conditioning_point: \
        The point in the parameter space around which the conditional distributions are
        evaluated. This point should correspond the posterior mode if the conditional
        distribution samples are to be used as approximate posterior samples.

    :param n_samples: \
        Number of samples to draw.

    :return samples: \
        The samples as a 2D numpy ``ndarray`` which has shape
        ``(n_samples, n_parameters)``.
    """
    axes, probs = get_conditionals(
        posterior=posterior, bounds=bounds, conditioning_point=conditioning_point
    )

    grid_size, n_params = probs.shape
    samples = zeros([n_samples, n_params])
    for i in range(n_params):
        samples[:, i] = piecewise_linear_sample(axes[:, i], probs[:, i], n_samples)
    return samples

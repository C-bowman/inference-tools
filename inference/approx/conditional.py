from numpy import array, ndarray, insert, exp, linspace, zeros
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


def conditional_sample(
    posterior: callable, bounds: list, mode: ndarray, n_samples: int
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
        A list of length-2 tuples specifying the lower and upper bounds to be set on
        each parameter, in the form (lower, upper).

    :param mode: \
        The parameters used for the conditioning - ideally these should correspond to
        the mode / MAP estimate.

    :param n_samples: \
        Number of samples to draw.

    :return samples: \
        The samples as a 2D numpy ``ndarray`` which has shape
        ``(n_samples, n_parameters)``.
    """
    conditional = Conditional(posterior=posterior, theta=mode, variable_index=0)

    n_params = mode.size
    assert len(bounds) == n_params

    n_search_points = 64
    samples = zeros([n_samples, n_params])
    for i in range(n_params):
        conditional.variable_index = i
        search_points = linspace(*bounds[i], n_search_points)
        x_cond, p_cond = evaluate_conditional(func=conditional, points=search_points)

        weights = p_cond / p_cond.sum()
        dx = x_cond[1] - x_cond[0]
        # As the grid is evenly spaced, we can first sample the indices of the grid-points,
        # then add samples from a triangle distribution with a half-width equal
        # to the grid-spacing. This is equivalent to sampling from a piecewise-linear
        # approximation of the conditional PDF.
        samples[:, i] = x_cond[rng.choice(p_cond.size, size=n_samples, p=weights)]
        samples[:, i] += rng.triangular(left=-dx, mode=0.0, right=dx, size=n_samples)
    return samples

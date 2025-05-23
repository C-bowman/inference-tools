from numpy import array, exp, log, pi, sqrt
from scipy.special import gammaln
from inference.approx.conditional import (
    get_conditionals,
    conditional_sample,
    conditional_moments,
)


def exponential(x, beta=1.0):
    return -x / beta - log(beta)


def normal(x, mu=0.0, sigma=1.0):
    return -0.5 * ((x - mu) / sigma) ** 2 - log(sigma * sqrt(2 * pi))


def log_normal(x, mu=0.0, sigma=0.65):
    return -0.5 * ((log(x) - mu) / sigma) ** 2 - log(x * sigma * sqrt(2 * pi))


def beta(x, a=2.0, b=2.0):
    norm = gammaln(a + b) - gammaln(a) - gammaln(b)
    return (a - 1) * log(x) + (b - 1) * log(1 - x) + norm


conditionals = [exponential, normal, log_normal, beta]


def conditional_test_distribution(theta):
    return sum(f(t) for f, t in zip(conditionals, theta))


def test_get_conditionals():
    bounds = [(0.0, 15.0), (-15, 100), (1e-2, 50), (1e-4, 1.0 - 1e-4)]
    conditioning_point = array([0.1, 3.0, 10.0, 0.8])
    axes, probs = get_conditionals(
        posterior=conditional_test_distribution,
        bounds=bounds,
        conditioning_point=conditioning_point,
        grid_size=128,
    )

    for i in range(axes.shape[1]):
        f = conditionals[i]
        analytic = exp(f(axes[:, i]))
        max_error = abs(probs[:, i] / analytic - 1.0).max()
        assert max_error < 1e-3


def test_conditional_sample():
    bounds = [(0.0, 15.0), (-15, 100), (1e-2, 50), (1e-4, 1.0 - 1e-4)]
    conditioning_point = array([0.1, 3.0, 10.0, 0.8])
    samples = conditional_sample(
        posterior=conditional_test_distribution,
        bounds=bounds,
        conditioning_point=conditioning_point,
        n_samples=1000,
    )

    # check that all samples produced are inside the bounds
    for i in range(samples.shape[1]):
        lwr, upr = bounds[i]
        assert (samples[:, i] >= lwr).all()
        assert (samples[:, i] <= upr).all()


def test_conditional_moments():
    # set parameters for some different beta distribution shapes
    beta_params = ((2, 5), (5, 1), (3, 3))
    bounds = [(1e-5, 1.0 - 1e-5)] * len(beta_params)
    conditioning_point = array([0.5] * len(beta_params))

    # make a posterior which is a product of these beta distributions
    def beta_posterior(theta, params=beta_params):
        return sum(beta(x, a=p[0], b=p[1]) for x, p in zip(theta, params))

    def beta_moments(a, b):
        mean = a / (a + b)
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        return mean, var

    means, variances = conditional_moments(
        posterior=beta_posterior, bounds=bounds, conditioning_point=conditioning_point
    )

    # verify numerical moments against analytic values
    for i, p in enumerate(beta_params):
        analytic_mean, analytic_var = beta_moments(*p)
        assert abs(means[i] / analytic_mean - 1) < 1e-3
        assert abs(variances[i] / analytic_var - 1) < 1e-2

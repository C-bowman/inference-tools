from numpy import array, sqrt
from inference.mcmc import GibbsChain, HamiltonianChain


def rosenbrock(t):
    # This is a modified form of the rosenbrock function, which
    # is commonly used to test optimisation algorithms
    X, Y = t
    X2 = X ** 2
    b = 15  # correlation strength parameter
    v = 3  # variance of the gaussian term
    return -X2 - b * (Y - X2) ** 2 - 0.5 * (X2 + Y ** 2) / v


class ToroidalGaussian(object):
    def __init__(self):
        self.R0 = 1.0  # torus major radius
        self.ar = 10.0  # torus aspect ratio
        self.w2 = (self.R0 / self.ar) ** 2

    def __call__(self, theta):
        x, y, z = theta
        r = sqrt(z ** 2 + (sqrt(x ** 2 + y ** 2) - self.R0) ** 2)
        return -0.5 * r ** 2 / self.w2

    def gradient(self, theta):
        x, y, z = theta
        R = sqrt(x ** 2 + y ** 2)
        K = 1 - self.R0 / R
        g = array([K * x, K * y, z])
        return -g / self.w2


def expected_len(length, start=1, step=1):
    """Expected length of an iterable"""
    real_length = length - (start - 1)
    return (real_length // step) + (real_length % step)


def test_gibbs_chain_get_replace_last():
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)

    replacements = [22.22, 44.44]
    chain.replace_last([22.22, 44.44])
    assert chain.get_last() == replacements


def test_gibbs_chain_take_step():
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    first_n = chain.n

    chain.take_step()

    assert chain.n == first_n + 1
    assert len(chain.params[0].samples) == chain.n
    assert len(chain.probs) == chain.n


def test_gibbs_chain_advance():
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    first_n = chain.n

    steps = 104
    chain.advance(steps)

    assert chain.n == first_n + steps
    assert len(chain.params[0].samples) == chain.n
    assert len(chain.probs) == chain.n


def test_gibbs_chain_get_parameter():
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)

    steps = 5
    chain.advance(steps)

    samples = chain.get_parameter(0)
    assert len(samples) == expected_len(steps)

    burn = 2
    samples = chain.get_parameter(0, burn=burn)
    assert len(samples) == expected_len(steps, burn)

    thin = 2
    samples = chain.get_parameter(1, thin=thin)
    assert len(samples) == expected_len(steps, step=thin)

    samples = chain.get_parameter(1, burn=burn, thin=thin)
    assert len(samples) == expected_len(steps, start=burn, step=thin)


def test_gibbs_chain_get_probabilities():
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    steps = 5
    chain.advance(steps)

    probabilities = chain.get_probabilities()
    assert len(probabilities) == steps
    assert probabilities[-1] > probabilities[0]

    burn = 2
    probabilities = chain.get_probabilities(burn=burn)
    assert len(probabilities) == expected_len(steps, start=burn)
    assert probabilities[-1] > probabilities[0]

    thin = 2
    probabilities = chain.get_probabilities(thin=thin)
    assert len(probabilities) == expected_len(steps, step=thin)
    assert probabilities[-1] > probabilities[0]

    probabilities = chain.get_probabilities(burn=burn, thin=thin)
    assert len(probabilities) == expected_len(steps, start=burn, step=thin)
    assert probabilities[-1] > probabilities[0]


def test_gibbs_chain_burn_in():
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    steps = 50
    chain.advance(steps)

    burn = chain.estimate_burn_in()

    assert 0 < burn <= steps

    chain.autoselect_burn()
    assert chain.burn == burn


def test_gibbs_chain_restore(tmp_path):
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    steps = 50
    chain.advance(steps)

    filename = tmp_path / "restore_file.npz"
    chain.save(filename)

    new_chain = GibbsChain.load(filename)

    assert new_chain.n == chain.n
    assert new_chain.probs == chain.probs
    assert new_chain.get_sample() == chain.get_sample()

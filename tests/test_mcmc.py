import datetime
import warnings


import numpy as np
import pytest
from freezegun import freeze_time
from numpy.random import default_rng

from inference.mcmc import GibbsChain, HamiltonianChain, PcaChain


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
        r = np.sqrt(z ** 2 + (np.sqrt(x ** 2 + y ** 2) - self.R0) ** 2)
        return -0.5 * r ** 2 / self.w2

    def gradient(self, theta):
        x, y, z = theta
        R = np.sqrt(x ** 2 + y ** 2)
        K = 1 - self.R0 / R
        g = np.array([K * x, K * y, z])
        return -g / self.w2


class LinePosterior(object):
    """
    This is a simple posterior for straight-line fitting
    with gaussian errors.
    """

    def __init__(self, x=None, y=None, err=None):
        self.x = x
        self.y = y
        self.err = err

    def __call__(self, theta):
        m, c = theta
        fwd = m * self.x + c
        ln_P = -0.5 * sum(((self.y - fwd) / self.err) ** 2)
        return ln_P


@pytest.fixture
def line_posterior():
    N = 25
    x = np.linspace(-2, 5, N)
    m = 0.5
    c = 0.05
    sigma = 0.3
    y = m * x + c + default_rng(1324).normal(size=N) * sigma
    return LinePosterior(x=x, y=y, err=np.ones(N) * sigma)


def expected_len(length, start=1, step=1):
    """Expected length of an iterable"""
    real_length = length - (start - 1)
    return (real_length // step) + (real_length % step)


def test_gibbs_chain_get_replace_last():
    start_location = np.array([2.0, -4.0])
    width_guesses = np.array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)

    replacements = [22.22, 44.44]
    chain.replace_last([22.22, 44.44])
    assert all(chain.get_last() == replacements)


def test_gibbs_chain_take_step():
    start_location = np.array([2.0, -4.0])
    width_guesses = np.array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    first_n = chain.n

    chain.take_step()

    assert chain.n == first_n + 1
    assert len(chain.params[0].samples) == chain.n
    assert len(chain.probs) == chain.n


def test_gibbs_chain_advance():
    start_location = np.array([2.0, -4.0])
    width_guesses = np.array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    first_n = chain.n

    steps = 104
    chain.advance(steps)

    assert chain.n == first_n + steps
    assert len(chain.params[0].samples) == chain.n
    assert len(chain.probs) == chain.n


def test_gibbs_chain_get_parameter():
    start_location = np.array([2.0, -4.0])
    width_guesses = np.array([5.0, 0.05])

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
    start_location = np.array([2.0, -4.0])
    width_guesses = np.array([5.0, 0.05])

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
    start_location = np.array([2.0, -4.0])
    width_guesses = np.array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    steps = 50
    chain.advance(steps)

    burn = chain.estimate_burn_in()

    assert 0 < burn <= steps

    chain.autoselect_burn()
    assert chain.burn == burn


def test_gibbs_chain_thin():
    start_location = np.array([2.0, -4.0])
    width_guesses = np.array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    steps = 50
    chain.advance(steps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chain.autoselect_thin()

    assert 0 < chain.thin <= steps


def test_gibbs_chain_restore(tmp_path):
    start_location = np.array([2.0, -4.0])
    width_guesses = np.array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    steps = 50
    chain.advance(steps)

    filename = tmp_path / "restore_file.npz"
    chain.save(filename)

    new_chain = GibbsChain.load(filename)

    assert new_chain.n == chain.n
    assert new_chain.probs == chain.probs
    assert new_chain.get_sample() == chain.get_sample()


def test_gibbs_chain_non_negative(line_posterior):
    chain = GibbsChain(posterior=line_posterior, start=[0.5, 0.1])

    chain.set_non_negative(1)
    chain.advance(100)

    offset = np.array(chain.get_parameter(1))

    assert all(offset >= 0)


def test_gibbs_chain_remove_non_negative(line_posterior):
    chain = GibbsChain(posterior=line_posterior, start=[0.5, 0.1])

    chain.set_non_negative(1, True)
    chain.set_non_negative(1, False)
    assert chain.params[1].non_negative is False
    chain.advance(100)

    offset = np.array(chain.get_parameter(1))

    assert all(offset >= 0)


def test_gibbs_chain_set_boundary(line_posterior):
    chain = GibbsChain(posterior=line_posterior, start=[0.5, 0.1])

    left, right = (0.45, 0.55)
    chain.set_boundaries(0, [left, right])
    chain.advance(100)

    gradient = np.array(chain.get_parameter(0))

    assert all(gradient >= left)
    assert all(gradient <= right)


def test_gibbs_chain_remove_boundary(line_posterior):
    chain = GibbsChain(posterior=line_posterior, start=[0.5, 0.1])

    left, right = (0.45, 0.4500000000001)
    chain.set_boundaries(0, [left, right])
    chain.set_boundaries(0, None, remove=True)
    chain.advance(100)

    gradient = np.array(chain.get_parameter(0))

    # Some values should be outside the original boundary
    assert not all(gradient >= left) or not all(gradient <= right)


def timedelta_to_days_hours_minutes(delta):
    return {
        "days": delta.days,
        "hours": delta.seconds // 3600,
        "minutes": delta.seconds // 60 % 60,
    }


@freeze_time("2nd Nov 2021", auto_tick_seconds=5)
def test_gibbs_chain_run_for_minute(line_posterior):
    expected_delta = datetime.timedelta(minutes=1)

    auto_tick_start = datetime.datetime.now()
    auto_tick = (datetime.datetime.now() - auto_tick_start).seconds
    start_time = datetime.datetime.now()
    chain = GibbsChain(posterior=line_posterior, start=[0.5, 0.1])
    chain.run_for(**timedelta_to_days_hours_minutes(expected_delta))
    end_time = datetime.datetime.now()

    # Extra 15s because two calls to `now()` in this function, plus initial
    # call to `time()` in `run_for`
    extra_delta = datetime.timedelta(seconds=3 * auto_tick)
    assert end_time - start_time == expected_delta + extra_delta
    # Probably get less due to multiple calls to `time()` per step
    assert chain.n <= expected_delta.seconds // auto_tick


@freeze_time("2nd Nov 2021", auto_tick_seconds=5)
def test_gibbs_chain_run_for_hour(line_posterior):
    expected_delta = datetime.timedelta(hours=1)

    auto_tick_start = datetime.datetime.now()
    auto_tick = (datetime.datetime.now() - auto_tick_start).seconds
    start_time = datetime.datetime.now()
    chain = GibbsChain(posterior=line_posterior, start=[0.5, 0.1])
    chain.run_for(**timedelta_to_days_hours_minutes(expected_delta))
    end_time = datetime.datetime.now()

    # Extra 15s because two calls to `now()` in this function, plus initial
    # call to `time()` in `run_for`
    extra_delta = datetime.timedelta(seconds=3 * auto_tick)
    assert end_time - start_time == expected_delta + extra_delta
    # Probably get less due to multiple calls to `time()` per step
    assert chain.n <= expected_delta.total_seconds() // auto_tick


@freeze_time("2nd Nov 2021", auto_tick_seconds=50)
def test_gibbs_chain_run_for_day(line_posterior):
    expected_delta = datetime.timedelta(days=1)

    auto_tick_start = datetime.datetime.now()
    auto_tick = (datetime.datetime.now() - auto_tick_start).seconds
    start_time = datetime.datetime.now()
    chain = GibbsChain(posterior=line_posterior, start=[0.5, 0.1])
    chain.run_for(**timedelta_to_days_hours_minutes(expected_delta))
    end_time = datetime.datetime.now()

    # Extra 15s because two calls to `now()` in this function, plus initial
    # call to `time()` in `run_for`
    extra_delta = datetime.timedelta(seconds=3 * auto_tick)
    assert end_time - start_time == expected_delta + extra_delta
    # Probably get less due to multiple calls to `time()` per step
    assert chain.n <= expected_delta.total_seconds() // auto_tick


@freeze_time("2nd Nov 2021", auto_tick_seconds=50)
def test_gibbs_chain_run_for_day_hour_minute(line_posterior):
    expected_delta = datetime.timedelta(days=1, hours=2, minutes=3)

    auto_tick_start = datetime.datetime.now()
    auto_tick = (datetime.datetime.now() - auto_tick_start).seconds
    start_time = datetime.datetime.now()
    chain = GibbsChain(posterior=line_posterior, start=[0.5, 0.1])
    chain.run_for(**timedelta_to_days_hours_minutes(expected_delta))
    end_time = datetime.datetime.now()

    # Extra 15s because two calls to `now()` in this function, plus initial
    # call to `time()` in `run_for`
    extra_delta = datetime.timedelta(seconds=4 * auto_tick)
    assert end_time - start_time >= expected_delta
    assert end_time - start_time <= expected_delta + extra_delta
    # Probably get less due to multiple calls to `time()` per step
    assert chain.n <= expected_delta.total_seconds() // auto_tick


def test_hamiltonian_chain_take_step():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(
        posterior=posterior, grad=posterior.gradient, start=[1, 0.1, 0.1]
    )
    first_n = chain.n

    chain.take_step()

    assert chain.n == first_n + 1
    chain.burn = 0
    assert len(chain.get_parameter(0)) == chain.n
    assert len(chain.get_parameter(1)) == chain.n
    assert len(chain.get_parameter(2)) == chain.n
    assert len(chain.probs) == chain.n


def test_hamiltonian_chain_advance():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(
        posterior=posterior, grad=posterior.gradient, start=[1, 0.1, 0.1]
    )
    first_n = chain.n
    steps = 10
    chain.advance(steps)

    assert chain.n == first_n + steps
    chain.burn = 0
    assert len(chain.get_parameter(0)) == chain.n
    assert len(chain.get_parameter(1)) == chain.n
    assert len(chain.get_parameter(2)) == chain.n
    assert len(chain.probs) == chain.n


def test_hamiltonian_chain_advance_no_gradient():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(posterior=posterior, start=[1, 0.1, 0.1])
    first_n = chain.n
    steps = 10
    chain.advance(steps)

    assert chain.n == first_n + steps
    chain.burn = 0
    assert len(chain.get_parameter(0)) == chain.n
    assert len(chain.get_parameter(1)) == chain.n
    assert len(chain.get_parameter(2)) == chain.n
    assert len(chain.probs) == chain.n


def test_hamiltonian_chain_burn_in():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(posterior=posterior, start=[1, 0.1, 0.1])
    steps = 10
    chain.advance(steps)
    burn = chain.estimate_burn_in()

    assert 0 < burn <= steps

    chain.autoselect_burn()
    assert chain.burn == burn


def test_hamiltonian_chain_advance_bounds(line_posterior):
    chain = HamiltonianChain(
        posterior=line_posterior,
        start=[0.5, 0.1],
        bounds=(np.array([0.45, 0.0]), np.array([0.55, 10.0])),
    )
    chain.advance(10)

    gradient = np.array(chain.get_parameter(0))
    assert all(gradient >= 0.45)
    assert all(gradient <= 0.55)

    offset = np.array(chain.get_parameter(1))
    assert all(offset >= 0)


def test_hamiltonian_chain_restore(tmp_path):
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(
        posterior=posterior, grad=posterior.gradient, start=[1, 0.1, 0.1]
    )
    steps = 10
    chain.advance(steps)

    filename = tmp_path / "restore_file.npz"
    chain.save(filename)

    new_chain = HamiltonianChain.load(filename)

    assert new_chain.n == chain.n
    assert new_chain.probs == chain.probs
    assert all(new_chain.get_last() == chain.get_last())


def test_pca_chain_take_step(line_posterior):
    chain = PcaChain(posterior=line_posterior, start=[0.5, 0.1])
    first_n = chain.n

    chain.take_step()

    assert chain.n == first_n + 1
    assert len(chain.params[0].samples) == chain.n
    assert len(chain.probs) == chain.n


def test_pca_chain_advance(line_posterior):
    chain = PcaChain(posterior=line_posterior, start=[0.5, 0.1])
    first_n = chain.n

    steps = 104
    chain.advance(steps)

    assert chain.n == first_n + steps
    assert len(chain.params[0].samples) == chain.n
    assert len(chain.probs) == chain.n


def test_pca_chain_restore(line_posterior, tmp_path):
    chain = PcaChain(posterior=line_posterior, start=[0.5, 0.1])
    steps = 200
    chain.advance(steps)

    filename = tmp_path / "restore_file.npz"
    chain.save(filename)

    new_chain = PcaChain.load(filename)

    assert new_chain.n == chain.n
    assert new_chain.probs == chain.probs
    assert all(new_chain.get_last() == chain.get_last())

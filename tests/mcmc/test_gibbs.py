import datetime
import warnings

from numpy import array
from inference.mcmc import GibbsChain
from mcmc_utils import line_posterior, rosenbrock, expected_len

from freezegun import freeze_time
import pytest


def test_gibbs_chain_get_replace_last():
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)

    replacements = [22.22, 44.44]
    chain.replace_last([22.22, 44.44])
    assert all(chain.get_last() == replacements)


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
    steps = 10
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


def test_gibbs_chain_thin():
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    steps = 50
    chain.advance(steps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chain.autoselect_thin()

    assert 0 < chain.thin <= steps


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


def test_gibbs_chain_non_negative(line_posterior):
    chain = GibbsChain(posterior=line_posterior, start=[0.5, 0.1])

    chain.set_non_negative(1)
    chain.advance(100)

    offset = array(chain.get_parameter(1))

    assert all(offset >= 0)


def test_gibbs_chain_remove_non_negative(line_posterior):
    chain = GibbsChain(posterior=line_posterior, start=[0.5, 0.1])

    chain.set_non_negative(1, True)
    chain.set_non_negative(1, False)
    assert chain.params[1].non_negative is False


def test_gibbs_chain_set_boundary(line_posterior):
    chain = GibbsChain(posterior=line_posterior, start=[0.5, 0.1])

    left, right = (0.45, 0.55)
    chain.set_boundaries(0, [left, right])
    chain.advance(100)

    gradient = array(chain.get_parameter(0))

    assert all(gradient >= left)
    assert all(gradient <= right)


def test_gibbs_chain_remove_boundary(line_posterior):
    chain = GibbsChain(posterior=line_posterior, start=[0.5, 0.1])

    left, right = (0.45, 0.4500000000001)
    chain.set_boundaries(0, [left, right])
    chain.set_boundaries(0, None, remove=True)
    chain.advance(100)

    gradient = array(chain.get_parameter(0))

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
    extra_delta = datetime.timedelta(seconds=4 * auto_tick)
    assert end_time - start_time == expected_delta + extra_delta
    # Probably get less due to multiple calls to `time()` per step
    # assert chain.n <= expected_delta.seconds // auto_tick


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
    extra_delta = datetime.timedelta(seconds=4 * auto_tick)
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
    extra_delta = datetime.timedelta(seconds=4 * auto_tick)
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
    extra_delta = datetime.timedelta(seconds=5 * auto_tick)
    assert end_time - start_time >= expected_delta
    assert end_time - start_time <= expected_delta + extra_delta
    # Probably get less due to multiple calls to `time()` per step
    assert chain.n <= expected_delta.total_seconds() // auto_tick

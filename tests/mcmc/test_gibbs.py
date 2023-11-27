import datetime
import warnings

from numpy import array
from inference.mcmc import GibbsChain
from mcmc_utils import line_posterior, rosenbrock, sliced_length
from itertools import product
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
    first_n = chain.chain_length

    chain.take_step()

    assert chain.chain_length == first_n + 1
    assert len(chain.params[0].samples) == chain.chain_length
    assert len(chain.probs) == chain.chain_length


def test_gibbs_chain_advance():
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    first_n = chain.chain_length

    steps = 104
    chain.advance(steps)

    assert chain.chain_length == first_n + steps
    assert len(chain.params[0].samples) == chain.chain_length
    assert len(chain.probs) == chain.chain_length


def test_gibbs_chain_results_access():
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    n_params = chain.n_parameters

    steps = 16
    chain.advance(steps)

    assert chain.chain_length == chain.get_parameter(0, burn=0, thin=1).size
    assert chain.chain_length == chain.get_probabilities(burn=0, thin=1).size
    assert (chain.chain_length, n_params) == chain.get_sample(burn=0, thin=1).shape

    burns = [0, 5, 8, 15]
    thins = [1, 3, 10, 50]
    for burn, thin in product(burns, thins):
        expected_len = sliced_length(chain.chain_length, start=burn, step=thin)
        assert expected_len == chain.get_parameter(0, burn=burn, thin=thin).size
        assert expected_len == chain.get_probabilities(burn=burn, thin=thin).size
        assert (expected_len, n_params) == chain.get_sample(burn=burn, thin=thin).shape


def test_gibbs_chain_burn_in():
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    steps = 50
    chain.advance(steps)

    burn = chain.estimate_burn_in()

    assert 0 < burn <= steps


def test_gibbs_chain_restore(tmp_path):
    start_location = array([2.0, -4.0])
    width_guesses = array([5.0, 0.05])

    chain = GibbsChain(posterior=rosenbrock, start=start_location, widths=width_guesses)
    steps = 50
    chain.advance(steps)

    filename = tmp_path / "restore_file.npz"
    chain.save(filename)

    new_chain = GibbsChain.load(filename)
    _ = GibbsChain.load(filename, posterior=rosenbrock)

    assert new_chain.chain_length == chain.chain_length
    assert (new_chain.get_probabilities() == chain.get_probabilities()).all()
    assert (new_chain.get_sample() == chain.get_sample()).all()


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
    assert chain.chain_length <= expected_delta.total_seconds() // auto_tick


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
    assert chain.chain_length <= expected_delta.total_seconds() // auto_tick


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
    assert chain.chain_length <= expected_delta.total_seconds() // auto_tick

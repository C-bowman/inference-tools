from mcmc_utils import line_posterior
from numpy import array
from numpy.random import default_rng
from inference.mcmc import EnsembleSampler, Bounds
import pytest


def test_ensemble_sampler_advance(line_posterior):
    n_walkers = 100
    guess = array([2.0, -4.0])
    rng = default_rng(256)
    starts = rng.normal(scale=0.01, loc=1.0, size=[n_walkers, 2]) * guess[None, :]

    bounds = Bounds(lower=array([-5.0, -10.0]), upper=array([5.0, 10.0]))

    chain = EnsembleSampler(
        posterior=line_posterior, starting_positions=starts, bounds=bounds
    )

    assert chain.n_walkers == n_walkers
    assert chain.n_parameters == 2

    n_iterations = 25
    chain.advance(iterations=n_iterations)
    assert chain.n_iterations == n_iterations

    sample = chain.get_sample()
    assert sample.shape == (n_iterations * n_walkers, 2)

    values = chain.get_parameter(1)
    assert values.shape == (n_iterations * n_walkers,)

    probs = chain.get_probabilities()
    assert probs.shape == (n_iterations * n_walkers,)


def test_ensemble_sampler_restore(line_posterior, tmp_path):
    n_walkers = 100
    guess = array([2.0, -4.0])
    rng = default_rng(256)
    starts = rng.normal(scale=0.01, loc=1.0, size=[n_walkers, 2]) * guess[None, :]
    bounds = Bounds(lower=array([-5.0, -10.0]), upper=array([5.0, 10.0]))

    chain = EnsembleSampler(
        posterior=line_posterior, starting_positions=starts, bounds=bounds
    )

    n_iterations = 25
    chain.advance(iterations=n_iterations)

    filename = tmp_path / "restore_file.npz"
    chain.save(filename)

    new_chain = EnsembleSampler.load(filename)

    assert new_chain.n_iterations == chain.n_iterations
    assert (new_chain.walker_positions == chain.walker_positions).all()
    assert (new_chain.walker_probs == chain.walker_probs).all()
    assert (new_chain.sample == chain.sample).all()
    assert (new_chain.sample_probs == chain.sample_probs).all()
    assert (new_chain.bounds.lower == chain.bounds.lower).all()
    assert (new_chain.bounds.upper == chain.bounds.upper).all()


def test_ensemble_sampler_input_parsing(line_posterior):
    n_walkers = 100
    guess = array([2.0, -4.0])
    rng = default_rng(256)

    # case where both variables are co-linear
    colinear_starts = (
        guess[None, :] * rng.normal(scale=0.05, loc=1.0, size=n_walkers)[:, None]
    )

    with pytest.raises(ValueError):
        chain = EnsembleSampler(
            posterior=line_posterior, starting_positions=colinear_starts
        )

    # case where one of the variables has zero variance
    zero_var_starts = (
        rng.normal(scale=0.01, loc=1.0, size=[n_walkers, 2]) * guess[None, :]
    )
    zero_var_starts[:, 0] = guess[0]

    with pytest.raises(ValueError):
        chain = EnsembleSampler(
            posterior=line_posterior, starting_positions=zero_var_starts
        )

    # test that incompatible bounds raise errors
    starts = rng.normal(scale=0.01, loc=1.0, size=[n_walkers, 2]) * guess[None, :]
    bounds = Bounds(lower=array([-5.0, -10.0, -1.0]), upper=array([5.0, 10.0, 1.0]))
    with pytest.raises(ValueError):
        chain = EnsembleSampler(
            posterior=line_posterior, starting_positions=starts, bounds=bounds
        )

    bounds = Bounds(lower=array([-5.0, 4.0]), upper=array([5.0, 10.0]))
    with pytest.raises(ValueError):
        chain = EnsembleSampler(
            posterior=line_posterior, starting_positions=starts, bounds=bounds
        )

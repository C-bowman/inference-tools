from mcmc_utils import line_posterior
from numpy import array
from numpy.random import default_rng
from inference.mcmc import EnsembleSampler


def test_ensemble_sampler_advance(line_posterior):
    n_walkers = 100
    guess = array([2.0, -4.0])
    rng = default_rng(256)
    starts = [guess * rng.normal(scale=0.01, loc=1.0, size=2) for i in range(n_walkers)]

    chain = EnsembleSampler(posterior=line_posterior, starting_positions=starts)

    assert chain.n_walkers == n_walkers
    assert chain.n_params == 2

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
    starts = [guess * rng.normal(scale=0.01, loc=1.0, size=2) for i in range(n_walkers)]

    chain = EnsembleSampler(posterior=line_posterior, starting_positions=starts)

    n_iterations = 25
    chain.advance(iterations=n_iterations)

    filename = tmp_path / "restore_file.npz"
    chain.save(filename)

    new_chain = EnsembleSampler.load(filename)

    assert new_chain.n_iterations == chain.n_iterations
    assert (new_chain.theta == chain.theta).all()
    assert (new_chain.probs == chain.probs).all()
    assert (new_chain.sample == chain.sample).all()
    assert (new_chain.sample_probs == chain.sample_probs).all()

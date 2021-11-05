from inference.posterior import Posterior

from unittest.mock import MagicMock
import pytest


def test_posterior_call():
    likelihood = MagicMock(return_value=4)
    prior = MagicMock(return_value=5)

    posterior = Posterior(likelihood, prior)

    result = posterior(44.4)
    assert result == 9
    cost = posterior.cost(55.5)
    assert result == -cost

    likelihood.assert_called()
    prior.assert_called()


def test_posterior_gradient():
    likelihood = MagicMock(return_value=4)
    prior = MagicMock(return_value=5)

    posterior = Posterior(likelihood, prior)

    posterior.gradient(44.4)

    likelihood.gradient.assert_called()
    prior.gradient.assert_called()


def test_posterior_cost():
    likelihood = MagicMock(return_value=4)
    prior = MagicMock(return_value=5)

    posterior = Posterior(likelihood, prior)

    assert posterior(44.4) == -posterior.cost(44.4)


def test_posterior_cost_gradient():
    likelihood = MagicMock(return_value=4)
    likelihood.gradient.return_value = 0
    prior = MagicMock(return_value=5)
    prior.gradient.return_value = 1

    posterior = Posterior(likelihood, prior)

    assert posterior.gradient(44.4) == -posterior.cost_gradient(44.4)


def test_posterior_bad_initial_guess_types():
    likelihood = MagicMock(return_value=4)
    prior = MagicMock(return_value=5)

    posterior = Posterior(likelihood, prior)

    with pytest.raises(TypeError):
        posterior.generate_initial_guesses(2.2)

    with pytest.raises(TypeError):
        posterior.generate_initial_guesses(2, 3.3)


def test_posterior_bad_initial_guess_values():
    likelihood = MagicMock(return_value=4)
    prior = MagicMock(return_value=5)

    posterior = Posterior(likelihood, prior)

    with pytest.raises(ValueError):
        posterior.generate_initial_guesses(-1)

    with pytest.raises(ValueError):
        posterior.generate_initial_guesses(0)

    with pytest.raises(ValueError):
        posterior.generate_initial_guesses(1, -3)

    with pytest.raises(ValueError):
        posterior.generate_initial_guesses(1, 0)

    with pytest.raises(ValueError):
        posterior.generate_initial_guesses(2, 1)


def test_posterior_initial_guess_default_args():
    likelihood = MagicMock()
    likelihood.side_effect = lambda x: x
    prior = MagicMock()
    prior.side_effect = lambda x: x
    prior.sample.side_effect = range(200)

    posterior = Posterior(likelihood, prior)

    samples = posterior.generate_initial_guesses()
    assert samples == [99]


def test_posterior_initial_guess():
    likelihood = MagicMock()
    likelihood.side_effect = lambda x: x
    prior = MagicMock()
    prior.side_effect = lambda x: x
    prior.sample.side_effect = range(100)

    posterior = Posterior(likelihood, prior)

    samples = posterior.generate_initial_guesses(n_guesses=2, prior_samples=10)
    assert samples == [9, 8]

    prior.sample.side_effect = range(100)

    samples = posterior.generate_initial_guesses(n_guesses=1, prior_samples=1)
    assert samples == [0]

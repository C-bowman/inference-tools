import pytest
from numpy import allclose, array, isfinite, log, nan
from itertools import product
from mcmc_utils import ToroidalGaussian, line_posterior, sliced_length
from inference.mcmc import HamiltonianChain, Bounds
from inference.mcmc.hmc.epsilon import EpsilonSelector
from inference.mcmc.hmc.transforms import IntervalTransform


class RejectingRng:
    def uniform(self, low, high):
        return 1.0

    def random(self):
        return 1.0


def test_hamiltonian_chain_take_step():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(
        posterior=posterior, start=array([1, 0.1, 0.1]), grad=posterior.gradient
    )
    first_n = chain.chain_length

    chain.take_step()

    assert chain.chain_length == first_n + 1
    for i in range(3):
        assert chain.get_parameter(i, burn=0).size == chain.chain_length
    assert len(chain.probs) == chain.chain_length


def test_hamiltonian_chain_records_rejected_proposal():
    chain = HamiltonianChain(
        posterior=lambda theta: -float(theta[0] ** 2),
        start=array([0.0]),
        grad=lambda theta: array([0.0]),
    )
    leapfrog_calls = 0

    def one_shot_leapfrog(theta, momentum, steps):
        nonlocal leapfrog_calls
        leapfrog_calls += 1
        if leapfrog_calls > 1:
            raise RuntimeError("proposal was retried")
        return array([10.0]), momentum

    chain.mass.sample_momentum = lambda rng: array([0.0])
    chain.run_leapfrog = one_shot_leapfrog
    chain.rng = RejectingRng()

    chain.take_step()

    assert leapfrog_calls == 1
    assert chain.get_parameter(0, burn=0).tolist() == [0.0, 0.0]
    assert chain.get_probabilities(burn=0).tolist() == [0.0, 0.0]


def test_hamiltonian_chain_accepts_highly_favorable_proposal():
    chain = HamiltonianChain(
        posterior=lambda theta: 1000.0 if theta[0] else 0.0,
        start=array([0.0]),
        grad=lambda theta: array([0.0]),
    )
    chain.mass.sample_momentum = lambda rng: array([0.0])
    chain.run_leapfrog = lambda theta, momentum, steps: (array([1.0]), momentum)
    chain.rng = RejectingRng()

    chain.take_step()

    assert chain.get_parameter(0, burn=0).tolist() == [0.0, 1.0]
    assert chain.get_probabilities(burn=0).tolist() == [0.0, 1000.0]


@pytest.mark.parametrize("proposed_probability", [float("inf"), -float("inf"), nan])
def test_hamiltonian_chain_rejects_non_finite_proposal(proposed_probability):
    chain = HamiltonianChain(
        posterior=lambda theta: proposed_probability if theta[0] else 0.0,
        start=array([0.0]),
        grad=lambda theta: array([0.0]),
    )
    chain.mass.sample_momentum = lambda rng: array([0.0])
    chain.run_leapfrog = lambda theta, momentum, steps: (array([1.0]), momentum)
    chain.rng = RejectingRng()

    chain.take_step()

    assert chain.get_parameter(0, burn=0).tolist() == [0.0, 0.0]
    assert chain.get_probabilities(burn=0).tolist() == [0.0, 0.0]


def test_hamiltonian_chain_advance():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(
        posterior=posterior, start=array([1, 0.1, 0.1]), grad=posterior.gradient
    )
    n_params = chain.n_parameters
    initial_length = chain.chain_length
    steps = 16
    chain.advance(steps)
    assert chain.chain_length == initial_length + steps

    for i in range(3):
        assert chain.chain_length == chain.get_parameter(i, burn=0, thin=1).size
    assert chain.chain_length == chain.get_probabilities(burn=0, thin=1).size
    assert (chain.chain_length, n_params) == chain.get_sample(burn=0, thin=1).shape

    burns = [0, 5, 8, 15]
    thins = [1, 3, 10, 50]
    for burn, thin in product(burns, thins):
        expected_len = sliced_length(chain.chain_length, start=burn, step=thin)
        assert expected_len == chain.get_parameter(0, burn=burn, thin=thin).size
        assert expected_len == chain.get_probabilities(burn=burn, thin=thin).size
        assert (expected_len, n_params) == chain.get_sample(burn=burn, thin=thin).shape


def test_hamiltonian_chain_advance_no_gradient():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(posterior=posterior, start=array([1, 0.1, 0.1]))
    first_n = chain.chain_length
    steps = 10
    chain.advance(steps)

    assert chain.chain_length == first_n + steps
    for i in range(3):
        assert chain.get_parameter(i, burn=0).size == chain.chain_length
    assert len(chain.probs) == chain.chain_length


def test_hamiltonian_finite_diff_is_untempered():
    posterior = lambda theta: -0.5 * float(theta @ theta)
    theta = array([1.0, -2.0])
    expected = -theta

    cold_chain = HamiltonianChain(
        posterior=posterior, start=theta, temperature=0.5
    )
    hot_chain = HamiltonianChain(
        posterior=posterior, start=theta, temperature=2.0
    )

    assert allclose(cold_chain.finite_diff(theta), expected, atol=1e-8)
    assert allclose(hot_chain.finite_diff(theta), expected, atol=1e-8)


def test_hamiltonian_finite_diff_at_zero():
    posterior = lambda theta: -0.5 * float(theta @ theta)
    theta = array([0.0, 1.0])
    chain = HamiltonianChain(posterior=posterior, start=theta)

    gradient = chain.finite_diff(theta)

    assert isfinite(gradient).all()
    assert allclose(gradient, -theta, atol=1e-8)


@pytest.mark.parametrize("temperature", [0.5, 2.0])
def test_hamiltonian_finite_diff_conserves_energy(temperature):
    posterior = lambda theta: -0.5 * float(theta @ theta)
    chain = HamiltonianChain(
        posterior=posterior,
        start=array([1.0, -0.5]),
        temperature=temperature,
        epsilon=0.01,
    )
    theta = array([1.0, -0.5])
    momentum = array([0.3, -0.2])
    initial_energy = chain.hamiltonian(theta, momentum)

    next_theta, next_momentum = chain.run_leapfrog(
        theta.copy(), momentum.copy(), n_steps=100
    )
    final_energy = chain.hamiltonian(next_theta, next_momentum)

    assert abs(final_energy - initial_energy) < 1e-3


def test_hamiltonian_chain_burn_in():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(
        posterior=posterior, start=array([2, 0.1, 0.1]), grad=posterior.gradient
    )
    steps = 500
    chain.advance(steps)
    burn = chain.estimate_burn_in()

    assert 0 < burn <= steps


def test_hamiltonian_chain_advance_bounds(line_posterior):
    chain = HamiltonianChain(
        posterior=line_posterior,
        start=array([0.5, 0.1]),
        bounds=(array([0.45, 0.0]), array([0.55, 10.0])),
    )
    chain.advance(10)

    gradient = chain.get_parameter(0)
    assert all(gradient >= 0.45)
    assert all(gradient <= 0.55)

    offset = chain.get_parameter(1)
    assert all(offset >= 0)


def test_interval_transform_target_inputs():
    transform = IntervalTransform(
        lower=array([0.0, -2.0]), upper=array([1.0, 2.0])
    )
    unconstrained = array([0.0, log(3.0)])

    value, log_jacobian = transform.log_prob_inputs(unconstrained)
    gradient_value, jacobian, log_jacobian_gradient = transform.gradient_inputs(
        unconstrained
    )

    assert value == pytest.approx(array([0.5, 1.0]))
    assert gradient_value == pytest.approx(value)
    assert jacobian == pytest.approx(array([0.25, 0.75]))
    assert log_jacobian == pytest.approx(log(0.25 * 0.75))
    assert log_jacobian_gradient == pytest.approx(array([0.0, -0.5]))
    assert transform.forward(value) == pytest.approx(unconstrained)
    assert transform.inverse(unconstrained) == pytest.approx(value)


def test_hamiltonian_transformed_gradient():
    posterior = lambda theta: -0.5 * float(theta @ theta)
    chain = HamiltonianChain(
        posterior=posterior,
        start=array([0.5, 0.5]),
        grad=lambda theta: -theta,
        temperature=2.0,
        bounds=(array([0.0, 0.0]), array([1.0, 1.0])),
    )
    unconstrained = array([-0.7, 0.8])
    step = 1e-6
    finite_difference = []
    for index in range(unconstrained.size):
        upper = unconstrained.copy()
        lower = unconstrained.copy()
        upper[index] += step
        lower[index] -= step
        finite_difference.append(
            (chain._target_log_prob(upper) - chain._target_log_prob(lower))
            / (2 * step)
        )

    assert chain._target_gradient(unconstrained) == pytest.approx(finite_difference)


def test_hamiltonian_jacobian_is_not_tempered():
    kwargs = {
        "posterior": lambda theta: 0.0,
        "start": array([0.5]),
        "grad": lambda theta: array([0.0]),
        "bounds": (array([0.0]), array([1.0])),
    }
    cold_chain = HamiltonianChain(temperature=0.5, **kwargs)
    hot_chain = HamiltonianChain(temperature=2.0, **kwargs)
    unconstrained = array([1.0])

    assert cold_chain._target_log_prob(unconstrained) == pytest.approx(
        hot_chain._target_log_prob(unconstrained)
    )
    assert cold_chain._target_gradient(unconstrained) == pytest.approx(
        hot_chain._target_gradient(unconstrained)
    )


def test_hamiltonian_bounds_use_unconstrained_internal_coordinates():
    chain = HamiltonianChain(
        posterior=lambda theta: -float(theta @ theta),
        start=array([0.25, 0.75]),
        grad=lambda theta: -2.0 * theta,
        bounds=(array([0.0, 0.0]), array([1.0, 1.0])),
        inverse_mass=array([[1.0, 0.5], [0.5, 1.0]]),
    )

    assert chain.theta[0] == pytest.approx(array([-log(3.0), log(3.0)]))
    assert chain.get_last() == pytest.approx(array([0.25, 0.75]))

    chain.replace_last(array([0.4, 0.6]))

    assert chain.get_last() == pytest.approx(array([0.4, 0.6]))


def test_hamiltonian_bounds_reject_boundary_start():
    with pytest.raises(ValueError, match="strictly inside"):
        HamiltonianChain(
            posterior=lambda theta: -float(theta @ theta),
            start=array([0.0]),
            bounds=(array([0.0]), array([1.0])),
        )


def test_hamiltonian_chain_restore(tmp_path):
    posterior = ToroidalGaussian()
    bounds = Bounds(lower=array([-2.0, -2.0, -1.0]), upper=array([2.0, 2.0, 1.0]))
    chain = HamiltonianChain(
        posterior=posterior,
        start=array([1.0, 0.1, 0.1]),
        grad=posterior.gradient,
        bounds=bounds,
    )
    steps = 10
    chain.advance(steps)

    filename = tmp_path / "restore_file.npz"
    chain.save(filename)

    new_chain = HamiltonianChain.load(
        filename, posterior=posterior, grad=posterior.gradient
    )

    assert new_chain.chain_length == chain.chain_length
    assert new_chain.probs == chain.probs
    assert (new_chain.get_last() == chain.get_last()).all()
    assert (new_chain.bounds.lower == chain.bounds.lower).all()
    assert (new_chain.bounds.upper == chain.bounds.upper).all()

    new_chain.take_step()

    assert new_chain.chain_length == chain.chain_length + 1
    assert new_chain.bounds.inside(new_chain.get_last())


def test_epsilon_selector_restores_running_statistics():
    selector = EpsilonSelector(initial_epsilon=0.1)
    for probability in [0.1, 0.9, 0.2]:
        selector.add_probability(probability)

    restored = EpsilonSelector(initial_epsilon=1.0)
    restored.load_items(selector.get_items())
    index = selector.current_index

    assert restored.stats[index].S == selector.stats[index].S
    restored.stats[index].add_sample(0.7)
    selector.stats[index].add_sample(0.7)
    assert restored.stats[index].variance == selector.stats[index].variance


def test_epsilon_selector_restores_unseen_bin_creation():
    selector = EpsilonSelector(initial_epsilon=0.1)
    for _ in range(selector.update_interval):
        selector.add_probability(0.0)
    assert selector.current_index not in selector.stats

    restored = EpsilonSelector(initial_epsilon=1.0)
    restored.load_items(selector.get_items())

    restored.add_probability(0.0)

    assert restored.stats[restored.current_index].count == 1


@pytest.mark.parametrize(
    "inverse_mass",
    [
        1.0,
        array([1.0, 2.0, 3.0]),
        array([[1.0, 0.1, 0.0], [0.1, 2.0, 0.1], [0.0, 0.1, 3.0]]),
    ],
)
def test_hamiltonian_chain_restore_and_continue(tmp_path, inverse_mass):
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(
        posterior=posterior,
        start=array([1.0, 0.1, 0.1]),
        grad=posterior.gradient,
        inverse_mass=inverse_mass,
    )
    chain.advance(chain.ES.update_interval)
    filename = tmp_path / "restore_and_continue.npz"
    chain.save(filename)

    restored = HamiltonianChain.load(
        filename, posterior=posterior, grad=posterior.gradient
    )
    restored.take_step()

    assert restored.chain_length == chain.chain_length + 1


def test_hamiltonian_chain_plots():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(
        posterior=posterior, start=array([2, 0.1, 0.1]), grad=posterior.gradient
    )

    # confirm that plotting with no samples raises error
    with pytest.raises(ValueError):
        chain.trace_plot()
    with pytest.raises(ValueError):
        chain.matrix_plot()

    # check that plots work with samples
    steps = 200
    chain.advance(steps)
    chain.trace_plot(show=False)
    chain.matrix_plot(show=False)

    # check plots raise error with bad burn / thin values
    with pytest.raises(ValueError):
        chain.trace_plot(burn=200)
    with pytest.raises(ValueError):
        chain.matrix_plot(thin=500)


def test_hamiltonian_chain_burn_thin_error():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(
        posterior=posterior, start=array([1, 0.1, 0.1]), grad=posterior.gradient
    )
    with pytest.raises(AttributeError):
        chain.burn = 10
    with pytest.raises(AttributeError):
        burn = chain.burn
    with pytest.raises(AttributeError):
        chain.thin = 5
    with pytest.raises(AttributeError):
        thin = chain.thin


def test_hamiltonian_posterior_validation():
    with pytest.raises(ValueError):
        chain = HamiltonianChain(posterior="posterior", start=array([1, 0.1]))

    with pytest.raises(ValueError):
        chain = HamiltonianChain(posterior=lambda x: 1, start=array([1, 0.1]))

    with pytest.raises(ValueError):
        chain = HamiltonianChain(posterior=lambda x: nan, start=array([1, 0.1]))

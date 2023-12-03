import pytest
from numpy import array
from itertools import product
from mcmc_utils import ToroidalGaussian, line_posterior, sliced_length
from inference.mcmc import HamiltonianChain, Bounds


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

    new_chain = HamiltonianChain.load(filename)

    assert new_chain.chain_length == chain.chain_length
    assert new_chain.probs == chain.probs
    assert (new_chain.get_last() == chain.get_last()).all()
    assert (new_chain.bounds.lower == chain.bounds.lower).all()
    assert (new_chain.bounds.upper == chain.bounds.upper).all()


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

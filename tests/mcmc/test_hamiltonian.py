from numpy import array
from mcmc_utils import ToroidalGaussian, line_posterior
from inference.mcmc import HamiltonianChain


def test_hamiltonian_chain_take_step():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(
        posterior=posterior, start=array([1, 0.1, 0.1]), grad=posterior.gradient
    )
    first_n = chain.chain_length

    chain.take_step()

    assert chain.chain_length == first_n + 1
    chain.burn = 0
    assert len(chain.get_parameter(0)) == chain.chain_length
    assert len(chain.get_parameter(1)) == chain.chain_length
    assert len(chain.get_parameter(2)) == chain.chain_length
    assert len(chain.probs) == chain.chain_length


def test_hamiltonian_chain_advance():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(
        posterior=posterior, start=array([1, 0.1, 0.1]), grad=posterior.gradient
    )
    first_n = chain.chain_length
    steps = 10
    chain.advance(steps)

    assert chain.chain_length == first_n + steps
    chain.burn = 0
    assert len(chain.get_parameter(0)) == chain.chain_length
    assert len(chain.get_parameter(1)) == chain.chain_length
    assert len(chain.get_parameter(2)) == chain.chain_length
    assert len(chain.probs) == chain.chain_length


def test_hamiltonian_chain_advance_no_gradient():
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(posterior=posterior, start=array([1, 0.1, 0.1]))
    first_n = chain.chain_length
    steps = 10
    chain.advance(steps)

    assert chain.chain_length == first_n + steps
    chain.burn = 0
    assert len(chain.get_parameter(0)) == chain.chain_length
    assert len(chain.get_parameter(1)) == chain.chain_length
    assert len(chain.get_parameter(2)) == chain.chain_length
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

    chain.autoselect_burn()
    assert chain.burn == burn


def test_hamiltonian_chain_advance_bounds(line_posterior):
    chain = HamiltonianChain(
        posterior=line_posterior,
        start=array([0.5, 0.1]),
        bounds=(array([0.45, 0.0]), array([0.55, 10.0])),
    )
    chain.advance(10)

    gradient = array(chain.get_parameter(0))
    assert all(gradient >= 0.45)
    assert all(gradient <= 0.55)

    offset = array(chain.get_parameter(1))
    assert all(offset >= 0)


def test_hamiltonian_chain_restore(tmp_path):
    posterior = ToroidalGaussian()
    chain = HamiltonianChain(
        posterior=posterior, start=array([1.0, 0.1, 0.1]), grad=posterior.gradient
    )
    steps = 10
    chain.advance(steps)

    filename = tmp_path / "restore_file.npz"
    chain.save(filename)

    new_chain = HamiltonianChain.load(filename)

    assert new_chain.chain_length == chain.chain_length
    assert new_chain.probs == chain.probs
    assert all(new_chain.get_last() == chain.get_last())

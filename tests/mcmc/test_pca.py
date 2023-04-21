from mcmc_utils import line_posterior
from inference.mcmc import PcaChain


def test_pca_chain_take_step(line_posterior):
    chain = PcaChain(posterior=line_posterior, start=[0.5, 0.1])
    first_n = chain.chain_length

    chain.take_step()

    assert chain.chain_length == first_n + 1
    assert len(chain.params[0].samples) == chain.chain_length
    assert len(chain.probs) == chain.chain_length


def test_pca_chain_advance(line_posterior):
    chain = PcaChain(posterior=line_posterior, start=[0.5, 0.1])
    first_n = chain.chain_length

    steps = 104
    chain.advance(steps)

    assert chain.chain_length == first_n + steps
    assert len(chain.params[0].samples) == chain.chain_length
    assert len(chain.probs) == chain.chain_length


def test_pca_chain_restore(line_posterior, tmp_path):
    chain = PcaChain(posterior=line_posterior, start=[0.5, 0.1])
    steps = 200
    chain.advance(steps)

    filename = tmp_path / "restore_file.npz"
    chain.save(filename)

    new_chain = PcaChain.load(filename)

    assert new_chain.chain_length == chain.chain_length
    assert new_chain.probs == chain.probs
    assert all(new_chain.get_last() == chain.get_last())

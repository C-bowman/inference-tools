from mcmc_utils import line_posterior
from inference.mcmc import PcaChain


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

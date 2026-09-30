from numpy import array, nan
from mcmc_utils import line_posterior
from inference.mcmc import PcaChain, Bounds
import pytest


class OneShotRejectingRng:
    def __init__(self):
        self.normal_calls = 0

    def normal(self):
        self.normal_calls += 1
        if self.normal_calls > 1:
            raise RuntimeError("proposal was retried")
        return 10.0

    def random(self):
        return 1.0


def test_pca_chain_take_step(line_posterior):
    chain = PcaChain(posterior=line_posterior, start=[0.5, 0.1])
    first_n = chain.chain_length

    chain.take_step()

    assert chain.chain_length == first_n + 1
    assert len(chain.params[0].samples) == chain.chain_length
    assert len(chain.probs) == chain.chain_length


def test_pca_chain_records_rejected_proposal():
    chain = PcaChain(
        posterior=lambda theta: -float(theta[0] ** 2),
        start=array([0.0]),
        widths=array([1.0]),
    )
    chain.rng = OneShotRejectingRng()

    chain.take_step()

    assert chain.get_parameter(0, burn=0).tolist() == [0.0, 0.0]
    assert chain.get_probabilities(burn=0).tolist() == [0.0, 0.0]


def test_pca_chain_advance(line_posterior):
    chain = PcaChain(posterior=line_posterior, start=[0.5, 0.1])
    first_n = chain.chain_length

    steps = 104
    chain.advance(steps)

    assert chain.chain_length == first_n + steps
    assert len(chain.params[0].samples) == chain.chain_length
    assert len(chain.probs) == chain.chain_length


def test_pca_chain_advance_bounded(line_posterior):
    bounds = [array([0.4, 0.0]), array([0.6, 0.5])]
    chain = PcaChain(posterior=line_posterior, start=[0.5, 0.1], bounds=bounds)
    first_n = chain.chain_length

    steps = 104
    chain.advance(steps)

    assert chain.chain_length == first_n + steps
    assert len(chain.params[0].samples) == chain.chain_length
    assert len(chain.probs) == chain.chain_length


def test_pca_chain_restore(line_posterior, tmp_path):
    bounds = Bounds(lower=array([0.4, 0.0]), upper=array([0.6, 0.5]))
    chain = PcaChain(posterior=line_posterior, start=[0.5, 0.1], bounds=bounds)
    steps = 200
    chain.advance(steps)

    filename = tmp_path / "restore_file.npz"
    chain.save(filename)

    new_chain = PcaChain.load(filename)
    _ = PcaChain.load(filename, posterior=line_posterior)

    assert new_chain.chain_length == chain.chain_length
    assert new_chain.probs == chain.probs
    assert (new_chain.get_last() == chain.get_last()).all()
    assert (new_chain.bounds.lower == chain.bounds.lower).all()
    assert (new_chain.bounds.upper == chain.bounds.upper).all()


def test_pca_posterior_validation():
    with pytest.raises(ValueError):
        chain = PcaChain(posterior="posterior", start=array([1, 0.1]))

    with pytest.raises(ValueError):
        chain = PcaChain(posterior=lambda x: 1, start=array([1, 0.1]))

    with pytest.raises(ValueError):
        chain = PcaChain(posterior=lambda x: nan, start=array([1, 0.1]))

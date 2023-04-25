import pytest
from numpy import array, allclose
from inference.mcmc import Bounds


def test_bounds_methods():
    bnds = Bounds(lower=array([0.0, 0.0]), upper=array([1.0, 1.0]))

    assert bnds.inside(array([0.2, 0.1]))
    assert not bnds.inside(array([-0.6, 0.1]))

    assert allclose(bnds.reflect(array([-0.6, 1.1])), array([0.6, 0.9]))
    assert allclose(bnds.reflect(array([-1.7, 1.2])), array([0.3, 0.8]))

    positions, reflects = bnds.reflect_momenta(array([-0.6, 0.1]))
    assert allclose(positions, array([0.6, 0.1]))
    assert allclose(reflects, array([-1, 1]))

    positions, reflects = bnds.reflect_momenta(array([-1.6, 3.1]))
    assert allclose(positions, array([0.4, 0.9]))
    assert allclose(reflects, array([1, -1]))


def test_bounds_error_handling():
    with pytest.raises(ValueError):
        Bounds(lower=array([3.0, 0.0]), upper=array([3.0, 1.0]))

    with pytest.raises(ValueError):
        Bounds(lower=array([0.0, 0.0]), upper=array([1.0, -1.0]))

    with pytest.raises(ValueError):
        Bounds(lower=array([0.0, 0.0]), upper=array([1.0, 1]).reshape([2, 1]))

    with pytest.raises(ValueError):
        Bounds(lower=array([0.0, 0.0]), upper=array([1.0, 1.0, 1.0]))

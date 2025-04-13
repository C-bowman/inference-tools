import pytest

from numpy import linspace, exp
from numpy.random import default_rng

from inference.mcmc.hmc.mass import ScalarMass, VectorMass, MatrixMass
from inference.mcmc.hmc.mass import get_particle_mass


def test_get_particle_mass():
    rng = default_rng(112358)
    n_params = 10

    scalar_mass = get_particle_mass(1.0, n_parameters=n_params)
    r = scalar_mass.sample_momentum(rng=rng)
    v = scalar_mass.get_velocity(r)
    assert isinstance(scalar_mass, ScalarMass)
    assert r.size == n_params and r.ndim == 1
    assert v.size == n_params and v.ndim == 1

    x = linspace(1, n_params, n_params)
    vector_mass = get_particle_mass(inverse_mass=x, n_parameters=n_params)
    r = vector_mass.sample_momentum(rng=rng)
    v = vector_mass.get_velocity(r)
    assert isinstance(vector_mass, VectorMass)
    assert r.size == n_params and r.ndim == 1
    assert v.size == n_params and v.ndim == 1

    cov = exp(-0.5 * (x[:, None] - x[None, :]) ** 2)
    matrix_mass = get_particle_mass(inverse_mass=cov, n_parameters=n_params)
    r = matrix_mass.sample_momentum(rng=rng)
    v = matrix_mass.get_velocity(r)
    assert isinstance(matrix_mass, MatrixMass)
    assert r.size == n_params and r.ndim == 1
    assert v.size == n_params and v.ndim == 1

    with pytest.raises(TypeError):
        get_particle_mass([5.0], n_parameters=4)

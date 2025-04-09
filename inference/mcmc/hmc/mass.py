from abc import ABC, abstractmethod
from typing import Union
from numpy import ndarray, sqrt, eye, isscalar
from numpy.random import Generator
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular


class ParticleMass(ABC):
    inv_mass: Union[float, ndarray]

    @abstractmethod
    def get_velocity(self, r: ndarray) -> ndarray:
        pass

    @abstractmethod
    def sample_momentum(self, rng: Generator) -> ndarray:
        pass


class ScalarMass(ParticleMass):
    def __init__(self, inv_mass: float, n_parameters: int):
        self.inv_mass = inv_mass
        self.sqrt_mass = 1 / sqrt(self.inv_mass)
        self.n_parameters = n_parameters

    def get_velocity(self, r: ndarray) -> ndarray:
        return r * self.inv_mass

    def sample_momentum(self, rng: Generator) -> ndarray:
        return rng.normal(size=self.n_parameters, scale=self.sqrt_mass)


class VectorMass(ScalarMass):
    def __init__(self, inv_mass: ndarray, n_parameters: int):
        super().__init__(inv_mass, n_parameters)
        assert inv_mass.ndim == 1
        assert inv_mass.size == n_parameters


class MatrixMass(ParticleMass):
    def __init__(self, inv_mass: ndarray, n_parameters: int):
        assert inv_mass.ndim == 2
        assert inv_mass.shape[0] == inv_mass.shape[1] == n_parameters
        assert (inv_mass == inv_mass.T).all()

        self.inv_mass = inv_mass
        self.n_parameters = n_parameters
        # find the cholesky decomp of the mass matrix
        iL = cholesky(inv_mass)
        self.L = solve_triangular(iL, eye(self.n_parameters), lower=True).T

    def get_velocity(self, r: ndarray) -> ndarray:
        return self.inv_mass @ r

    def sample_momentum(self, rng: Generator) -> ndarray:
        return self.L @ rng.normal(size=self.n_parameters)


def get_particle_mass(
    inverse_mass: Union[float, ndarray], n_parameters: int
) -> ParticleMass:
    if isscalar(inverse_mass):
        return ScalarMass(inverse_mass, n_parameters)

    if not isinstance(inverse_mass, ndarray):
        raise TypeError(
            f"""\n
            \r[ HamiltonianChain error ]
            \r>> The value given to the 'inverse_mass' keyword argument must be either
            \r>> a scalar type (e.g. int or float), or a numpy.ndarray.
            \r>> Instead, the given value has type:
            \r>> {type(inverse_mass)}
            """
        )

    if inverse_mass.ndim == 1:
        return VectorMass(inverse_mass, n_parameters)
    else:
        return MatrixMass(inverse_mass, n_parameters)

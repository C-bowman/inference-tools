from numpy import dot, zeros, ones, ndarray
from abc import ABC, abstractmethod


class MeanFunction(ABC):
    """
    Abstract base class for mean functions.
    """

    @abstractmethod
    def pass_spatial_data(self, x):
        pass

    @abstractmethod
    def estimate_hyperpar_bounds(self, y):
        pass

    @abstractmethod
    def __call__(self, q, theta):
        pass

    @abstractmethod
    def build_mean(self, theta):
        pass

    @abstractmethod
    def mean_and_gradients(self, theta):
        pass


class ConstantMean(MeanFunction):
    def __init__(self, hyperpar_bounds=None):
        self.bounds = hyperpar_bounds
        self.n_params = 1
        self.hyperpar_labels = ["ConstantMean"]

    def pass_spatial_data(self, x: ndarray):
        self.n_data = x.shape[0]

    def estimate_hyperpar_bounds(self, y: ndarray):
        w = y.max() - y.min()
        self.bounds = [(y.min() - w, y.max() + w)]

    def __call__(self, q, theta):
        return theta[0]

    def build_mean(self, theta):
        return zeros(self.n_data) + theta[0]

    def mean_and_gradients(self, theta):
        return zeros(self.n_data) + theta[0], [ones(self.n_data)]


class LinearMean(MeanFunction):
    def __init__(self, hyperpar_bounds=None):
        self.bounds = hyperpar_bounds

    def pass_spatial_data(self, x: ndarray):
        self.x_mean = x.mean(axis=0)
        self.dx = x - self.x_mean[None, :]
        self.n_data = x.shape[0]
        self.n_params = 1 + x.shape[1]
        self.hyperpar_labels = ["LinearMean background"]
        self.hyperpar_labels.extend(
            [f"LinearMean gradient {i}" for i in range(x.shape[1])]
        )

    def estimate_hyperpar_bounds(self, y: ndarray):
        w = y.max() - y.min()
        grad_bounds = 10 * w / (self.dx.max(axis=0) - self.dx.min(axis=0))
        self.bounds = [(y.min() - 2 * w, y.max() + 2 * w)]
        self.bounds.extend([(-b, b) for b in grad_bounds])

    def __call__(self, q, theta):
        return theta[0] + dot(q - self.x_mean, theta[1:]).squeeze()

    def build_mean(self, theta):
        return theta[0] + dot(self.dx, theta[1:])

    def mean_and_gradients(self, theta):
        grads = [ones(self.n_data)]
        grads.extend([v for v in self.dx.T])
        return theta[0] + dot(self.dx, theta[1:]), grads


class QuadraticMean(MeanFunction):
    def __init__(self, hyperpar_bounds=None):
        self.bounds = hyperpar_bounds

    def pass_spatial_data(self, x: ndarray):
        n = x.shape[1]
        self.x_mean = x.mean(axis=0)
        self.dx = x - self.x_mean[None, :]
        self.dx_sqr = self.dx**2
        self.n_data = x.shape[0]
        self.n_params = 1 + 2 * n
        self.hyperpar_labels = ["mean_background"]
        self.hyperpar_labels.extend([f"mean_linear_coeff_{i}" for i in range(n)])
        self.hyperpar_labels.extend([f"mean_quadratic_coeff_{i}" for i in range(n)])

        self.lin_slc = slice(1, n + 1)
        self.quad_slc = slice(n + 1, 2 * n + 1)

    def estimate_hyperpar_bounds(self, y: ndarray):
        w = y.max() - y.min()
        grad_bounds = 10 * w / (self.dx.max(axis=0) - self.dx.min(axis=0))
        self.bounds = [(y.min() - 2 * w, y.max() + 2 * w)]
        self.bounds.extend([(-b, b) for b in grad_bounds])
        self.bounds.extend([(-b, b) for b in grad_bounds])

    def __call__(self, q, theta):
        d = q - self.x_mean
        lin_term = dot(d, theta[self.lin_slc]).squeeze()
        quad_term = dot(d**2, theta[self.quad_slc]).squeeze()
        return theta[0] + lin_term + quad_term

    def build_mean(self, theta):
        lin_term = dot(self.dx, theta[self.lin_slc])
        quad_term = dot(self.dx_sqr, theta[self.quad_slc])
        return theta[0] + lin_term + quad_term

    def mean_and_gradients(self, theta):
        grads = [ones(self.n_data)]
        grads.extend([v for v in self.dx.T])
        grads.extend([v for v in self.dx_sqr.T])
        return self.build_mean(theta), grads

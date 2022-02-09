from numpy import dot, zeros, ones


class ConstantMean(object):
    def __init__(self):
        self.n_params = 1
        self.hyperpar_labels = ["mean"]

    def pass_data(self, x, y):
        self.n_data = y.size
        w = y.max() - y.min()
        self.bounds = [(y.min() - w, y.max() + w)]

    def __call__(self, q, theta):
        return theta[0]

    def build_mean(self, theta):
        return zeros(self.n_data) + theta[0]

    def mean_and_gradients(self, theta):
        return zeros(self.n_data) + theta[0], [ones(self.n_data)]


class LinearMean(object):
    def __init__(self):
        pass

    def pass_data(self, x, y):
        self.x_mean = x.mean(axis=0)
        self.dx = x - self.x_mean[None, :]
        self.n_data = y.size
        self.n_params = 1 + x.shape[1]
        w = y.max() - y.min()
        grad_bounds = 10 * w / (x.max(axis=0) - x.min(axis=0))
        self.bounds = [(y.min() - 2 * w, y.max() + 2 * w)]
        self.bounds.extend([(-b, b) for b in grad_bounds])

        self.hyperpar_labels = ["mean_background"]
        self.hyperpar_labels.extend([f"mean_gradient_{i}" for i in range(x.shape[1])])

    def __call__(self, q, theta):
        return theta[0] + dot(q - self.x_mean, theta[1:]).squeeze()

    def build_mean(self, theta):
        return theta[0] + dot(self.dx, theta[1:])

    def mean_and_gradients(self, theta):
        grads = [ones(self.n_data)]
        grads.extend([v for v in self.dx.T])
        return theta[0] + dot(self.dx, theta[1:]), grads


class QuadraticMean(object):
    def __init__(self):
        pass

    def pass_data(self, x, y):
        n = x.shape[1]
        self.x_mean = x.mean(axis=0)
        self.dx = x - self.x_mean[None, :]
        self.dx_sqr = self.dx**2
        self.n_data = y.size
        self.n_params = 1 + 2 * n
        w = y.max() - y.min()
        grad_bounds = 10 * w / (x.max(axis=0) - x.min(axis=0))
        self.bounds = [(y.min() - 2 * w, y.max() + 2 * w)]
        self.bounds.extend([(-b, b) for b in grad_bounds])
        self.bounds.extend([(-b, b) for b in grad_bounds])

        self.hyperpar_labels = ["mean_background"]
        self.hyperpar_labels.extend([f"mean_linear_coeff_{i}" for i in range(n)])
        self.hyperpar_labels.extend([f"mean_quadratic_coeff_{i}" for i in range(n)])

        self.lin_slc = slice(1, n + 1)
        self.quad_slc = slice(n + 1, 2 * n + 1)

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

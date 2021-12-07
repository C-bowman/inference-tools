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

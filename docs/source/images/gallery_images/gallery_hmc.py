from numpy import sqrt, array, argsort, exp
import matplotlib.pyplot as plt


class ToroidalGaussian:
    def __init__(self):
        self.R0 = 1.0  # torus major radius
        self.ar = 10.0  # torus aspect ratio
        self.iw2 = (self.ar / self.R0) ** 2

    def __call__(self, theta):
        x, y, z = theta
        r_sqr = z**2 + (sqrt(x**2 + y**2) - self.R0) ** 2
        return -0.5 * r_sqr * self.iw2

    def gradient(self, theta):
        x, y, z = theta
        R = sqrt(x**2 + y**2)
        K = 1 - self.R0 / R
        g = array([K * x, K * y, z])
        return -g * self.iw2


posterior = ToroidalGaussian()

from inference.mcmc import HamiltonianChain

hmc = HamiltonianChain(
    posterior=posterior, grad=posterior.gradient, start=[1, 0.1, 0.1]
)

hmc.advance(6000)
hmc.burn = 1000


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
ax.set_xticks([-1, -0.5, 0.0, 0.5, 1.0])
ax.set_yticks([-1, -0.5, 0.0, 0.5, 1.0])
ax.set_zticks([-1, -0.5, 0.0, 0.5, 1.0])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# ax.set_title('Hamiltonian Monte-Carlo')
L = 0.99
ax.set_xlim([-L, L])
ax.set_ylim([-L, L])
ax.set_zlim([-L, L])
probs = array(hmc.get_probabilities())
inds = argsort(probs)
colors = exp(probs - max(probs))
xs, ys, zs = [array(hmc.get_parameter(i)) for i in [0, 1, 2]]
ax.scatter(xs, ys, zs, c=colors, marker=".", alpha=0.5)
plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.03)
plt.savefig("gallery_hmc.png")
plt.show()

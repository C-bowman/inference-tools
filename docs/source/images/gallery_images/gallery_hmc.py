
from numpy import sqrt, array, argsort, exp
import matplotlib.pyplot as plt

class ToroidalGaussian(object):
    def __init__(self):
        self.R0 = 1.  # torus major radius
        self.ar = 10.  # torus aspect ratio
        self.w2 = (self.R0 / self.ar) ** 2

    def __call__(self, theta):
        x, y, z = theta
        r = sqrt(z ** 2 + (sqrt(x ** 2 + y ** 2) - self.R0) ** 2)
        return -0.5 * r ** 2 / self.w2

    def gradient(self, theta):
        x, y, z = theta
        R = sqrt(x ** 2 + y ** 2)
        K = 1 - self.R0 / R
        g = array([K * x, K * y, z])
        return -g / self.w2


posterior = ToroidalGaussian()

from inference.mcmc import HamiltonianChain

hmc = HamiltonianChain(posterior=posterior, grad=posterior.gradient, start=[1, 0.1, 0.1])

hmc.advance(6000)
hmc.burn = 1000


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (5,4))
ax = fig.add_subplot(111, projection='3d')
ax.set_xticks([-1,-0.5,0.,0.5,1.])
ax.set_yticks([-1,-0.5,0.,0.5,1.])
ax.set_zticks([-1,-0.5,0.,0.5,1.])
# ax.set_title('Hamiltonian Monte-Carlo')
L = 1.1
ax.set_xlim([-L,L]); ax.set_ylim([-L,L]); ax.set_zlim([-L,L])
probs = array(hmc.get_probabilities())
inds = argsort(probs)
colors = exp(probs - max(probs))
xs, ys, zs = [ array(hmc.get_parameter(i)) for i in [0,1,2] ]
ax.scatter(xs, ys, zs, c=colors, marker = '.', alpha = 0.5)
plt.subplots_adjust(left=0., right=1., top=1., bottom=0.03)
plt.savefig('gallery_hmc.png')
plt.show()
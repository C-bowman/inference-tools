from numpy import array, exp
import matplotlib.pyplot as plt
from inference.mcmc import GibbsChain


def rosenbrock(t):
    x, y = t
    x2 = x**2
    b = 15.  # correlation strength parameter
    v = 3.   # variance of the gaussian term
    return -x2 - b*(y - x2)**2 - 0.5*(x2 + y**2)/v


# create the chain object
gibbs = GibbsChain(posterior=rosenbrock, start=array([2., -4.]))
gibbs.advance(150000)
gibbs.burn = 10000
gibbs.thin = 70


p = gibbs.get_probabilities() # color the points by their probability value
fig = plt.figure(figsize=(6, 5))
ax1 = fig.add_subplot(111)
ax1.scatter(
    gibbs.get_parameter(0),
    gibbs.get_parameter(1),
    c=exp(p-max(p)),
    marker='.'
)
ax1.set_ylim([None, 2.8])
ax1.set_xlim([-1.8, 1.8])
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.grid()
plt.tight_layout()
plt.savefig('gallery_gibbs_sampling.png')
plt.show()
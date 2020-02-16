
from numpy import linspace, array
from numpy.random import normal, seed
import matplotlib.pyplot as plt
from inference.mcmc import PcaChain

class HdiPosterior(object):
    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.sigma = s

    @staticmethod
    def forward(x,theta):
        a,b,c = theta
        return a*x**2 + b*x + c

    def __call__(self, theta):
        prediction = self.forward(self.x, theta)
        return -0.5*(((prediction-self.y)/self.sigma)**2).sum()

seed(4)

x = linspace(1, 9, 9)
start = [-0.5,4.,30.]
y = HdiPosterior.forward(x, start)
s = y*0.1 + 2
y += normal(size=len(y))*s
p = HdiPosterior(x,y,s)

chain = PcaChain(posterior=p, start = start)#, parameter_boundaries=[(0,200),(0.1,10),(0.1,15)])
# chain = GibbsChain(posterior=p, start = [1.,1.,5.])#, parameter_boundaries=[(0,200),(0.1,10),(0.1,15)])
chain.advance(105000)
chain.burn = 5000
chain.thin = 2

# chain.plot_diagnostics()
# chain.trace_plot()
# chain.matrix_plot()


x_fits = linspace(0,10,100)
sample = chain.get_sample()
# pass each through the forward model
curves = array([HdiPosterior.forward(x_fits, theta) for theta in sample])

# We can use the hdi_plot function from the plotting module to plot
# highest-density intervals for each point where the model is evaluated:
from inference.plotting import hdi_plot

fig = plt.figure(figsize = (5,4))
ax = fig.add_subplot(111)

hdi_plot(x_fits, curves, axis=ax)
ax.errorbar(x, y, yerr = s, c = 'red', markeredgecolor = 'black', marker = 'D', ls = 'none', markersize=5, label = 'data')
# ax.set_ylim([20.,None])
ax.set_xlim([0,10])
# ax.set_xticks([])
# ax.set_yticks([])
# ax1.set_title('Gibbs sampling')
plt.tight_layout()
plt.legend()
plt.savefig('gallery_hdi.png')
plt.show()
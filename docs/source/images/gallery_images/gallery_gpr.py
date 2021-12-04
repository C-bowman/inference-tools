import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import exp, sin, sqrt
from numpy import linspace, zeros, array, meshgrid
from numpy.random import multivariate_normal as mvn
from numpy.random import normal, random, seed
from inference.gp import GpRegressor

seed(3) #4

"""
Code demonstrating the use of the GpRegressor class found in inference.gp_tools
"""

# create some testing data
Nx = 9*2
x = list( linspace(-3,1,Nx//2) )
x.extend( list( linspace(4,9,Nx//2) ) )
x = array(x)

# generate points q at which to evaluate the
# GP regression estimate
Nq = 200
q = linspace(-4, 10, Nq) # cover whole range, including the gap


sig = 0.1 # assumed normal error on the data points
y_c = ( 1. / (1 + exp(-q)) ) + 0.1*sin(2*q) # underlying function
y   = ( 1. / (1 + exp(-x)) ) + 0.1*sin(2*x) + sig*normal(size=len(x)) # sampled y data
errs = zeros(len(y)) + sig # y data errors


# initialise the class with the data and errors
GP = GpRegressor(x, y, y_err = errs)

# call the instance to get estimates for the points in q
mu_q, sig_q = GP(q)

# now plot the regression estimate and the data together
c1 = 'red'; c2 = 'blue'; c3 = 'green'
fig = plt.figure( figsize = (5,4) )
ax = fig.add_subplot(111)
ax.plot(q, mu_q, lw = 2, color = c2, label = 'posterior mean')
ax.fill_between(q, mu_q-sig_q, mu_q-sig_q*2, color = c2, alpha = 0.15, label = r'$\pm 2 \sigma$ interval')
ax.fill_between(q, mu_q+sig_q, mu_q+sig_q*2, color = c2, alpha = 0.15)
ax.fill_between(q, mu_q-sig_q, mu_q+sig_q, color = c2, alpha = 0.4, label = r'$\pm 1 \sigma$ interval')
# ax.plot(x, y, 'o', color = c1, label = 'data', markerfacecolor = 'none', markeredgewidth = 2)
ax.plot(x, y, 'o', color = c1, label = 'data', marker = 'D', ls = 'none', markeredgecolor = 'black')
ax.set_ylim([-0.1, 1.3])
ax.set_xlim([-4, 10])
# ax.set_title('Prediction using posterior mean and covariance', fontsize = 10)
ax.set_ylabel('function value', fontsize = 1)
ax.set_xlabel('spatial coordinate', fontsize = 10)
ax.grid()
ax.legend(loc=4, fontsize = 10)
plt.tight_layout()
plt.savefig('gallery_gpr.png')
plt.show()
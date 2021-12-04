
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import exp, sin, sqrt
from numpy import linspace, zeros, array, meshgrid
from numpy.random import multivariate_normal as mvn
from numpy.random import normal, random, seed
from inference.gp import GpRegressor

seed(4)

"""
Code demonstrating the use of the GpRegressor class found in inference.gp_tools
"""

# create some testing data
Nx = 24*2
x = list( linspace(-3,1,Nx//2) )
x.extend( list( linspace(4,9,Nx//2) ) )
x = array(x)

# generate points q at which to evaluate the
# GP regression estimate
Nq = 200
q = linspace(-4, 10, Nq) # cover whole range, including the gap


sig = 0.05 # assumed normal error on the data points
y_c = ( 1. / (1 + exp(-q)) ) + 0.1*sin(2*q) # underlying function
y   = ( 1. / (1 + exp(-x)) ) + 0.1*sin(2*x) + sig*normal(size=len(x)) # sampled y data
errs = zeros(len(y)) + sig # y data errors


# plot the data points plus the underlying function
# from which they are sampled
fig = plt.figure( figsize = (9,6) )
ax = fig.add_subplot(111)
ax.plot(q, y_c, lw = 2, color = 'black', label = 'test function')
ax.plot(x, y, 'o', color = 'red', label = 'sampled data')
ax.errorbar(x, y, yerr = errs, fmt = 'none', ecolor = 'red')
ax.set_ylim([-0.5, 1.5])
ax.set_xlim([-4, 10])
ax.set_title('Generate simulated data from a test function', fontsize = 12)
ax.set_ylabel('function value', fontsize = 12)
ax.set_xlabel('spatial coordinate', fontsize = 12)
ax.grid()
ax.legend(loc=2, fontsize = 12)
plt.tight_layout()
plt.savefig('sampled_data.png')
plt.close()


# initialise the class with the data and errors
GP = GpRegressor(x, y, y_err = errs)

# call the instance to get estimates for the points in q
mu_q, sig_q = GP(q)

# now plot the regression estimate and the data together
c1 = 'red'; c2 = 'blue'; c3 = 'green'
fig = plt.figure( figsize = (9,6) )
ax = fig.add_subplot(111)
ax.plot(q, mu_q, lw = 2, color = c2, label = 'posterior mean')
ax.fill_between(q, mu_q-sig_q, mu_q-sig_q*2, color = c2, alpha = 0.15, label = r'$\pm 2 \sigma$ interval')
ax.fill_between(q, mu_q+sig_q, mu_q+sig_q*2, color = c2, alpha = 0.15)
ax.fill_between(q, mu_q-sig_q, mu_q+sig_q, color = c2, alpha = 0.3, label = r'$\pm 1 \sigma$ interval')
ax.plot(x, y, 'o', color = c1, label = 'data', markerfacecolor = 'none', markeredgewidth = 2)
ax.set_ylim([-0.5, 1.5])
ax.set_xlim([-4, 10])
ax.set_title('Prediction using posterior mean and covariance', fontsize = 12)
ax.set_ylabel('function value', fontsize = 12)
ax.set_xlabel('spatial coordinate', fontsize = 12)
ax.grid()
ax.legend(loc=2, fontsize = 12)
plt.tight_layout()
plt.savefig('regression_estimate.png')
plt.close()


# As the estimate itself is defined by a multivariate normal distribution,
# we can draw samples from that distribution.
# to do this, we need to build the full covariance matrix and mean for the
# desired set of points using the 'build_posterior' method:
mu, sigma = GP.build_posterior(q)
# now draw samples
samples = mvn(mu, sigma, 100)
# and plot all the samples
fig = plt.figure( figsize = (9,6) )
ax = fig.add_subplot(111)
for i in range(100):
    ax.plot(q, samples[i,:], lw = 0.5)
ax.set_title('100 samples drawn from the posterior distribution', fontsize = 12)
ax.set_ylabel('function value', fontsize = 12)
ax.set_xlabel('spatial coordinate', fontsize = 12)
ax.set_xlim([-4, 10])
plt.grid()
plt.tight_layout()
plt.savefig('posterior_samples.png')
plt.close()


# The gradient of the Gaussian process estimate also has a multivariate normal distribution.
# The mean vector and covariance matrix of the gradient distribution for a series of points
# can be generated using the GP.gradient() method:
gradient_mean, gradient_variance = GP.gradient(q)
# in this example we have only one spatial dimension, so the covariance matrix has size 1x1
sigma = sqrt(gradient_variance) # get the standard deviation at each point in 'q'

# plot the distribution of the gradient
fig = plt.figure( figsize = (9,6) )
ax = fig.add_subplot(111)
ax.plot(q, gradient_mean, lw = 2, color = 'blue', label = 'gradient mean')
ax.fill_between(q, gradient_mean-sigma, gradient_mean+sigma, alpha = 0.3, color = 'blue', label = r'$\pm 1 \sigma$ interval')
ax.fill_between(q, gradient_mean+sigma, gradient_mean+2*sigma, alpha = 0.15, color = 'blue', label = r'$\pm 2 \sigma$ interval')
ax.fill_between(q, gradient_mean-sigma, gradient_mean-2*sigma, alpha = 0.15, color = 'blue')
ax.set_title('Distribution of the gradient of the GP', fontsize = 12)
ax.set_ylabel('function gradient value', fontsize = 12)
ax.set_xlabel('spatial coordinate', fontsize = 12)
ax.set_xlim([-4, 10])
ax.grid()
ax.legend(fontsize = 12)
plt.tight_layout()
plt.savefig('gradient_prediction.png')
plt.close()




# """
# 2D example
# """
# from mpl_toolkits.mplot3d import Axes3D
# # define an 2D function as an example
# def solution(v):
#     x, y = v
#     f = 0.5
#     return sin(x*0.5*f)+sin(y*f)
#
# # Sample the function value at some random points
# # to use as our data
# N = 50
# x = random(size=N) * 15
# y = random(size=N) * 15
#
# # build coordinate list for all points in the data grid
# coords = list(zip(x,y))
#
# # evaluate the test function at all points
# z = list(map(solution, coords))
#
# # build a colormap for the points
# colmap = cm.viridis((z - min(z)) / (max(z) - min(z)))
#
# # now 3D scatterplot the test data to visualise
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter([i[0] for i in coords], [i[1] for i in coords], z, color = colmap)
# plt.tight_layout()
# plt.show()
#
# # Train the GP on the data
# GP = GpRegressor(coords, z)
#
# # if we provide no error data, a small value is used (compared with
# # spread of values in the data) such that the estimate is forced to
# # pass (almost) through each data point.
#
# # make a set of axes on which to evaluate the GP estimate
# gp_x = linspace(0,15,40)
# gp_y = linspace(0,15,40)
#
# # build a coordinate list from these axes
# gp_coords = [ (i,j) for i in gp_x for j in gp_y ]
#
# # evaluate the estimate
# mu, sig = GP(gp_coords)
#
# # build a colormap for the surface
# Z = mu.reshape([40,40]).T
# Z = (Z-Z.min())/(Z.max()-Z.min())
# colmap = cm.viridis(Z)
# rcount, ccount, _ = colmap.shape
#
# # surface plot the estimate
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(*meshgrid(gp_x, gp_y), mu.reshape([40,40]).T, rcount=rcount,
#                        ccount=ccount, facecolors=colmap, shade=False)
# surf.set_facecolor((0,0,0,0))
#
# # overplot the data points
# ax.scatter([i[0] for i in coords], [i[1] for i in coords], z, color = 'black')
# plt.tight_layout()
# plt.show()
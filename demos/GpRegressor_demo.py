
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import exp, sin
from numpy import linspace, zeros, array, meshgrid
from numpy.random import multivariate_normal as mvn
from numpy.random import normal, random
from inference.gp_tools import GpRegressor

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


sig = 0.04 # assumed normal error on the data points
y_c = ( 1. / (1 + exp(-q)) ) + 0.1*sin(2*q) # underlying function
y   = ( 1. / (1 + exp(-x)) ) + 0.1*sin(2*x) + sig*normal(size=len(x)) # sampled y data
errs = zeros(len(y)) + sig # y data errors


# plot the data points plus the underlying function
# from which they are sampled
fig = plt.figure( figsize = (15,9) )
ax = fig.add_subplot(111)
ax.plot(q, y_c, lw = 2, color = 'black', label = 'test function')
ax.plot(x, y, 'D', color = 'blue', label = 'sampled data')
ax.errorbar(x, y, yerr = errs, fmt = 'none', ecolor = 'blue')
ax.set_ylim([-0.5, 1.5])
ax.set_title('Generate simulated data from a test function')
ax.grid()
ax.legend(loc=2)
plt.show()


# initialise the class with the data and errors
GP = GpRegressor(x, y, y_err = errs)

# call the class to get estimates for points in q
mu_q, sig_q = GP(q)

err_q = sig_q*2 # convert 1-sigma error to 95% credibility boundary


# now plot the regression estimate and the data together
c1 = 'blue'; c2 = 'red'; c3 = 'green'
fig = plt.figure( figsize = (15,9) )
ax = fig.add_subplot(111)
ax.plot(x, y, 'D', color = c1, label = 'data')
ax.plot(q, mu_q, lw = 2, color = c3, label = 'posterior mean')
ax.plot(q, mu_q-err_q, lw = 2, ls = 'dashed', color = c2, label = '95% credible bounds')
ax.plot(q, mu_q+err_q, lw = 2, ls = 'dashed', color = c2)
ax.set_ylim([-0.5, 1.5])
ax.set_title('Prediction using posterior mean and covariance')
ax.grid()
ax.legend(loc=2)
plt.show()


# now verify that the true function lies within the 95% envelope
fig = plt.figure( figsize = (15,9) )
ax = fig.add_subplot(111)
ax.plot(q, y_c, lw = 2, color = 'black', label = 'test function')
ax.plot(q, mu_q-err_q, lw = 2, ls = 'dashed', color = c2, label = '95% credible bounds')
ax.plot(q, mu_q+err_q, lw = 2, ls = 'dashed', color = c2)
ax.set_ylim([-0.5, 1.5])
ax.set_title('Verify that the true function lies within the 95% envelope')
ax.grid()
ax.legend(loc=2)
plt.show()


# finally, possible posterior functions can be sampled from a multivariate
# normal distribution
# to do this, we need to build the full covariance matrix and mean for the
# desired set of points using the 'build_posterior' method:
mu, sigma = GP.build_posterior(q)
fig = plt.figure( figsize = (15,9) )
ax = fig.add_subplot(111)
samples = mvn(mu, sigma, 100)
for i in range(100):
    ax.plot(q, samples[i,:])
ax.set_title('100 samples drawn from the posterior distribution')
plt.grid()
plt.show()


# this allows use to obtain reliable uncertainties for derived quantities.
# e.g. what if we wish to know the PDF of the gradient at x = 0?

# first generate samples around x = 0 to calculate the gradient
q = [-0.01, 0.01]
mu, sigma = GP.build_posterior(q)
samples = mvn(mu, sigma, 1000)
gradients = (samples[:,1]-samples[:,0])/0.02

# now we have a set of gradient samples, generate a PDF from the sample.
from inference.pdf_tools import UnimodalPdf
pdf = UnimodalPdf(gradients)

# plot the PDF
ax = linspace(0.2,0.8,300)
plt.plot( ax, pdf(ax), lw = 2)
plt.xlabel('Gradient at x = 0')
plt.ylabel('probability density')
plt.grid()
plt.show()




"""
2D example
"""
from mpl_toolkits.mplot3d import Axes3D

# define an 2D function as an example
def solution(v):
    x, y = v
    f = 0.5
    return sin(x*0.5*f)+sin(y*f)

# Sample the function value at some random points
# to use as our data
N = 50
x = random(size=N) * 15
y = random(size=N) * 15

# build coordinate list for all points in the data grid
coords = list(zip(x,y))

# evaluate the test function at all points
z = list(map(solution, coords))

# build a colormap for the points
colmap = cm.viridis((z - min(z)) / (max(z) - min(z)))

# now 3D scatterplot the test data to visualise
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([i[0] for i in coords], [i[1] for i in coords], z, color = colmap)
plt.show()

# Train the GP on the data
GP = GpRegressor(coords, z)

# if we provide no error data, a small value is used (compared with
# spread of values in the data) such that the estimate is forced to
# pass (almost) through each data point.

# make a set of axes on which to evaluate the GP estimate
gp_x = linspace(0,15,40)
gp_y = linspace(0,15,40)

# build a coordinate list from these axes
gp_coords = [ (i,j) for i in gp_x for j in gp_y ]

# evaluate the estimate
mu, sig = GP(gp_coords)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# build a colormap for the surface
Z = mu.reshape([40,40]).T
Z = (Z-Z.min())/(Z.max()-Z.min())
colmap = cm.viridis(Z)
rcount, ccount, _ = colmap.shape

# surface plot the estimate
surf = ax.plot_surface(*meshgrid(gp_x, gp_y), mu.reshape([40,40]).T, rcount=rcount, ccount=ccount,
                       facecolors=colmap, shade=False)
surf.set_facecolor((0,0,0,0))

# overplot the data points
ax.scatter([i[0] for i in coords], [i[1] for i in coords], z, color = 'black')
plt.show()

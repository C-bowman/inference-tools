
import matplotlib.pyplot as plt
from numpy import exp, sqrt, linspace, zeros, sum, dot, diag
from numpy.random import normal

from inference.gp_tools import GpInverter

"""
Code demonstrating the use of the gp_inverter class from inference.gp_tools
"""

# create a function which defines the solution to the problem
def solution(x):
	z1 = (x + 1.5)/0.5
	G1 = 3*exp(-0.5*(z1**2))

	z2 = (x - 1.5)/0.7
	G2 = 2*exp(-0.5*(z2**2))
	return G1 + G2

# create the solution grid and spatial axis
N = 225
spatial_points = linspace(-5, 5, N)
b = zeros([N,1])

# here we use map as it generalises easily to higher
# dimensional problems.
b[:,0] = list(map(solution, spatial_points))


# now define the data spatial grid
M = 400
x = linspace(-5,5,M)


# now create a smoothing matrix to blur the solution
G = zeros([M,N])
for i in range(M):
	z = (spatial_points - x[i]) / 0.5
	f = exp(-0.5*(z**2))
	G[i,:] = f / sum(f)


# create a diagonal covariance matrix for the data which
# contains the errors on the measured data
sigma = 0.01
S_y = zeros([M,M])
iS_y = zeros([M,M])
for i in range(M):
	S_y[i,i] = sigma**2
	iS_y[i,i] = sigma**(-2)


y = dot(G, b) # create the simulated data
y[:,0] += normal(size=M)*sigma # add noise to the simulated data


# now we use the gp_inverter class to attempt to recover the solution
solver = GpInverter(spatial_points, y, S_y, G)


S_b = solver.S_b # extract the posterior covariance
mu_b = solver.mu_b[:,0] # extract the posterior mean
err = 2*sqrt(diag(S_b)) # create 95% error bars based on covariance


# plot the real solution along with the projection
plt.figure(figsize=(12,9))
plt.plot(spatial_points, b, '.-', label = 'solution vector values')
plt.plot(x, dot(G, b), '.-', label = 'simulated data values')
plt.grid()
plt.legend()
plt.show()


"""
GP solution plot
"""
plt.figure(figsize=(12,9))
plt.plot(x, y, '.-', lw = 2, label = 'measured data')
plt.plot(spatial_points, b, '.-', lw = 2, color = 'green', label = 'true values')
plt.plot(spatial_points, mu_b, lw = 2, color = 'red', label = 'Gaussian process solution')
plt.plot(spatial_points, mu_b - err, ls = 'dashed', color = 'red')
plt.plot(spatial_points, mu_b + err, ls = 'dashed', color = 'red')
plt.title('gaussian process solution')
plt.grid()
plt.legend()
plt.show()

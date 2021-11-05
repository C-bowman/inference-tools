
from numpy import linspace, zeros, subtract, exp
from numpy.random import multivariate_normal

# Create a spatial axis and use it to define a Gaussian process
N = 8
x = linspace(1,N,N)
mean = zeros(N)
covariance = exp(-0.1*subtract.outer(x,x)**2)

# sample from the Gaussian process
samples = multivariate_normal(mean, covariance, size = 20000)
samples = [ samples[:,i] for i in range(N) ]

# use matrix_plot to visualise the sample data
from inference.plotting import matrix_plot
matrix_plot(samples, filename = 'matrix_plot_example.png')
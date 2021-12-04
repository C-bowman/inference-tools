from inference.gp import GpOptimiser

import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import sin, cos, linspace, array, meshgrid

mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

def example_plot_1d():
    mu, sig = GP(x_gp)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 3, 1]}, figsize = (10,8))

    ax1.plot(evaluations, max_values, marker = 'o', ls = 'solid', c = 'orange', label = 'highest observed value', zorder = 5)
    ax1.plot([2,12], [max(y_func), max(y_func)], ls = 'dashed', label = 'actual max', c = 'black')
    ax1.set_xlabel('function evaluations')
    ax1.set_xlim([2,12])
    ax1.set_ylim([max(y)-0.3, max(y_func)+0.3])
    ax1.xaxis.set_label_position('top')
    ax1.yaxis.set_label_position('right')
    ax1.xaxis.tick_top()
    ax1.set_yticks([])
    ax1.legend(loc=4)

    ax2.plot(GP.x, GP.y, 'o', c = 'red', label = 'observations', zorder = 5)
    ax2.plot(x_gp, y_func, lw = 1.5, c = 'red', ls = 'dashed', label = 'actual function')
    ax2.plot(x_gp, mu, lw = 2, c = 'blue', label = 'GP prediction')
    ax2.fill_between(x_gp, (mu-2*sig), y2=(mu+2*sig), color = 'blue', alpha = 0.15, label = '95% confidence interval')
    ax2.set_ylim([-1.5,4])
    ax2.set_ylabel('y')
    ax2.set_xticks([])

    aq = array([abs(GP.acquisition(array([k]))) for k in x_gp]).squeeze()
    proposal = x_gp[aq.argmax()]
    ax3.fill_between(x_gp, 0.9*aq/aq.max(), color = 'green', alpha = 0.15)
    ax3.plot(x_gp, 0.9*aq/aq.max(), color = 'green', label = 'acquisition function')
    ax3.plot([proposal]*2, [0.,1.], c = 'green', ls = 'dashed', label = 'acquisition maximum')
    ax2.plot([proposal]*2, [-1.5,search_function(proposal)], c = 'green', ls = 'dashed')
    ax2.plot(proposal, search_function(proposal), 'o', c = 'green', label = 'proposed observation')
    ax3.set_ylim([0,1])
    ax3.set_yticks([])
    ax3.set_xlabel('x')
    ax3.legend(loc=1)
    ax2.legend(loc=2)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()

def example_plot_2d():
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(10, 8))
    plt.subplots_adjust(hspace=0)

    ax1.plot(evaluations, max_values, marker='o', ls='solid', c='orange', label='optimum value', zorder=5)
    ax1.plot([5, 30], [z_func.max(), z_func.max()], ls='dashed', label='actual max', c='black')
    ax1.set_xlabel('function evaluations')
    ax1.set_xlim([5, 30])
    ax1.set_ylim([max(y) - 0.3, z_func.max() + 0.3])
    ax1.xaxis.set_label_position('top')
    ax1.yaxis.set_label_position('right')
    ax1.xaxis.tick_top()
    ax1.set_yticks([])
    ax1.legend(loc=4)

    ax2.contour(*mesh, z_func, 40)
    ax2.plot([i[0] for i in GP.x], [i[1] for i in GP.x], 'D', c='red', markeredgecolor='black')
    plt.show()





"""
GpOptimiser extends the functionality of GpRegressor to perform 'Bayesian optimisation'.

Bayesian optimisation is suited to problems for which a single evaluation of the function
being explored is expensive, such that the total number of function evaluations must be
made as small as possible.
"""

# define the function whose maximum we will search for
def search_function(x):
    return sin(0.5*x) + 3 / (1 + (x-1)**2)

# define bounds for the optimisation
bounds = [(-8.,8.)]

# create some initialisation data
x = array([-8,8])
y = search_function(x)

# create an instance of GpOptimiser
GP = GpOptimiser(x,y,bounds=bounds)


# here we evaluate the search function for plotting purposes
M = 500
x_gp = linspace(*bounds[0],M)
y_func = search_function(x_gp)
max_values = [max(GP.y)]
evaluations = [len(GP.y)]


for i in range(11):
    # plot the current state of the optimisation
    example_plot_1d()

    # request the proposed evaluation
    new_x = GP.propose_evaluation()

    # evaluate the new point
    new_y = search_function(new_x)

    # update the gaussian process with the new information
    GP.add_evaluation(new_x, new_y)

    # track the optimum value for plotting
    max_values.append(max(GP.y))
    evaluations.append(len(GP.y))










"""
2D example
"""
from mpl_toolkits.mplot3d import Axes3D

# define a new 2D search function
def search_function(v):
    x,y = v
    z = ((x-1)/2)**2 + ((y+3)/1.5)**2
    return sin(0.5*x) + cos(0.4*y) + 5/(1 + z)

# set bounds
bounds = [(-8,8), (-8,8)]

# evaluate function for plotting
N = 80
x = linspace(*bounds[0], N)
y = linspace(*bounds[1], N)
mesh = meshgrid(x, y)
z_func = search_function(mesh)



# create some initialisation data
# we've picked a point at each corner and one in the middle
x = [(-8,-8), (8,-8), (-8,8), (8,8), (0,0)]
y = [search_function(k) for k in x]

# initiate the optimiser
GP = GpOptimiser(x,y,bounds=bounds)


max_values = [max(GP.y)]
evaluations = [len(GP.y)]

for i in range(25):
    new_x = GP.propose_evaluation()
    new_y = search_function(new_x)
    GP.add_evaluation(new_x, new_y)

    # track the optimum value for plotting
    max_values.append(max(GP.y))
    evaluations.append(len(GP.y))

# plot the results
example_plot_2d()
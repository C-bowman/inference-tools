from inference.gp import GpOptimiser

import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import sin, linspace, array

mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

def example_plot_1d(filename):
    mu, sig = GP(x_gp)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 3, 1]}, figsize = (10,8))

    line, = ax1.plot(evaluations, max_values, c = 'purple', alpha = 0.3, zorder = 5)
    mark, = ax1.plot(evaluations, max_values, marker = 'o', ls = 'none', c = 'purple', zorder = 5)
    ax1.plot([2,12], [max(y_func), max(y_func)], ls = 'dashed', label = 'actual max', c = 'black')
    ax1.set_xlabel('function evaluations', fontsize = 12)
    ax1.set_xlim([2,12])
    ax1.set_ylim([max(y)-0.3, max(y_func)+0.3])
    ax1.xaxis.set_label_position('top')
    ax1.yaxis.set_label_position('right')
    ax1.xaxis.tick_top()
    ax1.set_yticks([])
    ax1.legend([(line, mark)], ['best observed value'], loc=4)

    ax2.plot(GP.x, GP.y, 'o', c = 'red', label = 'observations', zorder = 5)
    ax2.plot(x_gp, y_func, lw = 1.5, c = 'red', ls = 'dashed', label = 'actual function')
    ax2.plot(x_gp, mu, lw = 2, c = 'blue', label = 'GP prediction')
    ax2.fill_between(x_gp, (mu-2*sig), y2=(mu+2*sig), color = 'blue', alpha = 0.15, label = r'$\pm 2 \sigma$ interval')
    ax2.set_ylim([-1.5,4])
    ax2.set_ylabel('function value', fontsize = 12)
    ax2.set_xticks([])

    aq = array([abs(GP.acquisition(k)) for k in x_gp])
    proposal = x_gp[aq.argmax()]
    ax3.fill_between(x_gp, 0.9*aq/aq.max(), color='green', alpha=0.15)
    ax3.plot(x_gp, 0.9*aq/aq.max(), c = 'green', label = 'acquisition function')
    ax3.plot([proposal]*2, [0.,1.], c = 'green', ls = 'dashed', label = 'acquisition maximum')
    ax2.plot([proposal]*2, [-1.5,search_function(proposal)], c = 'green', ls = 'dashed')
    ax2.plot(proposal, search_function(proposal), 'D', c = 'green', label = 'proposed observation')
    ax3.set_ylim([0,1])
    ax3.set_yticks([])
    ax3.set_xlabel('spatial coordinate', fontsize = 12)
    ax3.legend(loc=1)
    ax2.legend(loc=2)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(filename)
    plt.close()




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
bounds = [(-8,8)]

# create some initialisation data
x = array([-8,8])
y = search_function(x)

# create an instance of GpOptimiser
GP = GpOptimiser(x,y,bounds=bounds)


# here we evaluate the search function for plotting purposes
M = 1000
x_gp = linspace(*bounds[0],M)
y_func = search_function(x_gp)
max_values = [max(GP.y)]
evaluations = [len(GP.y)]

N_iterations = 11
files = ['iteration_{}.png'.format(i) for i in range(N_iterations)]
for filename in files:
    # plot the current state of the optimisation
    example_plot_1d(filename)

    # request the proposed evaluation
    aq = array([abs(GP.acquisition(k)) for k in x_gp])
    new_x = x_gp[aq.argmax()]
    # evaluate the new point
    new_y = search_function(new_x)

    # update the gaussian process with the new information
    GP.add_evaluation(new_x, new_y)

    # track the optimum value for plotting
    max_values.append(max(GP.y))
    evaluations.append(len(GP.y))




from imageio import mimwrite, imread
from itertools import chain
from os import remove


images = []
for filename in chain(files, [files[-1]]):
    images.append(imread(filename))

mimwrite('GpOptimiser_iteration.gif', images, duration = 2.)

for filename in files:
    remove(filename)


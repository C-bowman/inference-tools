
"""
.. moduleauthor:: Chris Bowman <chris.bowman.physics@gmail.com>
"""
from numpy import array, meshgrid, linspace, sqrt, ceil
from itertools import product, cycle
from warnings import warn
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from inference.pdf_tools import GaussianKDE, KDE2D, sample_hdi


def matrix_plot(samples, labels = None, show = True, reference = None, filename = None, plot_style = 'contour',
                colormap = 'Blues', show_ticks = None, point_colors = None, point_size = 1, label_size = 10):
    """
    Construct a 'matrix plot' for a set of variables which shows all possible
    1D and 2D marginal distributions.

    :param samples: \
        A list of array-like objects containing the samples for each variable.

    :keyword labels: \
        A list of strings to be used as axis labels for each parameter being plotted.

    :keyword bool show: \
        Sets whether the plot is displayed.

    :keyword reference: \
        A list of reference values for each parameter which will be over-plotted.

    :keyword str filename: \
        File path to which the matrix plot will be saved (if specified).

    :keyword str plot_style: \
        Specifies the type of plot used to display the 2D marginal distributions.
        Available styles are 'contour' for filled contour plotting, 'histogram' for
        hexagonal-bin histogram, and 'scatter' for scatterplot.

    :keyword bool show_ticks: \
        By default, axis ticks are only shown when plotting less than 6 variables.
        This behaviour can be overridden for any number of parameters by setting
        show_ticks to either True or False.

    :keyword point_colors: \
        An array containing data which will be used to set the colors of the points
        if the plot_style argument is set to 'scatter'.

    :keyword point_size: \
        An array containing data which will be used to set the size of the points
        if the plot_style argument is set to 'scatter'.

    :keyword int label_size: \
        The font-size used for axis labels.
    """

    N_par = len(samples)
    if labels is None: # set default axis labels if none are given
        if N_par >= 10:
            labels = ['p' + str(i) for i in range(N_par)]
        else:
            labels = ['param ' + str(i) for i in range(N_par)]
    else:
        if len(labels) != N_par:
            raise ValueError('number of labels must match number of plotted parameters')

    if reference is not None:
        if len(reference) != N_par:
            raise ValueError('number of reference values must match number of plotted parameters')
    # check that given plot style is valid, else default to a histogram
    if plot_style not in ['contour', 'histogram', 'scatter']:
        plot_style = 'histogram'
        warn("""'plot_style' must be set as either 'contour', 'histogram' or 'scatter'""")

    # by default, we suppress axis ticks if there are 6 parameters or more to keep things tidy
    if show_ticks is None:
        show_ticks = N_par < 6

    L = 200
    cmap = get_cmap(colormap)
    # find the darker of the two ends of the colormap, and use it for the marginal plots
    marginal_color = sorted([cmap(10), cmap(245)], key = lambda x : sum(x[:-1]))[0]

    # build axis arrays and determine limits for all variables
    axis_limits = []
    axis_arrays = []
    for sample in samples:
        # get the 98% HDI to calculate plot limits
        lwr, upr = sample_hdi(sample, fraction = 0.98)
        # store the limits and axis array
        axis_limits.append([lwr-(upr-lwr)*0.3, upr+(upr-lwr)*0.3])
        axis_arrays.append(linspace(lwr-(upr-lwr)*0.35, upr+(upr-lwr)*0.35, L))

    fig = plt.figure(figsize = (8,8))
    # build a lower-triangular indices list in diagonal-striped order
    inds_list = [(N_par-1, 0)]  # start with bottom-left corner
    for k in range(1, N_par):
        inds_list.extend([(N_par-1-i, k-i) for i in range(k+1)])

    # now create a dictionary of axis objects with correct sharing
    axes = {}
    for tup in inds_list:
        i, j = tup
        x_share = None
        y_share = None

        if (i < N_par - 1):
            x_share = axes[(N_par - 1, j)]

        if (j > 0) and (i != j):  # diagonal doesnt share y-axis
            y_share = axes[(i, 0)]

        axes[tup] = plt.subplot2grid((N_par, N_par), (i, j), sharex = x_share, sharey = y_share)

    # now loop over grid and plot
    for tup in inds_list:
        i, j = tup
        ax = axes[tup]
        # are we on the diagonal?
        if i == j:
            sample = samples[i]
            pdf = GaussianKDE(sample)
            estimate = array( pdf(axis_arrays[i]) )
            ax.plot(axis_arrays[i], 0.9*(estimate/estimate.max()), lw = 1, color = marginal_color)
            ax.fill_between(axis_arrays[i], 0.9*(estimate/estimate.max()), color = marginal_color, alpha = 0.1)
            if reference is not None:
                ax.plot([reference[i], reference[i]], [0, 1], lw = 1.5, ls = 'dashed', color = 'red')
            ax.set_ylim([0, 1])
        else:
            x = samples[j]
            y = samples[i]

            # plot the 2D marginals
            if plot_style is 'contour':
                # Filled contour plotting using 2D gaussian KDE
                pdf = KDE2D(x = x, y = y)
                x_ax = axis_arrays[j][::4]
                y_ax = axis_arrays[i][::4]
                X, Y = meshgrid(x_ax, y_ax)
                prob = array(pdf(X.flatten(), Y.flatten())).reshape([L//4, L//4])
                ax.set_facecolor(cmap(256//20))
                ax.contourf(X, Y, prob, 10, cmap = cmap)
            elif plot_style is 'histogram':
                # hexagonal-bin histogram
                ax.set_facecolor(cmap(0))
                ax.hexbin(x, y, gridsize = 35, cmap = cmap)
            else:
                # scatterplot
                if point_colors is None:
                    ax.scatter(x, y, color = marginal_color, s=point_size)
                else:
                    ax.scatter(x, y, c = point_colors, s=point_size, cmap = cmap)

            # plot any reference points if given
            if reference is not None:
                ax.plot(reference[j], reference[i], marker = 'o', markersize = 7,
                        markerfacecolor = 'none', markeredgecolor = 'white', markeredgewidth = 3.5)
                ax.plot(reference[j], reference[i], marker = 'o', markersize = 7,
                        markerfacecolor = 'none', markeredgecolor = 'red', markeredgewidth = 2)

        # assign axis labels
        if i == N_par - 1: ax.set_xlabel(labels[j], fontsize = label_size)
        if j == 0 and i != 0: ax.set_ylabel(labels[i], fontsize = label_size)
        # impose x-limits on bottom row
        if i == N_par - 1: ax.set_xlim(axis_limits[j])
        # impose y-limits on left column, except the top-left corner
        if j == 0 and i != 0: ax.set_ylim(axis_limits[i])

        if show_ticks: # set up ticks for the edge plots if they are to be shown
            # hide x-tick labels for plots not on the bottom row
            if (i < N_par - 1): plt.setp(ax.get_xticklabels(), visible=False)
            # hide y-tick labels for plots not in the left column
            if j > 0: plt.setp(ax.get_yticklabels(), visible = False)
            # remove all y-ticks for 1D marginal plots on the diagonal
            if i == j: ax.set_yticks([])
        else: # else remove all ticks from all axes
            ax.set_xticks([])
            ax.set_yticks([])

    # set the plot spacing
    fig.tight_layout()
    fig.subplots_adjust(wspace = 0., hspace = 0.)
    # save/show the figure if required
    if filename is not None: plt.savefig(filename)
    if show:
        plt.show()
    else:
        fig.clear()
        plt.close(fig)




def trace_plot(samples, labels = None, show = True, filename = None):
    """
    Construct a 'trace plot' for a set of variables which displays the
    value of the variables as a function of step number in the chain.

    :param samples: \
        A list of array-like objects containing the samples for each variable.

    :keyword labels: \
        A list of strings to be used as axis labels for each parameter being plotted.

    :keyword bool show: \
        Sets whether the plot is displayed.

    :keyword str filename: \
        File path to which the matrix plot will be saved (if specified).
    """
    N_par = len(samples)
    if labels is None:
        if N_par >= 10:
            labels = ['p' + str(i) for i in range(N_par)]
        else:
            labels = ['param ' + str(i) for i in range(N_par)]
    else:
        if len(labels) != N_par:
            raise ValueError('number of labels must match the number of plotted parameters')

    # if for 'n' columns we allow up to m = 2*n rows, set 'n' to be as small as possible
    # given the number of parameters.
    n = int(ceil(sqrt(0.5*N_par)))
    # now given fixed n, make m as small as we can
    m = int(ceil(float(N_par) / float(n)))

    fig = plt.figure(figsize=(12,8))
    grid_inds = product(range(m),range(n))
    colors = cycle(['C0', 'C1', 'C2', 'C3', 'C4'])
    axes = {}
    for s, label, coords, col in zip(samples, labels, grid_inds, colors):
        i,j = coords
        if i==0 and j==0:
            axes[(i,j)] = plt.subplot2grid((m, n), (i, j))
        else:
            axes[(i,j)] = plt.subplot2grid((m, n), (i, j), sharex = axes[(0,0)])

        axes[(i,j)].plot(s, '.', markersize = 4, alpha = 0.15, c = col)
        axes[(i,j)].set_ylabel(label)
        # get the 98% HDI to calculate plot limits, and 10% HDI to estimate the mode
        lwr, upr = sample_hdi(s, fraction = 0.99)
        mid = 0.5 * sum(sample_hdi(s, fraction=0.10))
        axes[(i,j)].set_ylim([lwr-(mid-lwr)*0.7, upr+(upr-mid)*0.7])
        # get the 10% HDI to estimate the mode
        axes[(i,j)].set_yticks([lwr-(mid-lwr)*0.5, mid, upr+(upr-mid)*0.5])
        if (i < m-1):
            plt.setp(axes[(i,j)].get_xticklabels(), visible=False)
        else:
            axes[(i,j)].set_xlabel('chain step #')
    fig.tight_layout()
    if filename is not None: plt.savefig(filename)
    if show:
        plt.show()
    else:
        fig.clear()
        plt.close(fig)




def hdi_plot(x, sample, intervals=(0.35, 0.65, 0.95), colormap='Blues', axis=None, label_intervals=True):
    """
    Plot highest-density intervals for a given sample of model realisations.

    :param x: \
        The x-axis locations of the sample data.

    :param samples: \
        A ``numpy.ndarray`` containing the sample data, which has shape ``(n, len(x))`` where
        ``n`` is the number of samples.

    :keyword intervals: \
        A tuple containing the fractions of the total probability for each interval.

    :keyword colormap: \
        The colormap to be used for plotting the intervals. Must be a vaild argument
        of the ``matplotlib.cm.get_cmap`` function.

    :keyword axis: \
        A ``matplotlib.pyplot`` axis object which will be used to plot the intervals.

    :keyword bool label_intervals: \
        If ``True``, then labels will be assigned to each interval plot such that they appear
        in the legend when using ``matplotlib.pyplot.legend``.
    """
    # order the intervals from highest to lowest
    intervals = array(intervals)
    intervals.sort()
    intervals = intervals[::-1]

    # check that all the intervals are valid:
    if not all( (intervals > 0.) & (intervals < 1.) ):
        raise ValueError('All intervals must be greater than 0 and less than 1')

    # check the sample data has compatible dimensions
    s = array(sample)
    if s.shape[1] != len(x):
        if s.shape[0] == len(x):
            s = s.T
        else:
            raise ValueError('"x" and "sample" have incompatible dimensions')

    # sort the sample data
    s.sort(axis=0)
    n = s.shape[0]

    # construct the colors for each interval
    cmap = get_cmap(colormap)
    lwr = 0.2
    upr = 0.8
    colors = 1 - intervals
    colors += lwr - colors.min()
    colors *= upr/colors.max()
    colors = [cmap(int(255*c)) for c in colors]

    # if not plotting axis is given, then use default pyplot
    if axis is None: axis = plt

    from numpy import take_along_axis, expand_dims

    # iterate over the intervals and plot each
    for frac, col in zip(intervals, colors):
        L = int(frac * n)

        # check that we have enough samples to estimate the HDI for the chosen fraction
        if n > L:
            # find the optimal single HDI
            widths = s[L:,:] - s[:n-L,:]
            i = expand_dims(widths.argmin(axis=0), axis=0)
            lwr = take_along_axis(s,i,0).squeeze()
            upr = take_along_axis(s,i+L,0).squeeze()
        else:
            lwr = s[0,:]
            upr = s[-1,:]

        if label_intervals:
            axis.fill_between(x, lwr, upr, color=col, label = '{}% HDI'.format(int(100*frac)))
        else:
            axis.fill_between(x, lwr, upr, color=col)
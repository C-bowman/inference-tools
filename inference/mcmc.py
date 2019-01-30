
"""
.. moduleauthor:: Chris Bowman <chris.bowman@york.ac.uk>
"""

import sys
from warnings import warn
from copy import copy, deepcopy
from multiprocessing import Pool
from time import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import array, zeros, linspace, save

from numpy import exp, log, mean, sqrt, argmax, diff, dot
from numpy import meshgrid, isfinite, sort, argsort
from numpy.random import normal, random

from inference.pdf_tools import UnimodalPdf, GaussianKDE, KDE2D






class Parameter(object):
    """
    This class is used by the markov-chain samplers in this module
    to manage data specific to each model parameter which is being
    sampled.

    The class also adjusts the parameter's proposal distribution
    width automatically as the chain advances in order to ensure
    efficient sampling.
    """
    def __init__(self, value = None, sigma = None):
        self.samples = []  # list to store all samples for the parameter
        self.samples.append(value)  # add starting location as first sample
        self.sigma = sigma  # the width parameter for the proposal distribution

        # storage for proposal width adjustment algorithm
        self.avg = 0
        self.var = 0
        self.num = 0
        self.sigma_values = [copy(self.sigma)]  # sigma values after each assessment
        self.sigma_checks = [0.]  # chain locations at which sigma was assessed
        self.try_count = 0  # counter variable tracking number of proposals
        self.last_update = 0  # chain location where sigma was last updated

        # settings for proposal width adjustment algorithm
        self.target_rate = 0.25  # default of 0.25 is optimal for MH sampling
        self.max_tries = 50  # maximum allowed tries before width is cut in half
        self.chk_int = 100  # interval of steps at which proposal widths are adjusted
        self.growth_factor = 1.75  # factor by which self.chk_int grows when sigma is modified
        self.adjust_rate = 0.25

        # properties
        self._non_negative = False
        self.proposal = self.standard_proposal
        self.upper = None
        self.lower = None
        self.width = None

    def set_boundaries(self, lower, upper):
        if lower < upper:
            self.upper = upper
            self.lower = lower
            self.width = (upper - lower)
            self.proposal = self.boundary_proposal
        else:
            warn('Upper limit must be greater than lower limit')

    def remove_boundaries(self):
        self.proposal = self.standard_proposal
        self.upper = None
        self.lower = None
        self.width = None

    @property
    def non_negative(self):
        return self._non_negative

    @non_negative.setter
    def non_negative(self, value):
        if type(value) is bool:
            self._non_negative = value
            if self._non_negative is True:
                self.proposal = self.abs_proposal
            else:
                self.proposal = self.standard_proposal
        else:
            warn('non_negative must have a boolean value')

    def standard_proposal(self):
        # increment the try count
        self.try_count += 1
        # if tries climb too high, then cut sigma in half
        if self.try_count > self.max_tries: self.adjust_sigma(0.25)
        # return the proposed value
        return self.samples[-1] + self.sigma * normal()

    def abs_proposal(self):
        # increment the try count
        self.try_count += 1
        # if tries climb too high, then cut sigma in half
        if self.try_count > self.max_tries: self.adjust_sigma(0.25)
        # return the proposed value
        return abs(self.samples[-1] + self.sigma * normal())

    def boundary_proposal(self):
        # increment the try count
        self.try_count += 1
        # if tries climb too high, then cut sigma in half
        if self.try_count > self.max_tries: self.adjust_sigma(0.25)
        # generate the proposed value
        prop = self.samples[-1] + self.sigma * normal()

        # we now pass the proposal through a 'reflecting' function where
        # proposals falling outside the boundary are reflected inside
        d = prop - self.lower
        n = (d // self.width) % 2
        if n == 0:
            return self.lower + d % self.width
        else:
            return self.upper - d % self.width

    def submit_accept_prob(self, p):
        self.num += 1
        self.avg += p
        self.var += p*(1-p)

        if self.num >= self.chk_int:
            self.update_epsilon()

    def update_epsilon(self):
        """
        looks at average tries over recent steps, and adjusts proposal
        widths self.sigma to bring the average towards self.target_tries.
        """
        # normal approximation of poisson binomial distribution
        mu = self.avg / self.num
        std = sqrt(self.var) / self.num

        # now check if the desired success rate is within 2-sigma
        if ~(mu-2*std < self.target_rate < mu+2*std):
            adj = (log(self.target_rate) / log(mu))**(self.adjust_rate)
            adj = min(adj,3.)
            adj = max(adj,0.1)
            self.adjust_sigma(adj)
        else: # increase the check interval
            self.chk_int = int((self.growth_factor * self.chk_int) * 0.1) * 10

    def adjust_sigma(self, ratio):
        self.sigma *= ratio
        self.sigma_values.append(copy(self.sigma))
        # self.sigma_checks.append(self.sigma_checks[-1] + self.num)
        self.sigma_checks.append(len(self.samples))
        self.avg = 0
        self.var = 0
        self.num = 0

    def add_sample(self, s):
        self.samples.append(s)
        self.try_count = 0






class MarkovChain(object):
    """
    Implementation of the metropolis-hastings algorithm using a multivariate-normal proposal distribution.

    :param func posterior: a function which returns the log-posterior probability density for a \
                           given set of model parameters theta, which should be the only argument \
                           so that: ln(P) = posterior(theta)

    :param start: vector of model parameters which correspond to the parameter-space coordinates \
                  at which the chain will start.

    :param widths: vector of standard deviations which serve as initial guesses for the widths of \
                   the proposal distribution for each model parameter. If not specified, the starting \
                   widths will be approximated as 1% of the values in 'start'.
    """
    def __init__(self, posterior = None, start = None, widths = None):

        if start is None:
            start = []

        if posterior is not None:
            self.posterior = posterior

            # if widths are not specified, take 1% of the starting values:
            if widths is None:
                widths = [ abs(i)*0.05 for i in start ]

            # create a list of parameter objects
            self.params = [Parameter(value = v, sigma = s) for v, s in zip(start, widths)]

            # create storage
            self.n = 1  # tracks total length of the chain
            self.L = len(start)  # number of posterior parameters
            self.probs = list()  # list of probabilities for all steps

            # add starting point as first step in chain
            self.probs.append(self.posterior([p.samples[-1] for p in self.params]))

            # check posterior value of chain starting point is finite
            if not isfinite(self.probs[0]):
                ValueError('posterior returns a non-finite value for provided initial guess')

            # add default burn and thin values
            self.burn = 1 # remove the starting position by default
            self.thin = 1 # no thinning by default

            # flag for displaying completion of the advance() method
            self.print_status = True

    def take_step(self):
        """
        Draws samples from the proposal distribution until one is
        found which satisfies the metropolis-hastings criteria.
        """
        while True:
            proposal = [p.proposal() for p in self.params]
            pval = self.posterior(proposal)

            if pval > self.probs[-1]:
                break
            else:
                test = random()
                acceptance_prob = exp(pval-self.probs[-1])
                if test < acceptance_prob:
                    break

        for p, v in zip(self.params, proposal):
            p.add_sample(v)

        self.n += 1

    def advance(self, m):
        """
        Advances the chain by taking *m* new steps.

        :param int m: number of steps the chain will advance.
        """
        k = 100  # divide chain steps into k groups to track progress
        t_start = time()
        for j in range(k):
            for i in range(m//k):
                self.take_step()
            dt = time() - t_start

            # display the progress status message
            if self.print_status:
                pct = str(int(100*(j+1)/k))
                eta = str(int(dt*((k/(j+1)-1))))
                msg = '\r  advancing chain:   [ {}% complete   ETA: {} sec ]'.format(pct, eta)
                sys.stdout.write(msg)
                sys.stdout.flush()

        # cleanup
        if m % k != 0:
            for i in range(m % k):
                self.take_step()

        # this is a little ugly...
        sys.stdout.write('\r  advancing chain:   [ complete ]                         ')
        sys.stdout.flush()
        sys.stdout.write('\n')

    def run_for(self, minutes = 0, hours = 0, days = 0):
        """
        Advances the chain for a chosen amount of computation time

        :param int minutes: number of minutes for which to run the chain.
        :param int hours: number of hours for which to run the chain.
        :param int days: number of days for which to run the chain.
        """
        # first find the runtime in seconds:
        run_time = ((days*24. + hours)*60. + minutes)*60.
        start_time = time()
        end_time = start_time + run_time

        # get a rough estimate of the time per step
        step_time = time()
        __ = self.posterior([p.samples[-1] for p in self.params])
        step_time = time() - step_time
        step_time *= 2*self.L

        # choose an update interval that should take ~2 seconds
        update_interval = max( int(2. // step_time), 1)

        # store the starting length of the chain
        start_length = copy(self.n)

        while time() < end_time:
            for i in range(update_interval):
                self.take_step()


            # display the progress status message
            seconds_remaining = end_time - time()
            m, s = divmod(seconds_remaining, 60)
            h, m = divmod(m, 60)
            time_left = "%d:%02d:%02d" % (h, m, s)
            steps_taken = self.n - start_length
            msg = '\r  advancing chain:   [ {} steps taken, time remaining: {} ]'.format(steps_taken, time_left)
            sys.stdout.write(msg)
            sys.stdout.flush()

        # this is a little ugly...
        sys.stdout.write('\r  advancing chain:   [ complete ]                         ')
        sys.stdout.flush()
        sys.stdout.write('\n')

    def dump(self, thin = None, burn = None):
        """
        returns the theta values and associated log-probabilities
        for every step in the chain in a single numpy array
        for easier storage / analysis.
        """
        if burn is None: burn = self.burn
        if thin is None: thin = self.thin
        data = zeros([len(self.probs[burn::thin]), self.L+1])

        for i in range(self.L):
            data[:,i] = self.get_parameter(i, burn=burn, thin=thin)
        data[:,-1] = array(self.probs[burn::thin])
        return data

    def save(self, filepath, burn = None, thin = None):
        if burn is None: burn = self.burn
        if thin is None: thin = self.thin
        dat = self.dump(burn=burn, thin=thin)
        save(filepath, dat)

    def get_parameter(self, n, burn = None, thin = None):
        """
        Return sample values for a chosen parameter.

        :param int n: Index of the parameter for which samples are to be returned.

        :param int burn: Number of samples to discard from the start of the chain. If not \
                         specified, the value of self.burn is used instead.

        :param int thin: Instead of returning every sample which is not discarded as part \
                         of the burn-in, every *m*'th sample is returned for a specified \
                         integer *m*. If not specified, the value of self.thin is used instead.

        :return: List of samples for parameter *n*'th parameter.
        """
        if burn is None: burn = self.burn
        if thin is None: thin = self.thin
        return self.params[n].samples[burn::thin]

    def get_probabilities(self, burn = None, thin = None):
        """
        Return log-probability values for each step in the chain

        :param int burn: Number of steps to discard from the start of the chain. If not \
                         specified, the value of self.burn is used instead.

        :param int thin: Instead of returning every step which is not discarded as part \
                         of the burn-in, every *m*'th step is returned for a specified \
                         integer *m*. If not specified, the value of self.thin is used instead.

        :return: List of log-probability values for each step in the chain.
        """
        if burn is None: burn = self.burn
        if thin is None: thin = self.thin
        return self.probs[burn::thin]

    def get_sample(self, burn = None, thin = None):
        """
        Return the sample generated by the chain as a list of tuples

        :param int burn: Number of samples to discard from the start of the chain. If not \
                         specified, the value of self.burn is used instead.

        :param int thin: Instead of returning every sample which is not discarded as part \
                         of the burn-in, every *m*'th sample is returned for a specified \
                         integer *m*. If not specified, the value of self.thin is used instead.

        :return: List containing sample points stored as tuples.
        """
        if burn is None: burn = self.burn
        if thin is None: thin = self.thin
        return list(zip( *[ p.samples[burn::thin] for p in self.params] ))

    def get_interval(self, interval = None, burn = None, thin = None, samples = None):
        """
        Return the samples from the chain which lie inside a chosen highest-density interval.

        :param float interval: Total probability of the desired interval. For example, if \
                               interval = 0.95, then the samples corresponding to the top \
                               95% of posterior probability values are returned.

        :param int burn: Number of samples to discard from the start of the chain. If not \
                         specified, the value of self.burn is used instead.

        :param int thin: Instead of returning every sample which is not discarded as part \
                         of the burn-in, every *m*'th sample is returned for a specified \
                         integer *m*. If not specified, the value of self.thin is used instead.

        :param int samples: The number of samples that should be returned from the requested \
                            interval. Note that specifying *samples* overrides the value of *thin*.

        :return: List containing sample points stored as tuples, and a corresponding list of \
                 log-probability values
        """
        if burn is None: burn = self.burn
        if interval is None: interval = 0.95

        # get the sorting indices for the probabilities
        probs = array(self.probs[burn:])
        inds = probs.argsort()
        # sort the sample by probability
        arrays = [array(p.samples[burn:])[inds] for p in self.params]
        probs = probs[inds]
        # trim lowest-probability samples
        cutoff = int(len(probs) * (1 - interval))
        arrays = [a[cutoff:] for a in arrays]
        probs = probs[cutoff:]
        # if a specific number of samples is requested we override the thin value
        if samples is not None:
            thin = max(len(probs) // samples, 1)
        elif thin is None: thin = self.thin

        # thin the sample
        arrays = [a[::thin] for a in arrays]
        probs = probs[::thin]

        if samples is not None:
            # we may need to trim some extra samples to meet the requested number,
            # but as they arranged in order of increasing probability, we must remove
            # elements at random in order not to introduce bias.
            n_trim = len(probs) - samples
            if n_trim > 0:
                trim = sort( argsort( random(size=len(probs)) )[n_trim:] )
                arrays = [a[trim] for a in arrays]
                probs = probs[trim]

        return list(zip( *arrays )), probs

    def mode(self):
        """
        Return the sample with the current highest posterior probability.

        :return: Tuple containing parameter values.
        """
        ind = argmax(self.probs)
        return [p.samples[ind] for p in self.params]

    def set_non_negative(self, parameter, flag = True):
        """
        Constrain a particular parameter to have non-negative values.

        :param int parameter: Index of the parameter which is to be set \
                              as non-negative.
        """
        self.params[parameter].non_negative = flag

    def set_boundaries(self, parameter, boundaries, remove = False):
        """
        Constrain the value of a particular parameter to specified boundaries.

        :param int parameter: Index of the parameter for which boundaries \
                              are to be set.

        :param boundaries: Tuple of boundaries in the format (lower_limit, upper_limit)
        """
        if remove:
            self.params[parameter].remove_boundaries()
        else:
            self.params[parameter].set_boundaries(*boundaries)

    def marginalise(self, n, thin = None, burn = None, unimodal = False):
        """
        Estimate the 1D marginal distribution of a chosen parameter.

        :param int n: Index of the parameter for which the marginal distribution is to be estimated.

        :param int burn: Number of samples to discard from the start of the chain. If not \
                         specified, the value of self.burn is used instead.

        :param int thin: Rather than using every sample which is not discarded as part \
                         of the burn-in, every *m*'th sample is used for a specified \
                         integer *m*. If not specified, the value of self.thin is used \
                         instead, which has a default value of 1.

        :param bool unimodal: Selects the type of density estimation to be used. The default value \
                              is False, which causes a GaussianKDE object to be returned. If however \
                              the marginal distribution being estimated is known to be unimodal, \
                              setting `unimodal = True` will result in the UnimodalPdf class being \
                              used to estimate the density.

        Returns one of two 'density estimator' objects which can be
        called as functions to return the estimated PDF at any point.
        """
        if burn is None: burn = self.burn
        if thin is None: thin = self.thin

        if unimodal:
            return UnimodalPdf(self.get_parameter(n, burn=burn, thin=thin))
        else:
            return GaussianKDE(self.get_parameter(n, burn=burn, thin=thin))

    def plot_diagnostics(self, show = True, filename = None):
        """
        Plot diagnostic traces that give information on how the chain is progressing.

        Currently this method plots:

        - The posterior log-probability as a function of step number, which is useful
          for checking if the chain has reached a maximum. Any early parts of the chain
          where the probability is rising rapidly should be removed as burn-in.

        - The history of changes to the proposal widths for each parameter. Ideally, the
          proposal widths should converge, and the point in the chain where this occurs
          is often a good choice for the end of the burn-in. For highly-correlated pdfs,
          the proposal widths may never fully converge, but in these cases small fluctuations
          in the width values are acceptable.

        :param bool show: If set to True, the plot is displayed.

        :param str filename: File path to which the diagnostics plot will be saved. If left \
                             unspecified the plot won't be saved.
        """
        fig = plt.figure(figsize = (12,5))

        # probability history plot
        ax1 = fig.add_subplot(121)
        step_ax = [i * 1e-3 for i in range(len(self.probs))]  # TODO - avoid making this axis but preserve figure form
        ax1.plot(step_ax, self.probs, marker = '.', ls = 'none', markersize = 3)
        ax1.set_xlabel('chain step number ($10^3$)', fontsize = 12)
        ax1.set_ylabel('log posterior probability', fontsize = 12)
        ax1.set_title('Chain log-probability history')
        ax1.set_ylim([min(self.probs[self.n//2:]), max(self.probs)*1.1 - 0.1*min(self.probs[self.n//2:])])
        ax1.grid()

        # proposal widths plot
        ax2 = fig.add_subplot(122)
        for p in self.params:
            y = array(p.sigma_values)
            x = array(p.sigma_checks[1:]) * 1e-3
            ax2.plot(x, 1e2*diff(y)/y[:-1], marker = 'D', markersize = 3)
        ax2.plot([0, self.n*1e-3], [5, 5], ls = 'dashed', lw = 2, color = 'black')
        ax2.plot([0, self.n*1e-3], [-5,-5], ls = 'dashed', lw = 2, color = 'black')
        ax2.set_xlabel('chain step number ($10^3$)', fontsize = 12)
        ax2.set_ylabel('% change in proposal widths', fontsize = 12)
        ax2.set_title('Parameter proposal widths adjustment summary')
        ax2.set_ylim([-50,50])
        ax2.grid()

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()
        else:
            fig.clear()
            plt.close(fig)

    def matrix_plot(self, params = None, thin = None, burn = None, labels = None, show = True, reference = None, filename = None):
        """
        Construct a 'matrix plot' of the parameters (or a subset) which displays
        all 1D and 2D marginal distributions.

        :param params: A list of integers specifying the indices of parameters which \
                       are to be plotted.

        :param int burn: Number of samples to discard from the start of the chain. If not \
                         specified, the value of self.burn is used instead.

        :param int thin: Rather than using every sample which is not discarded as part \
                         of the burn-in, every *m*'th sample is used for a specified \
                         integer *m*. If not specified, the value of self.thin is used \
                         instead, which has a default value of 1.

        :param labels: A list or tuple of strings to be used axis labels for each parameter \
                       being plotted.

        :param bool show: If set to True, the plot is displayed.

        :param reference: A list of reference values for each parameter which \
                          will be over-plotted.

        :param str filename: File path to which the matrix plot will be saved. If unspecified \
                             the plot will be displayed but not saved.
        """
        if burn is None: burn = self.burn
        if thin is None: thin = self.thin

        # TODO - find a way to do this without changing environment variables
        mpl.rcParams['axes.xmargin'] = 0
        mpl.rcParams['axes.ymargin'] = 0

        if params is None:
            p = [ i for i in range(self.L) ]
        else:
            p = params

        if labels is None:
            labels = [ 'parameter ' + str(i) for i in p ]
        else:
            if len(labels) != len(p):
                raise ValueError('number of labels must match number of plotted parameters')

        if reference is not None:
            if len(reference) != len(p):
                raise ValueError('number of reference values must match number of plotted parameters')

        n = len(p)
        L = 200

        # Determine axis ranges for plotting
        axlims = []
        lins = []
        sample_data = []
        for i in p:
            A = array(self.get_parameter(i, burn=burn, thin=thin))

            mu = mean(A)
            sg = sqrt( mean(A*A) - mu**2 )
            axlims.append((mu-3.5*sg,mu+3.5*sg))
            lins.append(linspace(mu-4*sg, mu+4*sg, L))
            sample_data.append(A)

        fig = plt.figure( figsize = (8,8) )

        # lower-triangular indices list in diagonal-striped order
        inds_list = [(n-1, 0)] # start with bottom-left corner
        for k in range(1,n):
            inds_list.extend([ (n-1-i, k-i) for i in range(k+1) ])

        # now create a dictionary of axis objects with correct sharing
        axes = {}
        for tup in inds_list:
            i, j = tup
            x_share = None
            y_share = None

            if (i < n-1):
                x_share = axes[(n-1,j)]

            if (j > 0) and (i != j): # diagonal doesnt share y-axis
                y_share = axes[(i,0)]

            axes[tup] = plt.subplot2grid((n, n), (i, j), sharex = x_share, sharey = y_share)

        # now loop over grid and plot
        for tup in inds_list:
            i, j = tup
            ax = axes[tup]
            # are we on the diagonal?
            if i==j:
                sample = sample_data[i]
                pdf = GaussianKDE(sample)
                ax.plot(lins[i], 0.95*(pdf(lins[i])/pdf(pdf.mode)), lw = 2, color = 'C0')
                if reference is not None:
                    ax.plot([reference[i], reference[i]], [0,1], lw = 1.5, ls = 'dashed', color = 'red')
                ax.set_ylim([0,1])
            else:
                x = sample_data[j]
                y = sample_data[i]

                pdf = KDE2D(x=x, y=y)
                x_ax = lins[j][::4]
                y_ax = lins[i][::4]
                X, Y = meshgrid(x_ax, y_ax)
                prob = array(pdf(X.flatten(), Y.flatten())).reshape([L//4, L//4])
                ax.contour(X, Y, prob, 15)
                if reference is not None:
                    ax.plot(reference[j], reference[i], marker = 'o', markersize = 7,
                            markerfacecolor = 'none', markeredgecolor = 'white', markeredgewidth = 3.5)
                    ax.plot(reference[j], reference[i], marker = 'o', markersize = 7,
                            markerfacecolor = 'none', markeredgecolor = 'red', markeredgewidth = 2)

            if i == n-1:
                ax.set_xlim(axlims[j])
                ax.set_xlabel(labels[j])
            else: # not on the bottom row
                plt.setp(ax.get_xticklabels(), visible = False)

            if j == 0 and i != 0:
                ax.set_ylim(axlims[i])
                ax.set_ylabel(labels[i])
            else: # not on left column
                plt.setp(ax.get_yticklabels(), visible = False)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()
        else:
            fig.clear()
            plt.close(fig)






class GibbsChain(MarkovChain):
    """
    A Gibbs sampling class implemented as a child of the MarkovChain class.

    :param func posterior: a function which returns the log-posterior probability density for a \
                           given set of model parameters theta, which should be the only argument \
                           so that: ln(P) = posterior(theta)

    :param start: vector of model parameters which correspond to the parameter-space coordinates \
                  at which the chain will start.

    :param widths: vector of standard deviations which serve as initial guesses for the widths of \
                   the proposal distribution for each model parameter. If not specified, the starting \
                   widths will be approximated as 1% of the values in 'start'.

    In Gibbs sampling, each "step" in the chain consists of a series of 1D Metropolis-Hastings
    steps, one for each parameter, such that each step all parameters have been adjusted.

    This allows 1D step acceptance rate data to be collected independently for each parameter,
    thereby allowing the proposal width of each parameter to be tuned individually.
    """
    def __init__(self, *args, **kwargs):
        super(GibbsChain, self).__init__(*args, **kwargs)
        # we need to adjust the target acceptance rate to 50%
        # which is optimal for gibbs sampling:
        for p in self.params:
            p.target_rate = 0.5

    def take_step(self):
        """
        Take a 1D metropolis-hastings step for each parameter
        """
        p_old = self.probs[-1]
        prop = [p.samples[-1] for p in self.params]

        for i, p in enumerate(self.params):

            while True:
                prop[i] = p.proposal()
                p_new = self.posterior(prop)

                if p_new > p_old:
                    p.submit_accept_prob(1.)
                    break
                else:
                    test = random()
                    acceptance_prob = exp(p_new-p_old)
                    p.submit_accept_prob(acceptance_prob)
                    if test < acceptance_prob:
                        break

            p_old = deepcopy(p_new)  # NOTE - is deepcopy needed?

        for v, p in zip(prop, self.params):
            p.add_sample(v)

        self.probs.append(p_new)
        self.n += 1






class HamiltonianChain(MarkovChain):
    """
    Hamiltonian Monte-Carlo implemented as a child of the MarkovChain class.

    :param func posterior: \
        A function which returns the log-posterior probability density for a \
        given set of model parameters theta, which should be the only argument \
        so that: ln(P) = posterior(theta)

    :param func grad: \
        A function which returns the gradient of the log-posterior probability density \
        for a given set of model parameters theta. If this function is not given, the \
        gradient will instead be estimated by finite difference.

    :param start: \
        Vector of model parameters which correspond to the parameter-space coordinates \
        at which the chain will start.

    :param float epsilon: \
        Initial guess for the time-step of the Hamiltonian dynamics simulation.

    :param int temperature: \
        The temperature of the markov chain. This parameter is used for parallel \
        tempering and should be otherwise left unspecified.

    :param bounds: \
        A list or tuple containing two numpy arrays which specify the upper and lower \
        bounds for the parameters in the form (lower_bounds, upper_bounds).

    :param inv_mass: \
        A vector specifying the inverse-mass value to be used for each parameter. The \
        inverse-mass effectively re-scales the parameters to make the problem more isotropic, \
        which helps ensure good performance. The inverse-mass value for a given parameter \
        should be set to roughly the range over which that parameter is expected to vary.
    """
    def __init__(self, posterior = None, grad = None, start = None, epsilon = 0.1, temperature = 1, bounds = None, inv_mass = None):

        self.posterior = posterior

        if grad is None:
            self.grad = self.finite_diff
        else:
            self.grad = grad

        if bounds is None:
            self.leapfrog = self.standard_leapfrog
        else:
            self.leapfrog = self.bounded_leapfrog
            self.lwr_bounds = bounds[0]
            self.upr_bounds = bounds[1]
            if any((self.lwr_bounds > array(start)) | (self.upr_bounds < array(start))):
                raise ValueError('starting location for the chain is outside specified bounds')
            self.widths = self.upr_bounds - self.lwr_bounds
            if not all(self.widths > 0):
                raise ValueError('specified upper bounds must be greater than lower bounds')

        if inv_mass is None:
            self.inv_mass = 1
        else:
            self.inv_mass = inv_mass

        self.T = temperature
        self.theta = [start]
        self.probs = [self.posterior(start)/self.T]
        self.L = len(start)
        self.n = 1

        self.ES = EpsilonSelector(epsilon)
        self.steps = 50

        self.burn = 1
        self.thin = 1

        self.print_status = True

    def take_step(self):
        """
        Takes the next step in the HMC-chain
        """
        accept = False
        while not accept:
            r0 = normal(size = self.L)
            t0 = self.theta[-1]
            H0 = 0.5*dot(r0,r0) - self.probs[-1]

            r = copy(r0)
            t = copy(t0)
            self.g = self.grad(t) / self.T
            n_steps = int(self.steps * (1+(random()+0.5)*0.2))
            for i in range(n_steps):
                t, r = self.leapfrog(t, r)

            p = self.posterior(t) / self.T
            H = 0.5*dot(r,r) - p
            test = exp( H0 - H )

            if isfinite(test):
                self.ES.add_probability(min(test,1))
            else:
                self.ES.add_probability(0.)

            if (test >= 1):
                accept = True
            else:
                q = random()
                if (q <= test):
                    accept = True

        self.theta.append( t )
        self.probs.append( p )
        self.n += 1

    def finite_diff(self, t):
        p = self.posterior(t) / self.T
        G = zeros(self.L)
        for i in range(self.L):
            delta = zeros(self.L)+1
            delta[i] += 1e-5
            G[i] = (self.posterior(t * delta) / self.T - p) / ( t[i] * 1e-5 )
        return G

    def standard_leapfrog(self, t, r):
        r2 = r + (0.5*self.ES.epsilon)*self.g
        t2 = t + self.ES.epsilon * r2 * self.inv_mass

        self.g = self.grad(t2) / self.T
        r3 = r2 + (0.5*self.ES.epsilon)*self.g
        return t2, r3

    def bounded_leapfrog(self, t, r):
        r2 = r + (0.5*self.ES.epsilon)*self.g
        t2 = t + self.ES.epsilon * r2 * self.inv_mass

        # check for values outside bounds
        lwr_bools = t2 < self.lwr_bounds
        upr_bools = t2 > self.upr_bounds

        # calculate necessary adjustment
        lwr_adjust = ( lwr_bools*(self.lwr_bounds-t2) ) % self.widths
        upr_adjust = ( upr_bools*(t2-self.upr_bounds) ) % self.widths
        t2 += 2*lwr_adjust
        t2 -= 2*upr_adjust

        # reverse momenta where necessary
        reflect = 1 - 2 * (lwr_bools | upr_bools)
        r2 *= reflect

        self.g = self.grad(t2) / self.T
        r3 = r2 + (0.5*self.ES.epsilon)*self.g
        return t2, r3

    def get_parameter(self, n, burn = None, thin = None):
        """
        Return sample values for a chosen parameter.

        :param int n: Index of the parameter for which samples are to be returned.

        :param int burn: Number of samples to discard from the start of the chain. If not \
                         specified, the value of self.burn is used instead.

        :param int thin: Instead of returning every sample which is not discarded as part \
                         of the burn-in, every *m*'th sample is returned for a specified \
                         integer *m*. If not specified, the value of self.thin is used instead.

        :return: List of samples for parameter *n*'th parameter.
        """
        if burn is None: burn = self.burn
        if thin is None: thin = self.thin
        return [v[n] for v in self.theta[burn::thin]]

    def plot_diagnostics(self, show = True, filename = None):
        """
        Plot diagnostic traces that give information on how the chain is progressing.

        Currently this method plots:

        - The posterior log-probability as a function of step number, which is useful
          for checking if the chain has reached a maximum. Any early parts of the chain
          where the probability is rising rapidly should be removed as burn-in.

        - The history of changes to the proposal widths for each parameter. Ideally, the
          proposal widths should converge, and the point in the chain where this occurs
          is often a good choice for the end of the burn-in. For highly-correlated pdfs,
          the proposal widths may never fully converge, but in these cases small fluctuations
          in the width values are acceptable.

        :param bool show: If set to True, the plot is displayed.

        :param str filename: File path to which the diagnostics plot will be saved. If left \
                             unspecified the plot won't be saved.
        """
        fig = plt.figure(figsize = (12,5))

        # probability history plot
        ax1 = fig.add_subplot(121)
        step_ax = [i * 1e-3 for i in range(len(self.probs))]  # TODO - avoid making this axis but preserve figure form
        ax1.plot(step_ax, self.probs, marker = '.', ls = 'none', markersize = 3)
        ax1.set_xlabel('chain step number ($10^3$)', fontsize = 12)
        ax1.set_ylabel('log posterior probability', fontsize = 12)
        ax1.set_title('Chain log-probability history')
        ax1.set_ylim([min(self.probs[self.n//2:]), max(self.probs)*1.1 - 0.1*min(self.probs[self.n//2:])])
        ax1.grid()

        # epsilon plot
        ax2 = fig.add_subplot(122)
        ax2.plot(array(self.ES.epsilon_checks)*1e-3, self.ES.epsilon_values, '.-')
        ax2.set_xlabel('chain step number ($10^3$)', fontsize = 12)
        ax2.set_ylabel('Leapfrog step-size', fontsize = 12)
        ax2.set_title('Simulation time-step adjustment summary')
        ax2.grid()

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()
        else:
            fig.clear()
            plt.close(fig)

    def get_sample(self, burn = None, thin = None):
        raise ValueError('This method is not yet implemented for HamiltonianChain')

    def get_interval(self, interval = None, burn = None, thin = None, samples = None):
        raise ValueError('This method is not yet implemented for HamiltonianChain')






class EpsilonSelector(object):
    def __init__(self, epsilon):

        # storage
        self.epsilon = epsilon
        self.epsilon_values = [copy(epsilon)]  # sigma values after each assessment
        self.epsilon_checks = [0.]  # chain locations at which sigma was assessed

        # tracking variables
        self.avg = 0
        self.var = 0
        self.num = 0

        # settings for epsilon adjustment algorithm
        self.accept_rate = 0.65
        self.chk_int = 50  # interval of steps at which proposal widths are adjusted
        self.growth_factor = 1.5  # factor by which self.chk_int grows when sigma is modified

    def add_probability(self, p):
        self.num += 1
        self.avg += p
        self.var += p*(1-p)

        if self.num >= self.chk_int:
            self.update_epsilon()

    def update_epsilon(self):
        """
        looks at average tries over recent steps, and adjusts proposal
        widths self.sigma to bring the average towards self.target_tries.
        """
        # normal approximation of poisson binomial distribution
        mu = self.avg / self.num
        std = sqrt(self.var) / self.num

        # now check if the desired success rate is within 2-sigma
        if ~(mu-2*std < self.accept_rate < mu+2*std):
            adj = (log(self.accept_rate) / log(mu))**(0.15)
            adj = min(adj,2.)
            adj = max(adj,0.2)
            # print(adj)
            self.adjust_epsilon(adj)
        else: # increase the check interval
            self.chk_int = int((self.growth_factor * self.chk_int) * 0.1) * 10

    def adjust_epsilon(self, ratio):
        self.epsilon *= ratio
        self.epsilon_values.append(copy(self.epsilon))
        self.epsilon_checks.append(self.epsilon_checks[-1] + self.num)
        self.avg = 0
        self.var = 0
        self.num = 0






class ChainPool(object):
    def __init__(self, objects):
        self.chains = objects
        self.pool_size = len(self.chains)
        self.pool = Pool(self.pool_size)

    def advance(self, n):
        self.chains = self.pool.map(self.adv_func, [(n, chain) for chain in self.chains] )

    @staticmethod
    def adv_func(arg):
        n, chain = arg
        chain.advance(n)
        return chain






class TemperedChain(MarkovChain):
    """
    TemperedChain is identical to GibbsChain, other than the fact that
    the log-posterior probability is divided by the chain temperature
    (stored as self.temp) after evaluation.
    """
    def __init__(self, temperature = 1., *args, **kwargs):
        super(TemperedChain, self).__init__(*args, **kwargs)
        # we need to adjust the target acceptance rate to 50%
        # which is optimal for gibbs sampling:
        for p in self.params:
            p.target_tries = 2

        self.T = temperature
        self.probs[0] /= temperature

    def take_step(self):
        """
        Take a 1D metropolis-hastings step for each parameter
        """
        p_old = self.probs[-1]
        prop = [p.samples[-1] for p in self.params]

        for i, p in enumerate(self.params):

            while True:
                prop[i] = p.proposal()
                p_new = self.posterior(prop)/self.T

                if p_new > p_old:
                    break
                else:
                    test = random()
                    if test < exp(p_new-p_old):
                        break

            p_old = deepcopy(p_new)  # NOTE - is deepcopy needed?

        for v, p in zip(prop, self.params):
            p.add_sample(v)

        self.probs.append(p_new)
        self.n += 1
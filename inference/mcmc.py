
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

from numpy import exp, mean, sqrt, argmax, diff
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
        self.num = 0
        self.sigma_values = [copy(self.sigma)]  # sigma values after each assessment
        self.sigma_checks = [0.]  # chain locations at which sigma was assessed
        self.try_count = 0  # counter variable tracking number of proposals
        self.last_update = 0  # chain location where sigma was last updated

        # settings for proposal width adjustment algorithm
        self.target_tries = 4  # default of 4 is optimal for MH sampling
        self.max_tries = 50  # maximum allowed tries before width is cut in half
        self.chk_int = 200  # interval of steps at which proposal widths are adjusted
        self.growth_factor = 1.4  # factor by which self.chk_int grows when sigma is modified

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

    def update_sigma(self):
        """
        looks at average tries over recent steps, and adjusts proposal
        widths self.sigma to bring the average towards self.target_tries.
        """

        # here we model the try-reject process as binomial, and based on this
        # derive an approximate normal form for the success probability posterior,
        # the mean and standard deviation of which are given by
        mu = (1 + self.num) / (2 + self.num*self.avg)
        std = sqrt( (1+self.num)*(1+self.num*(self.avg-1)) / ( (3 + self.num*self.avg)*(2 + self.num*self.avg)**2 ) )

        # now check if the desired success rate is within 2-sigma
        if ~(mu-2*std < 1./self.target_tries < mu+2*std):

            # calculate the adjustment factor for sigma
            if self.avg == 1.: # rare case where every step was taken first try
                adj = 10.
            else:
                adj = ((self.target_tries - 1) / (self.avg - 1)) ** (0.65)
            self.adjust_sigma(adj)

        # increase the check interval
        self.chk_int = int((self.growth_factor * self.chk_int) * 0.1) * 10

    def adjust_sigma(self, ratio):
        self.sigma *= ratio
        if (self.width is not None) and (self.sigma > 2*self.width):
            self.sigma = copy(2*self.width)
        self.sigma_values.append(copy(self.sigma))
        self.sigma_checks.append(self.sigma_checks[-1] + self.num)
        self.avg = 0
        self.num = 0

    def add_sample(self, s):
        self.samples.append(s)
        # update the running average
        self.avg = (self.avg*self.num + self.try_count) / (self.num + 1)
        self.num += 1
        self.try_count = 0

        if self.num >= self.chk_int:
            self.update_sigma()




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
                widths = [ abs(i)*0.01 for i in start ]

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
                if test < exp(pval-self.probs[-1]):
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

    def dump(self, thin = None, burn = None):
        """
        returns the theta values and associated log-probabilities
        for every step in the chain in a single numpy array
        for easier storage / analysis.
        """
        if burn is None: burn = self.burn
        if thin is None: thin = self.thin
        data = zeros([len(self.probs), self.L+1])

        for i, p in enumerate(self.params):
            data[:,i] = p.samples
        data[:,-1] = array(self.probs)

        return data[burn::thin]

    def save(self, filepath, burn = None, thin = None):
        if burn is None: burn = self.burn
        if thin is None: thin = self.thin
        dat = (self.dump())[burn::thin]
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
            return UnimodalPdf(self.params[n].samples[burn::thin])
        else:
            return GaussianKDE(self.params[n].samples[burn::thin])

    def plot_diagnostics(self):
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
        plt.show()

    def matrix_plot(self, params = None, thin = None, burn = None, labels = None, filename = None):
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

        :param str filename: File path to which the matrix plot will be saved. If unspecified \
                             the plot will be displayed but not saved.
        """
        if burn is None: burn = self.burn
        if thin is None: thin = self.thin

        # TODO - find a way to do this without changing environment variables
        mpl.rcParams['axes.xmargin'] = 0
        mpl.rcParams['axes.ymargin'] = 0

        if params is None:
            p = [ i for i in range(len(self.params)) ]
        else:
            p = params

        if labels is None:
            labels = [ 'parameter ' + str(i) for i in p ]
        else:
            if len(labels) != len(p):
                ValueError('number of labels must match number of plotted parameters')

        n = len(p)
        L = 200

        # Determine axis ranges for plotting
        axlims = []
        lins = []
        for i in p:
            A = array(self.params[i].samples[burn::thin])

            mu = mean(A)
            sg = sqrt( mean(A*A) - mu**2 )
            axlims.append((mu-3.5*sg,mu+3.5*sg))
            lins.append(linspace(mu-4*sg, mu+4*sg, L))

        fig = plt.figure( figsize = (8,8))

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
                sample = self.params[p[i]].samples[burn::thin]
                pdf = GaussianKDE(sample)
                ax.plot(lins[i], 0.95*(pdf(lins[i])/pdf(pdf.mode)), lw = 2, color = 'purple')
                ax.set_ylim([0,1])
            else:
                x = self.params[p[j]].samples[burn::thin]
                y = self.params[p[i]].samples[burn::thin]

                pdf = KDE2D(x=x, y=y)
                x_ax = lins[j][::4]
                y_ax = lins[i][::4]
                X, Y = meshgrid(x_ax, y_ax)
                prob = array(pdf(X.flatten(), Y.flatten())).reshape([L//4, L//4])
                ax.contour(X, Y, prob, 15)

            if i == n-1:
                ax.set_xlim(axlims[j])
                ax.set_xlabel(labels[p[j]])
            else: # not on the bottom row
                plt.setp(ax.get_xticklabels(), visible = False)

            if j == 0 and i != 0:
                ax.set_ylim(axlims[i])
                ax.set_ylabel(labels[p[i]])
            else: # not on left column
                plt.setp(ax.get_yticklabels(), visible = False)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()




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
            p.target_tries = 2

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

"""
.. moduleauthor:: Chris Bowman <chris.bowman@york.ac.uk>
"""



from numpy import exp, log, mean, std, sqrt, tanh, cos, cov
from numpy import array, linspace, sort, searchsorted, pi
from scipy.integrate import quad, simps
from scipy.optimize import minimize, minimize_scalar
from itertools import product
from copy import copy
import matplotlib.pyplot as plt




class DensityEstimator(object):
    """
    Parent class for the 1D density estimation classes GaussianKDE and UnimodalPdf.
    """
    def __init__(self):
        self.lwr_limit = None
        self.upr_limit = None
        self.mode = None

    def __call__(self, x):
        return None

    def interval(self, frac = 0.95):
        p_max = self.__call__(self.mode)
        p_conf = self.binary_search(self.interval_prob, frac, [0., p_max], uphill = False)
        return self.get_interval(p_conf)

    def get_interval(self, z):
        lwr = self.binary_search(self.__call__, z, [self.lwr_limit, self.mode], uphill = True)
        upr = self.binary_search(self.__call__, z, [self.mode, self.upr_limit], uphill = False)
        return lwr, upr

    def interval_prob(self, z):
        lwr, upr = self.get_interval(z)
        return quad(self.__call__, lwr, upr, limit = 100)[0]

    def moments(self):
        pass

    def plot_summary(self, filename = None, show = True):
        """
        Plot the estimated PDF along with summary statistics.

        :param str filename: Filename to which the plot will be saved. If unspecified, the plot will not be saved.
        :param bool show: Boolean value indicating whether the plot should be displayed in a window. (Default is True)
        """
        sigma_1 = self.interval(frac = 0.68268)
        sigma_2 = self.interval(frac = 0.95449)

        mu, var, skw, kur = self.moments()

        if type(self).__name__ is 'GaussianKDE':
            lwr = self.s[2] - 2*self.h
            upr = self.s[-3] + 2*self.h
        else:
            sigma_3 = self.interval(frac = 0.9973)
            if hasattr(sigma_3[0], '__len__'):
                s_min = sigma_3[0][0]
                s_max = sigma_3[-1][1]
            else:
                s_min = sigma_3[0]
                s_max = sigma_3[1]

            lwr = s_min - 0.1*(s_max - s_min)
            upr = s_max + 0.1*(s_max - s_min)

        axis = linspace(lwr, upr, 500)

        plt.figure(figsize = (10,6))
        ax = plt.subplot2grid((1, 3), (0, 0), colspan = 2)
        ax.plot(axis, self.__call__(axis), lw = 2)
        ax.plot([self.mode, self.mode], [0., self.__call__(self.mode)], c = 'red', ls = 'dashed')

        ax.set_xlabel('argument', fontsize = 13)
        ax.set_ylabel('probability density', fontsize = 13)
        ax.grid()


        gap = 0.05
        h = 0.95
        x1 = 0.45
        x2 = 0.5
        ax = plt.subplot2grid((1, 3), (0, 2))

        ax.text(0., h, 'Basics', horizontalalignment = 'left', fontweight = 'bold')
        h -= gap
        ax.text(x1, h, 'Mode:', horizontalalignment='right')
        ax.text(x2, h, '{:.5G}'.format( self.mode ), horizontalalignment='left')
        h -= gap
        ax.text(x1, h, 'Mean:', horizontalalignment='right')
        ax.text(x2, h, '{:.5G}'.format( mu ), horizontalalignment='left')
        h -= gap
        ax.text(x1, h, 'Standard dev:', horizontalalignment='right')
        ax.text(x2, h, '{:.5G}'.format( sqrt(var) ), horizontalalignment='left')
        h -= 2*gap

        ax.text(0., h, 'Highest-density intervals', horizontalalignment = 'left', fontweight='bold')
        h -= gap
        ax.text(x1, h, '1-sigma:', horizontalalignment='right')
        if hasattr(sigma_1[0], '__len__'):
            for itvl in sigma_1:
                ax.text(x2, h, r'{:.5G} $\rightarrow$ {:.5G}'.format(itvl[0], itvl[1]), horizontalalignment = 'left')
                h -= gap
        else:
            ax.text(x2, h, r'{:.5G} $\rightarrow$ {:.5G}'.format(sigma_1[0], sigma_1[1]), horizontalalignment='left')
            h -= gap

        ax.text(x1, h, '2-sigma:', horizontalalignment='right')
        if hasattr(sigma_2[0], '__len__'):
            for itvl in sigma_2:
                ax.text(x2, h, r'{:.5G} $\rightarrow$ {:.5G}'.format(itvl[0], itvl[1]), horizontalalignment = 'left')
                h -= gap
        else:
            ax.text(x2, h, r'{:.5G} $\rightarrow$ {:.5G}'.format(sigma_2[0], sigma_2[1]), horizontalalignment='left')
            h -= gap

        if type(self).__name__ is not 'GaussianKDE':
            ax.text(x1, h, '3-sigma:', horizontalalignment='right')
            if hasattr(sigma_3[0], '__len__'):
                for itvl in sigma_3:
                    ax.text(x2, h, r'{:.5G} $\rightarrow$ {:.5G}'.format(itvl[0], itvl[1]), horizontalalignment = 'left')
                    h -= gap
            else:
                ax.text(x2, h, r'{:.5G} $\rightarrow$ {:.5G}'.format(sigma_3[0], sigma_3[1]), horizontalalignment='left')
                h -= gap

        h -= gap
        ax.text(0., h, 'Higher moments', horizontalalignment = 'left', fontweight = 'bold')
        h -= gap
        ax.text(x1, h, 'Variance:', horizontalalignment='right')
        ax.text(x2, h, '{:.5G}'.format( var ), horizontalalignment='left')
        h -= gap
        ax.text(x1, h, 'Skewness:', horizontalalignment='right')
        ax.text(x2, h, '{:.5G}'.format( skw ), horizontalalignment='left')
        h -= gap
        ax.text(x1, h, 'Kurtosis:', horizontalalignment='right')
        ax.text(x2, h, '{:.5G}'.format( kur ), horizontalalignment='left')

        ax.axis('off')

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()

    @staticmethod
    def binary_search(func, value, bounds, uphill = True):
        x_min, x_max = bounds
        x = (x_min + x_max) * 0.5

        converged = False
        while not converged:
            f = func(x)
            if f > value:
                if uphill:
                    x_max = x
                else:
                    x_min = x
            else:
                if uphill:
                    x_min = x
                else:
                    x_max = x

            x = (x_min + x_max) * 0.5
            if abs((x_max - x_min)/x) < 1e-3:
                converged = True

        # now linearly interpolate as a polish step
        f_max = func(x_max)
        f_min = func(x_min)
        df = f_max - f_min

        return x_min*((f_max-value)/df) + x_max*((value - f_min)/df)




class UnimodalPdf(DensityEstimator):
    """
    Construct an estimate of a univariate, unimodal probability distribution based on a given set of samples.

    :param sample: 1D array of samples from which to estimate the probability distribution
    :type sample: array-like


    The UnimodalPdf class is designed to robustly estimate univariate, unimodal
    probability distributions given a sample drawn from that distribution.

    This is a parametric method based on an extensively modified student-t
    distribution, which is extremely flexible.
    """
    def __init__(self, sample):

        self.sample = array(sample)
        self.n_samps = len(sample)

        # chebyshev quadtrature weights and axes
        self.sd = 0.2
        self.n_nodes = 128
        k = linspace(1, self.n_nodes, self.n_nodes)
        t = cos(0.5 * pi * ((2 * k - 1) / self.n_nodes))
        self.u = t / (1. - t**2)
        self.w = (pi / self.n_nodes) * (1 + t**2) / (self.sd * (1 - t**2)**(1.5))


        # first minimise based on a slice of the sample, if it's large enough
        self.cutoff = 2000
        self.skip = self.n_samps // self.cutoff
        if self.skip is 0:
            self.skip = 1

        self.x = self.sample[::self.skip]
        self.n = len(self.x)

        # makes guesses based on sample moments
        guesses = self.generate_guesses()

        # sort the guesses by the lowest score
        guesses = sorted(guesses, key = self.minfunc)

        # minimise based on the best guess
        self.min_result = minimize(self.minfunc, guesses[0])
        self.MAP = self.min_result.x
        self.mode = self.MAP[0] #: The mode of the pdf, calculated automatically when an instance of UnimodalPdf is created.

        # if we were using a reduced sample, use full sample
        if self.skip > 1:
            self.x = self.sample
            self.n = self.n_samps
            self.min_result = minimize(self.minfunc, self.MAP)
            self.MAP = self.min_result.x
            self.mode = self.MAP[0]

        # normalising constant for the MAP estimate curve
        self.map_lognorm = log(self.norm(self.MAP))

        # set some bounds for the confidence limits calculation
        x0, s0, v, f, k, q = self.MAP
        self.upr_limit = x0 + s0*(4*exp(f) + 1)
        self.lwr_limit = x0 - s0*(4*exp(-f) + 1)

    def generate_guesses(self):
        mu, sigma, skew = self.sample_moments()

        x0 = [mu, mu-sigma*skew*0.15, mu-sigma*skew*0.3]
        v = [0, 5.]
        s0 = [sigma, sigma*2]
        f = [0.5*skew, skew]
        k = [1., 4., 8.]
        q = [2.]

        return [ array(i) for i in product( x0, s0, v, f, k, q ) ]

    def sample_moments(self):
        mu = mean(self.x)
        x2 = self.x**2
        x3 = x2 * self.x
        sig = sqrt(mean(x2) - mu**2)
        skew = (mean(x3) - 3*mu*sig**2 - mu**3) / sig**3

        return mu, sig, skew

    def __call__(self, x):
        """
        Evaluate the PDF estimate at a set of given axis positions.

        :param x_vals: axis location(s) at which to evaluate the estimate.
        :return: values of the PDF estimate at the specified locations.
        """
        return exp(self.log_pdf_model(x, self.MAP) - self.map_lognorm)

    def posterior(self, paras):
        x0, s0, v, f, k, q = paras

        # prior checks
        if (s0 > 0) & (0 < k < 20) & (1 < q < 6):
            return self.log_pdf_model(self.x, paras).sum() - self.n*log(self.norm(paras))
        else:
            return -1e50

    def minfunc(self, paras):
        return -self.posterior(paras)

    def norm(self, pvec):
        v = self.pdf_model(self.u, [0., self.sd, *pvec[2:]])
        integral = (self.w * v).sum() * pvec[1]
        return integral

    def pdf_model(self, x, pvec):
        return exp(self.log_pdf_model(x, pvec))

    def log_pdf_model(self, x, pvec):
        x0, s0, v, f, k, q = pvec
        v = exp(v) + 1
        z0 = (x - x0)/s0
        ds = exp(f*tanh(z0/k))
        z = z0 / ds

        log_prob = - (0.5*(1+v))*log( 1 + (abs(z)**q)/v )
        return log_prob

    def moments(self):
        """
        Calculate the mean, variance skewness and excess kurtosis of the estimated PDF.

        :return: mean, variance, skewness, ex-kurtosis

        Note that these quantities are calculated directly from the estimated PDF, and
        note from the sample values.
        """
        s = self.MAP[1]
        f = self.MAP[3]

        lwr = self.mode - 5*max(exp(-f), 1.)*s
        upr = self.mode + 5*max(exp(f), 1.)*s
        x = linspace(lwr, upr, 1000)
        p = self.__call__(x)

        mu  = simps(p*x, x=x)
        var = simps(p*(x - mu)**2, x=x)
        skw = simps(p*(x - mu)**3, x=x) / var*1.5
        kur = (simps(p*(x - mu)**4, x=x) / var**2) - 3.
        return (mu, var, skw, kur)




class GaussianKDE(DensityEstimator):
    """
    Construct an estimate of a univariate probability distribution.

    :param sample: 1D array of samples from which to estimate the probability distribution
    :type sample: array-like

    :param float bandwidth: width of the Gaussian kernels used for the estimate. If not specified, \
                            an appropriate width is estimated based on sample data.

    The GaussianKDE class uses Gaussian kernel-density estimation to approximate a
    probability distribution given a sample drawn from that distribution.
    """
    def __init__(self, sample, bandwidth = None):

        self.s = sort(array(sample).flatten())

        if bandwidth is None:
            self.h = self.estimate_bandwidth(self.s)  # very simple bandwidth estimate
        else:
            self.h = bandwidth

        # define some useful constants
        self.norm = 1. / (len(self.s) * sqrt(2 * pi) * self.h)
        self.cutoff = self.h*4
        self.q = 1. / (sqrt(2)*self.h)
        self.lwr_limit = self.s[0]  - self.cutoff*0.5
        self.upr_limit = self.s[-1] + self.cutoff*0.5

        # decide how many regions the axis should be divided into
        n = int(log((self.s[-1] - self.s[0]) / self.h) / log(2)) + 1

        # now generate midpoints of these regions
        mids = linspace(self.s[0], self.s[-1], 2**n+1)
        mids = 0.5*(mids[1:] + mids[:-1])

        # get the cutoff indices
        lwr_inds = searchsorted(self.s, mids - self.cutoff)
        upr_inds = searchsorted(self.s, mids + self.cutoff)
        cuts = list(zip(lwr_inds, upr_inds))

        # now build a dict that maps midpoints to cut indices
        self.cut_map = dict(zip(mids, cuts))

        # build a binary tree which allows fast look-up of which
        # region contains a given value
        self.tree = BinaryTree(n, (self.s[0], self.s[-1]))

        #: The mode of the pdf, calculated automatically when an instance of GaussianKDE is created.
        self.mode = self.locate_mode()

    def __call__(self, x_vals):
        """
        Evaluate the PDF estimate at a set of given axis positions.

        :param x_vals: axis location(s) at which to evaluate the estimate.
        :return: values of the PDF estimate at the specified locations.
        """
        if hasattr(x_vals, '__iter__'):
            return [ self.density(x) for x in x_vals ]
        else:
            return self.density(x_vals)

    def density(self, x):
        # look-up the region
        region = self.tree.lookup(x)
        # look-up the cutting points
        cuts = self.cut_map[region[2]]
        # evaluate the density estimate from the slice
        return self.norm * exp(-((x - self.s[cuts[0]:cuts[1]])*self.q)**2).sum()

    def halley_update(self, x, y0):
        # look-up the region
        region = self.tree.lookup(x)
        # look-up the cutting points
        cuts = self.cut_map[region[2]]
        # pre-calculate some terms for speed
        z = (x - self.s[cuts[0]:cuts[1]])*self.q
        z2 = z**2
        exp_z = exp(-z2)
        # evaluate zeroth, first and second derivatives
        f0 = exp_z.sum() - y0/self.norm
        f1 = -2*self.q*(z*exp_z).sum()
        f2 = 4*(self.q**2)*((-0.5 + z2)*exp_z).sum()
        # find the required ratios
        f0f1 = f0/f1
        f2f1 = f2/f1
        # return the Halley's method update
        return -f0f1/(1 - 0.5*f0f1*f2f1)

    def find_root(self, x0, y0):
        x = x0
        for i in range(10): # max iterations
            z = x + self.halley_update(x,y0)

            # move any values outside the sample range
            if z < self.s[0]: z = self.s[0]
            if z > self.s[-1]: z = self.s[-1]

            # if value has converged, break the loop
            if x == z:
                break
            else:
                x = z
        return z

    def estimate_bandwidth(self, x): # could be static now, but not in future
        # TODO - we need to replace this with a more sophisticated bandwidth estimator
        return 1.06 * std(x) / (len(x)**0.2)

    def locate_mode(self):
        result = minimize_scalar(lambda x : -self.__call__(x), bounds = [self.s[0], self.s[-1]], method = 'bounded')
        return result.x

    def moments(self):
        """
        Calculate the mean, variance skewness and excess kurtosis of the estimated PDF.

        :return: mean, variance, skewness, ex-kurtosis

        Note that these quantities are calculated directly from the estimated PDF, and
        note from the sample values.
        """
        N = 1000
        x = linspace(self.lwr_limit, self.upr_limit, N)
        p = self.__call__(x)

        mu  = simps(p*x, x=x)
        var = simps(p*(x - mu)**2, x=x)
        skw = simps(p*(x - mu)**3, x=x) / var*1.5
        kur = (simps(p*(x - mu)**4, x=x) / var**2) - 3.
        return (mu, var, skw, kur)

    def interval(self, frac = 0.95):
        """
        Calculate the highest-density interval(s) which contain a given fraction of total probability.

        :param float frac: Fraction of total probability contained by the desired interval(s).
        :return: A list of tuples which specify the intervals.
        """
        # set target number of samples used to estimate the bounding density
        n = 2000
        # how much thinning is needed to get this number?
        skip = int(len(self.s)/n)
        if skip < 1: skip = 1
        # evaluate the PDF estimate for sub-sample and sort the result
        probs = sorted(self.__call__(self.s[::skip]))
        # estimate the bounding density of the region
        density = probs[int((1 - frac)*len(probs))]
        # now use newton's method to find roots
        roots = [ self.find_root(x, density) for x in linspace(self.s[0], self.s[-1], 12) ]
        # filter out anomalies
        roots = [ round(r,6) for r in roots if abs((self.density(r)/density)-1) < 1e-5 ]
        # remove duplicates
        roots = sorted(list(set(roots)))
        # are there an even number of roots?
        if len(roots)%2 == 1:
            print('## WARNING ## Odd number of roots detected in confidence interval calculation')
        return list(zip(roots[::2], roots[1::2]))




class KDE2D(object):
    def __init__(self, x = None, y = None):

        self.x = array(x)
        self.y = array(y)
        s_x, s_y = self.estimate_bandwidth(self.x, self.y)  # very simple bandwidth estimate
        self.q_x = 1. / (sqrt(2) * s_x)
        self.q_y = 1. / (sqrt(2) * s_y)
        self.norm = 1. / (len(self.x) * sqrt(2 * pi) * s_x * s_y)

    def __call__(self, x_vals, y_vals):
        if hasattr(x_vals, '__iter__') and hasattr(y_vals, '__iter__'):
            return [ self.density(x,y) for x,y in zip(x_vals, y_vals) ]
        else:
            return self.density(x_vals, y_vals)

    def density(self, x, y):
        z_x = ((self.x - x) * self.q_x)**2
        z_y = ((self.y - y) * self.q_y)**2
        return exp( -z_x - z_y ).sum() * self.norm

    def estimate_bandwidth(self, x, y):
        S = cov(x, y)
        p = S[0,1] / sqrt(S[0,0]*S[1,1])
        return 1.06 * sqrt(S.diagonal() * (1 - p**2)) / (len(x) ** 0.2)




class BinaryTree:
    """
    divides the range specified by limits into n = 2**layers equal regions,
    and builds a binary tree which allows fast look-up of which of region
    contains a given value.

    :param int layers: number of layers that make up the tree
    :param limits: tuple of the lower and upper bounds of the look-up region.
    """
    def __init__(self, layers, limits):
        self.n = layers
        self.lims = limits
        self.midpoint = 0.5*(limits[0] + limits[1])

        # first generate n trees of depth 1
        L = linspace(limits[0], limits[1], 2**self.n + 1)
        self.mids = 0.5*(L[1:] + L[:-1])
        L = [ [L[i], L[i+1], 0.5*(L[i]+L[i+1])] for i in range(2**self.n) ]

        # now recursively merge them until we have 1 tree of depth n
        for k in range(self.n-1):
            q = []
            for i in range(len(L)//2):
                q.append( [L[2*i], L[2*i+1], 0.5*(L[2*i][2] + L[2*i+1][2])] )
            L = copy(q)

        L.append(self.midpoint)
        self.tree = L

    def lookup(self, val):
        D = self.tree
        for i in range(self.n):
            D = D[val > D[2]]
        return D
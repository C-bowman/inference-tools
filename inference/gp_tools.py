
"""
.. moduleauthor:: Chris Bowman <chris.bowman.physics@gmail.com>
"""

from numpy import abs, exp, log, dot, sqrt, argmin, diagonal, ndarray
from numpy import zeros, ones, array, where, pi, diag, eye
from numpy import sum as npsum
from scipy.special import erf
from numpy.linalg import inv, slogdet, solve, cholesky
from scipy.linalg import solve_triangular
from scipy.optimize import minimize, differential_evolution
from itertools import product




class SquaredExponential(object):
    r"""
    ``SquaredExponential`` is a covariance-function class which can be passed to
    ``GpRegressor`` via the ``kernel`` keyword argument. It uses the 'squared-exponential'
    covariance function given by:

    .. math::

       K(\underline{u}, \underline{v}) = A^2 \exp \left( -\frac{1}{2} \sum_{i=1}^{n} \left(\frac{u_i - v_i}{l_i}\right)^2 \right)

    The hyper-parameter vector :math:`\underline{\theta}` used by ``SquaredExponential`` to define
    the above function is structured as follows:

    .. math::

       \underline{\theta} = [ \ln{A}, \ln{l_1}, \ldots, \ln{l_n}]
    """
    def __init__(self, x, y):
        # pre-calculates hyperparameter-independent part of the
        # data covariance matrix as an optimisation
        dx = x[:,None,:] - x[None,:,:]
        self.distances = -0.5*dx**2 # distributed outer subtraction using broadcasting

        # construct sensible bounds on the hyperparameter values
        s = log(y.std())
        self.bounds = [(s-4,s+4)]
        for i in range(x.shape[1]):
            lwr = log(abs(dx[:,:,i]).mean())-4
            upr = log(dx[:,:,i].max())+2
            self.bounds.append((lwr,upr))

    def __call__(self, u, v, theta):
        a = exp(theta[0])
        L = exp(theta[1:])
        D = -0.5*(u[:,None,:] - v[None,:,:])**2
        C = exp((D / L[None,None,:]**2).sum(axis=2))
        return (a**2)*C

    def build_covariance(self, theta):
        """
        Optimized version of self.matrix() specifically for the data
        covariance matrix where the vectors v1 & v2 are both self.x.
        """
        a = exp(theta[0])
        L = exp(theta[1:])
        C = exp((self.distances / L[None,None,:]**2).sum(axis=2))
        return (a**2)*C

    def gradient_terms(self, v, x, theta):
        """
        Calculates the covariance-function specific parts of
        the expression for the predictive mean and covariance
        of the gradient of the GP estimate.
        """
        a = exp(theta[0])
        L = exp(theta[1:])
        A = (x - v[None,:]) / L[None,:]**2
        return A.T, (a/L)**2

    def get_bounds(self):
        return self.bounds






class RationalQuadratic(object):
    r"""
    ``RationalQuadratic`` is a covariance-function class which can be passed to
    ``GpRegressor`` via the ``kernel`` keyword argument. It uses the 'rational quadratic'
    covariance function given by:

    .. math::

       K(\underline{u}, \underline{v}) = A^2 \left( 1 + \frac{1}{2\alpha} \sum_{i=1}^{n} \left(\frac{u_i - v_i}{l_i}\right)^2 \right)^{-\alpha}

    The hyper-parameter vector :math:`\underline{\theta}` used by ``RationalQuadratic`` to define
    the above function is structured as follows:

    .. math::

       \underline{\theta} = [ \ln{A}, \ln{\alpha}, \ln{l_1}, \ldots, \ln{l_n}]
    """
    def __init__(self, x, y):
        # pre-calculates hyperparameter-independent part of the
        # data covariance matrix as an optimisation
        dx = x[:,None,:] - x[None,:,:]
        self.distances = 0.5*dx**2 # distributed outer subtraction using broadcasting

        # construct sensible bounds on the hyperparameter values
        s = log(y.std())
        self.bounds = [(s-4,s+4), (-2,6)]
        for i in range(x.shape[1]):
            lwr = log(abs(dx[:,:,i]).mean())-4
            upr = log(dx[:,:,i].max())+2
            self.bounds.append((lwr,upr))

    def __call__(self, u, v, theta):
        a = exp(theta[0])
        k = exp(theta[1])
        L = exp(theta[2:])
        D = 0.5*(u[:,None,:] - v[None,:,:])**2
        Z = (D / L[None,None,:]**2).sum(axis=2)
        return (a**2)*(1 + Z/k)**(-k)

    def build_covariance(self, theta):
        a = exp(theta[0])
        k = exp(theta[1])
        L = exp(theta[2:])
        Z = (self.distances / L[None,None,:]**2).sum(axis=2)
        return (a**2)*(1 + Z/k)**(-k)

    def gradient_terms(self, v, x, theta):
        """
        Calculates the covariance-function specific parts of
        the expression for the predictive mean and covariance
        of the gradient of the GP estimate.
        """
        raise ValueError("""
        Gradient calculations are not yet available for the
        RationalQuadratic covariance function.
        """)

    def get_bounds(self):
        return self.bounds






class GpRegressor(object):
    """
    A class for performing Gaussian-process regression in one or more dimensions.

    Gaussian-process regression (GPR) is a non-parametric regression technique
    which can fit arbitrarily spaced data in any number of dimensions. A unique
    feature of GPR is its ability to account for uncertainties on the data
    (which must be assumed to be Gaussian) and propagate that uncertainty to the
    regression estimate by modelling the regression estimate itself as a multivariate
    normal distribution.

    :param x: \
        The spatial coordinates of the y-data values. For the 1-dimensional case,
        this should be a list or array of floats. For greater than 1 dimension,
        a list of coordinate arrays or tuples should be given.

    :param y: The y-data values as a list or array of floats.

    :param y_err: \
        The error on the y-data values supplied as a list or array of floats.
        This technique explicitly assumes that errors are Gaussian, so the supplied
        error values represent normal distribution standard deviations. If this
        argument is not specified the errors are taken to be small but non-zero.

    :param hyperpars: \
        An array specifying the hyper-parameter values to be used by the
        covariance function class, which by default is ``SquaredExponential``.
        See the documentation for the relevant covariance function class for
        a description of the required hyper-parameters. Generally this argument
        should be left unspecified, in which case the hyper-parameters will be
        selected automatically.

    :param class kernel: \
        The covariance function class which will be used to model the data. The
        covariance function classes can be imported from the gp_tools module and
        then passed to ``GpRegressor`` using this keyword argument.

    :param bool cross_val: \
        If set to `True`, leave-one-out cross-validation is used to select the
        hyper-parameters in place of the marginal likelihood.
    """
    def __init__(self, x, y, y_err = None, hyperpars = None, kernel = SquaredExponential, cross_val = False):

        self.N_points = len(x)
        # identify the number of spatial dimensions
        if hasattr(x[0], '__len__'):  # multi-dimensional case
            self.N_dimensions = len(x[0])
        else:  # 1D case
            self.N_dimensions = 1

        # load the spatial data into a 2D array
        self.x = zeros([self.N_points,self.N_dimensions])
        for i,v in enumerate(x): self.x[i,:] = v

        # data to fit
        self.y = array(y)

        # data errors covariance matrix
        self.sig = zeros([self.N_points, self.N_points])
        if y_err is not None:
            if len(y) == len(y_err):
                for i in range(len(self.y)):
                    self.sig[i,i] = y_err[i]**2
            else:
                raise ValueError('y_err must be the same length as y')
        else:
            err = ((self.y.max()-self.y.min()) * 1e-6)**2
            for i in range(len(self.y)):
                self.sig[i,i] = err

        # create an instance of the covariance function class
        self.cov = kernel(self.x, self.y)

        if cross_val:
            self.model_selector = self.loo_likelihood
        else:
            self.model_selector = self.marginal_likelihood

        # if hyper-parameters are specified manually, allocate them
        if hyperpars is None:
            hyperpars = self.optimize_hyperparameters()

        # build the covariance matrix
        self.set_hyperparameters(hyperpars)

    def __call__(self, points, theta = None):
        """
        Calculate the mean and standard deviation of the regression estimate at a series
        of specified spatial points.

        :param list points: \
            A list containing the spatial locations where the mean and standard deviation
            of the estimate is to be calculated. In the 1D case this would be a list of
            floats, or a list of coordinate tuples in the multi-dimensional case.

        :return: \
            Two 1D arrays, the first containing the means and the second containing the
            standard deviations.
        """
        if theta is not None:
            self.set_hyperparameters(theta)

        mu_q = []
        errs = []
        p = self.process_points(points)
        for v in p:
            K_qx = self.cov(v.reshape([1,self.N_dimensions]), self.x, self.hyperpars)
            K_qq = self.cov(v.reshape([1,self.N_dimensions]), v.reshape([1,self.N_dimensions]), self.hyperpars)
            mu_q.append(dot(K_qx, self.alpha)[0])
            v = solve_triangular(self.L, K_qx.T, lower = True)
            errs.append( K_qq[0,0] - npsum(v**2) )

        return array(mu_q), sqrt( abs(array(errs)) )

    def set_hyperparameters(self, theta):
        self.hyperpars = theta
        self.K_xx = self.cov.build_covariance(theta) + self.sig
        self.L = cholesky(self.K_xx)
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, self.y, lower = True))

    def process_points(self, points):
        if type(points) is ndarray:
            x = points
        else:
            x = array(points)

        m = len(x.shape)
        if self.N_dimensions == 1:
            if m == 0: # the case when given a float
                x = x.reshape([1,1])
            elif m == 1:
                x = x.reshape([x.shape[0],1])
            elif m == 2 and x.shape[1] != 1:
                raise ValueError('given spatial points have an incorrect number of dimensions')
        else:
            if m == 0:
                raise ValueError('given spatial points have an incorrect number of dimensions')
            elif m == 1 and x.shape[1] == self.N_dimensions:
                x = x.reshape([1, self.N_dimensions])
            elif m == 2 and x.shape[1] != self.N_dimensions:
                raise ValueError('given spatial points have an incorrect number of dimensions')
        return x

    def gradient(self, points):
        """
        Calculate the mean and covariance of the gradient of the regression estimate
        with respect to the spatial coordinates at a series of specified points.

        :param list points: \
            A list containing the spatial locations where the mean vector and and covariance
            matrix of the gradient of the regression estimate are to be calculated. In the 1D
            case this would be a list of floats, or a list of coordinate tuples in the
            multi-dimensional case.

        :return means, covariances: \
            A list of mean vectors and a list of covariance matrices for the gradient distribution
            at each given spatial point.
        """
        mu_q = []
        vars = []
        p = self.process_points(points)
        for v in p:
            K_qx = self.cov(v.reshape([1,self.N_dimensions]), self.x, self.hyperpars)
            A, R = self.cov.gradient_terms(v, self.x, self.hyperpars)

            B = (K_qx * self.alpha).T
            Q = solve_triangular(self.L, (A*K_qx).T, lower = True)

            # calculate the mean and covariance
            mean = dot(A,B)
            covariance = R - Q.T.dot(Q)

            # if there's only one spatial dimension, convert mean/covariance to floats
            if covariance.shape == (1,1): covariance = covariance[0,0]
            if mean.shape == (1,1): mean = mean[0,0]
            # store the results for the current point
            mu_q.append(mean)
            vars.append(covariance)
        return array(mu_q), vars

    def build_posterior(self, points):
        """
        Generates the full mean vector and covariance matrix for the GP fit at
        a set of specified points.

        :param points: \
            A list containing the spatial locations which will be used to construct
            the Gaussian process. In the 1D case this would be a list of floats, or
            a list of coordinate tuples in the multi-dimensional case.

        :return: The mean vector as a 1D array, followed by covariance matrix as a 2D array.
        """
        v = self.process_points(points)
        K_qx = self.cov(v, self.x, self.hyperpars)
        K_qq = self.cov(v, v, self.hyperpars)
        mu = dot(K_qx, self.alpha)
        sigma = K_qq - dot( K_qx, solve( self.K_xx, K_qx.T ) )
        return mu, sigma

    def loo_predictions(self):
        """
        Calculates the 'leave-one out' (LOO) predictions for the data,
        where each data point is removed from the training set and then
        has its value predicted using the remaining data.

        This implementation is based on equation (5.12) from Rasmussen &
        Williams.
        """
        # Use the Cholesky decomposition of the covariance to find its inverse
        I = eye(len(self.x))
        iK = solve_triangular(self.L.T, solve_triangular(self.L, I, lower = True))
        var = 1./diag(iK)

        mu = self.y - self.alpha * var
        sigma = sqrt(var)
        return mu, sigma

    def loo_likelihood(self, theta):
        """
        Calculates the 'leave-one out' (LOO) log-likelihood.

        This implementation is based on equations (5.10, 5.11, 5.12) from
        Rasmussen & Williams.
        """
        try:
            K_xx = self.cov.build_covariance(theta) + self.sig
            L = cholesky(K_xx)

            # Use the Cholesky decomposition of the covariance to find its inverse
            I = eye(len(self.x))
            iK = solve_triangular(L.T, solve_triangular(L, I, lower = True))
            alpha = solve_triangular(L.T, solve_triangular(L, self.y, lower = True))
            var = 1. / diag(iK)
            return -0.5*(var*alpha**2 + log(var)).sum()
        except:
            return -1e50

    def marginal_likelihood(self, theta):
        """
        returns the negative log marginal likelihood for the
        supplied hyperparameter values.
        """
        K_xx = self.cov.build_covariance(theta) + self.sig

        try: # protection against singular matrix error crash
            L = cholesky(K_xx)
            alpha = solve_triangular(L.T, solve_triangular(L, self.y, lower = True))
            return -0.5*dot( self.y.T, alpha ) - log(diagonal(L)).sum()
        except:
            return -1e50

    def optimize_hyperparameters(self):
        bnds = self.cov.get_bounds()
        # optimise the hyperparameters
        opt_result = differential_evolution(lambda x : -self.model_selector(x), bnds)
        return opt_result.x






class GpInverter(object):
    """
    Solves linear inverse problems of the form y = Gb, using a Gaussian-process
    prior which imposes spatial regularity on the solution.

    The solution vector 'b' must describe the value of a quantity everywhere
    on a grid, as the GP prior imposes covariance between these grid-points
    based on the 'distance' between them. The grid need not be a spatial one,
    only one over which regularity is desired, e.g. time, wavelength ect.

    > arguments

        x 	- array of position values/vectors for the model parameters

        y 	- array of data values

        cov	- covariance matrix for the data

        G	- the linearisation matrix
    """
    def __init__(self, x, y, cov, G, scale_length = None, mean = None, amplitude = None, selector = 'evidence'):
        self.x = x  # spatial location of the parameters, *not* the y data
        self.y = y  # data values
        self.S_y = cov  # data covariance matrix
        self.G = G  # geometry matrix

        self.selector = selector
        self.hyperpar_settings = (amplitude, scale_length, mean)

        # check inputs for compatability
        self.parse_inputs()

        self.I = ones([G.shape[1],1])
        self.f = dot( self.G, self.I )
        self.iS_y = inv(self.S_y)

        # generate square-distance matrix from self.x
        if hasattr(self.x[0], '__iter__'): # multi-dimensional case
            self.D = [ [ self.dist(i,j) for j in self.x] for i in self.x ]
        else: # 1D case
            self.D = [ [ (i-j)**2 for j in self.x] for i in self.x ]
        self.D = -0.5*array(self.D)

        self.A, self.L, self.mu_val = self.optimize_hyperparameters()

        # now we have determined the hyperparameters, generate the prior
        # mean and covariance matrices
        self.mu_p = self.mu_val * ones([len(x), 1])
        self.S_p = (self.A**2)*exp(self.D/(self.L**2))

        # we may now also generate the posterior mean and covariance.
        # To improve the numerical stability of calculating the posterior
        # covariance, we use the woodbury matrix identity:
        K = dot(self.G, self.S_p)

        V = self.S_y + dot(K, self.G.T)
        iVK = solve(V,K)
        self.S_b = self.S_p - dot( K.T, iVK )

        # posterior mean involves no further inversions so is stable
        self.mu_b = self.mu_p + dot( self.S_b, dot( self.G.T, dot( self.iS_y, (self.y - self.mu_val*self.f) ) ) )

    def parse_inputs(self):
        # first check input types
        if type(self.y) is not ndarray: self.y = array(self.y)
        if type(self.S_y) is not ndarray: self.S_y = array(self.S_y)
        if type(self.G) is not ndarray: self.G = array(self.G)

        # now check shapes / sizes are compatible
        if len(self.y.shape) is not 2: self.y = self.y.reshape([self.y.size,1])
        if self.S_y.shape[0] != self.S_y.shape[0]:
            raise ValueError('Data covariance matrix must be square')
        if self.S_y.shape[0] != self.y.shape[0]:
            raise ValueError('Dimensions of the data covariance matrix must equal the number of data points')
        if (self.G.shape[0] != self.y.shape[0]) or  (self.G.shape[1] != len(self.x)):
            raise ValueError('The operator matrix must have dimensions [# data points, # spatial points]')

    def dist(self, a, b):
        return sum( (i-j)**2 for i, j in zip(a, b) )

    def log_ev(self, h):
        # extract hyperparameters
        A, L, mu_p = [exp(v) for v in h]
        # first make the prior covariance
        S_p = (A**2)*exp(self.D/(L**2))
        # now the marginal likelihood covariance
        S_m = dot( self.G, dot(S_p, self.G.T) ) + self.S_y
        # and the marginal likelihood mean
        mu_m = mu_p * self.f
        # now calculate negative log marginal likelihood
        u = self.y - mu_m
        iSu = solve(S_m, u)
        L = dot( u.T, iSu ) + slogdet(S_m)[1]
        return L[0][0]

    def nn_maximum_likelihood(self, h):
        A, L, mu_p = [exp(v) for v in h]

        S_p = (A**2)*exp(self.D/(L**2))

        K = dot(self.G, S_p)
        V = self.S_y + dot(K, self.G.T)
        iVK = solve(V,K)
        S_b = S_p - dot( K.T, iVK )

        # posterior mean involves no further inversions so is stable
        mu_b = mu_p + dot( S_b, dot( self.G.T, dot( self.iS_y, (self.y - mu_p*self.f) ) ) )
        mu_b[where(mu_b < 0)] = 0.
        # find the residual
        res = self.y - self.G.dot(mu_b)
        LL = dot(res.T, self.iS_y.dot(res))
        return LL[0,0]

    def optimize_hyperparameters(self):
        # choose the selection criterion for the hyperparameters
        if self.selector is 'evidence':
            criterion = self.log_ev
        elif self.selector is 'NNML':
            criterion = self.nn_maximum_likelihood
        else:
            raise ValueError('The selector keyword must be given as either `evidence` or `NNML`')

        # Choose the correct inputs for the criterion based on which
        # hyperparameters have been given fixed values
        code = tuple([ x is None for x in self.hyperpar_settings ])
        log_vals = []
        for x in self.hyperpar_settings:
            if x is None:
                log_vals.append(None)
            else:
                log_vals.append(log(x))

        selection_functions = {
            (1,1,1) : lambda x : criterion(x),
            (1,1,0) : lambda x : criterion([x[0],x[1],log_vals[2]]),
            (1,0,1) : lambda x : criterion([x[0],log_vals[1],x[1]]),
            (0,1,1) : lambda x : criterion([log_vals[0],x[0],x[1]]),
            (1,0,0) : lambda x : criterion([x[0],log_vals[1],log_vals[2]]),
            (0,1,0) : lambda x : criterion([log_vals[0],x[0],log_vals[2]]),
            (0,0,1) : lambda x : criterion([log_vals[0],log_vals[1],x[0]]),
            (0,0,0) : None
        }

        minfunc = selection_functions[code]

        # if all the hyperparameters have been fixed, just return the fixed values
        if minfunc is None: return self.hyperpar_settings


        # make some guesses for the hyperparameters
        A_guess  = [-6,-4,-2, 0]
        L_guess  = [-6,-5,-4,-3,-2] # NOTE - should be data-determined in future
        mu_guess = [-8,-6,-4,-2, 0]

        # build a list of initial guesses again depending on what parameters are fixed
        guess_components = []
        if code[0]: guess_components.append(A_guess)
        if code[1]: guess_components.append(L_guess)
        if code[2]: guess_components.append(mu_guess)
        guesses = [ g for g in product(*guess_components) ]

        # sort the guesses by best score
        guesses = sorted(guesses, key = minfunc)

        LML_list   = []
        theta_list = []

        for g in guesses[:3]: # minimize the LML for the best guesses
            min_obj = minimize( minfunc, g, method = 'L-BFGS-B' )
            LML_list.append( min_obj['fun'] )
            theta_list.append( min_obj['x'] )

        # pick the solution the best score
        opt_params = theta_list[ argmin(array(LML_list)) ]
        paras = []
        k = 0
        for i in range(3):
            if code[i]:
                paras.append(opt_params[k])
                k += 1
            else:
                paras.append(log_vals[i])

        return [exp(v) for v in paras]






class GpOptimiser(object):
    """
    A class for performing Gaussian-process optimisation in one or more dimensions.

    GpOptimiser extends the functionality of GpRegressor to perform Gaussian-process
    optimisation, often also referred to as 'Bayesian optimisation'. This technique
    is suited to problems for which a single evaluation of the function being explored
    is expensive, such that the total number of function evaluations must be made as
    small as possible.

    In order to construct the gaussian-process regression estimate which is used to
    search for the global maximum, on initialisation GpOptimiser must be provided with
    at least two evaluations of the function which is to be maximised.

    :param x: \
        The spatial coordinates of the y-data values. For the 1-dimensional case,
        this should be a list or array of floats. For greater than 1 dimension,
        a list of coordinate arrays or tuples should be given.

    :param y: The y-data values as a list or array of floats.

    :keyword y_err: \
        The error on the y-data values supplied as a list or array of floats.
        This technique explicitly assumes that errors are Gaussian, so the supplied
        error values represent normal distribution standard deviations. If this
        argument is not specified the errors are taken to be small but non-zero.

    :keyword bounds: \
        A iterable containing tuples which specify the upper and lower bounds
        for the optimisation in each dimension in the format (lower_bound, upper_bound).

    :param hyperpars: \
        An array specifying the hyper-parameter values to be used by the
        covariance function class, which by default is ``SquaredExponential``.
        See the documentation for the relevant covariance function class for
        a description of the required hyper-parameters. Generally this argument
        should be left unspecified, in which case the hyper-parameters will be
        selected automatically.

    :param class kernel: \
        The covariance function class which will be used to model the data. The
        covariance function classes can be imported from the gp_tools module and
        then passed to ``GpOptimiser`` using this keyword argument.

    :param bool cross_val: \
        If set to `True`, leave-one-out cross-validation is used to select the
        hyper-parameters in place of the marginal likelihood.
    """
    def __init__(self, x, y, y_err = None, bounds = None, hyperpars = None, kernel = SquaredExponential, cross_val = False):
        self.x = list(x)
        self.y = list(y)
        self.y_err = y_err

        if y_err is not None: self.y_err = list(self.y_err)
        if bounds is None:
            ValueError('The bounds keyword argument must be specified')
        else:
            self.bounds = bounds

        self.kernel = kernel
        self.cross_val = cross_val
        self.gp = GpRegressor(x, y, y_err=y_err, hyperpars=hyperpars, kernel = kernel, cross_val = cross_val)

        self.ir2pi = 1 / sqrt(2*pi)
        self.ir2 = 1. / sqrt(2.)
        self.mu_max = max(self.y)
        self.expected_fractional_improvement_history = []

    def __call__(self, x):
        return self.gp(x)

    def add_evaluation(self, new_x, new_y, new_y_err=None):
        """
        Add the latest evaluation to the data set and re-build the \
        Gaussian process so a new proposed evaluation can be made.

        :param new_x: location of the new evaluation
        :param new_y: function value of the new evaluation
        :param new_y_err: Error of the new evaluation.
        """
        # update the data arrays
        self.x.append(new_x)
        self.y.append(new_y)
        if self.y_err is not None:
            if new_y_err is not None:
                self.y_err.append(new_y_err)
            else:
                raise ValueError('y_err must be specified for new evaluations if y_err was specified during __init__')

        # re-train the GP
        self.gp = GpRegressor(self.x, self.y, y_err=self.y_err, kernel = self.kernel, cross_val = self.cross_val)
        self.mu_max = max(self.y)

    def variance_aq(self,x):
        _, sig = self.gp([x])
        return -sig[0]**2

    def max_prediction(self,x):
        mu, _ = self.gp([x])
        return -mu[0]

    def maximise_acquisition(self, aq_func):
        opt_result = differential_evolution(aq_func, self.bounds, popsize = 30)
        return opt_result.x, opt_result.fun

    def learn_function(self):
        return self.maximise_acquisition(self.variance_aq)[0]

    def search_for_maximum(self):
        """
        Request a proposed location for the next evaluation. This proposal is \
        selected in order to maximise the "expected improvement" criteria which \
        searches for the global maximum value of the function.

        :return: location of the next proposed evaluation.
        """
        # find the evaluation point which maximises the acquisition function
        proposed_ev, max_EI = self.maximise_acquisition(self.expected_improvement)
        # store the expected fractional improvement to track convergence
        self.expected_fractional_improvement_history.append( abs(max_EI / self.mu_max) )
        # if the problem is 1D, but the result is returned as a length-1 array,
        # extract the result from the array
        if hasattr(proposed_ev, '__len__') and len(proposed_ev) == 1:
            proposed_ev = proposed_ev[0]
        return proposed_ev

    def expected_improvement(self,x):
        mu, sig = self.gp([x])
        Z  = (mu - self.mu_max) / sig
        pdf = self.normal_pdf(Z)
        cdf = self.normal_cdf(Z)
        return -(mu-self.mu_max)*cdf - sig*pdf

    def normal_pdf(self,z):
       return exp(-0.5*z**2)*self.ir2pi

    def normal_cdf(self,z):
        return 0.5*(1. + erf(z*self.ir2))
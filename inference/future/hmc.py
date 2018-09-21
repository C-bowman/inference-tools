from numpy import dot, exp, log, sqrt
from numpy import array, zeros, isfinite, where
from numpy.random import normal, random
from copy import copy
from inference.mcmc import MarkovChain





# HMC chain child-class
class HamiltonianChain(MarkovChain):
    """
    Hamiltonian Monte-Carlo implemented as a child of the MarkovChain class.

    :param func posterior: \
        a function which returns the log-posterior probability density for a \
        given set of model parameters theta, which should be the only argument \
        so that: ln(P) = posterior(theta)

    :param func posterior: \
        a function which returns the log-posterior probability density for a \
        given set of model parameters theta, which should be the only argument \
        so that: ln(P) = posterior(theta)

    :param start: \
        vector of model parameters which correspond to the parameter-space coordinates \
        at which the chain will start.

    :param float epsilon: \
        Initial guess for the time-step of the Hamiltonian dynamics simulation.

    :param int temperature: \
        The temperature of the markov chain.

    :param bounds: \
        A list or tuple containing two numpy arrays which specify the upper and lower \
        bounds for the parameters in the form (lower_bounds, upper_bounds).

    :param inv_mass: \
        The inverse-mass vector for the simulation.
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
            for i in range(self.steps):
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
        if burn is None: burn = self.burn
        if thin is None: thin = self.thin
        return [v[n] for v in self.theta[burn::thin]]




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



















if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from numpy import array

    def rosenbrock(t):
        # This is a modified form of the rosenbrock function, which
        # is commonly used to test optimisation algorithms
        X, Y = t
        X = 0.25*(X-2)
        Y *= 0.25
        X2 = X**2

        b = 25  # increase this to boost the 'correlatedness' of the function
        c = 2  # standard deviation of the gaussian part
        return -(X - 1) ** 2 - b * (Y - X2) ** 2 - 0.5 * ((X2 + Y ** 2) / c ** 2)


    def rosenbrock_grad(t):
        # gradient of the rosenbrock function as defined above
        X, Y = t
        X = 0.25*(X-2)
        Y *= 0.25

        b = 25  # increase this to boost the 'correlatedness' of the function
        c = 2  # standard deviation of the gaussian part

        t2 = -2*b*(Y - X**2)
        dx = -2*(X-1) + -2*X*t2 - X/c**2
        dy = t2 - Y/c**2
        return array([dx, dy]) * 0.25




    bnds = [array([2, -15]), array([10, 15])]
    chain = HamiltonianChain(posterior = rosenbrock, start = [3., 1.], grad = rosenbrock_grad, bounds = bnds)
    chain.advance(15000)

    x = chain.get_parameter(0)
    y = chain.get_parameter(1)
    p = chain.get_probabilities()
    plt.scatter(x, y, c=exp(p-max(p)), marker = '.')
    plt.show()

    plt.plot(chain.ES.epsilon_checks, chain.ES.epsilon_values, '.-')
    plt.grid()
    plt.show()
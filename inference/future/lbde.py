
from numpy import linspace, zeros, where, sort, array
from numpy import exp, log, sqrt, pi, mean, std
from copy import copy
from numpy.random import normal

import matplotlib.pyplot as plt


class LinearBasis(object):
    def __init__(self, x = None, s = None):
        self.x = x
        self.s = s

        self.N = len(x)
        self.M = len(s)

        self.G = zeros([self.M,self.N])
        for k in range(self.N):
            self.G[:,k] = self.kernel(self.s, k)

        self.I = zeros([self.M,self.N])
        for k in range(self.N):
            self.I[:,k] = self.ikernel(self.s, k)

        # self.D = zeros([self.N,self.N])
        # for k in range(1,self.N-1):
        #     self.D[k,k-1:k+2] = [1, -2, 1]

        self.D = zeros([self.N,self.N])
        for k in range(2,self.N-2):
            self.D[k,k-2:k+3] = [-1/12,4/3, -5/2, 4/3, -1/12]

        self.D = (self.D.T).dot(self.D)

    def kernel(self, v, k):
        f = zeros(len(v))
        if k != 0:
            inds = where( (v > self.x[k-1]) & (v <= self.x[k]) )
            f[inds] = (v[inds]-self.x[k-1]) / (self.x[k] - self.x[k-1])
        if k != self.N-1:
            inds = where( (v >= self.x[k]) & (v < self.x[k+1]))
            f[inds] = 1 - (v[inds] - self.x[k]) / (self.x[k+1] - self.x[k])
        return f

    def ikernel(self, v, k):
        f = zeros(len(v))
        if k != 0:
            L_area = 0.5*(self.x[k] - self.x[k-1])
        else: L_area = 0.

        if k != self.N - 1:
            R_area = 0.5*(self.x[k+1] - self.x[k])
            f[where(v >= self.x[k + 1])] = L_area + R_area
        else: R_area = 0.

        if k != 0:
            inds = where( (v > self.x[k-1]) & (v <= self.x[k]) )
            f[inds] = L_area*((v[inds]-self.x[k-1]) / (self.x[k] - self.x[k-1]))**2
        if k != self.N-1:
            inds = where( (v >= self.x[k]) & (v < self.x[k+1]))
            f[inds] = L_area + R_area*(1 - (1 - (v[inds] - self.x[k]) / (self.x[k+1] - self.x[k]))**2)
        return f

    def __call__(self, v):
        return self.G.dot(v)

    def integral(self, v):
        return self.I.dot(v)






class GL_density(object):
    def __init__(self, sample, N = 64, lam = 30):
        self.s = sort(sample)
        self.M = len(sample)
        self.N = N
        self.lam = lam

        self.x_min = self.s[0] #- 0.2*(self.s[-1] - self.s[0])
        self.x_max = self.s[-1] #+ 0.2*(self.s[-1] - self.s[0])
        self.x = linspace(self.x_min, self.x_max, N)

        self.basis = LinearBasis(x = self.x, s = self.s)

        self.theta = zeros(N + 1)

        self.initial_gamma_tuning()
        self.optimise(iter = 600)

        # lam0_prob = self.loglike(self.theta)
        #
        # probs = []
        # lam_vals = 10**linspace(-3,5,24)
        # lam_vals = lam_vals[::-1]
        # for L in lam_vals:
        #     self.lam = L
        #     self.optimise()
        #     probs.append(self.loglike(self.theta))
        #     print(L)
        #
        #
        #
        # plt.plot(lam_vals, probs,'.-')
        # plt.plot([lam_vals[0], lam_vals[-1]], [lam0_prob,lam0_prob], ls = 'dashed')
        # plt.xscale('log')
        # plt.grid()
        # plt.show()

    def optimise(self, iter = 120):
        alpha_0 = 0.5
        c = 0.5
        tau = 0.5

        p = []
        for i in range(iter):
            P0, grad = self.both(self.theta)
            unit = grad / sqrt(grad.dot(grad))
            m = unit.dot(grad)
            t = c*m
            alpha = copy(alpha_0)

            while True:
                P1 = self.log_prob(self.theta + alpha*unit)
                if (P1-P0) >= t*alpha:
                    break
                else:
                    alpha *= tau

            alpha_0 = copy(alpha / tau**2)
            self.theta += alpha*unit
            p.append(P1)

        self.P_max = P1
        plt.plot(p, '.-')
        plt.grid()
        plt.show()

    def initial_gamma_tuning(self):
        sig = std(self.s)

        a = 1 / (sqrt(2) * sig)
        self.theta[0] = 0.
        self.theta[1:] = a

        gamma = linspace(-6, 6, 20)
        result = []
        for G in gamma:
            self.theta[0] = G
            r = self.log_prob(self.theta)
            result.append(r)

        result = array(result)
        self.theta[0] = gamma[result.argmax()]

        plt.plot(gamma, result, '.-')
        plt.grid()
        plt.show()

    def loglike(self, theta):
        gamma = theta[0]
        alpha = theta[1:]
        f = self.basis(alpha)
        g = self.basis.integral(alpha) + gamma # TODO factor out gamma later

        return mean(log(f) - g**2)

    def log_prob(self, theta):
        gamma = theta[0]
        alpha = theta[1:]
        f = self.basis(alpha)
        g = self.basis.integral(alpha) + gamma # TODO factor out gamma later
        reg = (self.lam/self.N)*alpha.dot(self.basis.D.dot(alpha))

        return mean(log(f) - g**2) - reg

    def both(self, theta):
        gamma = theta[0]
        alpha = theta[1:]

        # calculate the probability
        f = self.basis(alpha)
        g = self.basis.integral(alpha) + gamma  # TODO factor out gamma later
        probability = mean(log(f) - g**2)

        # calculate gradient
        A = (self.basis.G.T).dot(1 / f)
        B = -2*(self.basis.I.T).dot(g)

        gradient = zeros(self.N + 1)
        gradient[0] = -2*sum(g)
        gradient[1:] = A + B
        gradient /= self.M # normalise to sample count

        # now include regularisation terms
        grad_reg = (self.lam/self.N)*self.basis.D.dot(alpha)
        regularisation = alpha.dot(grad_reg)

        gradient[1:] -= grad_reg
        probability -= regularisation

        return probability, gradient

    def __call__(self, x):
        gamma = self.theta[0]
        alpha = self.theta[1:]
        base = LinearBasis(x = self.x, s = x)
        f = base(alpha)
        g = base.integral(alpha) + gamma
        return f * exp(-g**2) / sqrt(pi)



if __name__ == "__main__":

    N = 30000
    sample = zeros(N)
    sample[:(N//3)] = normal(size=(N//3))+6
    sample[(N//3):] = normal(size=2*(N//3))+10
    # sample = 10 + (sample**2) / 10

    GLD = GL_density(sample = sample)

    ax = linspace(GLD.x_min, GLD.x_max, 1000)
    plt.plot(ax, GLD(ax))
    plt.grid()
    plt.show()

    plt.plot(GLD.theta[1:], '.-')
    plt.grid()
    plt.show()
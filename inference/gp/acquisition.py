from numpy import sqrt, log, exp, pi
from numpy import array, ndarray, minimum, maximum
from numpy.random import random
from scipy.special import erf, erfcx


class AcquisitionFunction:
    gp = None
    opt_func = None

    def starting_positions(self, bounds):
        lwr, upr = [array([k[i] for k in bounds], dtype=float) for i in [0, 1]]
        widths = upr - lwr

        lwr += widths * 0.01
        upr -= widths * 0.01
        starts = []
        L = len(widths)
        for x0 in self.gp.x:
            # first check if the point is inside the search bounds
            inside = ((x0 >= lwr) & (x0 <= upr)).all()
            if inside:
                # a small random search around the point to find a good start
                samples = [
                    x0 + 0.02 * widths * (2 * random(size=L) - 1) for i in range(20)
                ]
                samples = [minimum(upr, maximum(lwr, s)) for s in samples]
                samples = sorted(samples, key=self.opt_func)
                starts.append(samples[0])
            else:
                # draw a sample uniformly from the search bounds hypercube
                start = lwr + (upr - lwr) * random(size=L)
                starts.append(start)

        return starts


class ExpectedImprovement(AcquisitionFunction):
    r"""
    ``ExpectedImprovement`` is an acquisition-function class which can be passed to
    ``GpOptimiser`` via the ``acquisition`` keyword argument. It implements the
    expected-improvement acquisition function given by

    .. math::

       \mathrm{EI}(\underline{x}) = \left( z F(z) + P(z) \right) \sigma(\underline{x})

    where

    .. math::

       z = \frac{\mu(\underline{x}) - y_{\mathrm{max}}}{\sigma(\underline{x})},
       \qquad P(z) = \frac{1}{\sqrt{2\pi}}\exp{\left(-\frac{1}{2}z^2 \right)},
       \qquad F(z) = \frac{1}{2}\left[ 1 + \mathrm{erf}\left(\frac{z}{\sqrt{2}}\right) \right],

    :math:`\mu(\underline{x}),\,\sigma(\underline{x})` are the predictive mean and standard
    deviation of the Gaussian-process regression model at position :math:`\underline{x}`,
    and :math:`y_{\mathrm{max}}` is the current maximum observed value of the objective function.
    """

    def __init__(self):
        self.ir2pi = 1 / sqrt(2 * pi)
        self.ir2 = 1.0 / sqrt(2)
        self.rpi2 = sqrt(0.5 * pi)
        self.ln2pi = log(2 * pi)

        self.name = "Expected improvement"
        self.convergence_description = r"$\mathrm{EI}_{\mathrm{max}} \; / \; (y_{\mathrm{max}} - y_{\mathrm{min}})$"

    def update_gp(self, gp):
        self.gp = gp
        self.mu_max = gp.y.max()

    def __call__(self, x):
        mu, sig = self.gp(x)
        Z = (mu[0] - self.mu_max) / sig[0]
        if Z < -3:
            ln_EI = log(1 + Z * self.cdf_pdf_ratio(Z)) + self.ln_pdf(Z) + log(sig[0])
            EI = exp(ln_EI)
        else:
            pdf = self.normal_pdf(Z)
            cdf = self.normal_cdf(Z)
            EI = sig[0] * (Z * cdf + pdf)
        return EI

    def opt_func(self, x):
        mu, sig = self.gp(x)
        Z = (mu[0] - self.mu_max) / sig[0]
        if Z < -3:
            ln_EI = log(1 + Z * self.cdf_pdf_ratio(Z)) + self.ln_pdf(Z) + log(sig[0])
        else:
            pdf = self.normal_pdf(Z)
            cdf = self.normal_cdf(Z)
            ln_EI = log(sig[0] * (Z * cdf + pdf))
        return -ln_EI

    def opt_func_gradient(self, x):
        mu, sig = self.gp(x)
        dmu, dvar = self.gp.spatial_derivatives(x)
        Z = (mu[0] - self.mu_max) / sig[0]

        if Z < -3:
            R = self.cdf_pdf_ratio(Z)
            H = 1 + Z * R
            ln_EI = log(H) + self.ln_pdf(Z) + log(sig[0])
            grad_ln_EI = (0.5 * dvar / sig[0] + R * dmu) / (H * sig[0])
        else:
            pdf = self.normal_pdf(Z)
            cdf = self.normal_cdf(Z)
            EI = sig[0] * (Z * cdf + pdf)
            ln_EI = log(EI)
            grad_ln_EI = (0.5 * pdf * dvar / sig[0] + dmu * cdf) / EI

        # flip sign on the value and gradient since we're using a minimizer
        ln_EI = -ln_EI
        grad_ln_EI = -grad_ln_EI
        # make sure outputs are ndarray in the 1D case
        if type(ln_EI) is not ndarray:
            ln_EI = array(ln_EI)
        if type(grad_ln_EI) is not ndarray:
            grad_ln_EI = array(grad_ln_EI)

        return ln_EI, grad_ln_EI.squeeze()

    def normal_pdf(self, z):
        return exp(-0.5 * z**2) * self.ir2pi

    def normal_cdf(self, z):
        return 0.5 * (1.0 + erf(z * self.ir2))

    def cdf_pdf_ratio(self, z):
        return self.rpi2 * erfcx(-z * self.ir2)

    def ln_pdf(self, z):
        return -0.5 * (z**2 + self.ln2pi)

    def convergence_metric(self, x):
        return self.__call__(x) / (self.mu_max - self.gp.y.min())


class UpperConfidenceBound(AcquisitionFunction):
    r"""
    ``UpperConfidenceBound`` is an acquisition-function class which can be passed to
    ``GpOptimiser`` via the ``acquisition`` keyword argument. It implements the
    upper-confidence-bound acquisition function given by

    .. math::

       \mathrm{UCB}(\underline{x}) = \mu(\underline{x}) + \kappa \sigma(\underline{x})

    where :math:`\mu(\underline{x}),\,\sigma(\underline{x})` are the predictive mean and
    standard deviation of the Gaussian-process regression model at position :math:`\underline{x}`.

    :param float kappa: Value of the coefficient :math:`\kappa` which scales the contribution
        of the predictive standard deviation to the acquisition function. ``kappa`` should be
        set so that :math:`\kappa \ge 0`.
    """

    def __init__(self, kappa=2):
        self.kappa = kappa
        self.name = "Upper confidence bound"
        self.convergence_description = (
            r"$\mathrm{UCB}_{\mathrm{max}} - y_{\mathrm{max}}$"
        )

    def update_gp(self, gp):
        self.gp = gp
        self.mu_max = gp.y.max()

    def __call__(self, x):
        mu, sig = self.gp(x)
        return mu[0] + self.kappa * sig[0]

    def opt_func(self, x):
        mu, sig = self.gp(x)
        return -mu[0] - self.kappa * sig[0]

    def opt_func_gradient(self, x):
        mu, sig = self.gp(x)
        dmu, dvar = self.gp.spatial_derivatives(x)
        ucb = mu[0] + self.kappa * sig[0]
        grad_ucb = dmu + 0.5 * self.kappa * dvar / sig[0]
        # flip sign on the value and gradient since we're using a minimizer
        ucb = -ucb
        grad_ucb = -grad_ucb
        # make sure outputs are ndarray in the 1D case
        if type(ucb) is not ndarray:
            ucb = array(ucb)
        if type(grad_ucb) is not ndarray:
            grad_ucb = array(grad_ucb)
        return ucb, grad_ucb.squeeze()

    def convergence_metric(self, x):
        return self.__call__(x) - self.mu_max


class MaxVariance(AcquisitionFunction):
    r"""
    ``MaxVariance`` is an acquisition-function class which can be passed to
    ``GpOptimiser`` via the ``acquisition`` keyword argument. It selects new
    evaluations of the objective function by finding the spatial position
    :math:`\underline{x}` with the largest variance :math:`\sigma^2(\underline{x})`
    as predicted by the Gaussian-process regression model.

    This is a `pure learning' acquisition function which does not seek to find the
    maxima of the objective function, but only to minimize uncertainty in the
    prediction of the function.
    """

    def __init__(self):
        self.name = "Max variance"
        self.convergence_description = r"$\sqrt{\mathrm{Var}\left[x\right]}$"

    def update_gp(self, gp):
        self.gp = gp
        self.mu_max = gp.y.max()

    def __call__(self, x):
        _, sig = self.gp(x)
        return sig[0] ** 2

    def opt_func(self, x):
        _, sig = self.gp(x)
        return -sig[0] ** 2

    def opt_func_gradient(self, x):
        _, sig = self.gp(x)
        _, dvar = self.gp.spatial_derivatives(x)
        aq = -(sig**2)
        aq_grad = -dvar
        if type(aq) is not ndarray:
            aq = array(aq)
        if type(aq_grad) is not ndarray:
            aq_grad = array(aq_grad)
        return aq.squeeze(), aq_grad.squeeze()

    def convergence_metric(self, x):
        return sqrt(self.__call__(x))

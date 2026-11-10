from numbers import Integral
from numpy import sqrt, log, exp, pi, hypot, abs
from numpy import array, ndarray, asarray, empty, zeros, zeros_like
from numpy import array_equal, isfinite, isscalar, finfo, errstate
from numpy import argsort, diag, divide, median, minimum, maximum
from numpy.random import random, default_rng
from scipy.special import erf, erfcx, ndtr
from scipy.linalg import solve_triangular
from scipy.stats import qmc
from inference.gp.covariance import SquaredExponential
from inference.gp.regression import GpRegressor


class AcquisitionFunction:
    gp: GpRegressor
    mu_max: float
    opt_func: callable

    def set_bounds(self, bounds):
        """Receive the search bounds before the first call to ``update_gp``."""
        self.bounds = bounds

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

    def update_gp(self, gp: GpRegressor):
        self.gp = gp
        self.mu_max = gp.y.max()


class LevelSetLearning(AcquisitionFunction):
    r"""
    Select evaluations which reduce uncertainty near the level set ``f(x) = target``.

    Pass an instance to ``GpOptimiser`` using its ``acquisition`` argument. This
    implementation supports the ``SquaredExponential`` covariance function and both
    ``"bfgs"`` (with analytic gradients) and ``"diffev"`` optimizers.

    With fixed integration points :math:`z_j`, the score to maximize is

    .. math::

       a(x) = \frac{M^{-1}\sum_j w_j k_n(z_j,x)^2}{s_n^2(x)+\tau^2},
       \qquad
       w_j = \frac{\phi(t;\mu_n(z_j),\sqrt{s_n^2(z_j)+\epsilon^2})}
                        {M^{-1}\sum_i\phi(t;\mu_n(z_i),\sqrt{s_n^2(z_i)+\epsilon^2})}
                        + \mathrm{explore}.

    All predictions are latent (noise-free). Weights and fitted hyperparameters
    remain fixed during each look-ahead. Known historical observation errors enter
    through the GP; ``noise_sd`` is the expected standard error of a new observation.
    For a mean of independent Monte Carlo samples, use the sample standard
    deviation divided by the square root of the sample count, not the sample
    standard deviation itself.

    The criterion discourages redundant observations but does not enforce uniform
    contour spacing or guarantee discovery of every disconnected component.
    ``explore`` mixes in global variance reduction, not a fraction of evaluations.

    :param float target: \
        Function value defining the level set.
        
    :param int n_integration_points: \
        Number of fixed, scrambled Sobol integration points. Defaults to 4096.
        Powers of two retain Sobol balance properties. Check sensitivity to this
        setting for narrow bands or higher dimensions.

    :param float eps: \
        Non-negative target-band width in function-value units. Defaults to zero.
        A round-off-scale variance floor regularizes the density weights when
        latent uncertainty is zero.
        
    :param float explore: \
        Non-negative global-exploration weight, default 0.05.
        
    :param float noise_sd: \
        Expected standard deviation of a new observation. If omitted, use the median
        historical standard error, recomputed on update; if no errors were supplied
        to the GP, this is zero.

    :param int seed: \
        Optional seed for integration points and candidate starts.
        This does not seed GP hyperparameter fitting or differential evolution.

    Calling the instance returns a float for a single point, or a 1D score array
    for multiple points, using the same input conventions as ``GpRegressor``.
    Candidate batches are processed in bounded chunks. ``convergence_metric(x)``
    ignores ``x`` and returns the posterior expected misclassified fraction on the
    integration design, in [0, 0.5]. It need not decrease at every iteration.

    Based on targeted integrated mean squared error (Picheny et al., 2010,
    *Adaptive Designs of Experiments for Accurate Approximation of a Target Region*,
    Journal of Mechanical Design, 132(7), 071008), with normalized density weights,
    an additional exploration term, and a noisy-observation look-ahead.
    """

    def __init__(
        self,
        target: float,
        n_integration_points: int = 4096,
        eps: float = 0.0,
        explore: float = 0.05,
        noise_sd: float | None = None,
        seed: int | None = None,
    ):
        if not isscalar(target) or not isfinite(target):
            raise ValueError("target must be a finite scalar.")
        if (
            isinstance(n_integration_points, bool)
            or not isinstance(n_integration_points, Integral)
            or n_integration_points < 1
        ):
            raise ValueError("n_integration_points must be a positive integer.")
        for name, value in [("eps", eps), ("explore", explore), ("noise_sd", noise_sd)]:
            if name == "noise_sd" and value is None:
                continue
            if not isscalar(value) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative scalar.")

        self.target = float(target)
        self.n_integration_points = int(n_integration_points)
        self.eps = float(eps)
        self.explore = float(explore)
        self.noise_sd = noise_sd
        self.rng = default_rng(seed)
        self.name = "Level-set learning"
        self.convergence_description = "Expected misclassified fraction"

    def set_bounds(self, bounds):
        bounds = asarray(bounds, dtype=float)
        if (
            bounds.ndim != 2
            or bounds.shape[0] == 0
            or bounds.shape[1] != 2
            or not isfinite(bounds).all()
            or (bounds[:, 0] >= bounds[:, 1]).any()
        ):
            raise ValueError(
                "bounds must contain finite (lower, upper) pairs with lower < upper."
            )
        if hasattr(self, "bounds") and array_equal(self.bounds, bounds):
            return

        # Keep the integration design fixed so changes in the metric reflect the GP.
        self.bounds = bounds.copy()
        sampler = qmc.Sobol(d=len(bounds), scramble=True, seed=self.rng)
        sample_power = (self.n_integration_points - 1).bit_length()
        unit_points = sampler.random_base2(sample_power)
        self.integration_points = qmc.scale(
            unit_points[: self.n_integration_points], bounds[:, 0], bounds[:, 1]
        )

    def _checked_variance(self, variance):
        if not isfinite(variance).all() or (variance < -self._variance_tolerance).any():
            raise FloatingPointError("Invalid latent variance in LevelSetLearning.")
        # Clip only round-off-scale negatives, not invalid posterior variances.
        return maximum(variance, 0.0)

    def update_gp(self, gp: GpRegressor):
        if not hasattr(self, "bounds"):
            raise RuntimeError("Call set_bounds(bounds) before update_gp(gp).")
        if gp.n_dimensions != len(self.bounds):
            raise ValueError("Search bounds and GP data must have the same dimension.")
        if type(gp.cov) is not SquaredExponential:
            raise TypeError("LevelSetLearning currently requires SquaredExponential.")
        super().update_gp(gp)
        self._L = gp.L

        # SquaredExponential has the same prior variance at every location.
        self._prior_variance = gp.cov(gp.x[:1], gp.x[:1], gp.cov_hyperpars)[0, 0]
        self._variance_tolerance = 64 * finfo(float).eps * self._prior_variance
        self._length_squared = exp(2 * gp.cov_hyperpars[1:])

        # Cache L^-1 k(X, Z), shared by all candidate evaluations until the next fit.
        training_covariance = gp.cov(gp.x, self.integration_points, gp.cov_hyperpars)
        self._integration_projection = solve_triangular(
            self._L, training_covariance, lower=True
        )
        prior_mean = array(
            [gp.mean(point, gp.mean_hyperpars) for point in self.integration_points]
        )
        self._integration_mean = training_covariance.T @ gp.alpha + prior_mean
        explained_variance = (self._integration_projection**2).sum(axis=0)
        self._integration_variance = self._checked_variance(
            self._prior_variance - explained_variance
        )
        self._weights = self._calculate_weights()

        if self.noise_sd is None:
            historical_errors = sqrt(diag(gp.sig))
            self.expected_noise_sd = float(median(historical_errors))
        else:
            self.expected_noise_sd = float(self.noise_sd)
        self._noise_variance = self.expected_noise_sd**2
        if not isfinite(self._noise_variance):
            raise ValueError("noise_sd squared must be finite.")
        self._misclassified_fraction = self._expected_misclassified_fraction()

    def _calculate_weights(self):
        # A variance floor keeps the density defined even at a known target crossing.
        weight_variance = maximum(self._integration_variance, self._variance_tolerance)
        weight_sd = hypot(sqrt(weight_variance), self.eps)

        # Normalize in log space to avoid all densities underflowing for distant targets.
        with errstate(over="raise", invalid="raise", divide="raise"):
            target_distance = (self.target - self._integration_mean) / weight_sd
            log_weights = -0.5 * target_distance**2 - log(weight_sd)
            weights = exp(log_weights - log_weights.max())
            return weights / weights.mean() + self.explore

    def _expected_misclassified_fraction(self):
        # A deterministic prediction has no classification uncertainty.
        uncertainty = zeros_like(self._integration_variance)
        uncertain = self._integration_variance > 0
        target_distance = abs(self._integration_mean[uncertain] - self.target)
        posterior_sd = sqrt(self._integration_variance[uncertain])
        uncertainty[uncertain] = ndtr(-target_distance / posterior_sd)
        return float(uncertainty.mean())

    def _candidate_terms(self, points):
        """Return scores and the covariance terms needed for their gradients."""
        training_covariance = self.gp.cov(self.gp.x, points, self.gp.cov_hyperpars)
        candidate_projection = solve_triangular(
            self._L, training_covariance, lower=True
        )
        prior_covariance = self.gp.cov(
            self.integration_points, points, self.gp.cov_hyperpars
        )

        # Condition the integration-to-candidate covariance on the existing data.
        posterior_covariance = (
            prior_covariance - self._integration_projection.T @ candidate_projection
        )
        explained_variance = (candidate_projection**2).sum(axis=0)
        latent_variance = self._checked_variance(
            self._prior_variance - explained_variance
        )
        observation_variance = latent_variance + self._noise_variance
        weighted_covariance_squared = self._weights[:, None] * posterior_covariance**2
        numerator = weighted_covariance_squared.mean(axis=0)

        # A noiseless observation with zero posterior variance adds no information.
        scores = divide(
            numerator,
            observation_variance,
            out=zeros_like(numerator),
            where=observation_variance > 0,
        )
        return (
            scores,
            posterior_covariance,
            observation_variance,
            training_covariance,
            candidate_projection,
            prior_covariance,
        )

    def __call__(self, x) -> float | ndarray:
        points = self.gp.process_points(x)
        if not isfinite(points).all() or len(points) == 0:
            raise ValueError("Candidate points must be finite and non-empty.")
        scores = empty(len(points))

        # Bound the memory used by integration-point by candidate matrices.
        for start in range(0, len(points), 128):
            stop = min(start + 128, len(points))
            scores[start:stop] = self._candidate_terms(points[start:stop])[0]
        return float(scores[0]) if len(scores) == 1 else scores

    def opt_func(self, x) -> float:
        score = self(x)
        if not isscalar(score):
            raise ValueError("opt_func requires a single candidate point.")
        return -score

    def opt_func_gradient(self, x):
        points = self.gp.process_points(x)
        if len(points) != 1 or not isfinite(points).all():
            raise ValueError(
                "opt_func_gradient requires a single finite candidate point."
            )
        (
            scores,
            posterior_covariance,
            observation_variance,
            training_covariance,
            candidate_projection,
            prior_covariance,
        ) = self._candidate_terms(points)
        score = float(scores[0])
        denominator = observation_variance[0]
        if denominator == 0:
            return -score, zeros(self.gp.n_dimensions)

        # Differentiate the posterior covariance with respect to the candidate location.
        training_gradient = (
            training_covariance * (self.gp.x - points) / self._length_squared
        )
        projection_gradient = solve_triangular(self._L, training_gradient, lower=True)
        prior_gradient = (
            prior_covariance * (self.integration_points - points) / self._length_squared
        )
        covariance_gradient = (
            prior_gradient - self._integration_projection.T @ projection_gradient
        )
        weighted_gradient = (
            self._weights[:, None] * posterior_covariance * covariance_gradient
        )
        numerator_gradient = 2 * weighted_gradient.mean(axis=0)
        variance_gradient = -2 * (candidate_projection * projection_gradient).sum(
            axis=0
        )

        # Apply the quotient rule, then negate because the optimizer minimizes.
        gradient = (numerator_gradient - score * variance_gradient) / denominator
        return -score, -gradient

    def starting_positions(self, bounds):
        if not array_equal(asarray(bounds), self.bounds):
            raise ValueError(
                "Starting-position bounds must match the integration bounds."
            )
        candidates = self.rng.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(2048, len(self.bounds))
        )
        scores = self(candidates)
        # Limit the number of local searches independently of the training-set size.
        n_starts = min(8, 2 * self.gp.n_dimensions + 2)
        return candidates[argsort(scores)[-n_starts:]]

    def convergence_metric(self, x=None) -> float:
        return self._misclassified_fraction


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

    def __call__(self, x) -> float:
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

    def opt_func(self, x) -> float:
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

    def __init__(self, kappa: float = 2.0):
        self.kappa = kappa
        self.name = "Upper confidence bound"
        self.convergence_description = (
            r"$\mathrm{UCB}_{\mathrm{max}} - y_{\mathrm{max}}$"
        )

    def __call__(self, x) -> float:
        mu, sig = self.gp(x)
        return mu[0] + self.kappa * sig[0]

    def opt_func(self, x) -> float:
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

    This is a 'pure learning' acquisition function which does not seek to find the
    maxima of the objective function, but only to minimize uncertainty in the
    prediction of the function.
    """

    def __init__(self):
        self.name = "Max variance"
        self.convergence_description = r"$\sqrt{\mathrm{Var}\left[x\right]}$"

    def __call__(self, x) -> float:
        _, sig = self.gp(x)
        return sig[0] ** 2

    def opt_func(self, x) -> float:
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

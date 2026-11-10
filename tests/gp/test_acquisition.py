import pytest
from numpy import array, append, sqrt, exp, sin, cos, linspace, full, eye, vstack
from numpy import isfinite, median, abs, inf, nan, argmax, asarray, diag
from numpy.random import default_rng
from numpy.testing import assert_allclose
from scipy.special import ndtr

from inference.gp import (
    GpOptimiser,
    GpRegressor,
    ExpectedImprovement,
    UpperConfidenceBound,
    MaxVariance,
    RationalQuadratic,
    LevelSetLearning,
)

ACQUISITION_CLASSES = [
    ExpectedImprovement,
    UpperConfidenceBound,
    MaxVariance,
    LevelSetLearning,
]


def build_gp(dimensions=1, errors=True):
    rng = default_rng(42)
    x = rng.uniform(size=(12, dimensions))
    y = sin(5 * x[:, 0]) + 0.1 * x.sum(axis=1)
    y_err = linspace(0.01, 0.08, len(x)) if errors else None
    return GpRegressor(
        x,
        y,
        y_err=y_err,
        hyperpars=array([0.0, 0.0, *([-1.2] * dimensions)]),
    )


def build_acquisition(gp, acquisition_class=LevelSetLearning, **kwargs):
    if acquisition_class is LevelSetLearning:
        kwargs = {"target": 0.5, "n_integration_points": 64, "seed": 7} | kwargs
    acquisition = acquisition_class(**kwargs)
    acquisition.set_bounds([(0.0, 1.0)] * gp.n_dimensions)
    acquisition.update_gp(gp)
    return acquisition


@pytest.mark.parametrize("dimensions", [1, 2, 5])
@pytest.mark.parametrize("acquisition_class", ACQUISITION_CLASSES)
def test_acquisition_and_gradients(acquisition_class, dimensions):
    gp = build_gp(dimensions)
    acquisition = build_acquisition(gp, acquisition_class)
    points = default_rng(11).uniform(0.1, 0.9, (20, dimensions))
    shifts = eye(dimensions) * 1e-6

    for point in points:
        value, gradient = acquisition.opt_func_gradient(point)
        assert isfinite(acquisition(point))
        assert isfinite(value).all()
        assert isfinite(acquisition.convergence_metric(point))
        assert_allclose(value, acquisition.opt_func(point), rtol=1e-10)
        finite_difference = array(
            [
                (
                    acquisition.opt_func(point + shift)
                    - acquisition.opt_func(point - shift)
                )
                / 2e-6
                for shift in shifts
            ]
        )
        assert_allclose(
            asarray(gradient).reshape(-1), finite_difference, rtol=2e-5, atol=1e-8
        )

    starts = array(acquisition.starting_positions([(0, 1)] * dimensions))
    assert starts.ndim == 2 and starts.shape[1] == dimensions
    assert len(starts) > 0 and isfinite(starts).all()
    assert ((starts >= 0) & (starts <= 1)).all()


@pytest.mark.parametrize("dimensions", [1, 2, 5])
def test_level_set_vectorised_scores(dimensions):
    acquisition = build_acquisition(build_gp(dimensions))
    points = default_rng(11).uniform(0.1, 0.9, (260, dimensions))
    scores = acquisition(points)
    assert scores.shape == (len(points),)
    assert isfinite(scores).all()
    assert (scores >= 0).all()
    assert isinstance(acquisition(points[0]), float)
    assert isinstance(acquisition(points[:1]), float)
    assert_allclose(scores, [acquisition(p) for p in points], rtol=1e-10, atol=1e-14)
    if dimensions == 1:
        assert_allclose(acquisition(points[:, 0]), scores)


@pytest.mark.parametrize("noise_sd", [0.0, 0.03, 0.3])
def test_level_set_conditioned_variance_reduction(noise_sd):
    gp = build_gp(2)
    acquisition = build_acquisition(gp, noise_sd=noise_sd, eps=0.1, explore=0.2)
    points = array([[0.2, 0.4], [0.8, 0.7]])
    mu_Z, covariance = gp.build_posterior(acquisition.integration_points)
    var_Z = covariance.diagonal()
    sd = sqrt(var_Z + acquisition.eps**2)
    weights = exp(-0.5 * ((acquisition.target - mu_Z) / sd) ** 2) / sd
    weights = weights / weights.mean() + acquisition.explore

    for point in points:
        mean, _ = gp(point)
        augmented = GpRegressor(
            vstack([gp.x, point]),
            append(gp.y, mean),
            y_err=append(sqrt(gp.sig.diagonal()), noise_sd),
            hyperpars=gp.hyperpars,
        )
        _, updated_covariance = augmented.build_posterior(
            acquisition.integration_points
        )
        reduction = var_Z - updated_covariance.diagonal()
        assert acquisition(point) == pytest.approx(
            (weights * reduction).mean(), rel=1e-8, abs=1e-12
        )
    assert_allclose(acquisition._integration_mean, mu_Z)
    assert_allclose(acquisition._integration_variance, var_Z, atol=1e-14)
    assert_allclose(acquisition._weights, weights)
    assert acquisition._L is gp.L


def test_level_set_noise_and_fixed_design():
    gp = build_gp()
    acquisition = build_acquisition(gp)
    points = acquisition.integration_points.copy()
    assert acquisition.expected_noise_sd == pytest.approx(
        median(sqrt(gp.sig.diagonal()))
    )
    before = acquisition(0.4)
    quiet = build_acquisition(gp, noise_sd=0.0)
    loud = build_acquisition(gp, noise_sd=1.0)
    assert quiet(0.4) > before > loud(0.4)
    acquisition.update_gp(build_gp(errors=False))
    assert acquisition.expected_noise_sd == 0.0
    assert_allclose(acquisition.integration_points, points)
    acquisition.set_bounds([(0, 1)])
    assert_allclose(acquisition.integration_points, points)
    identical = build_acquisition(gp)
    assert_allclose(identical.integration_points, points)
    another_quiet = build_acquisition(gp, noise_sd=0.0)
    assert_allclose(
        quiet.starting_positions([(0, 1)]), another_quiet.starting_positions([(0, 1)])
    )


def test_level_set_distant_target():
    gp = build_gp(errors=False)
    acquisition = LevelSetLearning(
        target=1e4, n_integration_points=65, seed=3, explore=0.0
    )
    acquisition.set_bounds([(0, 1)])
    acquisition.update_gp(gp)
    assert isfinite(acquisition._weights).all()
    assert acquisition._weights.mean() == pytest.approx(1.0)
    assert isfinite(acquisition(linspace(0, 1, 100))).all()
    assert 0 <= acquisition.convergence_metric() <= 0.5


def test_level_set_zero_variance():
    gp = GpRegressor(
        array([0.0, 0.5, 1.0]),
        array([0.0, 1.0, 0.0]),
        hyperpars=array([0.0, 0.0, -1.0]),
    )
    acquisition = build_acquisition(gp, noise_sd=0.0)
    assert acquisition(gp.x).max() < 1e-9
    gp.cov.epsilon[:] = 0.0
    gp.set_hyperparameters(gp.hyperpars)
    acquisition.integration_points = gp.x.copy()
    acquisition.target = 1.0
    acquisition.update_gp(gp)
    assert isfinite(acquisition._weights).all()
    assert acquisition.convergence_metric() == 0.0
    assert acquisition(0.5) == 0.0
    value, gradient = acquisition.opt_func_gradient(0.5)
    assert value == 0.0
    assert_allclose(gradient, [0.0])


def test_level_set_convergence_metric():
    gp = build_gp()
    acquisition = build_acquisition(gp)
    mu, sigma = gp(acquisition.integration_points)
    expected = ndtr(-abs(mu - acquisition.target) / sigma).mean()
    assert acquisition.convergence_metric() == pytest.approx(expected)
    assert acquisition.convergence_metric(0.2) == acquisition.convergence_metric(0.8)


def test_level_set_proposal_near_root():
    x = array([0.0, 0.25, 0.75, 1.0])
    y = x - 0.5
    acquisition = LevelSetLearning(
        target=0.0, n_integration_points=512, explore=0.01, seed=23
    )
    optimizer = GpOptimiser(
        x,
        y,
        bounds=[(0, 1)],
        acquisition=acquisition,
        hyperpars=array([0.0, 0.0, -1.3]),
    )
    proposal = optimizer.propose_evaluation()
    assert abs(proposal - 0.5) < 0.05


@pytest.mark.parametrize(
    "method,n_processes", [("bfgs", 1), ("bfgs", 2), ("diffev", 1)]
)
@pytest.mark.parametrize("acquisition_class", ACQUISITION_CLASSES)
def test_optimizer_history(acquisition_class, method, n_processes):
    gp = build_gp()
    acquisition = build_acquisition(gp, acquisition_class)
    optimizer = GpOptimiser(
        gp.x,
        gp.y,
        bounds=[(0, 1)],
        y_err=sqrt(diag(gp.sig)),
        acquisition=acquisition,
        hyperpars=gp.hyperpars,
        n_processes=n_processes,
    )
    proposal = optimizer.propose_evaluation(optimizer=method)
    score = acquisition(proposal)
    metric = acquisition.convergence_metric(proposal)
    assert 0 <= proposal <= 1
    optimizer.add_evaluation(proposal, sin(5 * proposal) + 0.1 * proposal, 0.03)
    assert len(optimizer.y) == len(gp.y) + 1
    assert_allclose(optimizer.acquisition_max_history, [score])
    assert_allclose(optimizer.convergence_metric_history, [metric])
    assert acquisition.gp is optimizer.gp


def test_level_set_learning_reduces_uncertainty():
    rng = default_rng(12)
    x = rng.uniform(size=(8, 2))
    function = lambda p: sin(4 * p[:, 0]) + cos(3 * p[:, 1])
    y = function(x)
    acquisition = LevelSetLearning(target=0.5, n_integration_points=256, seed=14)
    hyperpars = array([0.0, 0.0, -0.8, -0.8])
    initial_metric = None
    for i in range(13):
        gp = GpRegressor(x, y, y_err=full(len(y), 0.01), hyperpars=hyperpars)
        if i == 0:
            acquisition.set_bounds([(0, 1), (0, 1)])
        acquisition.update_gp(gp)
        if i == 0:
            initial_metric = acquisition.convergence_metric()
        candidates = rng.uniform(size=(512, 2))
        point = candidates[argmax(acquisition(candidates))]
        x = vstack([x, point])
        y = append(y, function(point[None, :]))
    assert acquisition.convergence_metric() < 0.5 * initial_metric


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": nan},
        {"target": inf},
        {"n_integration_points": 0},
        {"n_integration_points": 1.5},
        {"n_integration_points": True},
        {"eps": -1},
        {"eps": inf},
        {"explore": -0.1},
        {"noise_sd": -1},
        {"noise_sd": nan},
    ],
)
def test_level_set_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        LevelSetLearning(**({"target": 0.0} | kwargs))


@pytest.mark.parametrize(
    "bounds",
    [
        [],
        [0, 1],
        [(1, 0)],
        [(0, 0)],
        [(0, inf)],
        [(0, 1, 2)],
    ],
)
def test_level_set_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        LevelSetLearning(0.0).set_bounds(bounds)


def test_level_set_invalid_usage():
    acquisition = LevelSetLearning(0.5, n_integration_points=32)
    gp = build_gp()
    with pytest.raises(RuntimeError, match="set_bounds"):
        acquisition.update_gp(gp)
    acquisition.set_bounds([(0, 1), (0, 1)])
    with pytest.raises(ValueError, match="dimension"):
        acquisition.update_gp(gp)
    acquisition.set_bounds([(0, 1)])
    gp.cov = RationalQuadratic()
    with pytest.raises(TypeError, match="SquaredExponential"):
        acquisition.update_gp(gp)
    acquisition.update_gp(build_gp())
    with pytest.raises(ValueError, match="single"):
        acquisition.opt_func([[0.2], [0.3]])
    with pytest.raises(ValueError, match="single"):
        acquisition.opt_func_gradient([[0.2], [0.3]])
    with pytest.raises(ValueError, match="finite"):
        acquisition([nan])
    with pytest.raises(ValueError, match="non-empty"):
        acquisition([])
    with pytest.raises(ValueError, match="match"):
        acquisition.starting_positions([(0, 2)])
    assert_allclose(acquisition._checked_variance(array([-1e-16])), [0.0])
    with pytest.raises(FloatingPointError, match="variance"):
        acquisition._checked_variance(array([-1.0]))

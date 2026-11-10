from numpy import array, sin, cos, meshgrid, linspace, column_stack
from inference.gp import (
    GpOptimiser,
    ExpectedImprovement,
    UpperConfidenceBound,
    MaxVariance,
    LevelSetLearning,
)
import pytest


def search_function_1d(x):
    return sin(0.5 * x) + 3 / (1 + (x - 1) ** 2)


def search_function_2d(v):
    x, y = v
    z = ((x - 1) / 2) ** 2 + ((y + 3) / 1.5) ** 2
    return sin(0.5 * x) + cos(0.4 * y) + 5 / (1 + z)


@pytest.mark.parametrize(
    ["acq_func", "opt_method"],
    [
        (ExpectedImprovement, "bfgs"),
        (UpperConfidenceBound, "bfgs"),
        (MaxVariance, "bfgs"),
        (ExpectedImprovement, "diffev"),
    ],
)
def test_optimizer_1d(acq_func, opt_method):
    x = array([-8, -6, 8])
    y = array([search_function_1d(k) for k in x])
    GpOpt = GpOptimiser(
        x=x, y=y, bounds=[(-8.0, 8.0)], acquisition=acq_func, optimizer=opt_method
    )

    for i in range(3):
        new_x = GpOpt.propose_evaluation()
        new_y = search_function_1d(new_x)
        GpOpt.add_evaluation(new_x, new_y)

    assert GpOpt.y.size == x.size + 3
    assert ((GpOpt.x >= -8) & (GpOpt.x <= 8)).all()


@pytest.mark.parametrize(
    ["acq_func", "opt_method"],
    [
        (ExpectedImprovement, "bfgs"),
        (UpperConfidenceBound, "bfgs"),
        (MaxVariance, "bfgs"),
        (ExpectedImprovement, "diffev"),
    ],
)
def test_optimizer_2d(acq_func, opt_method):
    x = array([(-8, -8), (8, -8), (-8, 8), (8, 8), (0, 0)])
    y = array([search_function_2d(k) for k in x])
    GpOpt = GpOptimiser(
        x=x, y=y, bounds=[(-8, 8), (-8, 8)], acquisition=acq_func, optimizer=opt_method
    )

    for i in range(3):
        new_x = GpOpt.propose_evaluation()
        new_y = search_function_2d(new_x)
        GpOpt.add_evaluation(new_x, new_y)

    assert GpOpt.y.size == x.shape[0] + 3
    assert ((GpOpt.x >= -8) & (GpOpt.x <= 8)).all()

    GpOpt.plot_results(show_plot=False)


def test_legacy_acquisition_without_bounds_hook():
    class LegacyAcquisition:
        def update_gp(self, gp):
            self.gp = gp

    legacy = LegacyAcquisition()
    optimizer = GpOptimiser(
        array([0, 1]),
        array([0, 1]),
        [(0, 1)],
        hyperpars=array([0.0, 0.0, -1.0]),
        acquisition=legacy,
    )
    assert legacy.gp is optimizer.gp


def test_demo_disconnected_contours_and_animation(tmp_path):
    from PIL import Image
    from scipy.ndimage import label
    from demos.scripts import level_set_acquisition_demo as demo

    mesh = meshgrid(linspace(0, 1, 50), linspace(0, 1, 50))
    grid = column_stack([axis.ravel() for axis in mesh])
    truth = demo.test_function(grid).reshape(mesh[0].shape)
    assert label(truth > demo.TARGET)[1] == 2
    acquisition = LevelSetLearning(demo.TARGET, n_integration_points=64, seed=12)
    optimizer, frames, errors, seconds = demo.run_experiment(
        acquisition, grid, iterations=2, n_mc=20, seed=12, record_frames=True
    )
    assert len(optimizer.y) == 10
    assert len(frames) == len(errors) == 3
    assert seconds.shape == (2,)
    for i, frame in enumerate(frames):
        assert frame.points.shape == (8 + i, 2)
        if frame.proposal is not None:
            assert frame.proposal.shape == (2,)
    assert frames[-1].proposal is None
    demo.save_animation(frames, mesh, tmp_path, mp4=False)
    with Image.open(tmp_path / "level_set_demo.gif") as image:
        assert image.n_frames == 3

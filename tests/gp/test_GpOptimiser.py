from numpy import array, sin, cos
from inference.gp import (
    GpOptimiser,
    ExpectedImprovement,
    UpperConfidenceBound,
    MaxVariance,
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

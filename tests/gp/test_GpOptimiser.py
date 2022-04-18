import numpy as np
from inference.gp import (
    GpOptimiser,
    ExpectedImprovement,
    UpperConfidenceBound,
    MaxVariance,
)
import pytest


def search_function_1d(x):
    return np.sin(0.5 * x) + 3 / (1 + (x - 1) ** 2)


def search_function_2d(v):
    x, y = v
    z = ((x - 1) / 2) ** 2 + ((y + 3) / 1.5) ** 2
    return np.sin(0.5 * x) + np.cos(0.4 * y) + 5 / (1 + z)


@pytest.mark.parametrize(
    "acq_func", [ExpectedImprovement, UpperConfidenceBound, MaxVariance]
)
def test_bfgs_1d(acq_func):
    x = [-8, -6, 8]
    y = [search_function_1d(k) for k in x]
    GP = GpOptimiser(x, y, bounds=[(-8.0, 8.0)], acquisition=acq_func, optimizer="bfgs")

    for i in range(3):
        new_x = GP.propose_evaluation()
        new_y = search_function_1d(new_x)
        GP.add_evaluation(new_x, new_y)

    x_array = np.array(GP.x)
    assert len(GP.y) == len(x) + 3
    assert all((x_array >= -8) & (x_array <= 8))


@pytest.mark.parametrize(
    "acq_func", [ExpectedImprovement, UpperConfidenceBound, MaxVariance]
)
def test_diffev_1d(acq_func):
    x = [-8, -6, 8]
    y = [search_function_1d(k) for k in x]
    GP = GpOptimiser(
        x, y, bounds=[(-8.0, 8.0)], acquisition=acq_func, optimizer="diffev"
    )

    for i in range(3):
        new_x = GP.propose_evaluation()
        new_y = search_function_1d(new_x)
        GP.add_evaluation(new_x, new_y)

    x_array = np.array(GP.x)
    assert len(GP.y) == len(x) + 3
    assert all((x_array >= -8) & (x_array <= 8))


@pytest.mark.parametrize(
    "acq_func", [ExpectedImprovement, UpperConfidenceBound, MaxVariance]
)
def test_bfgs_2d(acq_func):
    x = [(-8, -8), (8, -8), (-8, 8), (8, 8), (0, 0)]
    y = [search_function_2d(k) for k in x]
    GP = GpOptimiser(
        x, y, bounds=[(-8, 8), (-8, 8)], acquisition=acq_func, optimizer="bfgs"
    )

    for i in range(3):
        new_x = GP.propose_evaluation()
        new_y = search_function_2d(new_x)
        GP.add_evaluation(new_x, new_y)

    x_array = np.array(GP.x)
    assert len(GP.y) == len(x) + 3
    assert all((x_array[:, 0] >= -8) & (x_array[:, 0] <= 8))


@pytest.mark.parametrize(
    "acq_func", [ExpectedImprovement, UpperConfidenceBound, MaxVariance]
)
def test_diffev_2d(acq_func):
    x = [(-8, -8), (8, -8), (-8, 8), (8, 8), (0, 0)]
    y = [search_function_2d(k) for k in x]
    GP = GpOptimiser(
        x,
        y,
        bounds=[(-8, 8), (-8, 8)],
        acquisition=acq_func,
        optimizer="diffev",
    )

    for i in range(3):
        new_x = GP.propose_evaluation()
        new_y = search_function_2d(new_x)
        GP.add_evaluation(new_x, new_y)

    x_array = np.array(GP.x)
    assert len(GP.y) == len(x) + 3
    assert all((x_array[:, 0] >= -8) & (x_array[:, 0] <= 8))


import pytest
import unittest

from numpy import sin, cos
from inference.gp_tools import GpOptimiser, ExpectedImprovement, UpperConfidenceBound, MaxVariance

def search_function_1d(x):
    return sin(0.5*x) + 3/(1 + (x-1)**2)

def search_function_2d(v):
    x,y = v
    z = ((x-1)/2)**2 + ((y+3)/1.5)**2
    return sin(0.5*x) + cos(0.4*y) + 5/(1 + z)


class test_GpOptimiser(unittest.TestCase):

    def test_bfgs_1d(self):
        for acq_func in [ExpectedImprovement, UpperConfidenceBound, MaxVariance]:
            x = [-8,-6, 8]
            y = [search_function_1d(k) for k in x]
            GP = GpOptimiser(x, y, bounds = [(-8.,8.)], acquisition = acq_func, optimizer = 'bfgs')

            for i in range(3):
                new_x = GP.propose_evaluation()
                new_y = search_function_1d(new_x)
                GP.add_evaluation(new_x, new_y)

    def test_diffev_1d(self):
        for acq_func in [ExpectedImprovement, UpperConfidenceBound, MaxVariance]:
            x = [-8,-6, 8]
            y = [search_function_1d(k) for k in x]
            GP = GpOptimiser(x, y, bounds = [(-8.,8.)], acquisition = acq_func, optimizer = 'diffev')

            for i in range(3):
                new_x = GP.propose_evaluation()
                new_y = search_function_1d(new_x)
                GP.add_evaluation(new_x, new_y)

    def test_bfgs_2d(self):
        for acq_func in [ExpectedImprovement, UpperConfidenceBound, MaxVariance]:
            x = [(-8, -8), (8, -8), (-8, 8), (8, 8), (0, 0)]
            y = [search_function_2d(k) for k in x]
            GP = GpOptimiser(x, y, bounds = [(-8,8), (-8,8)], acquisition = acq_func, optimizer = 'bfgs')

            for i in range(3):
                new_x = GP.propose_evaluation()
                new_y = search_function_2d(new_x)
                GP.add_evaluation(new_x, new_y)

    def test_diffev_2d(self):
        for acq_func in [ExpectedImprovement, UpperConfidenceBound, MaxVariance]:
            x = [(-8, -8), (8, -8), (-8, 8), (8, 8), (0, 0)]
            y = [search_function_2d(k) for k in x]
            GP = GpOptimiser(x, y, bounds = [(-8, 8), (-8, 8)], acquisition = acq_func, optimizer = 'diffev')

            for i in range(3):
                new_x = GP.propose_evaluation()
                new_y = search_function_2d(new_x)
                GP.add_evaluation(new_x, new_y)


if __name__ == '__main__':

    unittest.main()

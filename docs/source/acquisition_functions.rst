
Acquisition functions
~~~~~~~~~~~~~~~~~~~~~
Acquisition functions are used to select new points in the search-space to evaluate in
Gaussian-process optimisation.

The available acquisition functions are implemented as classes within ``inference.gp_tools``,
and can be passed to ``GpOptimiser`` via the ``acquisition`` keyword argument as follows:

.. code-block:: python

   from inference.gp_tools import GpOptimiser, ExpectedImprovement
   GP = GpRegressor(x, y, bounds=bounds, acquisition=ExpectedImprovement)

The acquisition function classes can also be passed as instances, allowing settings of the
acquisition function to be altered:

.. code-block:: python

   from inference.gp_tools import GpOptimiser, UpperConfidenceBound
   UCB = UpperConfidenceBound(kappa = 2.)
   GP = GpRegressor(x, y, bounds=bounds, acquisition=UCB)

ExpectedImprovement
^^^^^^^^^^^^^^^^^^^

.. autoclass:: inference.gp_tools.ExpectedImprovement


UpperConfidenceBound
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: inference.gp_tools.UpperConfidenceBound
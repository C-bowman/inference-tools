
Acquisition functions
~~~~~~~~~~~~~~~~~~~~~
Acquisition functions are used to select new points in the search-space to evaluate in
Gaussian-process optimisation.

The available acquisition functions are implemented as classes within ``inference.gp``,
and can be passed to ``GpOptimiser`` via the ``acquisition`` keyword argument as follows:

.. code-block:: python

   from inference.gp import GpOptimiser, ExpectedImprovement
   GP = GpOptimiser(x, y, bounds=bounds, acquisition=ExpectedImprovement)

The acquisition function classes can also be passed as instances, allowing settings of the
acquisition function to be altered:

.. code-block:: python

   from inference.gp import GpOptimiser, UpperConfidenceBound
   UCB = UpperConfidenceBound(kappa = 2.)
   GP = GpOptimiser(x, y, bounds=bounds, acquisition=UCB)

ExpectedImprovement
^^^^^^^^^^^^^^^^^^^

.. autoclass:: inference.gp.ExpectedImprovement


UpperConfidenceBound
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: inference.gp.UpperConfidenceBound
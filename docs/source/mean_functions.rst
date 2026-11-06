
Mean functions
~~~~~~~~~~~~~~
Gaussian-process regression & optimisation model the data as a Gaussian process with a
mean function, which describes the expected value of the data in the absence of any
observations, and a covariance function, which describes the spatial structure of the
data around that mean.

The available mean functions are implemented as classes within ``inference.gp``,
and can be passed either to ``GpRegressor`` or ``GpOptimiser`` via the ``mean`` keyword
argument as follows

.. code-block:: python

   from inference.gp import GpRegressor, LinearMean
   GP = GpRegressor(x, y, mean=LinearMean)

If no mean function is specified, ``ConstantMean`` is used by default.


ConstantMean
^^^^^^^^^^^^

.. autoclass:: inference.gp.ConstantMean


LinearMean
^^^^^^^^^^

.. autoclass:: inference.gp.LinearMean


QuadraticMean
^^^^^^^^^^^^^

.. autoclass:: inference.gp.QuadraticMean

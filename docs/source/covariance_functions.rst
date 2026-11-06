
Covariance functions
~~~~~~~~~~~~~~~~~~~~
Gaussian-process regression & optimisation model the spatial structure of data using a
covariance function which specifies the covariance between any two points in the space.

The available covariance functions are implemented as classes within ``inference.gp``,
and can be passed either to ``GpRegressor`` or ``GpOptimiser`` via the ``kernel`` keyword
argument as follows

.. code-block:: python

   from inference.gp import GpRegressor, SquaredExponential
   GP = GpRegressor(x, y, kernel=SquaredExponential())


SquaredExponential
^^^^^^^^^^^^^^^^^^

.. autoclass:: inference.gp.SquaredExponential


RationalQuadratic
^^^^^^^^^^^^^^^^^

.. autoclass:: inference.gp.RationalQuadratic


WhiteNoise
^^^^^^^^^^

.. autoclass:: inference.gp.WhiteNoise


HeteroscedasticNoise
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: inference.gp.HeteroscedasticNoise

An example of using ``HeteroscedasticNoise`` to model noise which varies across the data
can be found in the
`heteroscedastic noise modelling jupyter notebook demo <https://github.com/C-bowman/inference-tools/blob/master/demos/heteroscedastic_noise.ipynb>`_.


ChangePoint
^^^^^^^^^^^

.. autoclass:: inference.gp.ChangePoint
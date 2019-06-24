Gaussian processes with inference.gp_tools
==========================================
This module provides implementations of some useful applications of 'Gaussian processes'.
This involves modelling data through multivariate normal distributions where the
covariance of any two points is defined by the 'distance' between them in some arbitrary
space.

.. _GpRegressor:

GpRegressor
~~~~~~~~~~~

.. autoclass:: inference.gp_tools.GpRegressor
   :members: __call__, build_posterior

.. _GpOptimiser:

GpOptimiser
~~~~~~~~~~~

.. autoclass:: inference.gp_tools.GpOptimiser
   :members: search_for_maximum, add_evaluation
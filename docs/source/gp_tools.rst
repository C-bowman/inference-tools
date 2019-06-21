The gp_tools module
===================
This module provides implementations of some useful applications of 'Gaussian processes'.
This involves modelling data through multivariate normal distributions where the
covariance of any two points is defined by the 'distance' between them in some arbitrary
space.

.. _GpRegressor:

GpRegressor
~~~~~~~~~~~
Gaussian-process regression has been implemented as the GpRegressor class, and can fit data which is
arbitrarily spaced (i.e. non-gridded) in any number of dimensions. A key advantage this
technique holds over other regression methods is its ability to properly account for
errors on the data and provide a corresponding error on the regression estimate.

A demonstration of how to use the GpRegressor class can be found in ``/demos/GpRegressor_demo.py``

.. autoclass:: inference.gp_tools.GpRegressor
   :members: __call__, build_posterior

.. _GpOptimiser:

GpOptimiser
~~~~~~~~~~~
Bounded global optimisation using gaussian-process regression has been implemented through the GpOptimiser class.
This algorithm, often referred to as "Bayesian optimisation" specifically suited to problems where a single evaluation
of the function which is to be maximised is very expensive, such that the total number of evaluations must be minimised.

A demonstration of how to use the GpOptimiser class can be found in ``/demos/GpOptimiser_demo.py``

.. autoclass:: inference.gp_tools.GpOptimiser
   :members: search_for_maximum, add_evaluation
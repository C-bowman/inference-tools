
Posterior
~~~~~~~~~
The ``Posterior`` class from the ``inference.posterior`` module provides a
simple way to combine a likelihood and a prior to form a posterior distribution.
Example code demonstrating its use can be found in the
the `Gaussian fitting jupyter notebook demo <https://github.com/C-bowman/inference-tools/blob/master/demos/gaussian_fitting_demo.ipynb>`_.

.. autoclass:: inference.posterior.Posterior
   :members: __call__, gradient, cost, cost_gradient
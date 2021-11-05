Likelihood classes
~~~~~~~~~~~~~~~~~~
The ``inference.likelihoods`` module provides tools for constructing likelihood functions.
Example code demonstrating their use can be found in the
`Gaussian fitting jupyter notebook demo <https://github.com/C-bowman/inference-tools/blob/master/demos/gaussian_fitting_demo.ipynb>`_.

GaussianLikelihood
^^^^^^^^^^^^^^^^^^

.. autoclass:: inference.likelihoods.GaussianLikelihood
   :members: __call__, gradient


CauchyLikelihood
^^^^^^^^^^^^^^^^

.. autoclass:: inference.likelihoods.CauchyLikelihood
   :members: __call__, gradient


LogisticLikelihood
^^^^^^^^^^^^^^^^^^

.. autoclass:: inference.likelihoods.LogisticLikelihood
   :members: __call__, gradient
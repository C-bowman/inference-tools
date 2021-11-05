Prior classes
~~~~~~~~~~~~~
The ``inference.priors`` module provides tools for constructing prior distributions over
the model variables. Example code demonstrating their use can be found in
the `Gaussian fitting jupyter notebook demo <https://github.com/C-bowman/inference-tools/blob/master/demos/gaussian_fitting_demo.ipynb>`_.

GaussianPrior
^^^^^^^^^^^^^

.. autoclass:: inference.priors.GaussianPrior
   :members: __call__, gradient, sample


UniformPrior
^^^^^^^^^^^^^

.. autoclass:: inference.priors.UniformPrior
   :members: __call__, gradient, sample


ExponentialPrior
^^^^^^^^^^^^^^^^

.. autoclass:: inference.priors.ExponentialPrior
   :members: __call__, gradient, sample


JointPrior
^^^^^^^^^^

.. autoclass:: inference.priors.JointPrior
   :members: __call__, gradient, sample
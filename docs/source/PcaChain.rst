
PcaChain
~~~~~~~~

.. autoclass:: inference.mcmc.PcaChain
   :members: advance, run_for, mode, get_marginal, get_sample, get_parameter, get_probabilities, get_interval, estimate_burn_in, plot_diagnostics, matrix_plot, trace_plot, save, load


PcaChain example code
^^^^^^^^^^^^^^^^^^^^^

``PcaChain`` is well suited to posteriors with strong linear correlations between
the parameters, which slow down standard Gibbs sampling. As a test case, define a
3D Gaussian posterior with strongly correlated parameters:

.. code-block:: python

   from numpy import array
   from numpy.linalg import inv

   covariance = array([
       [1.00, 0.95, 0.90],
       [0.95, 1.00, 0.95],
       [0.90, 0.95, 1.00]
   ])
   inverse_covariance = inv(covariance)

   def correlated_gaussian(theta):
       return -0.5 * theta @ inverse_covariance @ theta

Create the chain object and advance it to generate a sample:

.. code-block:: python

   from inference.mcmc import PcaChain
   chain = PcaChain(posterior=correlated_gaussian, start=array([1.0, -1.0, 0.5]))
   chain.advance(25000)

As the chain advances, it periodically re-estimates the covariance of the posterior
from the sample, and updates the directions in which it takes steps to be the
eigenvectors of that covariance. The ``plot_diagnostics`` method can be used to check
the progress of the chain, and the ``estimate_burn_in`` method gives an estimate of
how many samples should be discarded from the start of the chain:

.. code-block:: python

   chain.plot_diagnostics()
   burn = chain.estimate_burn_in()

We can now view the marginal distributions using ``matrix_plot``, and check that the
covariance of the sample is consistent with the covariance of the posterior:

.. code-block:: python

   chain.matrix_plot(burn=burn)

   from numpy import cov
   sample = chain.get_sample(burn=burn)
   print(cov(sample.T))

If the parameters must be restricted to lie within certain limits, bounds can be
specified using the ``bounds`` keyword argument (see :doc:`mcmc_utilities`).

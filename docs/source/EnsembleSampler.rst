
EnsembleSampler
~~~~~~~~~~~~~~~

.. autoclass:: inference.mcmc.EnsembleSampler
   :members: advance, run_for, mode, get_marginal, get_sample, get_parameter, get_probabilities, get_interval, plot_diagnostics, matrix_plot, trace_plot, save, load


EnsembleSampler example code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because the proposed moves for each walker are generated using the positions of the
other walkers, ``EnsembleSampler`` automatically adapts to the scale and correlation
of the posterior, without any proposal widths needing to be tuned.

As in the :doc:`GibbsChain` example, define the Rosenbrock density as a test case:

.. code-block:: python

   from numpy import array

   def rosenbrock(t):
       X, Y = t
       X2 = X**2
       return -X2 - 15.*(Y - X2)**2 - 0.5*(X2 + Y**2) / 3.

The sampler needs a starting position for each walker, given as a 2D array with shape
``(n_walkers, n_parameters)``. There must be at least one more walker than there are
parameters, and the starting positions should be spread over a region of reasonably
high probability (for example, by randomly scattering them around an initial guess):

.. code-block:: python

   from numpy.random import default_rng

   n_walkers = 32
   rng = default_rng()
   starting_positions = array([0.0, 0.0]) + 0.2 * rng.normal(size=[n_walkers, 2])

Create the sampler object and advance it to generate a sample:

.. code-block:: python

   from inference.mcmc import EnsembleSampler
   sampler = EnsembleSampler(posterior=rosenbrock, starting_positions=starting_positions)
   sampler.advance(iterations=2000)

Each iteration moves every walker once and stores the new walker positions, so the
total number of samples generated is the number of iterations multiplied by the
number of walkers. The ``plot_diagnostics`` method shows the acceptance rate and
log-probability of the walkers as the sampler progresses:

.. code-block:: python

   sampler.plot_diagnostics()

The ``burn`` argument of the sampler methods counts individual samples, so to discard
the first 200 iterations of every walker as burn-in we multiply by the number of walkers:

.. code-block:: python

   burn = 200 * n_walkers
   sampler.matrix_plot(burn=burn)
   sample = sampler.get_sample(burn=burn)

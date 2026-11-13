Markov-chain Monte-Carlo sampling
=================================
This module provides Markov-Chain Monte-Carlo (MCMC) samplers which can
be easily applied to inference problems.

.. rubric:: Choosing a sampler

.. list-table::
   :header-rows: 1
   :widths: 20 40 40
   :class: sampler-table

   * - Sampler
     - Best suited to
     - Notes
   * - :doc:`GibbsChain`
     - A good general-purpose default for problems with a low-to-moderate number of
       parameters which are only weakly correlated.
     - Requires only the log-posterior. Supports bounds and non-negativity
       constraints on individual parameters.
   * - :doc:`PcaChain`
     - Posteriors with strong linear correlations between the parameters.
     - Requires only the log-posterior. Supports bounds on the parameters.
   * - :doc:`HamiltonianChain`
     - Problems with many parameters, or with strong non-linear correlations
       between the parameters.
     - Works best when the gradient of the log-posterior can be provided. If it is not,
       the gradient is estimated using finite differences, which requires additional
       posterior evaluations. Supports bounds on the parameters.
   * - :doc:`EnsembleSampler`
     - Posteriors which are poorly scaled or have linear correlations between
       the parameters, as no proposal distribution needs to be tuned.
     - Requires only the log-posterior. Uses many walkers, and so requires many
       posterior evaluations. Supports bounds on the parameters.
   * - :doc:`ParallelTempering`
     - Multi-modal posteriors with well-separated peaks.
     - Advances a set of the other chains at different 'temperatures' in parallel
       processes, allowing the chains to exchange positions. Use as a context
       manager to shut down the worker processes automatically.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   GibbsChain
   PcaChain
   HamiltonianChain
   EnsembleSampler
   ParallelTempering
   mcmc_utilities
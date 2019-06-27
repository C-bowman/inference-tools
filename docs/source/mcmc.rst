MCMC sampling with inference.mcmc
=================================
This module provides Markov-Chain Monte-Carlo (MCMC) samplers which can
be easily applied to inference problems.

.. _GibbsChain:

GibbsChain
~~~~~~~~~~

.. autoclass:: inference.mcmc.GibbsChain
   :members: advance, run_for, mode, get_marginal, get_sample, get_parameter, get_interval, plot_diagnostics, matrix_plot, trace_plot, set_non_negative, set_boundaries


.. _PcaChain:

PcaChain
~~~~~~~~

.. autoclass:: inference.mcmc.PcaChain
   :members: advance, run_for, mode, get_marginal, get_sample, get_parameter, get_interval, plot_diagnostics, matrix_plot, trace_plot, parameter_boundaries


.. _HamiltonianChain:

HamiltonianChain
~~~~~~~~~~~~~~~~

.. autoclass:: inference.mcmc.HamiltonianChain
   :members: advance, run_for, mode, get_marginal, get_parameter, plot_diagnostics, matrix_plot, trace_plot
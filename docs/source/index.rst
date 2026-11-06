The inference-tools package
===========================

Introduction
------------
This package aims to provide a set of python-based tools for Bayesian data analysis
which are simple to use, allowing them to be applied quickly and easily.

Inference tools is not a framework for building Bayesian/probabilistic models - instead it
provides tools to characterise arbitrary posterior distributions (given a function which
maps model parameters to a log-probability) via MCMC sampling.

This type of 'black-box' functionality allows for inference without the requirement of
first implementing the problem within a modelling framework.

Additionally, the package provides tools for analysing and plotting sampling results, as
well as implementations of some useful applications of Gaussian processes.

Features
--------

- Implementations of MCMC algorithms like Gibbs sampling and Hamiltonian Monte-Carlo for
  sampling from user-defined posterior distributions.
- Density estimation and plotting tools for analysing and visualising inference results.
- Gaussian-process regression and optimisation.

.. container:: gallery

   .. container:: gallery-item

      .. image:: ./images/gallery_images/gallery_gibbs_sampling.png
         :alt: Gibbs sampling
         :target: GibbsChain.html

      :doc:`Gibbs sampling <GibbsChain>`

   .. container:: gallery-item

      .. image:: ./images/gallery_images/gallery_hmc.png
         :alt: Hamiltonian Monte-Carlo
         :target: HamiltonianChain.html

      :doc:`Hamiltonian Monte-Carlo <HamiltonianChain>`

   .. container:: gallery-item

      .. image:: ./images/gallery_images/gallery_density_estimation.png
         :alt: Density estimation
         :target: pdf.html

      :doc:`Density estimation <pdf>`

   .. container:: gallery-item

      .. image:: ./images/getting_started_images/matrix_plot_example.png
         :alt: Matrix plotting
         :target: plotting.html#matrix-plot

      :ref:`Matrix plotting <plotting-matrix-plot>`

   .. container:: gallery-item

      .. image:: ./images/gallery_images/gallery_hdi.png
         :alt: Highest-density intervals
         :target: plotting.html#hdi-plot

      :ref:`Highest-density intervals <plotting-hdi-plot>`

   .. container:: gallery-item

      .. image:: ./images/gallery_images/gallery_gpr.png
         :alt: Gaussian-process regression
         :target: GpRegressor.html

      :doc:`Gaussian-process regression <GpRegressor>`

Installation
------------

inference-tools is available from `PyPI <https://pypi.org/project/inference-tools/>`_,
and can be installed using pip:

.. code-block:: bash

   pip install inference-tools

See :doc:`getting_started` for a worked example of using the package.

Getting involved
----------------

Requests for features/improvements can be made via the
`issue tracker <https://github.com/C-bowman/inference-tools/issues>`_. If you have questions
or are interested in getting involved with the development of this package, please contact
me at ``chris.bowman.physics@gmail.com``.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   mcmc
   distributions
   pdf
   gp
   approx
   plotting
   citing
   release_notes

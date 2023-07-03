The inference-tools package
===========================

Introduction
------------
This package aims to provide a set of python-based tools for Bayesian data analysis
which are simple to use, allowing them to applied quickly and easily.

Inference tools is not a framework for building Bayesian/probabilistic models - instead it
provides tools to characterise arbitrary posterior distributions (given a function which
maps model parameters to a log-probability) via MCMC sampling.

This type of 'black-box' functionality allows for inference without the requirement of
first implementing the problem within a modelling framework.

Additionally, the package provides tools for analysing and plotting sampling results, as
well as implementations of some useful applications of Gaussian processes.

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

Getting started
===============

.. _Installation:

Installation
------------

inference-tools is available from the python package index `PyPi <https://pypi.org/project/inference-tools/>`_, so can
be easily installed using `pip <https://pip.pypa.io/en/stable/>`_: as follows:

.. code-block::

   pip install inference-tools

If pip is not available, you can clone from the GitHub `source repository <https://github.com/C-bowman/inference-tools>`_
or download the files from `PyPi <https://pypi.org/project/inference-tools/>`_ directly.


Simple example - straight-line fitting
--------------------------------------

Generate some noisy straight-line data:

.. code-block:: python

   from numpy import linspace, zeros
   from numpy.random import normal
   x = linspace(0,5,6)
   errors = zeros(6)+0.5
   y = 2*x + 1 + normal(size=6)*errors

Define a simple class to evaluate the posterior log-probability:

.. code-block:: python

   class StraightLinePosterior(object):
      def __init__(self, x, y, errors):
         self.x = x
         self.y = y
         self.errors = errors

      def __call__(self, theta):
         m, c = theta
         prediction = m*self.x + c
         return -0.5*(((self.y - prediction)/sigma)**2).sum()

import one of the Markov-chain Monte-Carlo samplers from the mcmc module:

.. code-block:: python

   from inference.mcmc import GibbsChain
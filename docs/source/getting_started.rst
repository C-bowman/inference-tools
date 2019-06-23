Getting started
===============

.. _Installation:

Installation
------------

inference-tools is available from the python package index `PyPi <https://pypi.org/project/inference-tools/>`_, so can
be easily installed using `pip <https://pip.pypa.io/en/stable/>`_: as follows:

.. code-block:: bash

   pip install inference-tools

If pip is not available, you can clone from the GitHub `source repository <https://github.com/C-bowman/inference-tools>`_
or download the files from `PyPi <https://pypi.org/project/inference-tools/>`_ directly.


Example - spectroscopy data fitting
-----------------------------------

Here we work through a toy problem of fitting the following synthetic spectroscopy, which contains two
peaks with known centres and a linear background:

.. image:: spectroscopy_data.png

First let's define a class to evaluate the log-posterior:

.. code-block:: python

   class SpectroPosterior(object):
      def __init__(self, wavelength, intensity, errors):
         self.x = wavelength
         self.y = intensity
         self.sigma = errors
         # Central wavelengths of the lines are known constants:
         self.c1 = 422.
         self.c2 = 428.

      def __call__(self, theta):
         return self.likelihood(theta) # omitting prior term here means our prior is uniform everywhere

      def likelihood(self, theta):
         return -0.5*sum( ((self.y - self.forward_model(self.x, theta)) / self.sigma)**2 )

      def forward_model(self, x, theta):
         # unpack the model parameters
         A1, w1, A2, w2, b0, b1 = theta
         # evaluate each term of the model
         peak_1 = (A1 / (pi*w1)) / (1 + ((x - self.c1)/w1)**2)
         peak_2 = (A2 / (pi*w2)) / (1 + ((x - self.c2)/w2)**2)
         d = (b1-b0)/(max(x) - min(x))
         background = d*x + (b0 - d*min(x))
         # return the prediction of the data
         return peak_1 + peak_2 + background

Create an instance of the posterior class, and import one of the Markov-chain Monte-Carlo samplers from the mcmc module:

.. code-block:: python

   from inference.mcmc import PcaChain
   posterior = SpectroPosterior(wavelength, intensity, errors)

As a minimum the MCMC sampling classes in `inference.mcmc` must be provided with a
log-posterior function and a starting location for the chain:

.. code-block:: python

   chain = PcaChain(posterior = posterior, start = [1000, 1, 1000, 1, 30, 30])

The chain can be advanced for a given number of steps using the `advance` method:

.. code-block:: python

   chain.advance(20000)

We can view diagnostics which give useful information regarding the convergence of the
sample using the `plot_diagnostics` method:

.. code-block:: python

   chain.plot_diagnostics()

.. image:: plot_diagnostics_example.png
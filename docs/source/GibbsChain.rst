
GibbsChain
~~~~~~~~~~

.. autoclass:: inference.mcmc.GibbsChain
   :members: advance, run_for, mode, get_marginal, get_sample, get_parameter, get_interval, plot_diagnostics, matrix_plot, trace_plot, set_non_negative, set_boundaries


GibbsChain example code
^^^^^^^^^^^^^^^^^^^^^^^

Define the Rosenbrock density to use as a test case:

.. code-block:: python

   from numpy import linspace, exp
   import matplotlib.pyplot as plt

   def rosenbrock(t):
      X, Y = t
      X2 = X**2
      return -X2 - 15.*(Y - X2)**2 - 0.5*(X2 + Y**2)/3.

Create the chain object:

.. code-block:: python

   from inference.mcmc import GibbsChain
   chain = GibbsChain(posterior = rosenbrock, start = [2.,-4.])

Advance the chain 150k steps to generate a sample from the posterior:

.. code-block:: python

   chain.advance(150000)

The samples for any parameter can be accessed through the
``get_parameter`` method. We could use this to plot the path of
the chain through the 2D parameter space:

.. code-block:: python

   p = chain.get_probabilities() # color the points by their probability value
   plt.scatter(chain.get_parameter(0), chain.get_parameter(1), c = exp(p-max(p)), marker = '.')
   plt.xlabel('parameter 1')
   plt.ylabel('parameter 2')
   plt.grid()
   plt.show()

.. image:: ./images/GibbsChain_images/initial_scatter.png

We can see from this plot that in order to take a representative sample,
some early portion of the chain must be removed. This is referred to as
the 'burn-in' period. This period allows the chain to both find the high
density areas, and adjust the proposal widths to their optimal values.

The ``plot_diagnostics`` method can help us decide what size of burn-in to use:

.. code-block:: python

   chain.plot_diagnostics()

.. image:: ./images/GibbsChain_images/gibbs_diagnostics.png

Occasionally samples are also 'thinned' by a factor of n (where only every
n'th sample is used) in order to reduce the size of the data set for
storage, or to produce uncorrelated samples.

Based on the diagnostics we can choose to manually set a global burn and
thin value, which is used (unless otherwise specified) by all methods which
access the samples:

.. code-block:: python

   chain.burn = 2000
   chain.thin = 10

Alternatively, the chain objects be asked to select appropriate
burn and thin values automatically using the ``autoselect_burn_and_thin``
method:

.. code-block:: python

   chain.autoselect_burn_and_thin()


After discarding burn-in, what we have left should be a representative
sample drawn from the posterior. Methods for accessing sample data like
``get_probabilities`` and ``get_parameter`` automatically implement
the burning and thinning once they are set, using the same code as before
we now generate a different plot:

.. code-block:: python

   p = chain.get_probabilities() # color the points by their probability value
   plt.scatter(chain.get_parameter(0), chain.get_parameter(1), c = exp(p-max(p)), marker = '.')
   plt.xlabel('parameter 1')
   plt.ylabel('parameter 2')
   plt.grid
   plt.show()

.. image:: ./images/GibbsChain_images/burned_scatter.png

We can easily estimate 1D marginal distributions for any parameter
using the ``get_marginal`` method:

.. code-block:: python

   pdf_1 = chain.get_marginal(0, unimodal = True)
   pdf_2 = chain.get_marginal(1, unimodal = True)

``get_marginal`` returns a density estimator object, which can be called
as a function to return the value of the pdf at any point:

.. code-block:: python

   ax = linspace(-3, 4, 500) # axis on which to evaluate the marginal PDFs
   # plot the marginal distributions
   plt.plot( ax, pdf_1(ax), label = 'param #1 marginal', lw = 2)
   plt.plot( ax, pdf_2(ax), label = 'param #2 marginal', lw = 2)
   plt.xlabel('parameter value')
   plt.ylabel('probability density')
   plt.legend()
   plt.grid()
   plt.show()

.. image:: ./images/GibbsChain_images/gibbs_marginals.png

MCMC utilities
~~~~~~~~~~~~~~
The ``inference.mcmc`` module also provides some utility classes which can be used
alongside the samplers.

ChainPool
^^^^^^^^^

.. autoclass:: inference.mcmc.ChainPool
   :members: advance

Because ``ChainPool`` uses separate processes to advance the chains, scripts which
use it should protect their entry point with an ``if __name__ == "__main__":`` guard,
as required by the python ``multiprocessing`` module on some platforms:

.. code-block:: python

   from inference.mcmc import GibbsChain, ChainPool

   if __name__ == "__main__":
       chains = [GibbsChain(posterior=posterior, start=s) for s in start_points]
       pool = ChainPool(chains)
       pool.advance(10000)
       chains = pool.chains


Bounds
^^^^^^

.. autoclass:: inference.mcmc.Bounds

``Bounds`` can be used to restrict the parameters sampled by a chain to lie
within specified limits:

.. code-block:: python

   from numpy import array
   from inference.mcmc import HamiltonianChain, Bounds

   bounds = Bounds(lower=array([0., 0.]), upper=array([10., 5.]))
   chain = HamiltonianChain(posterior=posterior, start=array([1., 1.]), bounds=bounds)

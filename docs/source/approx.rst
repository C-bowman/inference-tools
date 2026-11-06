Approximate inference
=====================

The ``inference.approx`` module provides tools for quickly approximating a posterior
distribution without sampling it via MCMC. They are based on the 1D 'conditional'
distributions of the posterior - the distribution of each parameter when all the other
parameters are held fixed at a chosen 'conditioning point', which will typically
be the posterior mode (e.g. as found by an optimiser).

Evaluating the conditional distributions requires only a modest number of posterior
evaluations, so these tools are useful when the posterior is expensive to evaluate, or
when a quick estimate of the parameter uncertainties is needed. However, because the
conditionals ignore correlations between the parameters, they will tend to underestimate
the uncertainties when significant correlations are present.

- ``conditional_sample`` draws samples from each of the conditional distributions, and
  combines them to approximate samples from the posterior itself.
- ``get_conditionals`` evaluates each of the conditional distributions on a grid.
- ``conditional_moments`` calculates the mean and variance of each of the conditional
  distributions.


conditional_sample
------------------

.. autofunction:: inference.approx.conditional_sample

get_conditionals
----------------

.. autofunction:: inference.approx.get_conditionals


conditional_moments
-------------------

.. autofunction:: inference.approx.conditional_moments
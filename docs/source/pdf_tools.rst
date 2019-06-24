Sample analysis with inference.pdf_tools
========================================
This module provides tools for reconstructing the probability density function from which
a given set of samples was drawn.

.. _GaussianKDE:

GaussianKDE
~~~~~~~~~~~
The GaussianKDE class provides an estimate of a univariate PDF for a given set of sample data using
Gaussian kernel density estimation. The class also provides a method which returns the
highest-density interval for a chosen percentile.

.. autoclass:: inference.pdf_tools.GaussianKDE
   :members: __call__, interval, plot_summary, locate_mode

.. _UnimodalPdf:

UnimodalPdf
~~~~~~~~~~~
UnimodalPdf provides smooth estimates of univariate, unimodal PDFs by fitting an extremely flexible
parametric model to a given set of sample data. The class also provides a method which returns the
highest-density interval for a chosen percentile.

.. autoclass:: inference.pdf_tools.UnimodalPdf
   :members: __call__, interval, plot_summary, mode

.. _sample_hdi:

sample_hdi
~~~~~~~~~~~
.. autofunction:: inference.pdf_tools.sample_hdi
The pdf_tools module
====================
This module provides tools for reconstructing the probability density function from which
a given set of samples was drawn.

.. _GaussianKDE:

GaussianKDE
~~~~~~~~~~~
The GaussianKDE class provides an estimate of a univariate PDF for a given set of sample data using
Gaussian kernel density estimation. An estimate of the mode is calulated automatically, and like
UnimodalPdf, the credible interval for a given percentile is available through a method call.

A demonstration of how to use the UnimodalPdf class can be found in ``/demos/GaussianKDE_demo.py``

.. autoclass:: inference.pdf_tools.GaussianKDE
   :members: __call__, interval, plot_summary, locate_mode

.. _UnimodalPdf:

UnimodalPdf
~~~~~~~~~~~
UnimodalPdf provides smooth estimates of univariate, unimodal PDFs by fitting an extremely flexible
parametric model to a given set of sample data. The class also provides a method which returns the
credible interval for a chosen percentile.

A demonstration of how to use the UnimodalPdf class can be found in ``/demos/UnimodalPdf_demo.py``

.. autoclass:: inference.pdf_tools.UnimodalPdf
   :members: __call__, interval, plot_summary, mode
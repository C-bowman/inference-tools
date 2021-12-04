Density estimation and sample analysis
======================================
The ``inference.pdf`` module provides tools for analysing sample data, including density
estimation and highest-density interval calculation. Example code for ``GaussianKDE``
and ``UnimodalPdf`` can be found in the `density estimation jupyter notebook demo <https://github.com/C-bowman/inference-tools/blob/master/demos/density_estimation_demo.ipynb>`_.

.. _GaussianKDE:

GaussianKDE
~~~~~~~~~~~

.. autoclass:: inference.pdf.GaussianKDE
   :members: __call__, interval, plot_summary, mode

.. _UnimodalPdf:

UnimodalPdf
~~~~~~~~~~~

.. autoclass:: inference.pdf.UnimodalPdf
   :members: __call__, interval, plot_summary, mode

.. _sample_hdi:

sample_hdi
~~~~~~~~~~~
.. autofunction:: inference.pdf.sample_hdi
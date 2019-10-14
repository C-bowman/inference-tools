
Covariance functions
~~~~~~~~~~~~~~~~~~~~


SquaredExponential
^^^^^^^^^^^^^^^^^^

.. math::

   K(\underline{u}, \underline{v}) = a^2 \exp \left( -\frac{1}{2} \sum_i \left(\frac{u_i - v_i}{l_i}\right)^2 \right)

.. autoclass:: inference.gp_tools.SquaredExponential


RationalQuadratic
^^^^^^^^^^^^^^^^^

.. autoclass:: inference.gp_tools.RationalQuadratic
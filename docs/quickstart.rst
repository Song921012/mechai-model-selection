Quick start
===========

Install the package with ``python -m pip install -e .``. The package does not
solve an ODE itself. Compute a differentiable observation map with your solver,
then pass its Jacobian or an information matrix to the geometry routines.

.. literalinclude:: ../examples/basic_geometry.py
   :language: python

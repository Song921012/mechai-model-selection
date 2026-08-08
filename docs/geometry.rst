Geometry and metric choice
==========================

For an information matrix ``G`` and reference metric ``R``, the package solves
``G v = mu R v``. At resolution ``lambda``, the effective dimension is the sum
of ``mu / (mu + lambda)`` and the relative log-volume is the sum of
``log(1 + mu/lambda)``.

``fisher_pullback`` is appropriate when distinguishability is defined by a
likelihood. ``wasserstein_pullback_1d`` measures transport of a normalized
one-dimensional distribution. These metrics answer different scientific
questions and should not be selected by which produces the preferred model.

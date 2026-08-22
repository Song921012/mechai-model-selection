Limitations
===========

The package reports local geometry of a specified statistical model manifold.
It does not establish global identifiability, universal model-selection
consistency, post-selection confidence coverage, exact marginal likelihood, or
the real log canonical threshold of a singular model.

gic_predictive requires a valid local information identity; under
misspecification, use the general sandwich complexity. gic_evidence is a
single-mode local Gaussian approximation and should be accompanied by posterior
or importance-sampling diagnostics in weakly identified models. Metric choice
must be fixed by the scientific loss, not by the preferred ranking.

Full Jacobian construction and dense eigendecomposition may be unsuitable for
large neural networks. Matrix-free approximations are not part of version
0.3.1.

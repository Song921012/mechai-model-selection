Derived criteria
================

The package distinguishes predictive risk from model evidence.

General loss-matched optimism
-----------------------------

For a penalized M-estimator, let A0 be the expected Hessian of the unpenalized
empirical loss, R the reference metric, lambda the fitted penalty strength, and
B the score covariance. The generalized_optimism function returns
trace((A0 + lambda R)^{-1} B). The corresponding criterion adds twice this
quantity to empirical deviance.

Predictive criterion
--------------------

When the likelihood is locally correct and the information identity applies,
B = G and A0 + lambda R = G + lambda R. The sandwich complexity becomes the effective
dimension. The gic_predictive function therefore estimates out-of-sample
deviance by adding 2 d_eff to the fitted deviance.

Evidence criterion
------------------

With the normalized Gaussian reference prior N(m, (lambda R)^{-1}), a local
Laplace approximation gives the relative log-volume
log det(G + lambda R) - log det(lambda R). The gic_evidence function adds this
term and the fitted prior energy to the deviance.

BIC-type approximation
----------------------

If resolved generalized eigenvalues grow linearly with sample size while the
remaining modes stay below the declared resolution, relative log-volume has
leading term rank times log(n). The gic_bic_approximation function implements
the finite-resolution approximation log(n) d_eff. It is auxiliary to the
predictive and evidence criteria, not an independent decision principle.

The penalty matrix and resolution must match the estimator or normalized prior.
A metric should likewise match the loss whose future risk is being estimated.

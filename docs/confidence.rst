Confidence and model averaging
==============================

``geometric_confidence_region`` defines a local geometric confidence region on the
identifiable quotient, represented numerically by the resolved generalized
eigenspace. Rank reduction does not guarantee Gaussian
coverage under weak identification. ``coverage_summary`` reports Wilson
intervals for repeated-sampling checks.

``model_average`` accepts scalar or trajectory-valued means and variances. It
returns within-model, between-model, and total predictive variance, with an
optional Gaussian interval. Criterion weights summarize a declared score;
``stacking_weights`` instead optimizes held-out predictive density.

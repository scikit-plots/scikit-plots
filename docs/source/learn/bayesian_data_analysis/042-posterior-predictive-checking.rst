.. _bda-posterior-predictive-checking:

========================================================================
Posterior predictive checking
========================================================================

**Part 2 · Stage 6 · 🔍 Model Checking & Comparison**  ·  Lesson 042 of 144  ·  *intermediate*

:doc:`◀ Previous · Do the Inferences from the Model Make Sense? <041-do-the-inferences-from-the-model-make-sense>`   ·   :doc:`Next · Graphical posterior predictive checks ▶ <043-graphical-posterior-predictive-checks>`   ·   :doc:`↑ Section <index>`


Let the model generate data
-----------------------------

The central technique of Part II is disarmingly simple: **if the model fits, data simulated from it
should look like the data you actually saw.** Simulate many replicated datasets from the fitted model,
compare them to the real one, and any **systematic** difference is a failure of the model.

The posterior predictive distribution
---------------------------------------

Replicated data :math:`y^{\text{rep}}` are drawn from the **posterior predictive distribution** —
the likelihood averaged over posterior uncertainty in the parameters:

.. math::

   p(y^{\text{rep}} \mid y) = \int p(y^{\text{rep}} \mid \theta) \; p(\theta \mid y) \; d\theta .

In simulation this is one line per draw: for each posterior draw :math:`\theta^{(s)}`, generate a full
dataset :math:`y^{\text{rep}(s)} \sim p(y \mid \theta^{(s)})`. Note what is averaged in — parameter
uncertainty **and** sampling variability, exactly the two sources present in the real data.

Test quantities
-----------------

You cannot compare whole datasets by eye at scale, so choose a **test quantity** :math:`T(y, \theta)`
summarising the feature you care about: the maximum, the number of zeros, a lag-1 autocorrelation, the
variance-to-mean ratio. Unlike a classical test statistic, :math:`T` may depend on the **parameters**.
The **posterior predictive** :math:`p`-value is the probability that the replicated data are more
extreme than what you observed:

.. math::

   p_B = \Pr\bigl(T(y^{\text{rep}}, \theta) \ge T(y, \theta) \;\bigm|\; y\bigr)
       \;\approx\; \frac{1}{S} \sum_{s=1}^{S}
         \mathbf{1}\bigl\{T(y^{\text{rep}(s)}, \theta^{(s)}) \ge T(y, \theta^{(s)})\bigr\} .

Values near 0 or 1 mean the observed data are **atypical** of the model in that respect.

.. code-block:: python

   import numpy as np, pymc as pm
   with model:
       ppc = pm.sample_posterior_predictive(idata)
   yrep = ppc.posterior_predictive["y"].values.reshape(-1, len(y))

   T = lambda d: d.max()                                # any feature you care about
   p_B = (np.array([T(r) for r in yrep]) >= T(y)).mean()
   p_B                                                  # ≈ 0.5: unremarkable; ≈ 0 or 1: misfit

Read it as a diagnostic, not a verdict
----------------------------------------

Two honest caveats. The check **uses the data twice** — to fit and to test — so :math:`p_B` is
conservative, tending to concentrate near 0.5; it is not calibrated like a frequentist :math:`p`-value
and is not "the probability the model is true". And a check can only find what its test quantity looks
for: choosing :math:`T` = the sample mean will almost always pass, because the model was fitted to match
the mean. **Choose test quantities the model does not automatically reproduce**, and ones that matter for
your purpose. The point is never to accept or reject, but to learn **where** the model misses.

.. hint::

   **Related lessons:** :doc:`The Place of Model Checking in Applied Bayesian Statistics <040-the-place-of-model-checking-in-applied-bayesian-statistics>`  ·  :doc:`Graphical posterior predictive checks <043-graphical-posterior-predictive-checks>`  ·  :doc:`Model checking for the educational testing example <044-model-checking-for-the-educational-testing-example>`  ·  :doc:`Measures of predictive accuracy <045-measures-of-predictive-accuracy>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/10/posterior-predictive-checking/ <https://insightful-data-lab.com/2025/11/10/posterior-predictive-checking/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate

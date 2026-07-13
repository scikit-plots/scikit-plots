.. _bda-aspects-of-robustness:

========================================================================
Aspects of robustness
========================================================================

**Part 4 · Stage 14 · 🛡️ Robustness & Missing Data**  ·  Lesson 113 of 144  ·  *advanced*

:doc:`◀ Previous · Loglinear models for multivariate discrete data <112-loglinear-models-for-multivariate-discrete-data>`   ·   :doc:`Next · Overdispersed versions of standard models ▶ <114-overdispersed-versions-of-standard-models>`   ·   :doc:`↑ Section <index>`


When the model is wrong in a particular way
---------------------------------------------

Every model is an approximation, but some approximations fail gracefully and others catastrophically.
**Robustness** is the study of how sensitive a conclusion is to assumptions that are not quite true —
and, more usefully, how to build models whose conclusions survive the failures that matter most. The
recurring culprit is the **normal likelihood** and its intolerance of outliers.

Why the normal is fragile
---------------------------

The normal distribution has **light tails**: its density falls off like :math:`e^{-x^2/2}`, so a value
five standard deviations out is astronomically improbable. When a real observation lands there anyway —
a recording error, a genuine anomaly — the normal model cannot treat it as a rare event. It must instead
**move the mean** to accommodate the point, because under the normal likelihood the far-out value is so
costly that shifting every estimate is cheaper than tolerating it. A single outlier can drag the
posterior a long way, and the sample mean, its estimator, is the classic non-robust statistic.

The Bayesian view of robustness
---------------------------------

Robustness in this framework is not a patch applied afterward; it is a **modelling choice about the
likelihood and prior**. Three questions organise it:

* **Likelihood robustness** — does one aberrant observation dominate the fit? The cure is a
  heavy-tailed likelihood, the next two lessons.
* **Prior robustness** — does the conclusion hinge on a specific prior? The check is to vary the prior
  and watch the posterior, the sensitivity analysis of Stage 6.
* **Structural robustness** — does it depend on the model's form? Answered by model expansion and
  predictive comparison.

The general principle
-----------------------

Heavy tails buy robustness. A likelihood or prior with **heavier tails than the normal** can assign a
far-out point reasonable probability *as an outlier*, so the point no longer forces the bulk of the model
to move — the estimate is determined by the mass of the data, not held hostage by its extremes. This is
the same insight that motivated the Cauchy prior for separation: heavy tails let a model **tolerate** the
unusual instead of contorting to explain it. The lessons that follow make it concrete for the
likelihood, where outliers do their damage.

One caution
-------------

Robustness is not free and not total. A heavy-tailed likelihood down-weights outliers automatically, but
it is only **partially** robust — an outlier in a *predictor* can still distort a regression, and
down-weighting a genuine signal as if it were noise loses information. Robustness is about surviving the
assumptions most likely to be wrong, not immunity to all of them; naming which failure you are guarding
against is part of the modelling.

.. hint::

   **Related lessons:** :doc:`Overdispersed versions of standard models <114-overdispersed-versions-of-standard-models>`  ·  :doc:`Continuous model expansion <048-continuous-model-expansion>`  ·  :doc:`Weakly informative priors for logistic regression <108-weakly-informative-priors-for-logistic-regression>`  ·  :doc:`Robust inference for the eight schools <116-robust-inference-for-the-eight-schools>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/07/aspects-of-robustness/ <https://insightful-data-lab.com/2025/12/07/aspects-of-robustness/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced

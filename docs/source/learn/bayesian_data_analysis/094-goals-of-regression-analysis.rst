.. _bda-goals-of-regression-analysis:

========================================================================
Goals of regression analysis
========================================================================

**Part 4 · Stage 11 · 📈 Regression Foundations**  ·  Lesson 094 of 144  ·  *advanced*

:doc:`◀ Previous · Regression for causal inference: incumbency and voting <093-regression-for-causal-inference-incumbency-and-voting>`   ·   :doc:`Next · Assembling the matrix of explanatory variables ▶ <095-assembling-the-matrix-of-explanatory-variables>`   ·   :doc:`↑ Section <index>`


Three questions, one equation
-------------------------------

The same fitted regression can serve three purposes, and confusing them is the most common error in
applied statistics. Ask which one you are pursuing **before** choosing predictors, priors, or a way to
evaluate the fit.

Prediction
------------

You want :math:`p(\tilde{y} \mid \tilde{x}, y)` to be accurate for new cases. Nothing else matters:
predictors may be proxies, collinear, or scientifically meaningless. **Include** anything that predicts,
regularise heavily, and evaluate by **out-of-sample predictive accuracy** (elpd/LOO, Stage 6).
Interpreting an individual coefficient is beside the point — and with collinear predictors, impossible.

Description and explanation
-----------------------------

You want to characterise **associations** in the population: how does income vary with education, given
age? Here coefficients are the output, so their **interpretation must be defensible**: they are
comparisons between units that differ in one predictor and agree on the others. No causal claim is
implied. Adjust for what makes the comparison meaningful, and beware that "controlling for" a variable
changes the meaning of every other coefficient.

Causal inference
------------------

You want the effect of **intervening** on a predictor. This is a different quantity — it concerns
potential outcomes, not conditional expectations — and it demands the machinery of Stage 7: a design, or
an assumption that assignment is ignorable given the covariates. Only some coefficients in a regression
are causal, and typically only the one you designed for.

.. code-block:: python

   import arviz as az
   az.loo(idata)                  # prediction: which model forecasts best?
   az.summary(idata, var_names=["beta"])   # description: which comparisons, with what uncertainty?
   # causation: the number above is causal only if the DESIGN says so

Choices that depend on the goal
---------------------------------

The purpose determines everything downstream:

* **Which predictors.** Prediction: everything useful. Description: what makes comparisons sensible.
  Causation: confounders in, **mediators and colliders out** — adjusting for a variable on the causal
  path removes part of the effect, and adjusting for a collider **creates** bias.
* **Which prior.** Prediction tolerates strong shrinkage; causal estimands usually want a weakly
  informative prior on the coefficient of interest so the data speak.
* **Which check.** Prediction: LOO. Description: posterior predictive checks. Causation: sensitivity to
  unmeasured confounding.

A model excellent for one goal can be worthless for another. The equation does not know which question
you are asking; **you must**.

.. hint::

   **Related lessons:** :doc:`Conditional modeling <091-conditional-modeling>`  ·  :doc:`Regression for causal inference: incumbency and voting <093-regression-for-causal-inference-incumbency-and-voting>`  ·  :doc:`Assembling the matrix of explanatory variables <095-assembling-the-matrix-of-explanatory-variables>`  ·  :doc:`Model comparison based on predictive performance <046-model-comparison-based-on-predictive-performance>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/23/goals-of-regression-analysis/ <https://insightful-data-lab.com/2025/11/23/goals-of-regression-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced

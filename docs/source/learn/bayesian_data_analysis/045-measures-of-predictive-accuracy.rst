.. _bda-measures-of-predictive-accuracy:

========================================================================
Measures of predictive accuracy
========================================================================

**Part 2 · Stage 6 · 🔍 Model Checking & Comparison**  ·  Lesson 045 of 144  ·  *intermediate*

:doc:`◀ Previous · Model checking for the educational testing example <044-model-checking-for-the-educational-testing-example>`   ·   :doc:`Next · Model comparison based on predictive performance ▶ <046-model-comparison-based-on-predictive-performance>`   ·   :doc:`↑ Section <index>`


Scoring a model by what it predicts
-------------------------------------

Checking asks whether a model fits. **Comparison** asks which of several models to prefer, and the
Bayesian answer is: the one that would **predict new data best**. That requires a score. The default
is the **log predictive density** — reward a model for putting high probability where the data actually
land, and it is *proper*, meaning it cannot be gamed by reporting a distribution you do not believe.

Expected log pointwise predictive density
-------------------------------------------

For future observations, the target quantity is the **elpd**, the expected log predictive density
summed over data points:

.. math::

   \mathrm{elpd} = \sum_{i=1}^{n} \int p_t(\tilde{y}_i) \,
                   \log p(\tilde{y}_i \mid y) \; d\tilde{y}_i ,

where :math:`p_t` is the (unknown) true data-generating process. Since :math:`p_t` is unavailable, elpd
must be **estimated**. The naive estimate uses the observed data themselves:

.. math::

   \mathrm{lppd} = \sum_{i=1}^{n} \log \left( \frac{1}{S} \sum_{s=1}^{S}
                   p(y_i \mid \theta^{(s)}) \right) .

This is the **log pointwise predictive density**, and it is **optimistic**: the same data fitted the
model, so the score is inflated by exactly the amount the model overfit.

Correcting the optimism
-------------------------

Information criteria subtract an estimate of that overfitting. **WAIC** uses the posterior variance of
the pointwise log-likelihood as an **effective number of parameters** :math:`p_{\text{WAIC}}`:

.. math::

   \widehat{\mathrm{elpd}}_{\text{WAIC}} = \mathrm{lppd} - p_{\text{WAIC}},
   \qquad
   p_{\text{WAIC}} = \sum_{i=1}^{n} \mathrm{var}_{s}\bigl(\log p(y_i \mid \theta^{(s)})\bigr).

Unlike AIC and DIC — which count parameters, or use a point estimate — WAIC is **fully Bayesian**: it
averages over the posterior and works when the effective number of parameters is not obvious, as in
hierarchical models where partial pooling makes :math:`J` groups behave like far fewer.

.. code-block:: python

   import arviz as az
   az.waic(idata)        # elpd_waic, p_waic, se
   az.loo(idata)         # elpd_loo — usually preferred (next lesson)

Pointwise is the point
------------------------

Note that every quantity here is a **sum over observations**. That decomposition is what makes standard
errors available (the sd of the :math:`n` components times :math:`\sqrt{n}`), what lets you see **which
observations** a model predicts badly, and what makes leave-one-out cross-validation — the more direct
estimate of the same elpd — computable from a single fit.

.. hint::

   **Related lessons:** :doc:`Posterior predictive checking <042-posterior-predictive-checking>`  ·  :doc:`Model comparison based on predictive performance <046-model-comparison-based-on-predictive-performance>`  ·  :doc:`Model comparison using Bayes factors <047-model-comparison-using-bayes-factors>`  ·  :doc:`Bayesian interpretations of other statistical methods <032-bayesian-interpretations-of-other-statistical-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/10/measures-of-predictive-accuracy/ <https://insightful-data-lab.com/2025/11/10/measures-of-predictive-accuracy/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate

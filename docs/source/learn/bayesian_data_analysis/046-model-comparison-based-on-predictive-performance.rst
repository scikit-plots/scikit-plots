.. _bda-model-comparison-based-on-predictive-performance:

========================================================================
Model comparison based on predictive performance
========================================================================

**Part 2 · Stage 6 · 🔍 Model Checking & Comparison**  ·  Lesson 046 of 144  ·  *intermediate*

:doc:`◀ Previous · Measures of predictive accuracy <045-measures-of-predictive-accuracy>`   ·   :doc:`Next · Model comparison using Bayes factors ▶ <047-model-comparison-using-bayes-factors>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Cross-validation, done cheaply
--------------------------------

WAIC corrects the optimism of in-sample scoring with an analytic penalty. **Leave-one-out
cross-validation** estimates the same elpd more directly: predict each observation from a posterior
fitted without it,

.. math::

   \mathrm{elpd}_{\text{loo}} = \sum_{i=1}^{n} \log p(y_i \mid y_{-i}) .

Done literally this means refitting :math:`n` times. The practical alternative reuses the **single**
posterior you already have, reweighting its draws by :math:`1/p(y_i \mid \theta^{(s)})` — importance
sampling — to approximate the leave-one-out posterior.

Pareto-smoothed importance sampling
-------------------------------------

Raw importance weights are unstable: a single influential observation gives one draw enormous weight.
**PSIS-LOO** (Vehtari, Gelman and Gabry) stabilises them by fitting a generalised Pareto distribution to
the largest weights and replacing them with smoothed order statistics. The fitted **shape parameter**
:math:`\hat{k}` is a free diagnostic, per observation:

* :math:`\hat{k} < 0.5` — the estimate is reliable;
* :math:`0.5 \le \hat{k} < 0.7` — acceptable, but the observation is influential;
* :math:`\hat{k} \ge 0.7` — the approximation may be **unreliable**; refit exactly for those points, or
  use :math:`k`-fold.

A high :math:`\hat{k}` is itself informative: it names an observation the model finds surprising.

.. code-block:: python

   import arviz as az
   loo_a, loo_b = az.loo(idata_a, pointwise=True), az.loo(idata_b, pointwise=True)
   az.compare({"simple": idata_a, "hierarchical": idata_b})   # elpd_diff, se_diff, weights
   (loo_a.pareto_k > 0.7).sum()                               # observations to worry about

Comparing honestly
--------------------

Report the **difference** in elpd together with the standard error of that difference — computed from
the pointwise components, which is far more precise than differencing two noisy totals. A rough rule:
if `elpd_diff` is not several times `se_diff`, the data do not distinguish the models. And the
comparison is **relative**: LOO ranks the candidates you supply and says nothing about whether the best
of them is any good. That question belongs to posterior predictive checking.

Two limits worth naming. LOO presupposes that predicting a **new observation** is the relevant task; for
time series or grouped data, leave-**future**-out or leave-one-**group**-out is the honest analogue. And
selecting a model by LOO from a large set reintroduces overfitting — to the selection criterion itself.
Where models are many and similar, **averaging** them (stacking) usually beats picking one.

.. hint::

   **Related lessons:** :doc:`Measures of predictive accuracy <045-measures-of-predictive-accuracy>`  ·  :doc:`Model comparison using Bayes factors <047-model-comparison-using-bayes-factors>`  ·  :doc:`Continuous model expansion <048-continuous-model-expansion>`  ·  :doc:`Graphical posterior predictive checks <043-graphical-posterior-predictive-checks>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/10/model-comparison-based-on-predictive-performance/ <https://insightful-data-lab.com/2025/11/10/model-comparison-based-on-predictive-performance/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate

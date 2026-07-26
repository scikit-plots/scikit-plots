.. _bda-regression-coefficients-exchangeable-in-batches:

========================================================================
Regression coeﬃcients exchangeable in batches
========================================================================

**Part 4 · Stage 12 · 🏗️ Hierarchical Regression**  ·  Lesson 099 of 144  ·  *advanced*

:doc:`◀ Previous · Including numerical prior information <098-including-numerical-prior-information>`   ·   :doc:`Next · Example: forecasting U.S. presidential elections ▶ <100-example-forecasting-u-s-presidential-elections>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Structure among the coefficients
----------------------------------

Ordinary regression treats coefficients as unrelated unknowns, each with its own flat prior. Often they
are not unrelated: a factor with many levels produces a **batch** of coefficients that are
**exchangeable** — the fifty state effects, the coefficients for twenty indicator categories, a set of
interactions. Treating a batch as exchangeable means giving its members a **common prior** whose
parameters are estimated, which is the hierarchical idea of Stage 5 applied inside a regression.

The model
-----------

Partition :math:`\beta` into batches. Within batch :math:`b`, the coefficients share a distribution:

.. math::

   \beta_j \sim \mathrm{N}(\mu_b, \tau_b^2) \quad \text{for } j \in \text{batch } b,
   \qquad \tau_b \sim \text{half-normal},

with :math:`\tau_b` — the batch's spread — **inferred from the data**. This is exactly a varying-intercept
model written in regression notation: each batch is a grouping factor, and :math:`\tau_b` controls how
much its coefficients are **pooled** toward the batch mean.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # fixed effects: their own weak priors
       gamma = pm.Normal("gamma", 0, 5, shape=n_fixed)
       # a batch of exchangeable coefficients: shared, inferred mean and scale
       mu_b = pm.Normal("mu_b", 0, 5)
       tau = pm.HalfNormal("tau", 1)
       z = pm.Normal("z", 0, 1, shape=n_batch)              # non-centred
       beta = pm.Deterministic("beta", mu_b + tau * z)      # pooled toward mu_b
       mu = Xf @ gamma + Xb @ beta
       pm.Normal("y", mu, pm.HalfNormal("s", 1), observed=y)

What the pooling buys
-----------------------

The batch scale :math:`\tau_b` is learned, so the amount of shrinkage is **adaptive**, exactly as in
Stage 5. A batch whose coefficients genuinely vary gets a large :math:`\tau_b` and little pooling; a
batch indistinguishable from noise gets a small :math:`\tau_b` and is shrunk hard toward its mean. The
data decide, per batch. This is far better than the two fixed alternatives: **no pooling** (ordinary
indicators, :math:`\tau_b = \infty`) overfits when levels are many and data per level are thin, while
**complete pooling** (:math:`\tau_b = 0`) ignores real differences.

Where it appears
------------------

The batched view organises much of applied modelling: the levels of every categorical predictor, the
coefficients of a spline basis (Stage 15), the many interactions in a deep model, varying slopes across
groups. Treating each such set as an exchangeable batch with its own variance is the unifying move of
this stage — and the varying-intercept, varying-slope, and ANOVA lessons that follow are all special
cases of it.

.. hint::

   **Related lessons:** :doc:`Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`  ·  :doc:`Example: forecasting U.S. presidential elections <100-example-forecasting-u-s-presidential-elections>`  ·  :doc:`Varying intercepts and slopes <102-varying-intercepts-and-slopes>`  ·  :doc:`Analysis of variance and the batching of coeﬃcients <104-analysis-of-variance-and-the-batching-of-coefficients>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/24/regression-coe%ef%ac%83cients-exchangeable-in-batches/ <https://insightful-data-lab.com/2025/11/24/regression-coe%ef%ac%83cients-exchangeable-in-batches/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced

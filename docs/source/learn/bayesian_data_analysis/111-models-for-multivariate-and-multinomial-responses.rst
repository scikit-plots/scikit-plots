.. _bda-models-for-multivariate-and-multinomial-responses:

========================================================================
Models for multivariate and multinomial responses
========================================================================

**Part 4 · Stage 13 · 🔗 Generalized Linear Models**  ·  Lesson 111 of 144  ·  *advanced*

:doc:`◀ Previous · State-level opinons from national polls <110-state-level-opinons-from-national-polls>`   ·   :doc:`Next · Loglinear models for multivariate discrete data ▶ <112-loglinear-models-for-multivariate-discrete-data>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

When the outcome is not a single number
-----------------------------------------

So far each observation had one response. Two common cases break that: a **multinomial** outcome, where
the response is one of several unordered (or ordered) categories, and a **multivariate** outcome, where
several correlated responses are measured together. Both extend the GLM, and both hinge on modelling the
**structure among the outcomes**.

Multinomial and categorical
-----------------------------

For an **unordered** categorical outcome — choice of transport, party, product — the multinomial
(softmax) logit gives each category :math:`k` its own linear predictor and normalises:

.. math::

   \Pr(y_i = k) = \frac{e^{X_i \beta_k}}{\sum_{k'} e^{X_i \beta_{k'}}},

with one category fixed as baseline for identifiability. For an **ordered** outcome — a Likert scale, a
severity grade — the **ordered logit** is better: a single linear predictor with a set of increasing
**cutpoints** slicing a latent continuum, which respects the ordering and spends far fewer parameters.

.. code-block:: python

   import pymc as pm
   # ordered categorical: shared slope, learned cutpoints
   with pm.Model():
       beta = pm.Normal("beta", 0, 1, shape=k)
       cut = pm.Normal("cut", 0, 5, shape=n_cat - 1,
                       transform=pm.distributions.transforms.ordered)   # increasing
       pm.OrderedLogistic("y", eta=X @ beta, cutpoints=cut, observed=y)

Multivariate responses
------------------------

When several responses are measured per unit — height and weight, a battery of test scores — they are
**correlated**, and modelling them jointly is more informative than one regression each. Give the vector
of responses a multivariate normal likelihood with a covariance to estimate, using the **LKJ**
decomposition from the hierarchy stage:

.. math::

   y_i \sim \mathrm{N}(X_i B, \; \Sigma), \qquad
   \Sigma = \mathrm{diag}(\sigma)\,\Omega\,\mathrm{diag}(\sigma), \quad \Omega \sim \mathrm{LKJ}(\eta).

The estimated correlations are often the point — how the outcomes move together, given the predictors.

Why model the joint structure
-------------------------------

Two payoffs. Modelling outcomes jointly **borrows information across them**: a missing component is
predicted from the others through the learned correlation, and estimates are more efficient than
separate fits. And the **dependence itself is substantive** — the correlation between test scores given
background, the covariance of symptoms — a quantity separate univariate models simply cannot report.
Categorical and multivariate responses complete the GLM's reach: with them, the framework covers binary,
count, ordered, unordered and vector-valued outcomes under one set of tools.

.. hint::

   **Related lessons:** :doc:`Standard generalized linear model likelihoods <106-standard-generalized-linear-model-likelihoods>`  ·  :doc:`Loglinear models for multivariate discrete data <112-loglinear-models-for-multivariate-discrete-data>`  ·  :doc:`Multivariate Normal with Unknown Mean and Variance <025-multivariate-normal-with-unknown-mean-and-variance>`  ·  :doc:`Working with generalized linear models <107-working-with-generalized-linear-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/06/models-for-multivariate-and-multinomial-responses/ <https://insightful-data-lab.com/2025/12/06/models-for-multivariate-and-multinomial-responses/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced

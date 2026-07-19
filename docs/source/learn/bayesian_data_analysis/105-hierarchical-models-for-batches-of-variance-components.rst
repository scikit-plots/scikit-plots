.. _bda-hierarchical-models-for-batches-of-variance-components:

========================================================================
Hierarchical models for batches of variance components
========================================================================

**Part 4 · Stage 12 · 🏗️ Hierarchical Regression**  ·  Lesson 105 of 144  ·  *advanced*

:doc:`◀ Previous · Analysis of variance and the batching of coeﬃcients <104-analysis-of-variance-and-the-batching-of-coefficients>`   ·   :doc:`Next · Standard generalized linear model likelihoods ▶ <106-standard-generalized-linear-model-likelihoods>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

When the variances themselves have structure
-----------------------------------------------

The ANOVA lesson gave every source of variation its own standard deviation: :math:`\sigma_A`,
:math:`\sigma_B`, :math:`\sigma_{AB}`, and so on. A complex design can have **many** such variance
components — crossed and nested factors, multiway interactions — and they are not unrelated. When there
are enough of them, the natural move is the one this whole part has been making: treat the **variance
components as a batch** and model them hierarchically too.

A hierarchy on the standard deviations
----------------------------------------

Give the collection of batch scales a common prior with its own parameters:

.. math::

   y_i = \mu + \sum_{m} \beta^{(m)}_{j_m[i]} + \epsilon_i, \qquad
   \beta^{(m)}_j \sim \mathrm{N}(0, \sigma_m^2), \qquad
   \sigma_m \sim \text{half-}\mathrm{N}(0, \tau^2),

where each :math:`\sigma_m` is the spread of variance-component batch :math:`m`, and the
:math:`\sigma_m` are themselves drawn from a shared distribution with hyperscale :math:`\tau`. The
variance components are now **partially pooled** toward each other — a component estimated from few levels
borrows from the others, exactly as the group means did one level down.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       tau = pm.HalfNormal("tau", 1)                        # hyperscale over components
       sigma = pm.HalfNormal("sigma", tau, shape=n_components)   # batch of scales, pooled
       # each factor's effects drawn from its own (pooled) scale, non-centred
       effects = [pm.Normal(f"b{m}", 0, 1, shape=n_levels[m]) * sigma[m]
                  for m in range(n_components)]
       mu_i = mu + sum(e[idx[m]] for m, e in enumerate(effects))
       pm.Normal("y", mu_i, pm.HalfNormal("sy", 1), observed=y)

Why pool variances
--------------------

The same logic that pools means, one storey higher. A variance component estimated from **few levels** is
badly determined on its own — a two-level factor gives almost no information about its own spread — and
pooling toward the other components stabilises it. This matters most in the designs where classical ANOVA
is least reliable: many factors, few levels each, unbalanced cells. The estimate of any one
:math:`\sigma_m` improves by borrowing from the ensemble.

The finite-population reminder
--------------------------------

The distinction from the ANOVA lesson persists at this level. For a factor with a handful of levels, the
**finite-population** standard deviation — the spread of the effects actually present — is often the
meaningful summary, and it is estimated more precisely than the **superpopulation** scale, which
describes hypothetical new levels. Report the one that answers the question. This closes the hierarchical
regression stage: the batching idea, applied first to coefficients and then to their variances, turns a
tangle of factors into one coherent model whose every scale is estimated with appropriate pooling. Part
IV now leaves the normal likelihood behind.

.. hint::

   **Related lessons:** :doc:`Analysis of variance and the batching of coeﬃcients <104-analysis-of-variance-and-the-batching-of-coefficients>`  ·  :doc:`Regression coeﬃcients exchangeable in batches <099-regression-coefficients-exchangeable-in-batches>`  ·  :doc:`Weakly Informative Priors for Variance Parameters <039-weakly-informative-priors-for-variance-parameters>`  ·  :doc:`Varying intercepts and slopes <102-varying-intercepts-and-slopes>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/06/hierarchical-models-for-batches-of-variance-components/ <https://insightful-data-lab.com/2025/12/06/hierarchical-models-for-batches-of-variance-components/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced

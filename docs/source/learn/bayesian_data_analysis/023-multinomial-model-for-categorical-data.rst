.. _bda-multinomial-model-for-categorical-data:

========================================================================
Multinomial Model for Categorical Data
========================================================================

**Part 1 · Stage 3 · 🧮 Multiparameter Models**  ·  Lesson 023 of 144  ·  *beginner*

:doc:`◀ Previous · Normal Data with a Conjugate Prior Distribution <022-normal-data-with-a-conjugate-prior-distribution>`   ·   :doc:`Next · Multivariate Normal Model with Known Variance ▶ <024-multivariate-normal-model-with-known-variance>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Beyond two categories
-----------------------

Generalise the binomial from two outcomes to :math:`k`. Each observation falls in exactly one of
:math:`k` categories with probabilities :math:`\theta = (\theta_1, \dots, \theta_k)`,
:math:`\sum_j \theta_j = 1`, and the counts :math:`y = (y_1, \dots, y_k)` follow a **multinomial**
likelihood:

.. math::

   p(y \mid \theta) \;\propto\; \prod_{j=1}^{k} \theta_j^{\,y_j} .

Poll responses, survey categories, and the components of the mixture models in Part V all live here.

The Dirichlet prior
---------------------

The conjugate prior on the simplex is the **Dirichlet**, the Beta's multivariate sibling:
:math:`\theta \sim \mathrm{Dirichlet}(\alpha_1, \dots, \alpha_k)`, with density proportional to
:math:`\prod_j \theta_j^{\,\alpha_j - 1}`. Same functional form as the likelihood — so the update is
again pure counting:

.. math::

   \theta \mid y \;\sim\; \mathrm{Dirichlet}(\alpha_1 + y_1,\; \dots,\; \alpha_k + y_k).

Each :math:`\alpha_j` is a **prior count** in category :math:`j`, and :math:`\sum_j \alpha_j` is the
prior sample size. :math:`\mathrm{Dirichlet}(1, \dots, 1)` is uniform on the simplex; Jeffreys' choice
is all :math:`\alpha_j = \tfrac12`.

Contrasts come free
---------------------

The real payoff is that questions about **differences** are answered directly from draws — no delta
method, no covariance algebra:

.. code-block:: python

   import numpy as np
   y = np.array([420, 380, 200])            # candidate A, B, undecided
   alpha = np.ones(3)                       # uniform prior
   draws = np.random.dirichlet(alpha + y, size=20_000)
   lead = draws[:, 0] - draws[:, 1]         # posterior of the A - B margin
   lead.mean(), (lead > 0).mean()           # P(A leads B | data) ≈ 0.91

Two structural facts
----------------------

Marginally, each :math:`\theta_j` is :math:`\mathrm{Beta}(\alpha_j,\ \alpha_0 - \alpha_j)` — collapse
the other categories and the binomial reappears. And the components are **negatively correlated** by
construction: they must sum to one, so probability given to one category is taken from another. That
constraint is exactly what makes the simplex the natural home for mixture weights in Stage 16.

.. hint::

   **Related lessons:** :doc:`Estimating a Probability from Binomial Data <011-estimating-a-probability-from-binomial-data>`  ·  :doc:`Setting up and interpreting mixture models <134-setting-up-and-interpreting-mixture-models>`  ·  :doc:`Dirichlet process prior distributions <140-dirichlet-process-prior-distributions>`  ·  :doc:`Models for multivariate and multinomial responses <111-models-for-multivariate-and-multinomial-responses>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/multinomial-model-for-categorical-data/ <https://insightful-data-lab.com/2025/11/09/multinomial-model-for-categorical-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

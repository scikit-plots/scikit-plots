.. _bda-multivariate-normal-model-with-known-variance:

========================================================================
Multivariate Normal Model with Known Variance
========================================================================

**Part 1 · Stage 3 · 🧮 Multiparameter Models**  ·  Lesson 024 of 144  ·  *beginner*

:doc:`◀ Previous · Multinomial Model for Categorical Data <023-multinomial-model-for-categorical-data>`   ·   :doc:`Next · Multivariate Normal with Unknown Mean and Variance ▶ <025-multivariate-normal-with-unknown-mean-and-variance>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Vectors instead of scalars
----------------------------

Extend the normal model to :math:`d` dimensions: observations :math:`y_i \sim \mathrm{N}(\theta,
\Sigma)` are now **vectors**, :math:`\theta` is a mean vector, and :math:`\Sigma` is a known
:math:`d \times d` covariance matrix. With a conjugate prior :math:`\theta \sim \mathrm{N}(\mu_0,
\Lambda_0)`, every scalar formula from Stage 2 survives — with matrices in place of numbers.

Precision matrices add
------------------------

Working with **precision matrices** (:math:`\Sigma^{-1}`, :math:`\Lambda_0^{-1}`) makes the result a
direct translation of the scalar case. The posterior is normal, precisions add, and the mean is a
precision-weighted average:

.. math::

   \theta \mid y \sim \mathrm{N}(\mu_n, \Lambda_n), \qquad
   \Lambda_n^{-1} = \Lambda_0^{-1} + n\,\Sigma^{-1},

.. math::

   \mu_n = \Lambda_n \left( \Lambda_0^{-1} \mu_0 + n\, \Sigma^{-1} \bar{y} \right).

Set :math:`d = 1` and these collapse to the scalar formulas exactly. A flat prior
(:math:`\Lambda_0^{-1} \to 0`) gives :math:`\theta \mid y \sim \mathrm{N}(\bar{y}, \Sigma / n)`.

.. code-block:: python

   import numpy as np
   Sigma = np.array([[1.0, 0.5], [0.5, 2.0]])          # known covariance
   L0    = np.array([[10.0, 0.0], [0.0, 10.0]])        # vague prior covariance
   mu0   = np.zeros(2)
   Y = np.random.multivariate_normal([1.0, -1.0], Sigma, size=30)
   n, ybar = len(Y), Y.mean(axis=0)

   Ln = np.linalg.inv(np.linalg.inv(L0) + n * np.linalg.inv(Sigma))
   mun = Ln @ (np.linalg.inv(L0) @ mu0 + n * np.linalg.inv(Sigma) @ ybar)

Correlation carries information
---------------------------------

The genuinely multivariate feature is that :math:`\Sigma` **couples** the components: observing one
coordinate informs the others whenever they are correlated. The posterior for :math:`\theta` inherits a
full covariance, so any **contrast** :math:`c^{\top}\theta` (a difference of means, a linear
combination) has posterior :math:`\mathrm{N}(c^{\top}\mu_n,\ c^{\top}\Lambda_n c)` — computed from
draws with a single dot product. Assuming :math:`\Sigma` known is of course a fiction, which the next
lesson removes.

.. hint::

   **Related lessons:** :doc:`Normal Distribution with Known Variance <015-normal-distribution-with-known-variance>`  ·  :doc:`Multivariate Normal with Unknown Mean and Variance <025-multivariate-normal-with-unknown-mean-and-variance>`  ·  :doc:`Averaging Over Nuisance Parameters <020-averaging-over-nuisance-parameters>`  ·  :doc:`Gaussian process regression <129-gaussian-process-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/multivariate-normal-model-with-known-variance/ <https://insightful-data-lab.com/2025/11/09/multivariate-normal-model-with-known-variance/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

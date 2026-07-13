.. _bda-standard-generalized-linear-model-likelihoods:

========================================================================
Standard generalized linear model likelihoods
========================================================================

**Part 4 · Stage 13 · 🔗 Generalized Linear Models**  ·  Lesson 106 of 144  ·  *advanced*

:doc:`◀ Previous · Hierarchical models for batches of variance components <105-hierarchical-models-for-batches-of-variance-components>`   ·   :doc:`Next · Working with generalized linear models ▶ <107-working-with-generalized-linear-models>`   ·   :doc:`↑ Section <index>`


Beyond the normal outcome
---------------------------

Linear regression assumes a **continuous, unbounded, normally-distributed** response. Most real outcomes
are none of these: a yes/no, a count, a category, a waiting time. The **generalized linear model** keeps
the linear predictor :math:`X\beta` but connects it to the outcome through two changes — a **link
function** and a non-normal **likelihood** — so the same machinery covers a whole zoo of data types.

The three ingredients
-----------------------

Every GLM is specified by:

1. a **linear predictor** :math:`\eta = X\beta`, the familiar weighted sum;
2. a **link function** :math:`g` mapping the mean of the outcome to that unbounded scale,
   :math:`g(\mu) = \eta`, so predictions respect the outcome's range;
3. a **distribution** for the outcome given its mean, from the exponential family.

.. math::

   g\bigl(\mathrm{E}[y_i]\bigr) = X_i \beta, \qquad
   y_i \sim \text{ExponentialFamily}(\mu_i).

The link is what keeps a probability in :math:`[0, 1]` and a rate positive: the linear predictor roams
the whole real line, and :math:`g^{-1}` folds it back into the legal range.

The common members
--------------------

Four cover most applied work:

* **Logistic** — binary :math:`y`. Link: **logit**, :math:`\log\frac{\mu}{1-\mu}`. Likelihood:
  Bernoulli/binomial. Coefficients are **log odds ratios**.
* **Poisson** — counts. Link: **log**, :math:`\log \mu`. Likelihood: Poisson. Coefficients are **log
  rate ratios**; an offset handles exposure.
* **Normal** — the identity link, the special case where the whole apparatus collapses back to linear
  regression.
* **Multinomial / ordinal** — categorical outcomes, via softmax or cumulative-logit links (next stage).

.. code-block:: python

   import pymc as pm
   # logistic regression
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])
       p = pm.Deterministic("p", pm.math.sigmoid(X @ beta))     # inverse logit
       pm.Bernoulli("y", p=p, observed=y)

   # Poisson regression with an exposure offset
   with pm.Model():
       beta = pm.Normal("beta", 0, 1, shape=X.shape[1])
       rate = pm.math.exp(X @ beta + log_exposure)              # log link + offset
       pm.Poisson("y", mu=rate, observed=counts)

Why one framework
-------------------

Unifying these under one structure is not mere tidiness. The **same** priors, the **same** hierarchical
extensions (varying coefficients, batching), the **same** diagnostics (PPC, LOO), and the **same**
sampler apply across every outcome type — only the link and likelihood change. Learn the pattern once and
binary, count and categorical data are variations on a theme, not separate subjects. The interpretation
of coefficients changes with the link, though, and that — along with the priors that keep GLMs
well-behaved — is what the rest of this stage is about.

.. hint::

   **Related lessons:** :doc:`Conditional modeling <091-conditional-modeling>`  ·  :doc:`Working with generalized linear models <107-working-with-generalized-linear-models>`  ·  :doc:`Weakly informative priors for logistic regression <108-weakly-informative-priors-for-logistic-regression>`  ·  :doc:`Models for multivariate and multinomial responses <111-models-for-multivariate-and-multinomial-responses>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/06/standard-generalized-linear-model-likelihoods/ <https://insightful-data-lab.com/2025/12/06/standard-generalized-linear-model-likelihoods/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced

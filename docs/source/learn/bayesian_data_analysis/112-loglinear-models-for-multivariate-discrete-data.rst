.. _bda-loglinear-models-for-multivariate-discrete-data:

========================================================================
Loglinear models for multivariate discrete data
========================================================================

**Part 4 · Stage 13 · 🔗 Generalized Linear Models**  ·  Lesson 112 of 144  ·  *advanced*

:doc:`◀ Previous · Models for multivariate and multinomial responses <111-models-for-multivariate-and-multinomial-responses>`   ·   :doc:`Next · Aspects of robustness ▶ <113-aspects-of-robustness>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Modelling a table of counts
-----------------------------

Cross-classify a sample by several categorical variables — sex by education by vote — and the data become
a **contingency table** of counts. **Loglinear models** describe such tables by modelling the log
expected count in each cell as a linear function of the variables and their interactions, turning
questions about **association** into questions about **coefficients**.

The model
-----------

For a two-way table with factors :math:`A` and :math:`B`, the log expected cell count is

.. math::

   \log \mu_{ij} = \lambda + \lambda^A_i + \lambda^B_j + \lambda^{AB}_{ij},

with main effects for each margin and an **interaction** term. The interaction is the object of interest:
:math:`\lambda^{AB}_{ij} = 0` for all :math:`i, j` is exactly the statement that :math:`A` and :math:`B`
are **independent**. Testing independence becomes testing whether an interaction batch is negligible —
and, Bayesianly, *estimating* how large it is.

.. code-block:: python

   import pymc as pm
   # loglinear model for a two-way table of counts
   with pm.Model():
       lam = pm.Normal("lam", 0, 5)
       aA = pm.Normal("aA", 0, 2, shape=I)                  # margin A effects
       aB = pm.Normal("aB", 0, 2, shape=J)                  # margin B effects
       s_ab = pm.HalfNormal("s_ab", 1)
       aAB = pm.Normal("aAB", 0, s_ab, shape=(I, J))        # interaction (association), pooled
       log_mu = lam + aA[:, None] + aB[None, :] + aAB
       pm.Poisson("y", pm.math.exp(log_mu).flatten(), observed=counts.flatten())

The connections
-----------------

Loglinear models tie together several threads. Their equivalence to **Poisson regression** with
categorical predictors is exact — a table of counts *is* count data with factor predictors — so
everything from the overdispersion lesson applies, including the caution that cell counts may vary more
than Poisson allows. For binary outcomes they connect directly to **logistic** regression: the loglinear
interaction involving the response reproduces the logistic coefficient. And treating the interaction
terms as an **exchangeable batch** with a shared scale — the batching idea one final time — gives a
hierarchical loglinear model that **pools** sparse cells, taming the zeros and small counts that make
large contingency tables notoriously unstable.

Where it sits
---------------

For **high-dimensional** discrete data — many categorical variables at once — loglinear models with
hierarchical interaction terms are a principled tool, letting the data determine which associations are
real while pooling away the noise. They close the generalized linear model stage by handling the last
data type on the list: whole tables of categorical counts. Part IV has taken the linear model from
continuous responses through binary, count, ordered, multivariate and tabular data; Part V now abandons
the linear predictor itself, for functions and infinite-dimensional models.

.. hint::

   **Related lessons:** :doc:`Models for multivariate and multinomial responses <111-models-for-multivariate-and-multinomial-responses>`  ·  :doc:`Overdispersed Poisson regression for police stops <109-overdispersed-poisson-regression-for-police-stops>`  ·  :doc:`Standard generalized linear model likelihoods <106-standard-generalized-linear-model-likelihoods>`  ·  :doc:`Working with generalized linear models <107-working-with-generalized-linear-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/06/loglinear-models-for-multivariate-discrete-data/ <https://insightful-data-lab.com/2025/12/06/loglinear-models-for-multivariate-discrete-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced

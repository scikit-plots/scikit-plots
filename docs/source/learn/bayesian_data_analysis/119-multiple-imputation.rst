.. _bda-multiple-imputation:

========================================================================
Multiple imputation
========================================================================

**Part 4 · Stage 14 · 🛡️ Robustness & Missing Data**  ·  Lesson 119 of 144  ·  *advanced*

:doc:`◀ Previous · Notation <118-notation>`   ·   :doc:`Next · Missing data in the multivariate normal and t models ▶ <120-missing-data-in-the-multivariate-normal-and-t-models>`   ·   :doc:`↑ Section <index>`


Fill in the blanks, honestly
------------------------------

The obvious fix for a missing value — plug in a single best guess (the mean, a regression prediction) —
is quietly wrong: it treats an **estimate as if it were known**, so every downstream standard error comes
out too small. **Multiple imputation** repairs this by filling the gaps not once but **many** times,
carrying the uncertainty about the missing values all the way through to the final answer.

The three steps
-----------------

1. **Impute.** Draw :math:`m` complete datasets, each with the missing entries replaced by values sampled
   from their **posterior predictive distribution** given the observed data — genuine draws, not point
   estimates, so the imputations differ from one another.
2. **Analyse.** Run the intended analysis on each completed dataset separately, giving :math:`m`
   estimates :math:`\hat{Q}_k` and their variances :math:`U_k`.
3. **Pool.** Combine them with **Rubin's rules**.

.. math::

   \bar{Q} = \frac{1}{m}\sum_k \hat{Q}_k, \qquad
   \bar{U} = \frac{1}{m}\sum_k U_k, \qquad
   B = \frac{1}{m-1}\sum_k (\hat{Q}_k - \bar{Q})^2,

.. math::

   T = \bar{U} + \Bigl(1 + \tfrac{1}{m}\Bigr) B .

The total variance :math:`T` is the heart of it. :math:`\bar{U}` is the ordinary within-imputation
uncertainty; :math:`B`, the **between-imputation** variance, measures how much the answer wobbles as the
imputations change — and that term is **exactly the extra uncertainty due to missingness** that single
imputation throws away.

.. code-block:: python

   import numpy as np
   Qk = np.array([analyze(dataset) for dataset in imputed_datasets])   # m estimates
   Uk = np.array([variance(dataset) for dataset in imputed_datasets])  # m variances
   Q_bar = Qk.mean()
   U_bar = Uk.mean()
   B = Qk.var(ddof=1)                                                  # between-imputation
   T = U_bar + (1 + 1/len(Qk)) * B                                     # total variance
   se = np.sqrt(T)

The Bayesian reading
----------------------

Multiple imputation *is* a Bayesian computation in disguise: the missing values are **unknown
parameters**, and the :math:`m` completed datasets are **draws from their posterior**. In a fully
Bayesian model you get this for free — sample :math:`y_{\text{mis}}` jointly with :math:`\theta` in one
MCMC run, and the posterior already integrates over the missing values with no separate pooling step.
Rubin's rules are what you use when the analysis model and the imputation model are **separate programs**;
a joint Bayesian model makes them one.

Practicalities
----------------

A few settled points. Classic advice said :math:`m = 5`; modern practice prefers **twenty or more**,
since it is cheap and stabilises :math:`B`. The **imputation model must be at least as rich as the
analysis model** — if you will study an interaction, the imputation must include it, or the imputations
will erase it. And it all rests on **MAR**: multiple imputation handles ignorable missingness, and MNAR
still demands an explicit model. Within those limits it is the standard, principled way to keep
incomplete data from silently understating what you do not know.

.. hint::

   **Related lessons:** :doc:`Notation <118-notation>`  ·  :doc:`Missing data in the multivariate normal and t models <120-missing-data-in-the-multivariate-normal-and-t-models>`  ·  :doc:`Example: multiple imputation for a series of polls <121-example-multiple-imputation-for-a-series-of-polls>`  ·  :doc:`Missing values with counted data <122-missing-values-with-counted-data>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/07/multiple-imputation/ <https://insightful-data-lab.com/2025/12/07/multiple-imputation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced

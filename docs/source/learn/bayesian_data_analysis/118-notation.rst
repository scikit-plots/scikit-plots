.. _bda-notation:

========================================================================
Notation
========================================================================

**Part 4 · Stage 14 · 🛡️ Robustness & Missing Data**  ·  Lesson 118 of 144  ·  *advanced*

:doc:`◀ Previous · Robust regression using t-distributed errors <117-robust-regression-using-t-distributed-errors>`   ·   :doc:`Next · Multiple imputation ▶ <119-multiple-imputation>`   ·   :doc:`↑ Section <index>`


A language for missingness
----------------------------

Missing data need notation before they need methods, because the **right analysis depends entirely on
why** the data are missing. This lesson sets up the framework — due to Rubin — that makes "why" a precise,
model-able quantity rather than a vague worry.

The pieces
------------

Split the complete data into what you see and what you do not, and add an indicator for which is which:

.. math::

   y = (y_{\text{obs}}, y_{\text{mis}}), \qquad
   R_{ij} = \begin{cases} 1 & y_{ij} \text{ observed} \\ 0 & y_{ij} \text{ missing.} \end{cases}

The **missingness indicator** :math:`R` is itself data — a matrix you fully observe — and the object that
makes the theory work is its distribution :math:`p(R \mid y, \phi)`, the **missingness mechanism**. The
question is how :math:`R` depends on the values :math:`y`, including the ones you cannot see.

Three mechanisms
------------------

Rubin's taxonomy, in decreasing order of convenience:

* **MCAR — missing completely at random.** :math:`p(R \mid y) = p(R)`: missingness is independent of all
  data, observed and missing alike. Dropped records are then a random subsample, so complete-case
  analysis is unbiased (if wasteful). Rarely true.
* **MAR — missing at random.** :math:`p(R \mid y) = p(R \mid y_{\text{obs}})`: missingness depends only
  on **observed** values. Income missing more often for the young is MAR *if age is recorded*. This is
  the workhorse assumption.
* **MNAR — missing not at random.** Missingness depends on the **unobserved** values themselves — income
  missing because it is high. Here the mechanism cannot be ignored and must be modelled explicitly.

Ignorability
--------------

The payoff is a precise condition. When data are **MAR** and the missingness parameters :math:`\phi` are
distinct from the model parameters :math:`\theta`, the mechanism is **ignorable**: the term
:math:`p(R \mid y_{\text{obs}}, \phi)` factors out of the likelihood for :math:`\theta`, so you may model
the data and simply **ignore** :math:`R`. This is the *same* ignorability condition met in Stage 7 for
data collection — MAR plus parameter distinctness — now applied to missing values. Under it, Bayesian
inference proceeds by treating :math:`y_{\text{mis}}` as **unknown parameters** and integrating them out,
which is exactly what imputation does.

The catch
-----------

MAR versus MNAR **cannot be tested from the data** — distinguishing them needs the very values that are
missing. The assumption rests on subject knowledge (why *would* these be missing?) and, where it is
doubtful, on a **sensitivity analysis** across plausible MNAR mechanisms. The notation's value is exactly
this: it turns an untestable worry into an explicit assumption you can state, defend, and vary.

.. hint::

   **Related lessons:** :doc:`Data-collection models and ignorability <051-data-collection-models-and-ignorability>`  ·  :doc:`Multiple imputation <119-multiple-imputation>`  ·  :doc:`Missing data in the multivariate normal and t models <120-missing-data-in-the-multivariate-normal-and-t-models>`  ·  :doc:`Sample surveys <052-sample-surveys>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/07/notation/ <https://insightful-data-lab.com/2025/12/07/notation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced

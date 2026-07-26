.. _dpa-logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds:

========================================================================
Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds
========================================================================

**Stage 6 · 🎯 Classification & Logistic Regression**  ·  Lesson 38 of 56  ·  *advanced*

:doc:`◀ Previous · How Shapley Values Work <37-how-shapley-values-work>`   ·   :doc:`Next · Maximum Likelihood (MLE): Fitting a Distribution to Observed Data ▶ <39-maximum-likelihood-mle-fitting-a-distribution-to-observed-data>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

When the outcome is yes or no
-------------------------------

Regression so far has predicted a **number** — a fare, a price. But many outcomes are **binary**: will
a customer **churn** or not? will a student **return** next year? Ordinary linear regression fails
here, because a straight line runs off to :math:`\pm\infty` and would predict "probabilities" **below
0 or above 1**. **Logistic regression** is the standard model for a **yes/no** outcome, and it works by
predicting a **probability** instead.

Odds and log-odds
-------------------

The trick is to transform the probability so a linear model fits. Start with the **odds** — the ratio
of the probability of the event to its complement, :math:`p / (1 - p)` — which stretches :math:`[0, 1]`
out to :math:`[0, \infty)`. Then take the logarithm, giving the **log-odds** or **logit**, which spans
**all** real numbers. Logistic regression makes *this* linear in the features:

.. math::

   \ln\!\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p.

The model is an ordinary linear equation — just on the **log-odds scale** rather than the probability
scale.

The logistic curve
--------------------

To get a probability back, invert the transform. Solving for :math:`p` gives the **logistic**
(sigmoid) function:

.. math::

   p = \frac{1}{1 + e^{-z}}, \qquad z = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p.

This S-shaped curve takes the linear combination :math:`z` — any real number — and squashes it
smoothly into a valid probability between 0 and 1. Large positive :math:`z` gives :math:`p` near 1,
large negative near 0, and :math:`z = 0` gives :math:`p = 0.5`.

Reading coefficients
----------------------

The coefficients read on the **odds** scale. A one-unit rise in :math:`x_j` adds :math:`\beta_j` to the
log-odds, which **multiplies** the odds by :math:`e^{\beta_j}` — the **odds ratio**. So
:math:`e^{\beta_j} > 1` means the feature raises the odds of the event, :math:`< 1` lowers them. In
Python it is ``LogisticRegression`` in scikit-learn, or ``Logit`` in statsmodels for the full
coefficient table. Unlike least squares, its coefficients have **no closed form** — they are found by
**maximum likelihood**, the subject of the next lesson.

.. hint::

   **Related lessons:** :doc:`Maximum Likelihood (MLE): Fitting a Distribution to Observed Data <39-maximum-likelihood-mle-fitting-a-distribution-to-observed-data>`  ·  :doc:`Assessing Model Fit in Logistic Regression <40-assessing-model-fit-in-logistic-regression>`  ·  :doc:`Binary Classification Models – Conceptual Framework and Evaluation Metrics <51-binary-classification-models-conceptual-framework-and-evaluation-metrics>`  ·  :doc:`Complete and Quasi-Complete Separation in Logistic Regression <41-complete-and-quasi-complete-separation-in-logistic-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds/ <https://insightful-data-lab.com/2026/01/16/logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: advanced

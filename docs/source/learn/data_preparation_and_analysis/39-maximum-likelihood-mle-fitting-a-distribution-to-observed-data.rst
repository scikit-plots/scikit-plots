.. _dpa-maximum-likelihood-mle-fitting-a-distribution-to-observed-data:

========================================================================
Maximum Likelihood (MLE): Fitting a Distribution to Observed Data
========================================================================

**Stage 6 · 🎯 Classification & Logistic Regression**  ·  Lesson 39 of 56  ·  *advanced*

:doc:`◀ Previous · Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds <38-logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds>`   ·   :doc:`Next · Assessing Model Fit in Logistic Regression ▶ <40-assessing-model-fit-in-logistic-regression>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What parameters best explain the data?
----------------------------------------

How does logistic regression actually **choose** its coefficients, when there is no closed-form
formula? The answer is a principle general enough to fit almost any model: **maximum likelihood
estimation** (MLE). Its question is simple and intuitive — *of all possible parameter values, which
ones make the* **data I actually observed** *most probable?*

The likelihood
----------------

The key object is the **likelihood**. For a candidate set of parameters :math:`\theta`, the likelihood
:math:`L(\theta)` is the probability of the observed data **computed under those parameters** — but
read as a function of :math:`\theta`, with the data held fixed. A parameter value under which the
observed data would be very **improbable** has low likelihood; one under which the data looks
**typical** has high likelihood. Because independent observations multiply, the likelihood is a
**product** of per-observation probabilities, and it is usually easier to work with its logarithm, the
**log-likelihood**, which turns the product into a sum without moving the maximum.

Maximising it
---------------

**Maximum likelihood** simply picks the :math:`\theta` that makes :math:`L(\theta)` — or the
log-likelihood — as **large as possible**. For simple cases this has a tidy answer: the MLE of a normal
distribution's mean is just the **sample mean**, and of a coin's bias the **observed proportion** of
heads. For models like logistic regression there is **no formula**, so the maximum is found
**numerically**, by iterative optimisation that climbs the log-likelihood to its peak.

Why it matters here
---------------------

MLE ties much of this course together. Logistic regression's coefficients **are** the maximum-likelihood
estimates. Least squares is itself MLE in disguise — minimising squared errors is *exactly* maximising
likelihood when the errors are **normally distributed**. And the precision of these estimates is
governed by the **Cramér–Rao bound** from Stage 2, which sets the best variance any unbiased estimator
can reach. The next lessons use likelihood again — to **assess** how well a fitted logistic model fits.

.. hint::

   **Related lessons:** :doc:`Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds <38-logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds>`  ·  :doc:`Least Squares Regression <31-least-squares-regression>`  ·  :doc:`Harald Cramér <14-harald-cramer>`  ·  :doc:`Assessing Model Fit in Logistic Regression <40-assessing-model-fit-in-logistic-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/maximum-likelihood-mle-fitting-a-distribution-to-observed-data/ <https://insightful-data-lab.com/2026/01/16/maximum-likelihood-mle-fitting-a-distribution-to-observed-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: advanced

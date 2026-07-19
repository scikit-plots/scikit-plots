.. _dpa-assessing-model-fit-in-logistic-regression:

========================================================================
Assessing Model Fit in Logistic Regression
========================================================================

**Stage 6 · 🎯 Classification & Logistic Regression**  ·  Lesson 40 of 56  ·  *advanced*

:doc:`◀ Previous · Maximum Likelihood (MLE): Fitting a Distribution to Observed Data <39-maximum-likelihood-mle-fitting-a-distribution-to-observed-data>`   ·   :doc:`Next · Complete and Quasi-Complete Separation in Logistic Regression ▶ <41-complete-and-quasi-complete-separation-in-logistic-regression>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

No R² to lean on
------------------

In ordinary regression, :math:`R^2` gives a quick read of fit — the fraction of variance explained.
Logistic regression has **no such natural measure**, because it predicts probabilities, not a
continuous quantity with variance to partition. Assessing a logistic model's fit therefore takes a
different toolkit, built on the **likelihood** the model was fitted to maximise.

Likelihood-based fit
----------------------

The foundation is the **log-likelihood** of the fitted model — higher means the model assigns greater
probability to what was actually observed. A closely related quantity is the **deviance**, defined as
:math:`-2` times the log-likelihood; **lower** deviance means **better** fit. On their own these
numbers are not interpretable, but **compared** — fitted model against a baseline — they become the
basis of every fit measure below.

Pseudo-R²
-----------

To mimic the familiar 0-to-1 feel of :math:`R^2`, several **pseudo-**:math:`R^2` measures compare the
fitted model's likelihood to that of the **null** (intercept-only) model. The most widely recommended
is **McFadden's**:

.. math::

   R^2_{\text{McF}} = 1 - \frac{\ln L_{\text{model}}}{\ln L_{\text{null}}}.

It is 0 when the predictors add nothing and approaches 1 as the fitted model explains the data far
better. It is **not** the proportion of variance explained — McFadden values of **0.2–0.4** already
indicate a very good fit — so it is read as a *relative* measure among models, not on the OLS scale.
(Cox–Snell, Nagelkerke and Efron's are common alternatives.)

Comparing to the null
-----------------------

That comparison can be made a formal test. The **likelihood-ratio test** asks whether the fitted model
fits **significantly** better than the null, using the drop in deviance as a **chi-square** statistic;
a significant result means the predictors, taken together, genuinely contribute. And to balance fit
against complexity across candidate models, **AIC** and **BIC** combine the log-likelihood with a
penalty for the number of predictors — the same parsimony principle from the selection lessons. The
next lesson turns this deviance-comparison idea into a tool for **choosing** features.

.. hint::

   **Related lessons:** :doc:`Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds <38-logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds>`  ·  :doc:`Maximum Likelihood (MLE): Fitting a Distribution to Observed Data <39-maximum-likelihood-mle-fitting-a-distribution-to-observed-data>`  ·  :doc:`Forward Selection with Nested Models and Deviance Tests <42-forward-selection-with-nested-models-and-deviance-tests>`  ·  :doc:`Assessing the Quality of Prediction Models <50-assessing-the-quality-of-prediction-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/assessing-model-fit-in-logistic-regression-predictive-power-and-r2-measures/ <https://insightful-data-lab.com/2026/01/16/assessing-model-fit-in-logistic-regression-predictive-power-and-r2-measures/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: advanced

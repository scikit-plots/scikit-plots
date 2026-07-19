.. _ts-best-linear-predictor-of-a-stationary-process:

========================================================================
Best Linear Predictor of a Stationary Process
========================================================================

**Stage 4 · 🎯 Prediction & the Sample ACF / PACF**  ·  Lesson 09 of 18  ·  *intermediate*

:doc:`◀ Previous · Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q) <08-understanding-acfs-via-difference-equations-for-ar-p-and-arma-p-q>`   ·   :doc:`Next · Sample ACF and Sample PACF ▶ <10-sample-acf-and-sample-pacf>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The prediction problem
------------------------

Given a stationary series observed up to now, the **best linear predictor** of the next value is
the linear combination of past observations
:math:`\hat{x}_{n+1} = \sum_{k=1}^{n} a_k\, x_{n+1-k}` that **minimises the mean-squared error**.
"Best" here means **within linear rules**; for a Gaussian process it is the best predictor of
**any** kind.

The projection principle
--------------------------

The solution is **geometric**: treat random variables as vectors with inner product
:math:`\langle X, Y\rangle = \mathbb{E}[XY]`, and the best predictor is the **orthogonal
projection** of the target onto the span of the predictors. Optimality is characterised by the
**orthogonality principle** — the prediction **error is uncorrelated with every predictor**:

.. math::

   \mathbb{E}\big[(x_{n+1} - \hat{x}_{n+1})\, x_{n+1-k}\big] = 0, \qquad k = 1, \dots, n.

Writing these conditions out gives the **prediction equations**, a linear system
:math:`\Gamma_n \mathbf{a} = \gamma_n` in the autocovariances — the same **Toeplitz** system as
Yule–Walker.

Solving efficiently
--------------------

Solving that system afresh at each order is wasteful. The **Durbin–Levinson recursion** builds the
order-:math:`h` predictor from the order-:math:`(h-1)` one **without inverting a matrix**, updating
the coefficients and the error variance in place — cheap and numerically stable.

From predictor to PACF
------------------------

The recursion delivers a bonus. The **last coefficient** :math:`\phi_{hh}` of the best
order-:math:`h` predictor **is the partial autocorrelation** at lag :math:`h` — the correlation
between :math:`x_t` and :math:`x_{t-h}` **after removing** the linear influence of the intervening
values. For an **AR(p)** these coefficients vanish for :math:`h > p`, so the **PACF cuts off at lag
p** — the property that makes it the tool for reading autoregressive order.

.. hint::

   **Related lessons:** :doc:`Sample ACF and Sample PACF <10-sample-acf-and-sample-pacf>`  ·  :doc:`Preliminary Estimation for AR Models and the Yule–Walker Equations <11-preliminary-estimation-for-ar-models-and-the-yule-walker-equations>`  ·  :doc:`Understanding ARMA Processes <06-understanding-arma-processes>`  ·  :doc:`Beyond One-Step Ahead Predictions <17-beyond-one-step-ahead-predictions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/17/best-linear-predictor-of-a-stationary-process-forecasting-principles-and-the-pacf/ <https://insightful-data-lab.com/2026/01/17/best-linear-predictor-of-a-stationary-process-forecasting-principles-and-the-pacf/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: time series, level: intermediate

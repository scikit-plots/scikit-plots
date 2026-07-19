.. _ts-sample-acf-and-sample-pacf:

========================================================================
Sample ACF and Sample PACF
========================================================================

**Stage 4 · 🎯 Prediction & the Sample ACF / PACF**  ·  Lesson 10 of 18  ·  *intermediate*

:doc:`◀ Previous · Best Linear Predictor of a Stationary Process <09-best-linear-predictor-of-a-stationary-process>`   ·   :doc:`Next · Preliminary Estimation for AR Models and the Yule–Walker Equations ▶ <11-preliminary-estimation-for-ar-models-and-the-yule-walker-equations>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

From process to sample
------------------------

In practice you never see the true ACF — you **estimate** it from one finite record. The **sample
autocovariance** and **sample ACF** are

.. math::

   \hat{\gamma}(h) = \frac{1}{n}\sum_{t=1}^{n-|h|} (x_t - \bar{x})(x_{t+|h|} - \bar{x}),
   \qquad \hat{\rho}(h) = \frac{\hat{\gamma}(h)}{\hat{\gamma}(0)}.

Both grow noisier at **large lags**, where few pairs contribute — a common rule keeps
:math:`n \ge 50` and lags :math:`h \le n/4`.

Significance bands
--------------------

How large is "large"? If the series were **white noise**, then for big :math:`n` each
:math:`\hat{\rho}(h)` is approximately :math:`\mathcal{N}(0, 1/n)`, so correlations beyond

.. math::

   \pm \frac{1.96}{\sqrt{n}}

are unlikely by chance — the dashed bands on every correlogram. For a genuinely correlated series
the bands widen, following **Bartlett's formula**
:math:`\operatorname{Var}(\hat{\rho}_k) \approx \frac{1}{n}\big(1 + 2\sum_{j<k}\rho_j^2\big)`.

The sample PACF
----------------

The **sample PACF** applies the **Durbin–Levinson recursion** to the *sample* autocovariances,
producing :math:`\hat{\phi}_{hh}` at each lag. It shares the same white-noise bands
:math:`\pm 1.96/\sqrt{n}`, so significant spikes stand out in the same way.

Reading them together
----------------------

Identification is a two-plot habit. An **ACF that cuts off** after lag :math:`q` alongside a
**PACF that tails off** suggests **MA(q)**; a **PACF that cuts off** after lag :math:`p` with a
**tailing ACF** suggests **AR(p)**; **both tailing off** suggests **ARMA**. In ``statsmodels`` these
plots are ``plot_acf`` and ``plot_pacf``.

.. hint::

   **Related lessons:** :doc:`Best Linear Predictor of a Stationary Process <09-best-linear-predictor-of-a-stationary-process>`  ·  :doc:`Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q) <08-understanding-acfs-via-difference-equations-for-ar-p-and-arma-p-q>`  ·  :doc:`Preliminary Estimation for AR Models and the Yule–Walker Equations <11-preliminary-estimation-for-ar-models-and-the-yule-walker-equations>`  ·  :doc:`Order Selection for Time Series Models <14-order-selection-for-time-series-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/17/sample-acf-and-sample-pacf/ <https://insightful-data-lab.com/2026/01/17/sample-acf-and-sample-pacf/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: time series, level: intermediate

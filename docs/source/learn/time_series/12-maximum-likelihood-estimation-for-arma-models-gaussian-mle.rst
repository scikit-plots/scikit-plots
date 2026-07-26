.. _ts-maximum-likelihood-estimation-for-arma-models-gaussian-mle:

========================================================================
Maximum Likelihood Estimation for ARMA Models (Gaussian MLE)
========================================================================

**Stage 5 · 🧮 Estimation**  ·  Lesson 12 of 18  ·  *advanced*

:doc:`◀ Previous · Preliminary Estimation for AR Models and the Yule–Walker Equations <11-preliminary-estimation-for-ar-models-and-the-yule-walker-equations>`   ·   :doc:`Next · Diagnostics After Fitting a Time Series Model ▶ <13-diagnostics-after-fitting-a-time-series-model>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The likelihood
----------------

If the ARMA process is driven by **Gaussian** white noise, then the whole sample
:math:`(x_1, \dots, x_n)` is **multivariate normal** with a covariance matrix determined by the
parameters. **Maximum likelihood** picks the :math:`(\boldsymbol{\phi}, \boldsymbol{\theta},
\sigma_w^2)` that make the observed data **most probable** under that model.

The innovations trick
----------------------

Working with the raw :math:`n \times n` covariance is expensive. The standard route rewrites the
likelihood through the **one-step prediction errors** ("innovations") and their variances —
computed cheaply by the **innovations algorithm** or a **Kalman filter**. This turns the likelihood
into a product over time and avoids inverting a large matrix.

A harder optimisation
----------------------

Unlike Yule–Walker, there is **no closed form** — the likelihood is **nonlinear**, especially in the
**moving-average** parameters, so it must be **maximised numerically**. That optimisation can be
finicky, which is exactly why good **starting values** (from Yule–Walker or the Hannan–Rissanen
method) matter. One further choice: the **conditional** likelihood fixes the initial values, while
the **exact** likelihood models them too.

Why it's the default
----------------------

Maximum likelihood is **consistent**, **asymptotically normal**, and **efficient** — the
lowest-variance estimates available — and, remarkably, its **asymptotic distribution is the same
even when the data are not Gaussian**. That combination is why ``statsmodels``' ``ARIMA`` fits by
**exact MLE** (via a state-space / Kalman filter) by default.

.. hint::

   **Related lessons:** :doc:`Preliminary Estimation for AR Models and the Yule–Walker Equations <11-preliminary-estimation-for-ar-models-and-the-yule-walker-equations>`  ·  :doc:`Understanding ARMA Processes <06-understanding-arma-processes>`  ·  :doc:`Order Selection for Time Series Models <14-order-selection-for-time-series-models>`  ·  :doc:`Diagnostics After Fitting a Time Series Model <13-diagnostics-after-fitting-a-time-series-model>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/17/maximum-likelihood-estimation-for-arma-models-gaussian-mle/ <https://insightful-data-lab.com/2026/01/17/maximum-likelihood-estimation-for-arma-models-gaussian-mle/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: time series, level: advanced

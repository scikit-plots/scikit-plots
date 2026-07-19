:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-bayesian-time-series:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Bayesian Time Series</b></div>`

======================
Bayesian Time Series
======================

*Time-series modelling in a Bayesian framework, yielding full posterior uncertainty over parameters and forecasts.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Why go Bayesian for time series
-------------------------------

Classical models — ARIMA, exponential smoothing — fix their parameters and estimate
them by maximum likelihood, returning a single point forecast. Real forecasting often
needs more: **uncertainty quantification** (a full distribution of future values, not
one number), **flexibility** (priors encoding trend, smoothness, seasonality, and
handling missing data), and **online updating** (re-forecasting as new data lands).
Bayesian methods deliver all three through **posterior distributions**.

The general framework
---------------------

Model a series :math:`y_{1:T}` with latent parameters :math:`\theta`:

.. math::

   p(\theta \mid y_{1:T}) \propto p(y_{1:T} \mid \theta)\, p(\theta),

where the prior :math:`p(\theta)` encodes structure (smoothness, trend, seasonality)
and the likelihood :math:`p(y_{1:T} \mid \theta)` is often Gaussian or Poisson.
Forecasts come from the **posterior predictive**, which integrates over parameter
uncertainty rather than plugging in a point estimate:

.. math::

   p(y_{T+h} \mid y_{1:T}) = \int p(y_{T+h} \mid \theta, y_{1:T})\, p(\theta \mid y_{1:T})\, d\theta.

The model families
------------------

- **Bayesian ARIMA/ARMA** — ARIMA with priors on :math:`\phi, \theta, \sigma^2`;
  inference by MCMC or variational methods, yielding posterior forecasts.
- **State-space models / dynamic linear models (DLMs)** — a hidden state evolves and
  emits observations,

  .. math::

     x_t = F x_{t-1} + w_t, \qquad y_t = H x_t + v_t,

  covering local-level and trend-plus-seasonality decompositions; inference via the
  **Kalman filter** (conjugate Gaussian) or a **particle filter** (nonlinear /
  non-Gaussian).
- **Bayesian structural time series (BSTS)** — decomposes into trend + seasonality +
  regressors + holiday effects; the engine behind Google's **CausalImpact** for
  estimating intervention effects (A/B tests, policy evaluation).
- **Gaussian-process time series** — a GP prior on the latent function,
  :math:`f \sim \mathcal{GP}(0, k(t,t'))`, with the kernel :math:`k` encoding
  smoothness and periodicity; flexible but heavy on long series.
- **Bayesian VAR** — multivariate autoregression with shrinkage (Minnesota) priors on
  the coefficient matrices to tame overfitting.

Inference toolkit
-----------------

Conjugate **Gibbs sampling** (Normal–Inverse-Gamma for AR models), **Hamiltonian
Monte Carlo** (Stan, PyMC), **variational inference** for speed, and **particle MCMC**
for nonlinear state spaces.

Trade-offs
----------

The payoff is full **posterior predictive distributions**, uncertainty over both
parameters and future values, principled priors for domain knowledge, and graceful
handling of missing data and regime shifts. The cost is **computation** (MCMC on large
data), the need for **careful prior choice**, and **scaling** limits (GPs on long
series).

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Time Series <010-time-series>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Posterior <063-posterior>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Bayesian Time Series <https://insightful-data-lab.com/2025/08/29/bayesian-time-series/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

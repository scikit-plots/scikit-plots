:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _time-series-index:

:raw-html:`<div style="text-align:center"><strong>` ⏱️ Time Series
|br| Modelling and forecasting data that arrives in order
|br| |full_version| - |today|
:raw-html:`</strong></div>`

======================================================================
Time Series
======================================================================

A **time series** is a sequence of observations indexed by time, where **order and
dependence carry information**. This hub walks the classical **Box–Jenkins** path the
source corpus follows — from stationarity and autocorrelation, through the
AR / MA / ARMA / ARIMA / SARIMA model family, to estimation, diagnostics and forecasting —
as an ordered, self-contained course of 18 lessons.

Read it at any depth:

* **newcomers** — what makes time-series data special, and stationarity;
* **practitioners** — reading the ACF / PACF and fitting ARIMA in ``statsmodels``;
* **researchers** — estimation (Yule–Walker, Gaussian MLE), order selection and residual diagnostics.

.. warning::

   Time series breaks the i.i.d. assumption behind ordinary cross-validation. **Never**
   shuffle: validate forward in time (**walk-forward**) so the future never leaks into the past.

.. note::

   Follow the lessons in order with **Next ▶**, or jump in by stage below. Snippets use
   real ``statsmodels`` / ``pandas`` / ``numpy`` calls. This course pairs with the
   :ref:`Terminology reference <terminology-index>` (Signal Processing & Time Series).

======================================================================

🧭 Stage 1 — Orientation
------------------------------------------------------------------------

*What time-series data is, why order carries information, and setting up the tools.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 01 · What Are Time Series, and How Are They Used?
      :link: ts-what-are-time-series-and-how-are-they-used
      :link-type: ref

      Sequences of time-ordered observations, their trend / seasonal / residual parts, and where forecasting applies.

   .. grid-item-card:: 02 · Getting Started with R
      :link: ts-getting-started-with-r
      :link-type: ref

      Setting up a working environment — the source's R tooling and the Python (statsmodels) path used here.

📐 Stage 2 — Stationarity
------------------------------------------------------------------------

*The property that makes a series learnable — how to recognise it and how to achieve it.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 03 · A Gentle Introduction to Stationarity
      :link: ts-a-gentle-introduction-to-stationarity
      :link-type: ref

      Why a stable mean, variance and autocovariance make a series learnable — and how differencing helps.

   .. grid-item-card:: 04 · Weak and Strong Stationarity
      :link: ts-weak-and-strong-stationarity
      :link-type: ref

      The precise definitions: strict distributional invariance versus the weaker second-order (covariance) form.

🔗 Stage 3 — Linear & ARMA Processes
------------------------------------------------------------------------

*The building blocks: linear processes, the AR / MA / ARMA family, and their autocorrelation.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 05 · Linear Processes
      :link: ts-linear-processes
      :link-type: ref

      Series written as a linear filter of white noise — the general form underlying AR, MA and ARMA.

   .. grid-item-card:: 06 · Understanding ARMA Processes
      :link: ts-understanding-arma-processes
      :link-type: ref

      Combining autoregressive and moving-average terms; causality, invertibility and what each parameter does.

   .. grid-item-card:: 07 · Computing ACFs of Causal AR(2) Processes Using Difference Equations
      :link: ts-computing-acfs-of-causal-ar-2-processes-using-difference-equations
      :link-type: ref

      Solving the autocorrelation of an AR(2) by treating its recursion as a linear difference equation.

   .. grid-item-card:: 08 · Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q)
      :link: ts-understanding-acfs-via-difference-equations-for-ar-p-and-arma-p-q
      :link-type: ref

      Generalising the difference-equation method to the autocorrelation of any AR(p) or ARMA(p, q).

🎯 Stage 4 — Prediction & the Sample ACF / PACF
------------------------------------------------------------------------

*Optimal linear forecasting and the empirical correlation tools used to identify model order.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 09 · Best Linear Predictor of a Stationary Process
      :link: ts-best-linear-predictor-of-a-stationary-process
      :link-type: ref

      The optimal linear forecast, the projection principle, and how the PACF emerges from it.

   .. grid-item-card:: 10 · Sample ACF and Sample PACF
      :link: ts-sample-acf-and-sample-pacf
      :link-type: ref

      Estimating autocorrelation from data, their sampling behaviour, and reading them to pick model order.

🧮 Stage 5 — Estimation
------------------------------------------------------------------------

*Fitting parameters: Yule–Walker for AR models, Gaussian maximum likelihood for ARMA.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 11 · Preliminary Estimation for AR Models and the Yule–Walker Equations
      :link: ts-preliminary-estimation-for-ar-models-and-the-yule-walker-equations
      :link-type: ref

      Method-of-moments AR fitting by solving the Yule–Walker system from sample autocovariances.

   .. grid-item-card:: 12 · Maximum Likelihood Estimation for ARMA Models (Gaussian MLE)
      :link: ts-maximum-likelihood-estimation-for-arma-models-gaussian-mle
      :link-type: ref

      Fitting ARMA by maximising the Gaussian likelihood — the standard, efficient estimator.

🏗️ Stage 6 — Building & Forecasting Models
------------------------------------------------------------------------

*Diagnostics, order selection, ARIMA / SARIMA, multi-step forecasting and exponential smoothing.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 13 · Diagnostics After Fitting a Time Series Model
      :link: ts-diagnostics-after-fitting-a-time-series-model
      :link-type: ref

      Checking standardized residuals for leftover structure: normality, autocorrelation and the Ljung–Box test.

   .. grid-item-card:: 14 · Order Selection for Time Series Models
      :link: ts-order-selection-for-time-series-models
      :link-type: ref

      Choosing p, d, q with information criteria (AIC / BIC) balanced against parsimony and diagnostics.

   .. grid-item-card:: 15 · ARIMA Models: How Nonstationary Models Are Built from Stationary Ones
      :link: ts-arima-models-how-nonstationary-models-are-built-from-stationary-ones
      :link-type: ref

      Differencing to remove trend, turning a nonstationary series into an ARMA-modellable one.

   .. grid-item-card:: 16 · SARIMA Models: Seasonal ARIMA
      :link: ts-sarima-models-seasonal-arima
      :link-type: ref

      Extending ARIMA with seasonal AR, differencing and MA terms for periodic patterns.

   .. grid-item-card:: 17 · Beyond One-Step Ahead Predictions
      :link: ts-beyond-one-step-ahead-predictions
      :link-type: ref

      Multi-step forecasting, how uncertainty compounds with horizon, and forecast intervals.

   .. grid-item-card:: 18 · Exponential Smoothing Models
      :link: ts-exponential-smoothing-models
      :link-type: ref

      Weighted-average forecasting (simple, Holt, Holt–Winters) as a practical complement to ARIMA.

.. toctree::
   :hidden:
   :maxdepth: 1

   01-what-are-time-series-and-how-are-they-used
   02-getting-started-with-r
   03-a-gentle-introduction-to-stationarity
   04-weak-and-strong-stationarity
   05-linear-processes
   06-understanding-arma-processes
   07-computing-acfs-of-causal-ar-2-processes-using-difference-equations
   08-understanding-acfs-via-difference-equations-for-ar-p-and-arma-p-q
   09-best-linear-predictor-of-a-stationary-process
   10-sample-acf-and-sample-pacf
   11-preliminary-estimation-for-ar-models-and-the-yule-walker-equations
   12-maximum-likelihood-estimation-for-arma-models-gaussian-mle
   13-diagnostics-after-fitting-a-time-series-model
   14-order-selection-for-time-series-models
   15-arima-models-how-nonstationary-models-are-built-from-stationary-ones
   16-sarima-models-seasonal-arima
   17-beyond-one-step-ahead-predictions
   18-exponential-smoothing-models

.. tags:: purpose: reference, topic: time series

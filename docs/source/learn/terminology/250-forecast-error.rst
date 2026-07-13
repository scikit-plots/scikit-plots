:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-forecast-error:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Forecast Error</b></div>`

================
Forecast Error
================

*The difference between a forecast and the realised value.*

What it is
----------

**Forecast error** is the gap between what happened and what was predicted — the residual

.. math::

   e_t = y_t - \hat{y}_t.

A single error means little; forecast quality is summarized by **aggregating** errors into metrics.

Common metrics
--------------

**MAE** :math:`= \frac{1}{N}\sum_t |y_t - \hat{y}_t|` is robust and interpretable, and the forecast that
minimizes it is the **median**. **RMSE** :math:`= \sqrt{\frac{1}{N}\sum_t (y_t - \hat{y}_t)^2}` penalizes
large misses more and is minimized by the **mean**, but is harder to read. **MAPE** (mean absolute
*percentage* error) is scale-free but **explodes** when actuals are near zero; **sMAPE** is a bounded
symmetric variant, still shaky near zero. **MASE** scales MAE by a naive forecast's error, making it
**scale-free** and interpretable (:math:`<1` beats naive).

Use several
-----------

No single metric tells the whole story — **MAPE** can look great while **bias** quietly builds, and
**MAE** can hide a few enormous misses — so report several: an absolute metric (MAE / RMSE), a scaled one
(MASE), and a **bias** measure.

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Time Series Forecasting <256-time-series-forecasting>` · :doc:`Relative accuracy <258-relative-accuracy>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Forecast Error <https://insightful-data-lab.com/2025/08/22/forecast-error/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

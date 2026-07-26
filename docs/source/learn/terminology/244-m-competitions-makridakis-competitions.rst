:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-m-competitions-makridakis-competitions:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>M-Competitions (Makridakis Competitions)</b></div>`

==========================================
M-Competitions (Makridakis Competitions)
==========================================

*Influential forecasting competitions benchmarking time-series methods.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What they are
-------------

The **M-competitions** are a series of large-scale **forecasting competitions** (M1 through M6) organized
by **Spyros Makridakis** and colleagues to gather **empirical evidence** about which forecasting methods
actually work best in practice — not just in theory.

Key findings
------------

Across the early competitions **no single method dominated**, and **simple methods** (naïve, exponential
smoothing, ARIMA) proved **tough baselines** that often matched or beat more complex statistical models;
**combining** forecasts reliably improved accuracy. **M4** (2018; 100,000 series, 61 methods) found the
best results came from **hybrid statistical + ML** approaches and combinations, while pure-ML methods
fared poorly. **M5** (2020; Walmart hierarchical retail data on Kaggle, roughly ``$100,000`` in prizes)
was the first in which **ML methods dominated** the leaderboard, and it put **probabilistic / uncertainty**
forecasting center stage.

Why they matter
---------------

For four decades the M-competitions have shaped forecasting — establishing that **combinations and
hybrids** win, that **simple baselines** must always be checked, and that **probabilistic forecasting** is
now the standard. They directly inspired modern ML forecasting competitions (such as those on Kaggle).

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Time Series Forecasting <256-time-series-forecasting>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `M-Competitions (Makridakis Competitions) <https://insightful-data-lab.com/2025/08/22/m-competitions-makridakis-competitions/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

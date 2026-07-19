:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-lstm-long-short-term-memory-networks:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>LSTM — Long Short-Term Memory Networks</b></div>`

========================================
LSTM — Long Short-Term Memory Networks
========================================

*A recurrent network with gating that captures long-range dependencies in sequences.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

An **LSTM (Long Short-Term Memory network)** is a special **recurrent neural network** for sequential
data that overcomes the **vanishing- and exploding-gradient** problem of vanilla RNNs. It adds a
**gated memory** structure that learns what to **keep, update and forget** across long sequences.

The gated cell
--------------

A **cell state** runs through the sequence like a conveyor belt, and three **gates** — sigmoids in
:math:`[0, 1]` that decide *how much* to let through — govern it: a **forget gate**
:math:`f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)`, an **input gate** :math:`i_t` with candidate
values :math:`\tilde{C}_t = \tanh(\cdot)`, and an **output gate** :math:`o_t`. The updates are

.. math::

   C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t, \qquad h_t = o_t \odot \tanh(C_t),

with :math:`C_t` the long-term cell state and :math:`h_t` the short-term hidden state.

Variants
--------

Common variants include the **bidirectional LSTM** (reads a sequence forwards and backwards), the
**stacked LSTM** (several layers for depth), the **peephole LSTM** (gates can see the cell state), and
the **GRU**, a simpler cousin that merges the forget and input gates.

Uses, strengths, weaknesses
---------------------------

LSTMs power **NLP** (text generation, translation, sentiment), **speech recognition**, **time-series
forecasting** (demand, stock prices, anomalies) and **control systems**. They capture dependencies
across **50-100+ steps**, but they are **computationally heavy**, slower than GRUs, and now often
**outperformed by Transformers** on very long sequences.

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Autoencoder <171-autoencoder>` · :doc:`Embedding <173-embedding>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Time Series <010-time-series>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `LSTM — Long Short-Term Memory Networks <https://insightful-data-lab.com/2025/08/23/lstm-long-short-term-memory-networks/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

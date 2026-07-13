:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-low-pass-filtering:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Low-pass Filtering</b></div>`

====================
Low-pass Filtering
====================

*A filter that keeps low-frequency content and attenuates high-frequency noise in a signal.*

What it is
----------

A **low-pass filter (LPF)** lets *low-frequency* content through while attenuating
*high-frequency* content. In practice that means smoothing a signal and removing
high-frequency noise while keeping the slow-moving structure.

Frequency-domain view
---------------------

An ideal low-pass filter keeps everything below a **cutoff frequency** :math:`f_c`
and removes everything above it:

.. math::

   H(f) = \begin{cases} 1 & |f| \le f_c \\ 0 & |f| > f_c \end{cases}

Real filters approximate this brick wall with a smoother roll-off.

Time-domain view
----------------

Equivalently, low-pass filtering is **convolution with a smoothing kernel** — a
moving average or a Gaussian window — which is exactly the smoothing used in
time-series analysis.

Common filter types
-------------------

- **Ideal** — perfect sharp cutoff (theoretical only).
- **Butterworth** — flat passband, smooth roll-off.
- **Chebyshev** — sharper cutoff at the cost of passband ripple.
- **Digital FIR / IIR** — the workhorses of practical DSP.
- **Moving average** — the simplest crude low-pass filter.

Where it's used
---------------

Removing hiss from audio, blurring images, extracting long-term trends from noisy
time series, isolating frequency bands in communications, and cleaning ECG/EEG
signals in biomedicine.

Example
-------

Daily stock prices wobble with short-term noise; a low-pass filter strips the
wobble and leaves the longer-term trend visible.

.. code-block:: python

   import numpy as np
   from scipy.signal import butter, filtfilt

   t = np.linspace(0, 1, 500)
   signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*50*t)   # 5 Hz + 50 Hz
   noisy = signal + 0.3*np.random.randn(len(t))

   b, a = butter(N=4, Wn=0.1)        # cutoff at 0.1 x Nyquist
   clean = filtfilt(b, a, noisy)     # zero-phase filtering

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Subsampling <001-subsampling>` · :doc:`Downsampling <368-downsampling>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Time Series <010-time-series>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Low-pass Filtering <https://insightful-data-lab.com/2025/08/30/low-pass-filtering/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

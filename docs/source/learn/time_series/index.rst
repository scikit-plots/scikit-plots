:html_theme.sidebar_secondary.remove:

..
   ##################################################################
   learn/time_series/index.rst
   ==================================================================
   Classical (Box–Jenkins) time-series hub, framed for scikit-plots
   users.  Source context (framing only, re-expressed here):
   https://insightful-data-lab.com/category/introduction-to-time-series/ (18 posts)
   ------------------------------------------------------------------
   Extensions: sphinx_design, sphinx_tags (bottom), sphinx_copybutton.
   Underlines: = section (overline)  - subsection  ^ subsubsection
   ##################################################################

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

A **time series** is a sequence of observations indexed by time, where
order and dependence matter. This hub walks the classical **Box–Jenkins**
path that the source corpus follows: from stationarity and autocorrelation,
through the AR / MA / ARMA / ARIMA / SARIMA model family, to estimation,
diagnostics and forecasting.

Read it at any depth:

* **newcomers** — what makes time-series data special, and stationarity;
* **practitioners** — reading ACF/PACF and fitting ARIMA in statsmodels;
* **researchers** — estimation (Yule–Walker, MLE), order selection and
  residual diagnostics.

.. warning::

   Time series breaks the i.i.d. assumption behind ordinary
   cross-validation. **Never** shuffle: validate forward in time
   (walk-forward) to avoid leaking the future into the past.

.. note::

   Open a dropdown for detail and follow **See also** links. Snippets use
   real ``statsmodels`` / ``pandas`` / ``scikit-learn`` calls. This page
   pairs with the :ref:`Terminology reference <terminology-index>` (Signal
   Processing & Time Series) and the
   :ref:`Bayesian Data Analysis hub <bayesian-data-analysis-index>`.

----------------------------------------------------------------------

.. _ts-discovery:

Discovery at a Glance
----------------------------------------------------------------------

.. tab-set::
   :class: sd-width-100

   .. tab-item:: 🟢 Start Here — Foundations
      :sync: level-foundations

      What is different about ordered data.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: 📈 What is a Time Series?
            :link: ts-what-is
            :link-type: ref
            :class-card: sd-border-1

            Trend, seasonality and noise — the components hiding in a
            sequence.

         .. grid-item-card:: ⚖️ Stationarity
            :link: ts-stationarity
            :link-type: ref
            :class-card: sd-border-1

            The property most classical models assume, and how to get it
            by differencing.

         .. grid-item-card:: 🔗 ACF & PACF
            :link: ts-acf-pacf
            :link-type: ref
            :class-card: sd-border-1

            The two correlation fingerprints that reveal model order.

   .. tab-item:: 🔵 Core — The Model Family
      :sync: level-core

      AR, MA and their combinations.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: 🔁 AR & MA
            :link: ts-ar-ma
            :link-type: ref
            :class-card: sd-border-1

            Regress on the past (AR) or on past shocks (MA) — the two
            atoms.

         .. grid-item-card:: 🧱 ARIMA
            :link: ts-arima
            :link-type: ref
            :class-card: sd-border-1

            How a nonstationary model is built from a stationary ARMA via
            differencing.

         .. grid-item-card:: 🌗 SARIMA
            :link: ts-sarima
            :link-type: ref
            :class-card: sd-border-1

            Adding a seasonal layer for weekly / yearly periodicity.

   .. tab-item:: 🔴 Advanced — Estimate, Select, Forecast
      :sync: level-advanced

      Fit it, check it, project it forward.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: 🧮 Estimation
            :link: ts-estimation
            :link-type: ref
            :class-card: sd-border-1

            Yule–Walker and Gaussian maximum likelihood for ARMA
            parameters.

         .. grid-item-card:: 🎯 Order Selection & Diagnostics
            :link: ts-order-diagnostics
            :link-type: ref
            :class-card: sd-border-1

            AIC/BIC to pick (p, d, q); residual checks to trust the fit.

         .. grid-item-card:: 🔮 Forecasting
            :link: ts-forecasting
            :link-type: ref
            :class-card: sd-border-1

            Best linear prediction, multi-step horizons, and exponential
            smoothing.

----------------------------------------------------------------------

.. _ts-foundations:

Part 1 — Time Series Foundations
----------------------------------------------------------------------

.. dropdown:: What is a Time Series?
   :color: primary
   :icon: graph
   :name: ts-what-is
   :open:

   **What is it?**

   An ordered sequence :math:`\{x_t\}_{t=1}^{T}` of observations sampled
   over time. It is usually decomposed into:

   * **Trend** — long-run direction;
   * **Seasonality** — fixed-period cycles (daily, weekly, yearly);
   * **Residual / noise** — what is left after trend and seasonality.

   **pandas**

   .. code-block:: python

      import pandas as pd
      from statsmodels.tsa.seasonal import seasonal_decompose

      s = pd.read_csv("series.csv", parse_dates=["date"], index_col="date")
      result = seasonal_decompose(s["value"], model="additive", period=12)
      result.plot()

   .. seealso::

      :ref:`ts-stationarity` · :ref:`terminology-signal-timeseries`

.. dropdown:: Stationarity
   :color: primary
   :icon: meter
   :name: ts-stationarity

   **What is it?**

   A series is (weakly) **stationary** when its mean, variance and
   autocovariance do not change over time. Most classical models assume
   this, so a trending/seasonal series is first **differenced** to remove
   the changing parts:

   .. math::

      \nabla x_t = x_t - x_{t-1}

   The **ADF test** checks for a unit root (nonstationarity):

   .. code-block:: python

      from statsmodels.tsa.stattools import adfuller
      stat, pvalue, *_ = adfuller(s["value"])
      # small p-value → reject unit root → treat as stationary

   .. seealso::

      :ref:`ts-acf-pacf` · :ref:`ts-arima`

.. dropdown:: Autocorrelation — ACF & PACF
   :color: primary
   :icon: pulse
   :name: ts-acf-pacf

   **What is it?**

   * **ACF** (autocorrelation function) — correlation between the series
     and its own lag :math:`k`:

   .. math::

      \rho(k) = \frac{\gamma(k)}{\gamma(0)}, \qquad
      \gamma(k) = \operatorname{Cov}(x_t, x_{t-k})

   * **PACF** (partial autocorrelation) — the correlation at lag :math:`k`
     *after removing* the effect of shorter lags.

   Their decay/cut-off patterns are the classic fingerprint for choosing
   AR vs. MA order: a PACF that cuts off after lag *p* suggests AR(*p*); an
   ACF that cuts off after lag *q* suggests MA(*q*).

   .. code-block:: python

      from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
      plot_acf(s["value"], lags=40)
      plot_pacf(s["value"], lags=40, method="ywm")   # Yule–Walker

   .. seealso::

      :ref:`ts-ar-ma` · :ref:`ts-estimation`

----------------------------------------------------------------------

.. _ts-models:

Part 2 — The Classical Model Family
----------------------------------------------------------------------

.. dropdown:: AR & MA Models
   :color: info
   :icon: sync
   :name: ts-ar-ma

   **Autoregressive — AR(p)** regresses the present on its own past:

   .. math::

      x_t = c + \sum_{i=1}^{p} \phi_i\, x_{t-i} + \varepsilon_t

   **Moving average — MA(q)** regresses the present on past shocks:

   .. math::

      x_t = \mu + \varepsilon_t + \sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j}

   **ARMA(p, q)** combines both on a stationary series.

   .. seealso::

      :ref:`ts-arima` · :ref:`ts-acf-pacf`

.. dropdown:: ARIMA — Integrating Nonstationary Series
   :color: info
   :icon: stack
   :name: ts-arima

   **What is it?**

   **ARIMA(p, d, q)** applies an ARMA(p, q) model to a series that has
   been **differenced** :math:`d` times to make it stationary — exactly
   the "build a nonstationary model from a stationary one" idea in the
   source.

   **statsmodels**

   .. code-block:: python

      from statsmodels.tsa.arima.model import ARIMA

      model = ARIMA(s["value"], order=(1, 1, 1))   # (p, d, q)
      fit = model.fit()
      print(fit.summary())

   .. seealso::

      :ref:`ts-sarima` · :ref:`ts-order-diagnostics`

.. dropdown:: SARIMA — Adding Seasonality
   :color: info
   :icon: calendar
   :name: ts-sarima

   **What is it?**

   **SARIMA** extends ARIMA with a seasonal :math:`(P, D, Q)_m` component
   (period :math:`m`) to capture repeating cycles on top of the
   non-seasonal dynamics.

   .. code-block:: python

      from statsmodels.tsa.statespace.sarimax import SARIMAX

      model = SARIMAX(s["value"], order=(1, 1, 1),
                      seasonal_order=(1, 1, 1, 12))   # monthly seasonality
      fit = model.fit(disp=False)

   .. seealso::

      :ref:`ts-forecasting`

----------------------------------------------------------------------

.. _ts-fit:

Part 3 — Estimate, Select & Forecast
----------------------------------------------------------------------

.. dropdown:: Estimation — Yule–Walker & Gaussian MLE
   :color: secondary
   :icon: number
   :name: ts-estimation

   **What is it?**

   * **Yule–Walker** — solves the linear system linking AR coefficients to
     the autocovariances; a fast, closed-form *preliminary* estimate for
     AR models.
   * **Gaussian MLE** — maximises the likelihood under a Gaussian
     innovation assumption; the standard estimator for full ARMA/ARIMA
     models (what ``statsmodels`` reports).

   .. seealso::

      :ref:`ts-order-diagnostics`

.. dropdown:: Order Selection & Residual Diagnostics
   :color: secondary
   :icon: checklist
   :name: ts-order-diagnostics

   **Selecting (p, d, q)** — fit candidates and compare information
   criteria; lower is better:

   .. math::

      \text{AIC} = 2k - 2\ln \hat{L}, \qquad
      \text{BIC} = k\ln n - 2\ln \hat{L}

   **Diagnostics after fitting** — the residuals should look like white
   noise: no autocorrelation (Ljung–Box test), roughly normal, constant
   variance.

   .. code-block:: python

      import statsmodels.api as sm
      fit.plot_diagnostics(figsize=(10, 8))            # built-in panel
      sm.stats.acorr_ljungbox(fit.resid, lags=[10])    # whiteness test

   .. seealso::

      :ref:`ts-forecasting` · :ref:`data-prep-residuals`

.. dropdown:: Forecasting — Linear Prediction & Smoothing
   :color: secondary
   :icon: telescope
   :name: ts-forecasting

   **Best linear predictor** — under stationarity, the minimum-MSE linear
   forecast is built from the autocovariance structure (and the PACF gives
   the one-step coefficients). Forecasts extend to **multi-step** horizons
   with widening uncertainty bands.

   **Exponential smoothing** — a complementary family that forecasts by
   exponentially weighting recent observations (Holt–Winters adds trend
   and seasonality):

   .. code-block:: python

      from statsmodels.tsa.holtwinters import ExponentialSmoothing

      hw = ExponentialSmoothing(s["value"], trend="add",
                                seasonal="add", seasonal_periods=12).fit()
      forecast = hw.forecast(12)

   **Validate forward in time:**

   .. code-block:: python

      from sklearn.model_selection import TimeSeriesSplit
      for tr_idx, te_idx in TimeSeriesSplit(n_splits=5).split(s):
          ...   # train on the past, test on the next block

   .. seealso::

      :ref:`ts-sarima` · :ref:`terminology-signal-timeseries`

----------------------------------------------------------------------

.. _ts-stack-map:

Map to the Python Time-Series Stack
----------------------------------------------------------------------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: statsmodels — tsa
      :link: https://www.statsmodels.org/stable/tsa.html

      ARIMA, SARIMAX, exponential smoothing, ACF/PACF, diagnostics.

   .. grid-item-card:: pandas — time series
      :link: https://pandas.pydata.org/docs/user_guide/timeseries.html

      Datetime indexing, resampling, rolling windows.

   .. grid-item-card:: scikit-learn — TimeSeriesSplit
      :link: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

      Leakage-free walk-forward cross-validation.

   .. grid-item-card:: scikit-plots — residual diagnostics
      :link: https://scikit-plots.github.io/dev/auto_examples/stats/plot_residuals_distribution_script.html

      Distribution / Q–Q checks for model residuals.

----------------------------------------------------------------------

.. _ts-sources:

Sources
----------------------------------------------------------------------

Verified during preparation of this page; resolvable at build date.

**Source context (framing only, re-expressed in our own words)**

* Introduction to Time Series category (18 posts):
  https://insightful-data-lab.com/category/introduction-to-time-series/

**Official documentation (API calls used above)**

* statsmodels — time-series analysis (``tsa``):
  https://www.statsmodels.org/stable/tsa.html
* pandas — time-series / date functionality:
  https://pandas.pydata.org/docs/user_guide/timeseries.html
* scikit-learn — ``TimeSeriesSplit``:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

**scikit-plots (this project)**

* Example gallery: https://scikit-plots.github.io/dev/auto_examples/index.html
* Terminology reference: :ref:`terminology-index`

**Standard references**

* Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd
  ed., free): https://otexts.com/fpp3/
* Brockwell & Davis, *Introduction to Time Series and Forecasting*.

..
   ##################################################################
   Tags — bottom of page, project controlled vocabulary only.
   (Promote to `domain: time series` once multiple TS pages exist.)
   ##################################################################

.. tags::
   purpose: reference,
   domain: statistics,
   level: beginner,
   level: intermediate,
   level: advanced

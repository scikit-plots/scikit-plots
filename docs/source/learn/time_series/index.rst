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

.. raw:: html

   <div style="text-align:center;margin:0.4rem 0 0.4rem">
   <input id="term-filter" type="search" autocomplete="off" spellcheck="false"
          placeholder="&#128269;&nbsp; Type to filter 18 lessons &mdash; by title or keyword&hellip;"
          style="width:100%;max-width:100%;padding:0.55rem 1rem;font-size:1rem;
                 border:1px solid var(--pst-color-border,#ccc);border-radius:0.55rem;box-sizing:border-box;
                 background:transparent;color:inherit"/>
   <div id="term-filter-count" style="opacity:0.65;font-size:0.85rem;
        min-height:1.2em;margin-top:0.35rem"></div>
   </div>
   <script>
   document.addEventListener('DOMContentLoaded',function(){
     var inp=document.getElementById('term-filter');if(!inp){return;}
     var dds=[].slice.call(document.querySelectorAll('details.sd-dropdown'));
     var az=document.querySelector('details.term-az');
     var items=[];
     dds.forEach(function(d){[].slice.call(d.querySelectorAll('li')).forEach(
       function(li){items.push({li:li,d:d,t:li.textContent.toLowerCase()});});});
     var cnt=document.getElementById('term-filter-count');
     inp.addEventListener('input',function(){
       var q=inp.value.trim().toLowerCase();var n=0;
       dds.forEach(function(d){d.tHits=0;});
       items.forEach(function(it){
         var hit=!q||it.t.indexOf(q)!==-1;
         it.li.style.display=hit?'':'none';
         if(hit){it.d.tHits+=1;if(az&&it.d===az){n+=1;}}});
       dds.forEach(function(d){
         if(q){d.style.display=d.tHits?'':'none';d.open=d.tHits>0;}
         else{d.style.display='';d.open=false;}});
       if(cnt){cnt.textContent=(q&&az)?(n+' of 18 match'+(n===1?'':'s')):'';}
     });
   });
   </script>

.. _ts-stage-orient:

.. dropdown:: 🧭 Stage 1 — Orientation  ·  2 lessons
   :animate: fade-in-slide-down

   *What time-series data is, why order carries information, and setting up the tools.*  ·  *beginner*

   * :doc:`01 · What Are Time Series, and How Are They Used? <01-what-are-time-series-and-how-are-they-used>` — Sequences of time-ordered observations, their trend / seasonal / residual parts, and where forecasting applies.
   * :doc:`02 · Getting Started with R <02-getting-started-with-r>` — Setting up a working environment — the source's R tooling and the Python (statsmodels) path used here.

.. _ts-stage-stationarity:

.. dropdown:: 📐 Stage 2 — Stationarity  ·  2 lessons
   :animate: fade-in-slide-down

   *The property that makes a series learnable — how to recognise it and how to achieve it.*  ·  *beginner*

   * :doc:`03 · A Gentle Introduction to Stationarity <03-a-gentle-introduction-to-stationarity>` — Why a stable mean, variance and autocovariance make a series learnable — and how differencing helps.
   * :doc:`04 · Weak and Strong Stationarity <04-weak-and-strong-stationarity>` — The precise definitions: strict distributional invariance versus the weaker second-order (covariance) form.

.. _ts-stage-arma:

.. dropdown:: 🔗 Stage 3 — Linear & ARMA Processes  ·  4 lessons
   :animate: fade-in-slide-down

   *The building blocks: linear processes, the AR / MA / ARMA family, and their autocorrelation.*  ·  *intermediate*

   * :doc:`05 · Linear Processes <05-linear-processes>` — Series written as a linear filter of white noise — the general form underlying AR, MA and ARMA.
   * :doc:`06 · Understanding ARMA Processes <06-understanding-arma-processes>` — Combining autoregressive and moving-average terms; causality, invertibility and what each parameter does.
   * :doc:`07 · Computing ACFs of Causal AR(2) Processes Using Difference Equations <07-computing-acfs-of-causal-ar-2-processes-using-difference-equations>` — Solving the autocorrelation of an AR(2) by treating its recursion as a linear difference equation.
   * :doc:`08 · Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q) <08-understanding-acfs-via-difference-equations-for-ar-p-and-arma-p-q>` — Generalising the difference-equation method to the autocorrelation of any AR(p) or ARMA(p, q).

.. _ts-stage-prediction:

.. dropdown:: 🎯 Stage 4 — Prediction & the Sample ACF / PACF  ·  2 lessons
   :animate: fade-in-slide-down

   *Optimal linear forecasting and the empirical correlation tools used to identify model order.*  ·  *intermediate*

   * :doc:`09 · Best Linear Predictor of a Stationary Process <09-best-linear-predictor-of-a-stationary-process>` — The optimal linear forecast, the projection principle, and how the PACF emerges from it.
   * :doc:`10 · Sample ACF and Sample PACF <10-sample-acf-and-sample-pacf>` — Estimating autocorrelation from data, their sampling behaviour, and reading them to pick model order.

.. _ts-stage-estimation:

.. dropdown:: 🧮 Stage 5 — Estimation  ·  2 lessons
   :animate: fade-in-slide-down

   *Fitting parameters: Yule–Walker for AR models, Gaussian maximum likelihood for ARMA.*  ·  *advanced*

   * :doc:`11 · Preliminary Estimation for AR Models and the Yule–Walker Equations <11-preliminary-estimation-for-ar-models-and-the-yule-walker-equations>` — Method-of-moments AR fitting by solving the Yule–Walker system from sample autocovariances.
   * :doc:`12 · Maximum Likelihood Estimation for ARMA Models (Gaussian MLE) <12-maximum-likelihood-estimation-for-arma-models-gaussian-mle>` — Fitting ARMA by maximising the Gaussian likelihood — the standard, efficient estimator.

.. _ts-stage-building:

.. dropdown:: 🏗️ Stage 6 — Building & Forecasting Models  ·  6 lessons
   :animate: fade-in-slide-down

   *Diagnostics, order selection, ARIMA / SARIMA, multi-step forecasting and exponential smoothing.*  ·  *advanced*

   * :doc:`13 · Diagnostics After Fitting a Time Series Model <13-diagnostics-after-fitting-a-time-series-model>` — Checking standardized residuals for leftover structure: normality, autocorrelation and the Ljung–Box test.
   * :doc:`14 · Order Selection for Time Series Models <14-order-selection-for-time-series-models>` — Choosing p, d, q with information criteria (AIC / BIC) balanced against parsimony and diagnostics.
   * :doc:`15 · ARIMA Models: How Nonstationary Models Are Built from Stationary Ones <15-arima-models-how-nonstationary-models-are-built-from-stationary-ones>` — Differencing to remove trend, turning a nonstationary series into an ARMA-modellable one.
   * :doc:`16 · SARIMA Models: Seasonal ARIMA <16-sarima-models-seasonal-arima>` — Extending ARIMA with seasonal AR, differencing and MA terms for periodic patterns.
   * :doc:`17 · Beyond One-Step Ahead Predictions <17-beyond-one-step-ahead-predictions>` — Multi-step forecasting, how uncertainty compounds with horizon, and forecast intervals.
   * :doc:`18 · Exponential Smoothing Models <18-exponential-smoothing-models>` — Weighted-average forecasting (simple, Holt, Holt–Winters) as a practical complement to ARIMA.

🔤 Every lesson, A–Z index
---------------------------

.. dropdown:: 🔠 Open the full alphabetical index
   :class-container: term-az

   .. hlist::
      :columns: 2

      * :doc:`A Gentle Introduction to Stationarity <03-a-gentle-introduction-to-stationarity>`
      * :doc:`ARIMA Models: How Nonstationary Models Are Built from Stationary Ones <15-arima-models-how-nonstationary-models-are-built-from-stationary-ones>`
      * :doc:`Best Linear Predictor of a Stationary Process <09-best-linear-predictor-of-a-stationary-process>`
      * :doc:`Beyond One-Step Ahead Predictions <17-beyond-one-step-ahead-predictions>`
      * :doc:`Computing ACFs of Causal AR(2) Processes Using Difference Equations <07-computing-acfs-of-causal-ar-2-processes-using-difference-equations>`
      * :doc:`Diagnostics After Fitting a Time Series Model <13-diagnostics-after-fitting-a-time-series-model>`
      * :doc:`Exponential Smoothing Models <18-exponential-smoothing-models>`
      * :doc:`Getting Started with R <02-getting-started-with-r>`
      * :doc:`Linear Processes <05-linear-processes>`
      * :doc:`Maximum Likelihood Estimation for ARMA Models (Gaussian MLE) <12-maximum-likelihood-estimation-for-arma-models-gaussian-mle>`
      * :doc:`Order Selection for Time Series Models <14-order-selection-for-time-series-models>`
      * :doc:`Preliminary Estimation for AR Models and the Yule–Walker Equations <11-preliminary-estimation-for-ar-models-and-the-yule-walker-equations>`
      * :doc:`Sample ACF and Sample PACF <10-sample-acf-and-sample-pacf>`
      * :doc:`SARIMA Models: Seasonal ARIMA <16-sarima-models-seasonal-arima>`
      * :doc:`Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q) <08-understanding-acfs-via-difference-equations-for-ar-p-and-arma-p-q>`
      * :doc:`Understanding ARMA Processes <06-understanding-arma-processes>`
      * :doc:`Weak and Strong Stationarity <04-weak-and-strong-stationarity>`
      * :doc:`What Are Time Series, and How Are They Used? <01-what-are-time-series-and-how-are-they-used>`

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

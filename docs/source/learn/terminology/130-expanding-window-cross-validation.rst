:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-expanding-window-cross-validation:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧷&nbsp;&nbsp;<b>Expanding Window Cross-Validation</b></div>`

===================================
Expanding Window Cross-Validation
===================================

*Time-series CV that grows the training window as it walks forward.*

What it is
----------

**Expanding-window** cross-validation is a time-series scheme where the **training set grows**
over time while the test set always takes the **next** period. The rule is the usual one —
**train on the past, predict the future** — and crucially, **once data enters training it
stays** in every later fold.

How it works
------------

Order the data chronologically, start from an initial training period, and validate on the
next. Then **expand** the training window to absorb more past data and validate on the
following period, repeating to the end of the series.

Example
-------

Over **2020–2024**: train **2020** → test **2021**; train **2020–2021** → test **2022**;
train **2020–2022** → test **2023**; train **2020–2023** → test **2024**. The training block
**keeps growing**; the test always sits just after it.

vs the rolling window
---------------------

Both respect time order; they differ on memory. **Expanding** accumulates **all** history,
which helps when **older data is still relevant** (finance, macroeconomic forecasting). The
**rolling/sliding** window holds a **fixed** size and drops the oldest data, which helps when
**recent data dominates** (stock trading, demand forecasting). The choice is really a
question of whether the process is stationary.

----

*Theme:* :ref:`Validation & Cross-Validation <term-theme-validation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Temporal autocorrelation (Serial Correlation) <127-temporal-autocorrelation-serial-correlation>` · :doc:`Time Series <010-time-series>` · :doc:`Data Leakage <131-data-leakage>`

----

.. hint::
   **More in Validation & Cross-Validation**

   :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Data Leakage <131-data-leakage>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Expanding Window Cross-Validation <https://insightful-data-lab.com/2025/08/24/expanding-window-cross-validation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-sliding-window-rolling-window-cross-validation:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧷&nbsp;&nbsp;<b>Sliding Window (Rolling Window) Cross-Validation</b></div>`

==================================================
Sliding Window (Rolling Window) Cross-Validation
==================================================

*Time-series CV using a fixed-size window that moves forward through time.*

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

**Sliding-window** (or **rolling-window**) cross-validation is a time-series validation
scheme where the training set has a **fixed size** and **slides forward** through time: as
new data enters the window, the **oldest data drops out**. Like all time-series CV, it
**respects time order** — past trains, future tests.

How it works
------------

Fix a **window size**, train on that window, validate on the next step, then **slide forward
and repeat**. Each fold uses a same-length training block ending just before its test block.

Example
-------

With a 2-year window over **2020–2024**: train **2020–2021** → test **2022**; train
**2021–2022** → test **2023**; train **2022–2023** → test **2024**. Notice 2020 is
**dropped** once the window moves past it.

vs the expanding window
-----------------------

The contrast is what happens to old data. An **expanding window keeps all history**, suiting
cases where old data stays relevant (macroeconomics, cumulative learning). A **sliding window
discards it**, suiting **non-stationary** settings where recent data is more predictive —
financial markets, demand forecasting, IoT sensor streams.

Benefits and the size trade-off
-------------------------------

It keeps the model **focused on recent patterns** and **bounds compute** (the training set
never grows without limit). The cost: it can **forget useful long-run history**, and the
**window size is a critical knob** — too short is noisy, too long is unresponsive.

----

*Theme:* :ref:`Validation & Cross-Validation <term-theme-validation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Temporal autocorrelation (Serial Correlation) <127-temporal-autocorrelation-serial-correlation>` · :doc:`Time Series <010-time-series>` · :doc:`Data Leakage <131-data-leakage>`

----

.. hint::
   **More in Validation & Cross-Validation**

   :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Data Leakage <131-data-leakage>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Sliding Window (Rolling Window) Cross-Validation <https://insightful-data-lab.com/2025/08/24/sliding-window-rolling-window-cross-validation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

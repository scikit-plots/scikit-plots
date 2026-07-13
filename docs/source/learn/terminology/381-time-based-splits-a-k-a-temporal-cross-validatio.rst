:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-time-based-splits-a-k-a-temporal-cross-validation-rolling-window-validation:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧷&nbsp;&nbsp;<b>Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation)</b></div>`

=================================================================================
Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation)
=================================================================================

*Validation that respects time order to avoid using the future to predict the past.*

What it is
----------

A **time-based split** orders data by **time** and trains on the **past** while validating and testing on the
**future** — the earliest records for training, the most recent held out. It reproduces the reality of
deployment, where **future data doesn't exist** at training time.

Why it's needed
---------------

**Time-series** data violates the **i.i.d.** assumption behind ordinary splitting — observations depend on
**prior** ones. Shuffling or random k-fold would let the model **train on the future** to predict the past, a
**temporal leakage** that badly **overstates** accuracy.

How it's done
-------------

Schemes like a **rolling forecasting origin** (walk-forward) or an **expanding / sliding window** repeatedly
move the training window forward in time, so every evaluation always predicts **later** data than it trained
on. Look-ahead **features** must be avoided too.

----

*Theme:* :ref:`Validation & Cross-Validation <term-theme-validation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`Time Series Forecasting <256-time-series-forecasting>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>`

----

.. hint::
   **More in Validation & Cross-Validation**

   :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Data Leakage <131-data-leakage>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <https://insightful-data-lab.com/2025/08/19/time-based-splits-a-k-a-temporal-cross-validation-rolling-window-validation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-blocked-splits-single-holdout:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧷&nbsp;&nbsp;<b>Blocked Splits (Single Holdout)</b></div>`

=================================
Blocked Splits (Single Holdout)
=================================

*Splitting time-ordered data into contiguous train/test blocks to avoid leakage.*

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

A **blocked split** (or **single holdout** for time series) is the simplest time-series
validation: cut the data into **one contiguous training block** and **one later contiguous
test block**. Unlike a random split, it **respects temporal order** — train on the past, test
on the future — which is mandatory when observations are autocorrelated.

How it works
------------

Order the data chronologically, take an **early block to train** and the **later block to
test**, fit on the first and evaluate on the second. For data spanning **2018–2022**, train
on **2018–2020** and test on **2021–2022** — a single cut, one split.

Why and when
------------

It is **simple, fast**, and the natural **first baseline** for a time-series model — a good
fit when retraining is infrequent or the series is stable.

The limitation
--------------

There is **only one split**, so the estimate is **sensitive to where you cut** and gives a
less robust read on generalisation than rolling schemes. An unusual test window (a pandemic
spike, say) can misrepresent the model. The more robust alternatives keep the time order but
add splits: an **expanding window** grows the train set forward over many folds, and a
**sliding window** rolls a fixed-size train set forward — trading simplicity for a steadier
estimate.

----

*Theme:* :ref:`Validation & Cross-Validation <term-theme-validation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Temporal autocorrelation (Serial Correlation) <127-temporal-autocorrelation-serial-correlation>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Time Series <010-time-series>`

----

.. hint::
   **More in Validation & Cross-Validation**

   :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Data Leakage <131-data-leakage>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Blocked Splits (Single Holdout) <https://insightful-data-lab.com/2025/08/24/blocked-splits-single-holdout/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

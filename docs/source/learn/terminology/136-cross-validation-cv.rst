:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-cross-validation-cv:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧷&nbsp;&nbsp;<b>Cross-Validation (CV)</b></div>`

=======================
Cross-Validation (CV)
=======================

*Estimating generalisation by repeatedly training and testing on different splits.*

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

**Cross-validation (CV)** is the umbrella **model-evaluation technique**: split the data into
multiple subsets, rotate which subset is held out for testing, and **average** the results so
the performance estimate does not hinge on a single lucky or unlucky split. Its goal is to
gauge how well a model **generalises to unseen data**, and it underpins **model selection,
hyperparameter tuning and overfitting prevention**.

How it works
------------

Partition into **k folds**; for each, train on the rest and test on the held-out fold; repeat
so every fold serves as test once; average across folds for the final estimate.

The main flavours
-----------------

**k-fold** is the balanced default; **stratified k-fold** holds class proportions steady for
imbalanced classification; **leave-one-out (LOOCV)** uses one sample per fold — accurate but
costly; **time-series CV** (rolling or expanding windows) respects time order for sequential
data; and **nested CV** wraps an inner tuning loop inside an outer evaluation loop so that
**hyperparameter selection does not leak** into the performance estimate.

Why it matters, and the costs
-----------------------------

CV gives a **robust, lower-variance** read on generalisation and is the standard harness for
**grid, random and Bayesian** hyperparameter search. The price is **compute** (the model is
trained many times) and the need to **match the fold type to the data** — most importantly,
never shuffling a time series.

----

*Theme:* :ref:`Validation & Cross-Validation <term-theme-validation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`Data Leakage <131-data-leakage>`

----

.. hint::
   **More in Validation & Cross-Validation**

   :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Data Leakage <131-data-leakage>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Cross-Validation (CV) <https://insightful-data-lab.com/2025/08/24/cross-validation-cv/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

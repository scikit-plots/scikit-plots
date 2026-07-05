:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-evaluation-set:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧷&nbsp;&nbsp;<b>Evaluation Set</b></div>`

================
Evaluation Set
================

*Held-out data used to measure model performance.*

What it is
----------

An **evaluation set** is data **held out** from training so a model can be scored on examples it has **never
seen** — the only honest way to estimate how it will **generalize**. In practice it splits into two roles.

Validation vs test
------------------

The **validation set** is used **repeatedly** during development — tuning hyperparameters, early stopping,
choosing between models; the **test set** is touched **once**, at the very end, for a final **unbiased**
estimate. Any peek at the test set during development **contaminates** it and inflates the reported score.

The cardinal rule
-----------------

**Split first**, then fit every preprocessing step (scaling, encoding) on the **training data only** and
apply it to the held-out sets. Fitting on all the data before splitting leaks information from the evaluation
set into training — the classic **data leakage** that makes scores look better than reality.

----

**Mind map — connected ideas**

   :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Model Score <364-model-score>` · :doc:`Model Stability <187-model-stability>` · :doc:`Data Drift <331-data-drift>`

----

**More in Validation & Cross-Validation**

   :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Data Leakage <131-data-leakage>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`

----

*Theme:* :ref:`Validation & Cross-Validation <term-theme-validation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Evaluation Set <https://insightful-data-lab.com/2025/08/20/evaluation-set/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate

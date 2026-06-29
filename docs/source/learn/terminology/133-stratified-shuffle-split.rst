:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-stratified-shuffle-split:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧷&nbsp;&nbsp;<b>Stratified Shuffle Split</b></div>`

==========================
Stratified Shuffle Split
==========================

*Repeated random splits that preserve class proportions in each split.*

What it is
----------

**Stratified Shuffle Split** repeatedly carves a dataset into **random train/test splits
while preserving the class distribution**. Unlike k-fold, it does **not** partition into fixed
folds — it **reshuffles and resamples** as many times as you ask, each split a fresh random
draw with the original class ratios intact.

How it works
------------

Set **n_splits** (how many reshuffles) and a **train/test size**; for each split, shuffle,
partition keeping the class proportions, and evaluate — then average across splits. Because
test sets can overlap between splits (they are independent draws), it is not a partition the
way k-fold is.

Example
-------

For 1,000 samples at **80% class A, 20% class B** with ``test_size=0.2`` over 5 splits, each
split yields train = 800 (A=640, B=160) and test = 200 (A=160, B=40) — the **80/20** ratio
holds every time.

In scikit-learn
---------------

.. code-block:: python

   from sklearn.model_selection import StratifiedShuffleSplit

   sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
   for train_idx, test_idx in sss.split(X, y):
       ...

It shines on **imbalanced or small** data where you want **many randomized splits** rather
than a fixed fold structure — the stratified counterpart to a plain shuffle split, which
randomizes but does *not* preserve class balance.

----

**Mind map — connected ideas**

   :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Class Weighting <002-class-weighting>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>`

----

**More in Validation & Cross-Validation**

   :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Data Leakage <131-data-leakage>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`

----

*Theme:* :ref:`Validation & Cross-Validation <term-theme-validation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Stratified Shuffle Split <https://insightful-data-lab.com/2025/08/24/stratified-shuffle-split/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate

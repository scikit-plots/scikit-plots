:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-k-fold-cross-validation:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧷&nbsp;&nbsp;<b>k-fold cross-validation</b></div>`

=========================
k-fold cross-validation
=========================

*Train on k-1 folds and test on the held-out fold, rotating through all k.*

What it is
----------

**k-fold cross-validation** splits the data into **k roughly equal folds** and trains and
tests the model **k times**, each run holding out a **different fold** as the test set and
training on the other :math:`k-1`. Averaging the k scores gives a lower-variance estimate of
performance than any single train/test split — which is why it is the **default** CV method.

How it works
------------

Shuffle (if order is irrelevant), split into **k folds** (commonly 5 or 10), and for each
fold :math:`i` train on the other folds and test on fold :math:`i`. Collect the k scores and
**average** them for the final metric.

Example
-------

With 1,000 samples and ``k = 5``, each fold is 200 samples: every run trains on **800** and
validates on **200**, rotating which 200 is held out, and the result is the **mean** across
the five runs.

Variations
----------

**Stratified k-fold** preserves class balance per fold (vital for imbalanced data);
**repeated k-fold** re-runs the whole process with new splits for a steadier estimate; and
**leave-one-out (LOOCV)** is the extreme :math:`k = N`, one sample per fold — very accurate,
very expensive.

In scikit-learn, and the trade-offs
-----------------------------------

.. code-block:: python

   from sklearn.model_selection import cross_val_score

   scores = cross_val_score(model, X, y, cv=5)
   print(scores.mean(), scores.std())

The gains — a **reliable estimate**, less dependence on one random split, full use of the
data, and a backbone for **hyperparameter tuning** — cost **k model fits**, and plain k-fold
is **wrong for time series**, where time-aware CV is required instead.

----

*Theme:* :ref:`Validation & Cross-Validation <term-theme-validation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Data Leakage <131-data-leakage>`

----

.. hint::
   **More in Validation & Cross-Validation**

   :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Data Leakage <131-data-leakage>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `k-fold cross-validation <https://insightful-data-lab.com/2025/08/24/k-fold-cross-validation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

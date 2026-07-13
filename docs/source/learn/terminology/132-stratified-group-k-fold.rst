:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-stratified-group-k-fold:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧷&nbsp;&nbsp;<b>Stratified Group K-Fold</b></div>`

=========================
Stratified Group K-Fold
=========================

*K-fold CV preserving class balance while keeping groups intact across folds.*

What it is
----------

**Stratified Group K-Fold** is a cross-validation scheme that fuses **three requirements** at
once: **k-fold** splitting, **stratification** (preserve the class balance in every fold), and
**grouping** (keep every group — same patient, user, session — entirely on one side of each
split). It is the right tool for **grouped *and* imbalanced** classification.

Why it's needed
---------------

Each simpler scheme covers only part of the problem. **Stratified k-fold** balances classes
but can let one group's rows fall into both train and validation, **leaking** information.
**Group k-fold** prevents that overlap but can wreck the class balance. **Stratified group
k-fold** does both — class proportions held *and* group boundaries respected.

How it works and an example
---------------------------

Identify the group key, then build folds that are simultaneously class-balanced and
group-clean. For 1,000 samples from **100 patients** with a 20/80 disease split and ``k = 5``,
each fold holds about 20 patients, preserves the ~20/80 ratio, and shares **no patient**
between train and validation.

In scikit-learn
---------------

.. code-block:: python

   from sklearn.model_selection import StratifiedGroupKFold

   cv = StratifiedGroupKFold(n_splits=5)
   for train_idx, test_idx in cv.split(X, y, groups):  # groups = patient IDs
       ...

The comparison is clean: plain **k-fold** is neither stratified nor group-aware, **stratified
k-fold** adds class balance, **group k-fold** adds group safety, and **stratified group
k-fold** is the only one with both.

----

*Theme:* :ref:`Validation & Cross-Validation <term-theme-validation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Data Leakage <131-data-leakage>` · :doc:`Class Weighting <002-class-weighting>`

----

.. hint::
   **More in Validation & Cross-Validation**

   :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Data Leakage <131-data-leakage>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Stratified Group K-Fold <https://insightful-data-lab.com/2025/08/24/stratified-group-k-fold/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

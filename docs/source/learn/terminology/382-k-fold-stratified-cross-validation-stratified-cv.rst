:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-k-fold-stratified-cross-validation-stratified-cv:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧷&nbsp;&nbsp;<b>k-fold Stratified Cross-Validation (Stratified CV)</b></div>`

====================================================
k-fold Stratified Cross-Validation (Stratified CV)
====================================================

*K-fold CV that preserves class proportions in every fold.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Stratified k-fold cross-validation** splits the data into **k folds** while **preserving each class's
proportion** in every fold — so a fold of a 5%-positive dataset stays about **5% positive**. It combines the
stability of k-fold CV with balanced folds.

Why stratify
------------

Plain **k-fold** can, by chance, build folds with **too few or missing** minority-class examples, giving
**biased** or unstable metrics — especially on **imbalanced** data. Stratification makes each fold **mirror**
the overall distribution, so the k scores are **reliable** and comparable.

How it's used
-------------

It is the **default** for classification (often **repeated stratified 10-fold**); scikit-learn provides
``StratifiedKFold``. Two cautions carry over from any CV: fit preprocessing on the **training folds only** to
avoid **leakage**, and don't use it on **time-series** data, where **time-based** splits are required instead.

----

*Theme:* :ref:`Validation & Cross-Validation <term-theme-validation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>` · :doc:`Model Stability <187-model-stability>` · :doc:`Data Drift <331-data-drift>`

----

.. hint::
   **More in Validation & Cross-Validation**

   :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Data Leakage <131-data-leakage>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `k-fold Stratified Cross-Validation (Stratified CV) <https://insightful-data-lab.com/2025/08/19/stratified-cross-validation-stratified-cv/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

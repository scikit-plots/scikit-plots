:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-multiclass-stratified-cv:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧷&nbsp;&nbsp;<b>Multiclass stratified CV</b></div>`

==========================
Multiclass stratified CV
==========================

*Stratified cross-validation maintaining each class's proportion across folds.*

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

**Multiclass stratified CV** is stratified k-fold extended **beyond two classes**: every fold
keeps approximately the **same distribution across all classes** as the full dataset. It is
the natural generalisation of binary stratification to **three or more** labels.

How it works
------------

Measure the overall class mix — say **A = 60%, B = 30%, C = 10%** — and build each fold to
mirror it, so every fold carries A, B and C in roughly those proportions. Both training and
validation sets then **represent all classes**.

Example
-------

For 1,000 samples split **A = 600, B = 300, C = 100** with ``k = 5``, a **regular k-fold**
might leave some folds with almost no class-C examples. **Multiclass stratified k-fold** gives
each fold about **A = 120, B = 60, C = 20** — the original shape, fold after fold.

Why it matters
--------------

In **imbalanced multiclass** problems, plain k-fold can starve a minority class in some folds,
producing **unstable, misleading metrics** (a fold with no class-C samples cannot measure
class-C performance). Stratification makes the evaluation **fair and stable**. It applies to
**classification only** — there is nothing to stratify in a regression target.

----

*Theme:* :ref:`Validation & Cross-Validation <term-theme-validation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Class Weighting <002-class-weighting>` · :doc:`Multiclass AUROC <022-multiclass-auroc>`

----

.. hint::
   **More in Validation & Cross-Validation**

   :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Data Leakage <131-data-leakage>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Multiclass stratified CV <https://insightful-data-lab.com/2025/08/24/multiclass-stratified-cv/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

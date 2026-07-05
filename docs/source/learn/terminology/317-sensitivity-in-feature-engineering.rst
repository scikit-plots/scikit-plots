:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-sensitivity-in-feature-engineering:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Sensitivity in Feature Engineering</b></div>`

====================================
Sensitivity in Feature Engineering
====================================

*How much a model's output responds to changes in a feature.*

What it is
----------

**Sensitivity** in feature engineering is the degree to which a model's predictions **depend on how features
are represented** — their **scale**, distribution, and encoding. Some algorithms are highly **sensitive** to
these choices; others are nearly **invariant** — and that difference decides how much preprocessing you must
do.

Who's sensitive
---------------

**Distance-based** models (KNN, SVM), **gradient-descent** learners (linear / logistic regression, neural
nets), and **regularized** models are **scale-sensitive** — a feature with a large range will **dominate**
distances or gradients unless it's **normalized**. **Tree-based** models (decision trees, random forests,
gradient boosting) split one feature at a time and are essentially **scale-invariant**.

Why it matters
--------------

Knowing a model's sensitivity tells you what preprocessing is **required** versus **wasted** — you **must**
scale for KNN or a neural net, but scaling for a random forest changes little. The same lens underlies
**feature-sensitivity analysis**: measuring how much the output moves when a feature changes reveals which
features the model actually **relies on**.

----

**Mind map — connected ideas**

   :doc:`Encode (in Feature Engineering) <318-encode-in-feature-engineering>` · :doc:`Normalize (in Feature Engineering) <319-normalize-in-feature-engineering>` · :doc:`Feature Values <188-feature-values>` · :doc:`Outlier <307-outlier>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Decision Trees <340-decision-trees>`

----

**More in Data Preparation & Features**

   :doc:`Advanced Sorting in Spreadsheets <431-advanced-sorting-in-spreadsheets>` · :doc:`Encode (in Feature Engineering) <318-encode-in-feature-engineering>` · :doc:`Normalize (in Feature Engineering) <319-normalize-in-feature-engineering>`

----

*Theme:* :ref:`Data Preparation & Features <term-theme-features>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Sensitivity in Feature Engineering <https://insightful-data-lab.com/2025/08/20/sensitivity-in-feature-engineering/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-one-vs-rest-ovr:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>One-vs-Rest (OvR)</b></div>`

===================
One-vs-Rest (OvR)
===================

*A multiclass strategy fitting one binary classifier per class.*

What it is
----------

**One-vs-Rest** (also **one-vs-all**) reduces a **K-class** problem to **K binary** ones — in each, a single
class is the **positive** and all the others are lumped together as the **negative**. It's the simplest way
to let binary tools handle many classes.

How it's used
-------------

For a K-class model you get **K** ROC curves and AUCs, one per class, each answering *how well does the model
separate this class from everything else?* scikit-learn exposes it as ``multi_class='ovr'``; it also matches
the **multilabel** setting, where classes aren't exclusive.

The catch
---------

Each binary split is **imbalanced** — the positive class is only about **1/K** of the data, and the "rest"
group's makeup shifts with the class distribution, so OvR scores are **sensitive to class imbalance**. The
alternative, **One-vs-One**, compares class **pairs** and is less imbalance-prone but trains
:math:`O(K^2)` classifiers.

----

**Mind map — connected ideas**

   :doc:`Multiclass Classification <311-multiclass-classification>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Micro AUC <313-micro-auc>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <427-roc-auc-receiver-operating-characteristic-area-u>` · :doc:`Multiclass AUROC <022-multiclass-auroc>`

----

**More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `One-vs-Rest (OvR) <https://insightful-data-lab.com/2025/08/21/one-vs-rest-ovr/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate

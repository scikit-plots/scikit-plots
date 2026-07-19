:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-multiclass-classification:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Multiclass Classification</b></div>`

===========================
Multiclass Classification
===========================

*Assigning each instance to one of three or more classes.*

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

**Multiclass classification** predicts one of **more than two** mutually exclusive classes — a handwritten
digit (0–9), a species, a product category. It generalizes **binary** classification, and its models usually
end in a **softmax** layer that outputs a probability over the **K** classes.

The evaluation twist
--------------------

Metrics built for two classes — **ROC-AUC**, **precision**, recall — have no direct multiclass definition,
because "**positive** vs negative" is ambiguous with many classes. To use them, the problem is **binarized**
(one class vs the others) and the per-class scores are **averaged**.

The two decompositions
----------------------

**One-vs-Rest** turns K classes into K binary problems (each class against the rest); **One-vs-One** compares
every **pair**. Either produces a set of per-class or per-pair scores that a **micro** or **macro** average
then collapses into a single number.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`One-vs-Rest (OvR) <310-one-vs-rest-ovr>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Micro AUC <313-micro-auc>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Softmax Function <296-softmax-function>` · :doc:`Multiclass AUROC <022-multiclass-auroc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Multiclass Classification <https://insightful-data-lab.com/2025/08/21/multiclass-classification/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-partial-auc-pauc:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Partial AUC (pAUC)</b></div>`

====================
Partial AUC (pAUC)
====================

*AUC restricted to a region of interest of the ROC curve.*

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

**Partial AUC** is the area under **only a slice** of the ROC curve — typically a confined range of **false
positive rate** (say :math:`\text{FPR} \le 0.1`), sometimes of TPR, or both. It focuses the metric on the
**operating region** that actually matters.

Why restrict
------------

Full AUC weights **all** FPR regions equally, but many are operationally irrelevant — a radiologist doesn't
care about performance at 80% FPR, a bank won't run a fraud model that flags half of transactions. pAUC
scores the model **where it will be used** (low FPR / high TPR for screening), and is especially apt for
**low-prevalence** data needing high specificity.

The trade-off
-------------

pAUC is more **decision-relevant** than full AUC and distinguishes curves that **cross** yet share the same
total AUC, but it **ignores** the ROC outside the band and its raw value depends on the **interval width**
(so it's often **standardized**, e.g. McClish, back to :math:`[0,1]`). The bounds must be justified by the
use case.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <427-roc-auc-receiver-operating-characteristic-area-u>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Micro AUC <313-micro-auc>` · :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Precision–Recall AUC (PR-AUC) <430-precisionrecall-auc-pr-auc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Partial AUC (pAUC) <https://insightful-data-lab.com/2025/08/21/partial-auc-pauc/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

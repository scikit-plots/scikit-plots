:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-macro-averaging:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Macro Averaging</b></div>`

=================
Macro Averaging
=================

*Computing a metric per class then averaging them equally.*

What it is
----------

**Macro averaging** computes the metric **separately for each class** (one-vs-rest), then takes the
**unweighted arithmetic mean**. Every **class** counts the **same**, no matter how many samples it has:

.. math::

   P_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C}\frac{TP_c}{TP_c + FP_c}.

Its behavior
------------

Because each class contributes equally, macro averaging **punishes ignoring minorities** — a model that aces
the majority but fails a rare class gets a **low** macro score. That makes it sensitive to **rare-class**
performance and a natural **fairness**-oriented headline.

When to use it
--------------

Use macro averaging when **all classes matter equally**, especially on **imbalanced** data where you don't
want the majority to drown out the rest. It can, however, look **pessimistic** if some tiny classes are
inherently hard.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Micro Averaging <369-micro-averaging>` · :doc:`Weighted Averaging <361-weighted-averaging>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`One-vs-Rest (OvR) <310-one-vs-rest-ovr>` · :doc:`F1-score <363-f1-score>` · :doc:`Multiclass Classification <311-multiclass-classification>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Macro Averaging <https://insightful-data-lab.com/2025/08/20/macro-averaging/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

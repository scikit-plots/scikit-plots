:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-micro-recall:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Micro Recall</b></div>`

==============
Micro Recall
==============

*Recall computed from globally pooled true positives and false negatives.*

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

**Recall** answers: of all the items that are *actually* positive, how many did the
model catch?

.. math::

   \text{Recall} = \frac{TP}{TP + FN},

so it is sensitive to **false negatives** (missed positives). **Micro recall**
extends this to :math:`K` classes by pooling counts across classes rather than
averaging per-class recall (which is macro recall).

How it's computed
-----------------

Sum true positives and false negatives over every class, then divide:

.. math::

   \text{Recall}_{\text{micro}}
   = \frac{\sum_{i=1}^{K} TP_i}{\sum_{i=1}^{K} (TP_i + FN_i)}.

This treats the whole multi-class problem as one pooled "caught vs missed"
question, so **majority classes dominate** the result.

The micro identity
------------------

In single-label problems, micro recall equals micro precision equals micro F1 (and
equals accuracy): once everything is pooled, the denominators coincide. The three
diverge only in the multi-label setting.

Worked example
--------------

Three classes with TP = (40, 30, 10), FN = (10, 20, 30):

- Recall(A)=0.80, Recall(B)=0.60, Recall(C)=0.25 → **Macro recall** = 0.55.
- **Micro recall** :math:`= 80/140 \approx 0.571`.

Macro weights every class equally (exposing class C's weak 0.25); micro is a global,
sample-weighted figure.

When recall matters most
------------------------

Favour recall when a **missed positive is costly** — disease screening, fraud
detection, safety alerts — where you would rather tolerate false alarms than let a
true case slip through.

In code
-------

.. code-block:: python

   from sklearn.metrics import recall_score

   micro = recall_score(y_true, y_pred, average="micro")
   macro = recall_score(y_true, y_pred, average="macro")

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Micro Precision <016-micro-precision>` · :doc:`Micro F1 <013-micro-f1>` · :doc:`Macro Recall <020-macro-recall>` · :doc:`Micro AUROC <011-micro-auroc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Micro Recall <https://insightful-data-lab.com/2025/08/30/micro-recall/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

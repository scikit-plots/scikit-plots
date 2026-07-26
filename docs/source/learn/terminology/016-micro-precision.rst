:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-micro-precision:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Micro Precision</b></div>`

=================
Micro Precision
=================

*Precision computed from globally pooled true positives and false positives.*

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

**Precision** answers: of all the items the model *flagged* positive, how many were
right?

.. math::

   \text{Precision} = \frac{TP}{TP + FP},

so it is sensitive to **false positives** (false alarms). **Micro precision**
extends it to :math:`K` classes by pooling counts before dividing, rather than
averaging per-class precision (macro precision).

How it's computed
-----------------

.. math::

   \text{Precision}_{\text{micro}}
   = \frac{\sum_{i=1}^{K} TP_i}{\sum_{i=1}^{K} (TP_i + FP_i)}.

Summing across classes first makes the metric a single global "of all predictions,
how many correct", so **frequent classes carry the most weight**.

The micro identity
------------------

As with micro recall and micro F1, in single-label classification micro precision
equals the other two (and accuracy), because the pooled denominators line up. They
differ only under multi-label evaluation.

Worked example
--------------

Three classes with TP = (40, 30, 10), FP = (10, 20, 20):

- Precision(A)=0.80, Precision(B)=0.60, Precision(C)=0.33 → **Macro precision** = 0.58.
- **Micro precision** :math:`= 80/130 \approx 0.615`.

The micro value sits near the large classes' contribution; macro surfaces class C's
weaker 0.33.

When precision matters most
---------------------------

Favour precision when a **false positive is costly** — spam filters (blocking real
mail), recommending a bad product, flagging an innocent transaction — where the
price of a wrong "yes" is high.

In code
-------

.. code-block:: python

   from sklearn.metrics import precision_score

   micro = precision_score(y_true, y_pred, average="micro")
   macro = precision_score(y_true, y_pred, average="macro")

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Micro Recall <015-micro-recall>` · :doc:`Micro F1 <013-micro-f1>` · :doc:`Macro Precision <021-macro-precision>` · :doc:`Micro AUROC <011-micro-auroc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Micro Precision <https://insightful-data-lab.com/2025/08/30/micro-precision/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-one-vs-rest-ovr-auroc:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>One-vs-Rest (OvR) AUROC</b></div>`

=========================
One-vs-Rest (OvR) AUROC
=========================

*Multiclass AUROC obtained by scoring each class against all others and averaging.*

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

AUROC is only defined for a binary problem, so to score a :math:`K`-class model we
reduce it to :math:`K` binary ones with **One-vs-Rest (OvR)**: for each class
:math:`i`, treat class :math:`i` as positive and *all other classes together* as
negative, and compute that binary AUROC.

.. math::

   \text{AUROC}_{\text{OvR}}(i) = \text{AUROC}(\text{class}_i \;\text{vs}\; \text{rest}).

Each value measures how well the model separates *that one class* from everything
else.

From per-class to a single number
---------------------------------

You can report the per-class AUROCs directly, or average them into **macro AUROC**:

.. math::

   \text{AUROC}_{\text{macro}}
   = \frac{1}{K} \sum_{i=1}^{K} \text{AUROC}_{\text{OvR}}(i).

(Pooling the OvR decisions instead of averaging gives *micro* AUROC; averaging, as
here, weights every class equally.)

Worked example
--------------

Three classes with AUROC(A vs rest)=0.83, AUROC(B vs rest)=0.76,
AUROC(C vs rest)=0.70:

- Per class, these show A is the easiest to separate and C the hardest.
- **Macro AUROC** :math:`= (0.83 + 0.76 + 0.70)/3 = 0.763`.

When it's useful
----------------

OvR AUROC gives a **per-class, threshold-free** read on discrimination — exactly
what you want to answer "how well does the model pick out *this* class from the
rest?". It is especially informative for imbalanced problems (e.g. a rare fraud
class), where a single global score can hide a weak minority.

In code
-------

.. code-block:: python

   from sklearn.metrics import roc_auc_score

   per_class = roc_auc_score(y_true, y_score, multi_class="ovr", average=None)
   macro = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Micro AUROC <011-micro-auroc>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Single-label Classification <014-single-label-classification>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `One-vs-Rest (OvR) AUROC <https://insightful-data-lab.com/2025/08/30/one-vs-rest-ovr-auroc/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

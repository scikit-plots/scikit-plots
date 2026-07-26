:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-multiclass-auroc:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Multiclass AUROC</b></div>`

==================
Multiclass AUROC
==================

*AUROC extended beyond two classes via One-vs-Rest or One-vs-One schemes.*

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

AUROC — the probability a random positive outscores a random negative, from 0.5
(chance) to 1.0 (perfect) — is defined only for a binary problem. **Multiclass
AUROC** is the umbrella for the strategies that extend it to :math:`K` classes by
reducing the problem to many binary ones and aggregating.

Two reductions
--------------

one-vs-rest (OvR)
^^^^^^^^^^^^^^^^^

For each class, score it against *all others* combined — :math:`K` binary AUROCs:

.. math::

   \text{AUROC}_{\text{macro}}
   = \frac{1}{K} \sum_{i=1}^{K} \text{AUROC}(\text{class}_i \;\text{vs}\; \text{rest}).

one-vs-one (OvO)
^^^^^^^^^^^^^^^^

Score *every pair* of classes and average over all :math:`\binom{K}{2}` pairs:

.. math::

   \text{AUROC}_{\text{ovo}}
   = \frac{2}{K(K-1)} \sum_{i<j} \text{AUROC}(\text{class}_i \;\text{vs}\; \text{class}_j).

OvO gives a more balanced picture under heavy imbalance, at :math:`O(K^2)` cost.

Two averagings
--------------

Independently of OvR / OvO, the per-binary results can be combined as **macro**
(equal weight per class — fair to minorities), **micro** (pool all decisions into
one global AUROC — sample-weighted, majority-driven), or **weighted** (by class
frequency).

Worked example
--------------

OvR AUROCs of 0.82, 0.75, 0.70 over three classes:

.. math::

   \text{AUROC}_{\text{macro}} = \frac{0.82 + 0.75 + 0.70}{3} = 0.7567.

Interpretation
--------------

All variants keep the binary AUROC scale (0.5 random, 1.0 perfect). Read **macro**
for fairness across classes, **micro** for overall sample-level discrimination, and
**OvO** for pairwise separability. Reporting more than one avoids being misled by a
single summary.

In code
-------

.. code-block:: python

   from sklearn.metrics import roc_auc_score

   ovr_macro = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
   ovo_macro = roc_auc_score(y_true, y_score, multi_class="ovo", average="macro")

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`One-vs-Rest (OvR) AUROC <017-one-vs-rest-ovr-auroc>` · :doc:`Micro AUROC <011-micro-auroc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Gini Coefficient <023-gini-coefficient>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Multiclass AUROC <https://insightful-data-lab.com/2025/08/30/multiclass-auroc/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

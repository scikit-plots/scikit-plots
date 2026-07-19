:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-gini-coefficient:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Gini Coefficient</b></div>`

==================
Gini Coefficient
==================

*A ranking-quality score linearly tied to AUROC: Gini = 2 x AUROC - 1.*

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

The **Gini coefficient** has two lives. In economics (Corrado Gini, 1912) it
measures **inequality** in a distribution such as income, via the **Lorenz curve** —
the cumulative share of income against the cumulative share of population. It is the
normalised area between the line of perfect equality and the Lorenz curve:

.. math::

   \text{Gini} = \frac{A}{A + B},

where :math:`A` is the area between the equality line and the Lorenz curve and
:math:`B` is the area under the Lorenz curve.

In machine learning
-------------------

For a binary classifier, Gini measures **discriminatory power** and is a simple
linear rescaling of AUROC:

.. math::

   \text{Gini} = 2 \cdot \text{AUROC} - 1.

So AUROC 0.5 (random) gives Gini 0, AUROC 1.0 (perfect) gives Gini 1, and an AUROC
below 0.5 gives a negative Gini. It carries **no information beyond AUROC** — the
same ranking quality on a stretched scale — but it is the convention in finance.

Why credit risk uses it
-----------------------

Credit-scoring models (loan default, churn, fraud) usually report Gini rather than
AUROC: a higher Gini means the model separates "goods" (non-defaulters) from "bads"
(defaulters) better, and regulatory frameworks (Basel II/III) often expect it in
model validation. The ML Lorenz curve simply replaces "income" with the predicted
score and "population" with cases sorted by that score.

Worked example
--------------

- Model A: AUROC 0.72 → :math:`\text{Gini} = 2(0.72) - 1 = 0.44`.
- Model B: AUROC 0.85 → :math:`\text{Gini} = 0.70`.

Model B ranks defaulters above non-defaulters far better; a Gini of 0.70 is
considered excellent in credit risk.

Rules of thumb and edge cases
-----------------------------

- Typical bands: 0.20–0.30 weak, 0.40–0.50 useful, 0.60–0.70 strong.
- **0.80+ is suspicious** — usually overfitting or target leakage, not a genuinely
  great model.
- A **negative Gini** means predictions are inverted (worse than random); flipping
  the score sign fixes it.

In code
-------

.. code-block:: python

   from sklearn.metrics import roc_auc_score

   gini = 2 * roc_auc_score(y_true, y_score) - 1

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`One-vs-Rest (OvR) AUROC <017-one-vs-rest-ovr-auroc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Gini Coefficient <https://insightful-data-lab.com/2025/08/30/gini-coefficient/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

.. _dpa-nominal-classification-models-model-state-and-evaluation-metrics:

========================================================================
Nominal Classification Models: Model State and Evaluation Metrics
========================================================================

**Stage 8 · 📊 Model Evaluation**  ·  Lesson 52 of 56  ·  *advanced*

:doc:`◀ Previous · Binary Classification Models – Conceptual Framework and Evaluation Metrics <51-binary-classification-models-conceptual-framework-and-evaluation-metrics>`   ·   :doc:`Next · Binary Classification Model Evaluation and Threshold Optimization ▶ <53-binary-classification-model-evaluation-and-threshold-optimization>`   ·   :doc:`↑ Section <index>`


Beyond two classes
--------------------

Many classification problems have **more than two** unordered labels — payment type (cash / card /
mobile), a product category, a species. These are **nominal** (multi-class) problems: the classes have
**no natural order**, and every prediction must pick one of :math:`K` labels. Evaluation carries over
from the binary case, but with two twists worth understanding.

The K×K confusion matrix
--------------------------

The confusion matrix generalises directly: it becomes a :math:`K \times K` table, actual classes down
the rows and predicted classes across the columns. The **diagonal** still holds the correct
predictions, and every **off-diagonal** cell now names a **specific confusion** — how often class A was
mistaken for class B. This is the matrix's great virtue in multi-class work: it shows **which pairs**
of classes the model muddles (cash vs mobile, say), pointing directly at what to fix. scikit-plots
renders it as the same labelled heat-map plot, just larger.

Per-class metrics
-------------------

Precision, recall and F1 are **binary** notions, so for :math:`K` classes they are computed **per
class**, one-vs-rest: for class A, treat A as "positive" and everything else as "negative", and read
TP, FP and FN from the matrix's row and column for A. Each class gets its own precision (of the
predictions of A, how many were right?) and recall (of the true A's, how many were found?) —
scikit-learn's ``classification_report`` prints exactly this table, one row per class.

Averaging: macro, micro, weighted
-----------------------------------

To summarise the per-class scores in **one number**, three averages are standard, and they answer
**different questions**:

* **Macro** — the plain mean of the per-class scores: every **class** counts equally, however rare, so it exposes weakness on small classes;
* **Weighted** — the mean weighted by class **frequency**: large classes dominate, mirroring overall behaviour;
* **Micro** — pool all the TP / FP / FN counts first, then compute: every **instance** counts equally. For single-label problems micro-precision, micro-recall and accuracy all **coincide**.

On **imbalanced** data the choice matters: a model useless on a rare class can still post a high
weighted or micro score, while its **macro** score collapses — so pick the average that matches whether
classes or instances are what the business weighs equally.

.. hint::

   **Related lessons:** :doc:`Binary Classification Models – Conceptual Framework and Evaluation Metrics <51-binary-classification-models-conceptual-framework-and-evaluation-metrics>`  ·  :doc:`Assessing the Quality of Prediction Models <50-assessing-the-quality-of-prediction-models>`  ·  :doc:`Binary Classification Model Evaluation and Threshold Optimization <53-binary-classification-model-evaluation-and-threshold-optimization>`  ·  :doc:`AUC–ROC Curve: Evaluating Classification Model Performance <55-auc-roc-curve-evaluating-classification-model-performance>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/nominal-classification-models-model-state-and-evaluation-metrics/ <https://insightful-data-lab.com/2026/01/16/nominal-classification-models-model-state-and-evaluation-metrics/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: advanced

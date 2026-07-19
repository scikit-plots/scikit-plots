.. _dpa-binary-classification-model-evaluation-and-threshold-optimization:

========================================================================
Binary Classification Model Evaluation and Threshold Optimization
========================================================================

**Stage 8 · 📊 Model Evaluation**  ·  Lesson 53 of 56  ·  *advanced*

:doc:`◀ Previous · Nominal Classification Models: Model State and Evaluation Metrics <52-nominal-classification-models-model-state-and-evaluation-metrics>`   ·   :doc:`Next · Identifying Outliers Using Residuals and Studentized Residuals ▶ <54-identifying-outliers-using-residuals-and-studentized-residuals>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The hidden dial
-----------------

A binary classifier rarely outputs a label directly — it outputs a **probability** (or score), like the
logistic model's :math:`p`. The label comes from a **threshold**: predict positive when the score
exceeds a **cutoff**, conventionally **0.5**. That cutoff is a **dial**, not a law of nature — and
turning it changes every metric of the last lessons without retraining anything. Choosing it well is
**threshold optimisation**.

Turning the dial
------------------

The trade-off is mechanical. **Lower** the threshold and more cases are called positive: **recall
rises** (fewer misses) but **precision falls** (more false alarms). **Raise** it and the reverse —
predictions become conservative, precision improves, recall drops. The confusion matrix **morphs**
continuously as the dial turns, trading false negatives for false positives. No single threshold is
"correct"; each is a different **operating point** on the same fitted model.

Cost decides
--------------

The right operating point comes from the **relative cost of the two errors** — the domain judgement
from the confusion-matrix lesson, now made operational. If a **miss** is expensive (fraud, disease
screening), push the threshold **down** and accept false alarms — a fraud team might demand recall
above 0.85 and tune precision after. If a **false alarm** is expensive (spam filters binning real
mail), push it **up**. When costs can be written down, the threshold can even be chosen to **minimise
expected cost** directly.

Choosing it in practice
-------------------------

Practically, you **sweep** the threshold on validation data and plot the metrics against it: precision
and recall curves crossing, F1 peaking, or expected cost bottoming out. Pick the threshold that
optimises the criterion you care about — the F1-maximising point, the smallest threshold hitting a
target recall, or the cost minimum. scikit-plots' evaluation charts exist for exactly this sweep, and
the next lessons introduce the two most important of them: the **ROC curve**, which displays *every*
operating point at once, and the **lift** chart, which ranks rather than cuts.

.. hint::

   **Related lessons:** :doc:`Binary Classification Models – Conceptual Framework and Evaluation Metrics <51-binary-classification-models-conceptual-framework-and-evaluation-metrics>`  ·  :doc:`AUC–ROC Curve: Evaluating Classification Model Performance <55-auc-roc-curve-evaluating-classification-model-performance>`  ·  :doc:`Lift Analysis for Direct Mail Campaigns: Concept, Process, and Business Value <56-lift-analysis-for-direct-mail-campaigns-concept-process-and-business-value>`  ·  :doc:`Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds <38-logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/binary-classification-model-evaluation-and-threshold-optimization/ <https://insightful-data-lab.com/2026/01/16/binary-classification-model-evaluation-and-threshold-optimization/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: advanced

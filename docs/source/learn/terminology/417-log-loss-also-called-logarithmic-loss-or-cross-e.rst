:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-log-loss-also-called-logarithmic-loss-or-cross-entropy-loss:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Log Loss (also called Logarithmic Loss or Cross-Entropy Loss)</b></div>`

===============================================================
Log Loss (also called Logarithmic Loss or Cross-Entropy Loss)
===============================================================

*A loss penalising confident wrong probabilistic predictions.*

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

**Log loss** (logarithmic loss / **cross-entropy** loss) scores a probabilistic classifier by the **negative
log-likelihood** of the true labels — how surprised the model is by reality. For binary labels:

.. math::

   \text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\big[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\big].

It runs from **0** (perfect) to **∞**.

Its defining trait
------------------

The **logarithm** makes log loss punish **confident** mistakes **brutally** — predicting 0.01 for a true
positive costs far more than predicting 0.4. This **exponential** penalty for overconfidence is exactly why
it is the standard **training objective** for logistic regression and neural nets, which optimize it directly.

When to use it (and a caveat)
-----------------------------

Reach for log loss whenever **calibrated** probabilities matter — fraud, diagnosis, risk. It **requires**
genuine probabilities, so raw **logits** must be squashed (softmax / sigmoid) first, and it is harsher on
overconfidence than the **Brier score**. Compare a model's log loss to the **base-rate** log loss to confirm
it adds value.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Brier Score <418-brier-score>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Softmax Function <296-softmax-function>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Strictly Proper Scoring Rules <234-strictly-proper-scoring-rules>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <https://insightful-data-lab.com/2025/08/19/log-loss-also-called-logarithmic-loss-or-cross-entropy-loss/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

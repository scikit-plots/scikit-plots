:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-binary-cross-entropy-bce:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Binary Cross-Entropy (BCE)</b></div>`

============================
Binary Cross-Entropy (BCE)
============================

*The standard loss for binary classification, penalising confident wrong predictions.*

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

**Binary cross-entropy** (log loss) is the standard **loss** for **binary classification**, for a true
label :math:`y \in \{0,1\}` and predicted probability :math:`p = \sigma(z)`:

.. math::

   \text{BCE} = -\big[\,y \log(p) + (1 - y)\log(1 - p)\,\big], \qquad p = \sigma(z),

averaged over the data.

How it behaves
--------------

The **logarithm** gives an **asymmetric** penalty — a confident-correct prediction costs almost nothing
(:math:`-\log 0.99 \approx 0.01`), a confident-wrong one costs a lot (:math:`-\log 0.01 \approx 4.6`). This
pressures the model to be **confident when right and hesitant when wrong**, pushing toward **calibrated**
probabilities.

Why it fits
-----------

BCE is the **negative log-likelihood** of the **Bernoulli** distribution, so minimizing it is **maximum
likelihood** — the natural partner of the **sigmoid**, with a clean gradient :math:`(p - y)`. Its
multi-class analogue is **categorical cross-entropy** with **softmax**.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Loss Functions <289-loss-functions>` · :doc:`Logit Space <291-logit-space>` · :doc:`Sigmoid Function <297-sigmoid-function>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Binary Cross-Entropy (BCE) <https://insightful-data-lab.com/2025/08/21/binary-cross-entropy-bce/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

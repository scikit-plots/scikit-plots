:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-logits:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Logits</b></div>`

========
Logits
========

*Raw pre-activation scores before a sigmoid or softmax.*

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

**Logits** are the **raw, unnormalized** scores a classifier's final layer produces **before** they're turned
into probabilities. They range over **all** real numbers — positive, negative, unbounded — and live in
**log-odds** space, not probability space.

From logits to probabilities
-----------------------------

A **softmax** turns a vector of logits into a probability **distribution** that sums to 1 (for multiclass),
while a **sigmoid** maps a single logit to one probability (for binary). Because softmax **normalizes**,
raising one logit **lowers** the others' probabilities — the competition that sharpens a prediction.

Why keep them raw
-----------------

Exposing logits enables **numerically stable** training (log-softmax beats probabilities-then-log, which is
why frameworks feed **cross-entropy** raw logits) and **post-hoc calibration** — **temperature scaling**
divides logits by T *before* softmax, which is only possible when the logits are available.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Softmax Function <296-softmax-function>` · :doc:`Sigmoid Function <297-sigmoid-function>` · :doc:`Logit Space <291-logit-space>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Logits <https://insightful-data-lab.com/2025/08/17/logits/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

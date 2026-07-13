:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-model-distillation-knowledge-distillation:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Model Distillation (Knowledge Distillation)</b></div>`

=============================================
Model Distillation (Knowledge Distillation)
=============================================

*Training a small student model to mimic a larger teacher for efficiency.*

What it is
----------

**Model distillation** (knowledge distillation) is a **compression** technique: a large,
accurate **teacher** model transfers its knowledge to a smaller, faster **student**. The aim
is a lighter model that **keeps most of the teacher's accuracy** while being cheap enough to
deploy on phones and edge devices.

How it works
------------

Train the big teacher (a large transformer, say), then have it produce **soft targets** — full
probability distributions like ``[0.7 cat, 0.2 dog, 0.1 rabbit]`` rather than a bare hard
label. The student trains to **mimic those soft outputs**, usually under a loss that blends a
**distillation term** (student vs teacher) with a **supervised term** (student vs true labels).

The objective
-------------

.. math::

   L = \alpha \, L_{\text{hard}}(y, p_s) \; + \; (1 - \alpha)\, L_{\text{soft}}(p_t, p_s, T),

where :math:`y` are true labels, :math:`p_t` and :math:`p_s` the teacher and student
probabilities, :math:`T` a **temperature** that softens the teacher's distribution to expose
its "dark knowledge", and :math:`\alpha` the balance between the two terms.

Why, examples, and costs
------------------------

It buys **efficiency** (faster, cheaper inference), **deployability** (edge devices), and even
better generalisation from the teacher's soft probabilities. **DistilBERT** is ~40% smaller and
~60% faster than BERT at ~97% of its performance; **ResNet-50** distills into ResNet-18 at
similar accuracy. The catches: the student **cannot capture everything**, :math:`T` and
:math:`\alpha` need tuning, and you still pay to **train the teacher** first.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Quantization <343-quantization>` · :doc:`Frozen Encoder <172-frozen-encoder>` · :doc:`Embedding <173-embedding>` · :doc:`Autoencoder <171-autoencoder>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Epochs <141-epochs>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Model Distillation (Knowledge Distillation) <https://insightful-data-lab.com/2025/08/24/model-distillation-knowledge-distillation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

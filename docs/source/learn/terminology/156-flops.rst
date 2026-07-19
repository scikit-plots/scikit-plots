:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-flops:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>FLOPs</b></div>`

=======
FLOPs
=======

*Floating-point operations — a measure of a model's compute cost.*

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

**FLOPs — floating-point operations** — count the **mathematical operations** (adds, multiplies,
divides) a model or algorithm performs. In ML, FLOPs are a standard **proxy for computational
complexity** and for training/inference **cost** — they measure *workload*, not memory access or
I/O.

The units
---------

One operation is a **FLOP**, and the scale climbs fast: **MFLOPs** (millions), **GFLOPs**
(billions), **TFLOPs** (trillions), **PFLOPs** (quadrillions). A single large matrix
multiplication already involves enormous counts.

Counting them
-------------

Multiplying an :math:`m \times n` matrix by an :math:`n \times p` matrix costs about
:math:`2 \, m \, n \, p` FLOPs — so two :math:`1000 \times 1000` matrices run to roughly **2
billion**. In deep learning, **training** FLOPs scale with dataset size × model size × epochs,
while **inference** FLOPs per forward pass drive deployment latency. For comparison, ResNet-50 is
~4 GFLOPs per image, BERT-base ~22 GFLOPs per sequence, and GPT-3 took an estimated
:math:`3 \times 10^{23}` FLOPs to train.

FLOPs vs FLOPS
--------------

Two letters, two meanings. **FLOPs** is a **count** of operations (the workload); **FLOPS** is
**floating-point operations per second**, the **speed** of hardware — an NVIDIA A100 delivers
~312 TFLOPS on its tensor cores. FLOPs is the job; FLOPS is how fast the worker does it. Because
training cost scales almost linearly with FLOPs, the count is a key lever in the
efficiency-versus-accuracy trade-off.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Model Weights <155-model-weights>` · :doc:`Quantization <343-quantization>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`OpEx <157-opex>` · :doc:`Neural Networks <287-neural-networks>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `FLOPs <https://insightful-data-lab.com/2025/08/24/flops/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

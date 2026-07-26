:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-quantization:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Quantization</b></div>`

==============
Quantization
==============

*Reducing numeric precision of weights to shrink and speed up models.*

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

**Quantization** lowers the **numerical precision** of a model's **weights** (and often activations) — from
32-bit floating point (**FP32**) down to **INT8**, **FP8**, or even **INT4**. Fewer bits per number means a
**smaller, faster, cheaper** model.

The payoff
----------

An **INT8** model uses about **75% less memory** than FP32, and because decoding is often
**memory-bandwidth-bound**, 4-bit weights can be read up to **4× faster** than 16-bit — directly cutting
**latency** and **inference cost** on hardware that supports low precision.

Managing the trade-off
----------------------

Naive quantization **degrades accuracy**. **Post-training quantization (PTQ)** converts a trained model
quickly (with a small calibration set), while **quantization-aware training (QAT)** bakes precision loss into
training to preserve accuracy; advanced schemes like **AWQ** and **GPTQ** protect the most **sensitive**
weights to reach near-FP16 quality at INT4 speeds.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Caching <342-caching>` · :doc:`TPU Clusters <347-tpu-clusters>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`ONNX (Open Neural Network Exchange) <344-onnx-open-neural-network-exchange>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Quantization <https://insightful-data-lab.com/2025/08/20/quantization/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

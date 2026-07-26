:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-hyperparameter:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Hyperparameter</b></div>`

================
Hyperparameter
================

*A setting fixed before training (e.g. learning rate) rather than learned.*

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

A **hyperparameter** is a setting **chosen before training** that controls **how a model
learns or how it is structured** — as opposed to **parameters** (weights and biases), which the
optimiser **learns from the data**. Hyperparameters are the model's *learning strategy*;
parameters are its *learned knowledge*.

The main kinds
--------------

They span four areas. **Model structure**: number of layers, neurons per layer, a tree's max
depth. **Training process**: learning rate, batch size, number of epochs, dropout rate.
**Regularisation**: L1/L2 penalty strength, weight decay. **Optimisation**: SGD momentum, the
Adam beta values.

Why they matter
---------------

The same model with different hyperparameters can perform **very differently** — poor choices
cause underfitting, overfitting or painfully slow training, so **tuning is critical** for
accuracy and generalisation.

Tuning, and an example
----------------------

Common strategies escalate in sophistication: **manual** trial, **grid search** (every
combination), **random search** (sample combinations — often more efficient), **Bayesian
optimisation** (a probabilistic model guides the search), and **Hyperband / population-based**
methods at scale. Training a net on MNIST one might fix learning rate ``0.001``, batch size
``64``, dropout ``0.5`` and ``20`` epochs; the weights are learned automatically, but
performance hinges on those chosen hyperparameters.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Epochs <141-epochs>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>` · :doc:`Quantization <343-quantization>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Hyperparameter <https://insightful-data-lab.com/2025/08/24/hyperparameter/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

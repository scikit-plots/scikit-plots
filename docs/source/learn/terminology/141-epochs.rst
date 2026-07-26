:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-epochs:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Epochs</b></div>`

========
Epochs
========

*One full pass of the training algorithm over the entire dataset.*

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

An **epoch** is **one complete pass of the entire training set through the model**. In each
epoch the model sees every training sample once — in **mini-batches**, not all at once — and
across many epochs it refines its weights through repeated exposure.

Epoch vs batch vs iteration
---------------------------

Three terms that are easy to conflate. A **batch** is a subset of the data processed together;
an **iteration** is one update step (a forward and backward pass on one batch); and an
**epoch** is one full cycle through *all* batches. So with **10,000 samples** and a **batch
size of 100**, it takes **100 iterations to complete 1 epoch**.

Too few, too many
-----------------

Epoch count trades **underfitting against overfitting**. Too few and the model **hasn't
learned enough**; too many and it begins to **memorise** the training data and generalises
worse. Loss falls with more epochs only **up to a point**.

The training loop and an example
--------------------------------

The loop is: initialise weights; for each epoch, run forward/backward passes over the
mini-batches and update, then evaluate on a validation set; stop when validation stops
improving (**early stopping**). Training an image classifier on **CIFAR-10** for 20 epochs,
the model sees all 50,000 images each epoch and, by epoch 20, the loss has stabilised. Since
the right count is data-dependent — roughly **50–200** for small sets, **5–30** with early
stopping for large ones — the **epoch count is itself a hyperparameter**.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Quantization <343-quantization>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Epochs <https://insightful-data-lab.com/2025/08/24/epochs/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

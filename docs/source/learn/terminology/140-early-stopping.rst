:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-early-stopping:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Early Stopping</b></div>`

================
Early Stopping
================

*Halting training when validation performance stops improving, to curb overfitting.*

What it is
----------

**Early stopping** is a **regularisation** technique: stop training **before the model
overfits**. Rather than running a fixed number of epochs, you watch a **validation metric** and
halt once it stops improving — capturing the model at its generalisation "sweet spot" and
saving the wasted epochs beyond it.

How it works
------------

Hold out a validation set, train across epochs, and after each one **score the validation
metric** and remember the **best** so far. If it fails to improve for **N consecutive epochs** —
the **patience** — stop. The key parameters are the **monitored metric**, the **mode**
(``min`` for loss, ``max`` for accuracy or AUC), the **patience**, and **restore-best-weights**,
which rolls the model back to its best epoch.

Example
-------

Capped at 100 epochs, suppose validation loss improves until **epoch 25** and then climbs as
the model starts overfitting. With **patience = 3**, training stops at **epoch 28** and restores
the weights from epoch 25 — the genuine best.

Benefits and drawbacks
----------------------

It **prevents overfitting**, **cuts training time**, and finds a near-optimal stopping point
automatically. The costs are minor: it **needs a validation set**, and a noisy metric can trip
it too soon — which is exactly what **patience** is there to absorb.

----

**Mind map — connected ideas**

   :doc:`Epochs <141-epochs>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>` · :doc:`Frozen Encoder <172-frozen-encoder>` · :doc:`Autoencoder <171-autoencoder>` · :doc:`Quantization <343-quantization>`

----

**More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Early Stopping <https://insightful-data-lab.com/2025/08/24/early-stopping/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate

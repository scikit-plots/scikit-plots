:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-loss-functions:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Loss Functions</b></div>`

================
Loss Functions
================

*Objectives quantifying prediction error that training seeks to minimise.*

What it is
----------

A **loss function** (cost or objective) measures **how wrong** a model's predictions are on the training
data — a single number the model **minimizes**. Lower loss means predictions closer to targets; it is the
signal that **gradient descent** follows to update parameters.

Matching loss to task
---------------------

The loss encodes what "wrong" means — **regression** uses **mean squared error** or MAE; **binary
classification** uses **binary cross-entropy**; **multi-class** uses **categorical cross-entropy**. Squared
error can't tell a bad classification from a disastrous one, which is why classification uses cross-entropy.

Why it matters
--------------

The loss defines what the model actually **optimizes**, so a mismatched loss silently optimizes the wrong
thing (MSE on a sigmoid gives a **non-convex** surface). A good loss is **differentiable**, fits the task,
and aligns with the real objective.

----

**Mind map — connected ideas**

   :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Logit Space <291-logit-space>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Sigmoid Function <297-sigmoid-function>` · :doc:`Softmax Function <296-softmax-function>` · :doc:`Neural Networks <287-neural-networks>`

----

**More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Loss Functions <https://insightful-data-lab.com/2025/08/21/loss-functions/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-model-weights:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Model Weights</b></div>`

===============
Model Weights
===============

*The learned parameters that define a trained model's behaviour.*

What it is
----------

**Model weights** are a model's **trainable parameters** — the numbers that determine how input
features are turned into predictions. Each weight encodes the **importance** of a feature, and
training **adjusts** them to minimise loss.

By model type
-------------

In **linear regression**, :math:`y = w_1 x_1 + w_2 x_2 + b`, the weights :math:`w_1, w_2` say
how much each feature contributes — if :math:`w_1 = 200`, every extra square foot adds ``$200``
to the predicted price. In a **neural network**, every connection between neurons has a weight;
the forward pass multiplies inputs by weights and applies an activation, and CNN filter weights
learn patterns like edges. In **logistic regression**, weights are the **log-odds**
contribution — positive pushes toward the positive class, negative away.

How they're learned
-------------------

Weights start **random** (or from a heuristic), then a loop refines them: **forward pass** to
predict, a **loss** against the truth, **backpropagation** for the gradients of loss with
respect to each weight, and an **optimiser** (SGD, Adam) to update them — repeated until the
loss settles.

Why they matter
---------------

Weights **are** the model's learned knowledge: saving or loading a model *is* saving or loading
its weights. In transfer learning we often **freeze** the encoder's weights and fine-tune only
the last layers. Concretely, a spam classifier might learn a weight of **+2.5** for "free"
(strongly spammy) and **-1.0** for "invoice" (less so).

----

**Mind map — connected ideas**

   :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Frozen Encoder <172-frozen-encoder>` · :doc:`Regression Coefficient <090-regression-coefficient>` · :doc:`Epochs <141-epochs>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`FLOPs <156-flops>`

----

**More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>`

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Model Weights <https://insightful-data-lab.com/2025/08/24/model-weights/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate

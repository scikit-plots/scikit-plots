:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-logit-space:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Logit Space</b></div>`

=============
Logit Space
=============

*The pre-activation, log-odds scale on which linear models and nets operate.*

What it is
----------

**Logit space** means working with the **raw log-odds** scores :math:`z` (the **logits**) instead of the
probabilities :math:`p = \sigma(z)` — the "pre-sigmoid" world, where values span the **whole real line**
rather than being squeezed into (0,1).

Why it's used
-------------

Computing the loss **directly from logits** is far more **numerically stable**. Converting a logit to a
probability and then taking its log can **underflow** (a tiny :math:`p` rounds to 0, and :math:`\log 0 =
-\infty`) or **overflow** (:math:`e^{z}` for a large logit exceeds the float range); staying in logit space
with the **log-sum-exp** trick avoids both. This is why frameworks fuse sigmoid + BCE
(``BCEWithLogitsLoss``) and softmax + cross-entropy into a single ``from_logits`` op.

The payoff
----------

Better stability and cleaner gradients — the same reason **log-space** helps elsewhere. Logit space ties
the **log-odds**, the **sigmoid**, and the **cross-entropy** loss into one numerically safe computation.

----

**Mind map — connected ideas**

   :doc:`Log-Odds <295-log-odds>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Sigmoid Function <297-sigmoid-function>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Underflow <290-underflow>` · :doc:`Log-Space <257-log-space>`

----

**More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Logit Space <https://insightful-data-lab.com/2025/08/21/logit-space/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate

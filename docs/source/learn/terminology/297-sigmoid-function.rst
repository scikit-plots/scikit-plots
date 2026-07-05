:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-sigmoid-function:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Sigmoid Function</b></div>`

==================
Sigmoid Function
==================

*Maps any real value to (0, 1), used for probabilities and gating.*

What it is
----------

The **sigmoid** function :math:`\sigma` maps **any** real number to the open interval **(0, 1)**, tracing an
**S-shaped** curve:

.. math::

   \sigma(z) = \frac{1}{1 + e^{-z}}.

At :math:`z=0` it returns **0.5**; large positive :math:`z` → near **1**, large negative :math:`z` → near
**0**.

Its role
--------

It is the **inverse of the logit** — it turns a **log-odds** score back into a **probability** — which makes
it the output activation of **logistic regression** and of **binary**-classification output layers, and the
basis of **binary cross-entropy** loss. Each output is an **independent** probability, so sigmoid also
serves **multi-label** problems.

Watch out
---------

In the **hidden** layers of deep networks the sigmoid causes **vanishing gradients** (its slope flattens for
large :math:`|z|`), so ReLU-family activations are preferred there; sigmoid is kept for the **output**.

----

**Mind map — connected ideas**

   :doc:`Log-Odds <295-log-odds>` · :doc:`Softmax Function <296-softmax-function>` · :doc:`Squashing Function <298-squashing-function>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>`

----

**More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>`

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Sigmoid Function <https://insightful-data-lab.com/2025/08/21/sigmoid-function/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-softmax-function:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Softmax Function</b></div>`

==================
Softmax Function
==================

*Converts a vector of scores into a probability distribution over classes.*

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

The **softmax** function generalizes the sigmoid to **many** classes. It takes a vector of **K** raw scores
(**logits**), exponentiates each and normalizes by their sum, producing **K** probabilities that each lie
in (0,1) and **sum to 1** — a full **distribution** over mutually exclusive classes:

.. math::

   \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.

Its role
--------

It is the standard **output layer** for **multi-class** classification, trained with **categorical
cross-entropy**. It **amplifies** the largest score toward 1 while **dampening** the rest — a soft "winner."
When :math:`K=2` it **reduces to the sigmoid**.

Watch out
---------

Because the outputs are coupled (they must sum to 1), softmax assumes classes are **mutually exclusive** —
for **multi-label** problems (independent classes) use per-class sigmoids instead. Its probabilities can
also be **poorly calibrated**.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Sigmoid Function <297-sigmoid-function>` · :doc:`Squashing Function <298-squashing-function>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Classification Models <294-classification-models>` · :doc:`Binary Classification <293-binary-classification>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Softmax Function <https://insightful-data-lab.com/2025/08/21/softmax-function/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

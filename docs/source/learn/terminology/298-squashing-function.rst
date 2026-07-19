:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-squashing-function:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Squashing Function</b></div>`

====================
Squashing Function
====================

*Any bounded nonlinearity (sigmoid, tanh) that compresses its input range.*

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

A **squashing function** is any function that **compresses** an unbounded input — any real number in
:math:`(-\infty, \infty)` — into a **bounded** range, giving the characteristic **S-shape**. It "squashes"
an infinite domain into a finite interval.

Examples
--------

The **sigmoid** squashes to **(0, 1)** (a probability), **tanh** to **(−1, 1)**, and **softmax** squashes a
vector of logits into probabilities on (0,1). They are the classic **non-linear activations** that let a
network turn raw scores into interpretable, constrained outputs.

Why it matters
--------------

Squashing is what converts an **unbounded** linear score (like the **log-odds**) into something usable — a
probability, or a normalized signal — and the non-linearity is what lets stacked layers model **complex**
patterns. Its flat tails are also the source of **saturation** and vanishing gradients.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Sigmoid Function <297-sigmoid-function>` · :doc:`Softmax Function <296-softmax-function>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Squashing Function <https://insightful-data-lab.com/2025/08/21/squashing-function/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-maximum-mean-discrepancy-mmd:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Maximum Mean Discrepancy (MMD)</b></div>`

================================
Maximum Mean Discrepancy (MMD)
================================

*A kernel-based distance between distributions for two-sample and drift tests.*

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

**Maximum mean discrepancy (MMD)** is a statistical test for the **difference between two
distributions** :math:`P` and :math:`Q` from samples. Under a chosen kernel, **MMD is 0 exactly
when the distributions match**, and grows as they diverge. It is widely used to detect **drift**,
compare **train versus test** data, and evaluate **generative models** (GANs, VAEs, diffusion).

The idea
--------

MMD compares the **mean embeddings** of the two distributions in a **reproducing kernel Hilbert
space (RKHS)**:

.. math::

   \text{MMD}^2(P, Q) = \left\| \mu_P - \mu_Q \right\|_{\mathcal{H}}^2,

where :math:`\mu_P = \mathbb{E}_{x \sim P}[\phi(x)]` is the mean embedding of :math:`P` under the
feature map :math:`\phi` of a kernel :math:`k(x, y)`.

Estimating it
-------------

The **kernel trick** gives an estimate from samples without ever touching the infinite-dimensional
space — averaging within-:math:`P` kernels plus within-:math:`Q` kernels minus twice the cross
kernels:

.. math::

   \widehat{\text{MMD}}^2 = \frac{1}{m^2}\sum_{i,i'} k(x_i, x_{i'}) + \frac{1}{n^2}\sum_{j,j'} k(y_j, y_{j'}) - \frac{2}{mn}\sum_{i,j} k(x_i, y_j),

usually with a Gaussian RBF kernel.

Where it's used
---------------

A **small** MMD means similar samples, a **large** one means different. It powers **drift detection**
(train vs production, covariate drift), **GAN evaluation** (generated vs real), and **domain
adaptation** (aligning source and target features, as in MMD-regularised networks). Train a fraud
model on last year's data :math:`P`; if this year's :math:`Q` gives a high MMD, you have covariate
drift and likely need to retrain.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Energy Distance <176-energy-distance>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Representation Shift <174-representation-shift>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Maximum Mean Discrepancy (MMD) <https://insightful-data-lab.com/2025/08/23/maximum-mean-discrepancy-mmd/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

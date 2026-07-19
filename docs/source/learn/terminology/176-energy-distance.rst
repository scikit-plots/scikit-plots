:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-energy-distance:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Energy Distance</b></div>`

=================
Energy Distance
=================

*A distance between distributions used to test whether two samples differ.*

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

**Energy distance** is a **statistical distance** between two probability distributions :math:`P`
and :math:`Q`, built from **expected pairwise distances** between samples. It is **0 exactly when
the distributions are identical**, and grows as they differ — making it a natural **two-sample
test**, in the same family as MMD.

The formula
-----------

For :math:`X \sim P` and :math:`Y \sim Q`,

.. math::

   D_E^2(P, Q) = 2\, \mathbb{E}\|X - Y\| - \mathbb{E}\|X - X'\| - \mathbb{E}\|Y - Y'\|,

where :math:`X, X'` are independent draws from :math:`P`, :math:`Y, Y'` from :math:`Q`, and
:math:`\|\cdot\|` is the Euclidean norm. The empirical version replaces these expectations with
averages over the two samples — cross-distances minus within-distances.

Properties
----------

It is a true **metric**: non-negative, symmetric, and zero iff :math:`P = Q`. Unlike MMD it is
**kernel-free**, working directly with Euclidean distances, and like MMD it stays sensitive in
**high dimensions** where the KS test fails. Both energy distance and MMD are **integral
probability metrics**.

Where it's used, with an example
----------------------------------

It serves **two-sample testing**, **drift detection** (training vs production), **GAN evaluation**
and **clustering validation**. Comparing two customer-age distributions, an energy distance of
**0.15** says they are fairly similar, while **1.2** signals a real difference — perhaps a much
younger incoming sample.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Representation Shift <174-representation-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Maximum Mean Discrepancy (MMD) <177-maximum-mean-discrepancy-mmd>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Data Drift <331-data-drift>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Energy Distance <https://insightful-data-lab.com/2025/08/23/energy-distance/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

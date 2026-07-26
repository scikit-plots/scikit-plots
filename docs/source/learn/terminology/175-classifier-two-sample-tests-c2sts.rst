:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-classifier-two-sample-tests-c2sts:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Classifier Two-Sample Tests (C2STs)</b></div>`

=====================================
Classifier Two-Sample Tests (C2STs)
=====================================

*Testing for shift by checking whether a classifier can tell two samples apart.*

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

A **classifier two-sample test (C2ST)** checks whether two datasets come from the **same
distribution** by **training a classifier to tell them apart**. If the classifier **can't** beat
chance, the distributions are likely the same; if it **can** separate them well, they differ — a
distribution shift. It is a modern, high-dimensional-friendly alternative to classical tests like
the KS test, MMD or energy distance.

How it works
------------

Three steps. **Label** dataset A as 0 (say training data) and dataset B as 1 (production). **Train**
a classifier — logistic regression, random forest, neural net — to predict which set a sample came
from. **Evaluate**: accuracy near **50%** means the two are indistinguishable (same distribution),
while accuracy well above 50% (or a high AUC) signals they differ.

The hypothesis test
-------------------

Formally it tests :math:`H_0: P = Q` against :math:`H_1: P \neq Q`, using **classifier accuracy or
AUC** as the test statistic and **permutation testing or bootstrapping** to get a p-value. Its
strengths are working in **very high dimensions** (where KS or chi-square fail) and using
off-the-shelf classifiers; its costs are the **compute** of training, **sensitivity to classifier
choice**, and the need for enough data.

Where it's used
---------------

C2STs power **data-drift detection** (train vs production), **generative-model evaluation** (real vs
generated), and **bias detection** (comparing subgroups). Concretely: with 10,000 samples from 2024
and 10,000 from 2025, an XGBoost separator reaching **AUC 0.90** means the two are easily
distinguishable — strong drift.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Representation Shift <174-representation-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Data Drift <331-data-drift>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Maximum Mean Discrepancy (MMD) <177-maximum-mean-discrepancy-mmd>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Classifier Two-Sample Tests (C2STs) <https://insightful-data-lab.com/2025/08/23/classifier-two-sample-tests-c2sts/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

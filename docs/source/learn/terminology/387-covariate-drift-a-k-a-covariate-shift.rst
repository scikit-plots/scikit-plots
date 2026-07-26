:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-covariate-drift-a-k-a-covariate-shift:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Covariate Drift (a.k.a. Covariate Shift)</b></div>`

==========================================
Covariate Drift (a.k.a. Covariate Shift)
==========================================

*A change in the input distribution while the input-output relationship holds.*

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

**Covariate drift** (covariate shift) is a change in the distribution of the **input features** a model sees
— the production inputs no longer look like the training inputs — while the feature-to-label rule stays the
same:

.. math::

   p_{\text{train}}(x) \neq p_{\text{prod}}(x), \qquad p(y \mid x)\ \text{unchanged}.

The model is being asked about a **different population** than it learned on.

How it differs
--------------

It is one of three **dataset shifts**. **Covariate drift** moves **p(x)** (the inputs), **label drift** moves
**p(y)** (the target mix), and **concept drift** moves **p(y | x)** (the relationship itself). Only concept
drift changes the *rule*; covariate drift changes *who* you're scoring.

Detecting and fixing it
-----------------------

It is caught by comparing feature distributions per column with **PSI** or the **KS** test. Remedies include
**importance weighting** — reweighting training points by the density ratio
:math:`w(x) = p_{\text{prod}}(x) / p_{\text{train}}(x)` — retraining on **recent** data, and building
**robust** features (winsorized, log-scaled, sensible bins).

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Covariate Drift (a.k.a. Covariate Shift) <https://insightful-data-lab.com/2025/08/19/covariate-drift-a-k-a-covariate-shift/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

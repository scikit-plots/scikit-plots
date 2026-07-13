:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-concept-drift:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Concept Drift</b></div>`

===============
Concept Drift
===============

*When the relationship between inputs and target changes over time.*

What it is
----------

**Concept drift** is a change in the **relationship** between inputs and the outcome — formally a shift in
:math:`P(Y \mid X)`. The inputs can look **identical**, but what they *mean* for the target has changed: the
rules the model learned no longer hold.

Why it's dangerous
------------------

Because the input distribution may look **normal**, concept drift is **hard to detect** — the model keeps
predicting **confidently** while being **wrong**. It shows up as a **decline** in accuracy, F1, or business
KPIs, which is why performance is monitored on **labeled** or delayed data, aided by detectors like
**ADWIN**, **DDM**, or **Page-Hinkley**.

Its forms and fix
-----------------

Drift can be **sudden** (a regime change), **gradual**, **incremental**, or **recurring** (seasonal patterns
that revert). The remedy is **retraining** on fresh labeled data that reflects the new relationship — the
reason production models need continuous **monitoring** and update loops.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Data Drift <331-data-drift>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Model Stability <187-model-stability>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Concept Drift <https://insightful-data-lab.com/2025/08/20/concept-drift/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

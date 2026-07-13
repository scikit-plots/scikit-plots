:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-drift-guardrails:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Drift Guardrails</b></div>`

==================
Drift Guardrails
==================

*Automated thresholds that flag or block on detected drift.*

What it is
----------

**Drift guardrails** are automated monitoring rules that watch a deployed model's **inputs and predictions**
for **distribution shift** and **trigger action** — an alert, an investigation, or a **retrain** — when the
drift crosses a threshold. They turn passive monitoring into a **response**.

How they're set
---------------

They compare live data to a **rolling baseline** with drift metrics like **PSI** and the **KS** test — for
example, **PSI above 0.2–0.25** on a key feature raises an alert, and **prediction drift that stays over
threshold for several consecutive days** kicks off **automatic retraining** on fresh labels.

Why they matter
---------------

A model silently **degrades** as the world moves away from its training data, and no aggregate dashboard
catches that on its own. Guardrails make the degradation **actionable** — often gating a retrained model
through a **registry** that re-evaluates it against production before it sees traffic.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Fairness Guardrails <351-fairness-guardrails>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Data Drift <331-data-drift>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Drift Guardrails <https://insightful-data-lab.com/2025/08/20/drift-guardrails/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

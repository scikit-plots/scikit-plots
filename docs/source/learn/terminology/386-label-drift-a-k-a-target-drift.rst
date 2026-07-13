:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-label-drift-a-k-a-target-drift:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Label Drift (a.k.a. Target Drift)</b></div>`

===================================
Label Drift (a.k.a. Target Drift)
===================================

*A change over time in the distribution of the target variable.*

What it is
----------

**Label drift** (target drift) is a change in the distribution of the **target** itself — the **class
balance** or outcome mix shifts between training and production, :math:`p_{\text{train}}(y) \neq
p_{\text{prod}}(y)`, even when the feature-to-label relationship may be unchanged. A fraud rate that creeps
from 1% to 3% is label drift.

How it differs
--------------

Like covariate drift it is a **dataset shift**, but it moves **p(y)** rather than **p(x)** or **p(y | x)**.
Because most classifiers implicitly assume the **base rate** they trained on, a shifted target distribution
can throw off **calibrated probabilities** and **thresholds** even if each input still maps to the right
answer.

Detecting and fixing it
-----------------------

Monitor the **label** or **prediction** distribution over time (PSI on predicted classes, tracked class
proportions). Fixes include **recalibrating** decision thresholds to the new base rate, **reweighting** or
resampling to the current mix, and **retraining** on recent labels.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Data Drift <331-data-drift>` · :doc:`Drift Detection <138-drift-detection>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Label Drift (a.k.a. Target Drift) <https://insightful-data-lab.com/2025/08/19/label-drift-a-k-a-target-drift/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

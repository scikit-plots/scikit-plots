:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-data-drift:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Data Drift</b></div>`

============
Data Drift
============

*When the distribution of input data shifts away from training.*

What it is
----------

**Data drift** — also called **covariate shift** — is a change in the **input** distribution :math:`P(X)`
while the model itself stays **fixed**. Its weights and logic are unchanged, but the data arriving at
inference no longer **resembles** the training data, so predictions grow **less reliable**.

Drift vs noise
--------------

Data drift is **systematic** — a sustained, directional shift — not the random fluctuation that's normal and
expected. Examples: a fraud model meeting **new devices and geographies**, or a credit model trained on
salaried workers now scoring **gig workers**.

Detecting and relating it
-------------------------

It's caught by comparing a **production** window to a **reference** window with statistical tests (**KS**,
**Chi-square**), **PSI**, or divergence metrics — the input side, so it's detectable **before** labels
arrive. Data drift can **evolve into** concept drift, which is why teams monitor :math:`P(X)` first, then
investigate the input–output relationship if performance drops.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Categorical Drift <179-categorical-drift>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Data Drift <https://insightful-data-lab.com/2025/08/20/data-drift/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

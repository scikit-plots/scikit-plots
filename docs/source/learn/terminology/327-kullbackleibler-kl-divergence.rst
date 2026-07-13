:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-kullbackleibler-kl-divergence:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Kullback–Leibler (KL) Divergence</b></div>`

==================================
Kullback–Leibler (KL) Divergence
==================================

*An asymmetric measure of how one distribution differs from another.*

What it is
----------

**KL divergence** measures how much one probability distribution **P** differs from a reference **Q** — the
**relative entropy**, the expected extra "surprise" from using Q when the truth is P:

.. math::

   D_{\text{KL}}(P \,\|\, Q) = \sum_{x} P(x)\,\log\!\frac{P(x)}{Q(x)} \;\ge\; 0,

(an integral for continuous distributions). It is **0** only when P = Q.

Its quirks
----------

KL is **asymmetric** — :math:`D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)` — so it is **not a true
distance**, and it **blows up** when Q assigns zero probability where P doesn't (it needs **absolute
continuity**). It is a **directed** measure, not a symmetric metric.

Where it's used
---------------

It is the core of **variational** methods (the VAE loss), of **feature selection**, and of **drift
detection**, where it behaves much like **PSI** — a solid default on large datasets, though its asymmetry
means you read it as a **degree** of drift, not a comparable distance.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Data Drift <331-data-drift>` · :doc:`Statistical Tests <328-statistical-tests>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Cramér's V <180-cramer-s-v>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Kullback–Leibler (KL) Divergence <https://insightful-data-lab.com/2025/08/20/kullback-leibler-kl-divergence/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

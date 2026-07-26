:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-cardinality-in-categorical-data:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Cardinality in Categorical Data</b></div>`

=================================
Cardinality in Categorical Data
=================================

*The number of distinct values a categorical feature can take.*

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

**Cardinality** is the number of **unique categories** in a categorical feature. ``Gender`` has just
two values — **low cardinality**; a ``Zip Code`` field has thousands — **high cardinality**. The
distinction matters because it changes how features should be measured and encoded.

Why it matters for association
--------------------------------

Cardinality directly affects measures like **Cramér's V**, which divides by the smaller table
dimension :math:`k = \min(\text{rows}, \text{cols})`. With **low cardinality** (Gender, Yes/No) the
statistic is easy to read — Gender versus product preference giving V = 0.3 is a clear moderate
association. With **high cardinality** (zip codes, product IDs) it grows unreliable: many categories
have tiny counts, the chi-square statistic inflates, and an association can look strong when it is
really just **sparsity**.

Handling high cardinality
---------------------------

Three remedies. **Group** rare categories into an "Other" bucket. Use **target encoding** or
**frequency encoding** rather than raw category comparison. And if using Cramér's V, ensure the
**sample is large enough** that expected cell counts are not tiny.

An example
----------

Comparing ``City`` (100 categories) against ``Purchase`` (Yes/No) might yield Cramér's V of **0.6**,
suggesting a strong link — but that can simply reflect **too few samples per city**, not a real
effect of city on purchasing.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`Embedding <173-embedding>` · :doc:`Data Drift <331-data-drift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Cardinality in Categorical Data <https://insightful-data-lab.com/2025/08/23/cardinality-in-categorical-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

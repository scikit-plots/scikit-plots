:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-categorical-drift:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Categorical Drift</b></div>`

===================
Categorical Drift
===================

*Shifts in the distribution of categorical feature values over time.*

What it is
----------

**Categorical drift** is a change over time in the **distribution of categories** between training
and production data. It is a form of data drift that specifically affects **categorical features**
rather than continuous ones.

What happens
------------

The **frequency of categories** shifts — if 80% of customers came from *Region A* in training but
only 40% do in production, the feature has drifted. This **hurts models** trained on the old mix:
predictions skew, once-rare categories become common, and entirely **unseen categories** can appear
in production that the model never learned.

Detecting it
------------

Standard tools compare category frequencies. A **chi-square test** weighs observed against expected
counts; **Cramér's V** measures the strength of the shift; and the **Population Stability Index
(PSI)** quantifies how much a categorical distribution has moved.

Where it bites
--------------

The effects are concrete. In **e-commerce**, a recommender fails when new products dominate. In
**healthcare**, a diagnosis model degrades as disease-code frequencies change. In **finance**, fraud
detection weakens as transaction types (online, POS, crypto) shift — each a categorical drift the
monitoring must catch.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`Data Drift <331-data-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Drift Detection <138-drift-detection>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Categorical Drift <https://insightful-data-lab.com/2025/08/23/categorical-drift/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

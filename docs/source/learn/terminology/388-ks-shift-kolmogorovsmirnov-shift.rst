:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-ks-shift-kolmogorovsmirnov-shift:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>KS shift (Kolmogorov–Smirnov shift)</b></div>`

=====================================
KS shift (Kolmogorov–Smirnov shift)
=====================================

*Using the KS statistic to quantify distribution shift in a feature.*

What it is
----------

**KS shift** detects **data drift** in a **continuous** feature with the **two-sample Kolmogorov–Smirnov
test** — it compares the feature's **cumulative distribution** in a reference window against the current
production window and measures their **largest** gap:

.. math::

   D = \sup_{x}\,\big| F_{\text{ref}}(x) - F_{\text{prod}}(x) \big|.

A large **D** means the two samples likely come from **different** distributions.

Why it's used
-------------

The KS test is **non-parametric** — it assumes **no** particular distribution shape — so it flags
**arbitrary** changes in a numeric feature's distribution. When **D** exceeds a **critical value** (or its
p-value falls below a threshold), the shift is **statistically significant**.

Using it in practice
--------------------

At production scale, tiny shifts become "significant" on **huge** samples, so the threshold is **calibrated**
to batch size to avoid **alert fatigue**. KS shift complements **PSI** (which grades severity) and
**Chi-square** (for categorical features) as part of a drift-monitoring suite.

----

**Mind map — connected ideas**

   :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Data Drift <331-data-drift>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Concept Drift <330-concept-drift>`

----

**More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `KS shift (Kolmogorov–Smirnov shift) <https://insightful-data-lab.com/2025/08/19/ks-shift-kolmogorov-smirnov-shift/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced

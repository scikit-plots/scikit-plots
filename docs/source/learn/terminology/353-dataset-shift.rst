:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-dataset-shift:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Dataset Shift</b></div>`

===============
Dataset Shift
===============

*Any mismatch between training and deployment data distributions.*

What it is
----------

**Dataset shift** is when the **training** data distribution differs from the **test / production**
distribution — formally :math:`P_{\text{train}}(X, Y) \neq P_{\text{test}}(X, Y)`. Because a model
learns its patterns from training data, a shift makes it **perform worse in the real world**. It is the
formal **umbrella** over the whole drift family.

The three types
---------------

**Covariate shift**: :math:`P(X)` changes but :math:`P(Y \mid X)` stays — a spam filter trained on old
emails, tested on new ones. **Prior (label) shift**: :math:`P(Y)` changes but :math:`P(X \mid Y)`
stays — fraud is 1% in training but 5% in production. **Concept shift**: :math:`P(Y \mid X)` itself
changes — the meaning of a label evolves, the hardest case to handle.

Detecting it
------------

Use **statistical tests** (KS, chi-square, PSI, KL-divergence), a **train-versus-test discriminator**
(if a classifier can tell the two sets apart, they differ), and **production monitoring** of accuracy,
AUC and calibration.

Coping
------

**Reweight** samples by importance, :math:`w(x) = \frac{P_{\text{test}}(x)}{P_{\text{train}}(x)}`, for
covariate shift; **resample** to match real prevalence; apply **domain adaptation**; **retrain
continually**; or build **robust, invariant** models. A model trained on one hospital's older, less
diverse patients loses accuracy on another's younger, more diverse population.

----

**Mind map — connected ideas**

   :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Data Drift <331-data-drift>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Label Noise <354-label-noise>`

----

**More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Dataset Shift <https://insightful-data-lab.com/2025/08/20/dataset-shift/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced

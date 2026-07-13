:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-drift-detection:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Drift Detection</b></div>`

=================
Drift Detection
=================

*Monitoring for changes in data or target distributions that degrade a model.*

What it is
----------

**Drift detection** is the practice of **monitoring for changes in the data distribution or
model behaviour over time** that can erode performance. Because real-world data evolves —
seasonality, new users, shifting business conditions — a model that was accurate at launch can
silently decay, and drift detection is what **triggers retraining or recalibration** before
that decay bites.

The three drifts
----------------

**Covariate (feature) drift**: the input distribution moves — 80% desktop users become 60%
mobile. **Label drift** (prior shift): the target mix changes — fraud rises from 1% to 3%.
**Concept drift**: the *relationship* between features and label changes — the same features
no longer predict fraud because the tactics evolved. Only concept drift necessarily breaks the
learned mapping; the others may or may not.

How it's detected
-----------------

Four families. **Statistical tests** compare old and new distributions (Kolmogorov–Smirnov,
chi-square, **PSI** — population stability index). **Distance measures** quantify the gap (KL
and Jensen–Shannon divergence, Wasserstein, MMD). **Classifier two-sample tests** train a
model to tell "old" from "new" — if it succeeds, the distributions differ. And **performance
monitoring** tracks loss or AUC over time, though as a *lagging* signal it only fires once
labels arrive.

In practice
-----------

Set **baselines** from training data, compare **rolling windows** (last 24h vs last 4 weeks),
fire **alerts** past a threshold (e.g. PSI > 0.2), and wire the signal into the pipeline to
**retrain or recalibrate**. A credit model trained on 2022 data might, by 2025, see income
shift toward gig workers — PSI flags the covariate drift and the monitor flags concept drift,
triggering a retrain. The payoff is avoiding **silent degradation**, protecting **fairness**
(drift can hit subgroups unevenly), and meeting **compliance** demands for monitoring.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Data Drift <331-data-drift>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Recalibration <159-recalibration>` · :doc:`Re-scoring <137-re-scoring>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Drift Detection <https://insightful-data-lab.com/2025/08/24/drift-detection/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

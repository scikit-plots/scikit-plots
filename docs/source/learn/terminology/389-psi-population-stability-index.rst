:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-psi-population-stability-index:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>PSI (Population Stability Index)</b></div>`

==================================
PSI (Population Stability Index)
==================================

*A widely used score measuring how much a distribution has shifted.*

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

The **Population Stability Index** measures how much a variable's distribution has **shifted** between a
**reference** ("expected") sample and a **current** ("actual") one — usually training vs production. It bins
the variable and sums the per-bin discrepancy:

.. math::

   \text{PSI} = \sum_{b} (A_b - E_b)\,\ln\!\frac{A_b}{E_b},

over bins :math:`b`. It is **0** when the distributions match and grows without bound as they diverge.

How it's computed
-----------------

Choose **bins** (often 10, by quantile or equal width) using the **same edges** for both samples, take each
bin's **proportion** in the expected and actual data, and sum the term above across bins. It is closely
related to a **symmetrized KL divergence**.

Reading it
----------

The rules of thumb are **< 0.1** stable, **0.1–0.25** moderate drift (**watch**), and **> 0.25** significant
drift (**retrain**). Two cautions: an **empty bin** makes PSI undefined or unbounded (so proportions are
clipped), and PSI tends to **rise with sample size**, so thresholds may need tuning.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Data Drift <331-data-drift>` · :doc:`Drift Detection <138-drift-detection>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `PSI (Population Stability Index) <https://insightful-data-lab.com/2025/08/19/psi-population-stability-index/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

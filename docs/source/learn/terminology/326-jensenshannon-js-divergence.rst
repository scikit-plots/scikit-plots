:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-jensenshannon-js-divergence:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Jensen–Shannon (JS) Divergence</b></div>`

================================
Jensen–Shannon (JS) Divergence
================================

*A symmetric, bounded measure of difference between two distributions.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**JS divergence** is the **symmetric, bounded** repair of KL. It averages the KL of each distribution to
their **mixture** :math:`M = \tfrac{1}{2}(P+Q)`:

.. math::

   D_{\text{JS}}(P \,\|\, Q) = \tfrac{1}{2}D_{\text{KL}}(P \,\|\, M) + \tfrac{1}{2}D_{\text{KL}}(Q \,\|\, M).

Unlike KL it is always **finite** and **symmetric**.

Its properties
--------------

JS ranges from **0** (identical) to a bounded maximum (**1** in bits, :math:`\log 2` in nats, when the
distributions are **disjoint**). Its **square root** is the **Jensen–Shannon distance**, which *is* a proper
metric — so JS gives a well-behaved, comparable measure of distributional difference.

Where it's used
---------------

It works on **numerical and categorical** features alike and is a popular **drift** signal — stable, less
noisy, and slightly **more sensitive** than KL or PSI — which is why monitoring systems favour it when a
symmetric, bounded score is wanted.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Data Drift <331-data-drift>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Statistical Tests <328-statistical-tests>` · :doc:`Cramér's V <180-cramer-s-v>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Jensen–Shannon (JS) Divergence <https://insightful-data-lab.com/2025/08/20/jensen-shannon-js-divergence/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

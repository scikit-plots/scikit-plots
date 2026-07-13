:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-kolmogorovsmirnov-ks-test:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Kolmogorov–Smirnov (KS) Test</b></div>`

==============================
Kolmogorov–Smirnov (KS) Test
==============================

*A test comparing distributions via their largest cumulative gap.*

What it is
----------

The **KS test** is a **non-parametric** test of whether two samples come from the **same distribution**
(two-sample), or whether a sample matches a **reference** distribution (goodness-of-fit). It compares their
**cumulative distribution functions (CDFs)**.

The statistic
-------------

Its **D-statistic** is the **largest vertical gap** between the two CDFs:

.. math::

   D = \sup_{x} \,\big|F_1(x) - F_2(x)\big|.

A bigger D means the distributions are further apart. Because it uses the CDF directly, it makes **no
assumptions** about the distribution's shape — its great strength.

Where it's used
---------------

With a null of "same distribution," a small p-value flags a **significant** difference — making the KS test
a standard tool for **drift detection** on continuous features and for **goodness-of-fit** checks. It
underlies the **KS statistic** used as a drift metric.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`Data Drift <331-data-drift>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Statistical Tests <328-statistical-tests>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>` · :doc:`Significance Level (α) <105-significance-level>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Kolmogorov–Smirnov (KS) Test <https://insightful-data-lab.com/2025/08/20/kolmogorov-smirnov-ks-test/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner

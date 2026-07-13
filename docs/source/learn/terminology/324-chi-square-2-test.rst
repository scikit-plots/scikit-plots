:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-chi-square-2-test:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Chi-square (χ²) Test</b></div>`

======================
Chi-square (χ²) Test
======================

*A test of association between categorical variables using expected counts.*

What it is
----------

The **chi-square test** works on **categorical** data, comparing **observed** counts to the **expected**
counts under a null hypothesis:

.. math::

   \chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}.

A large :math:`\chi^2` means observations stray far from expectation.

Its two forms
-------------

**Goodness-of-fit** asks whether one categorical variable follows a **specified distribution** (do dice
rolls look fair?); **independence** asks whether two categorical variables in a **contingency table** are
**associated** (is purchase related to region?). The statistic is compared to the **χ² distribution** with
the appropriate **degrees of freedom**.

Reading it and its limits
-------------------------

A small **p-value** rejects the null (a real fit failure or association); the strength of an association is
then summarized by **Cramér's V**. The test needs **adequate expected counts** per cell and has **low
power** on small samples — a non-significant result is weak evidence, not confirmation.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Statistical Tests <328-statistical-tests>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Data Drift <331-data-drift>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>` · :doc:`Significance Level (α) <105-significance-level>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Chi-square (χ²) Test <https://insightful-data-lab.com/2025/08/20/chi-square-%cf%87%c2%b2-test/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner

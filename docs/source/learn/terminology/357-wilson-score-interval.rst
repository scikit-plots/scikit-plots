:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-wilson-score-interval:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Wilson Score Interval</b></div>`

=======================
Wilson Score Interval
=======================

*An accurate confidence interval for a proportion, robust for small samples.*

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

The **Wilson score interval** is a well-calibrated confidence interval for a **binomial proportion**, derived
by improving the crude **normal-approximation (Wald)** interval. Introduced by E. B. Wilson in 1927, it is
**asymmetric** and always stays **within [0, 1]**.

Why it's better
---------------

Unlike the **Wald** interval, it doesn't **overshoot** past 0 or 1 and doesn't collapse to **zero width**
when the observed proportion is 0 or 1; and unlike **Clopper–Pearson**, it isn't overly **conservative** —
its coverage sits **close to nominal**, so its intervals are **narrower**. That balance makes it the
**recommended default** in most applications.

The caveats
-----------

Its coverage can dip **slightly below** nominal for a few awkward proportions, and for **extremely small**
samples the guaranteed **Clopper–Pearson** may still be safer. A continuity-corrected variant exists for
tighter coverage.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Z-Score <097-z-score>` · :doc:`Statistical Tests <328-statistical-tests>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Wilson Score Interval <https://insightful-data-lab.com/2025/08/20/wilson-score-interval/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner

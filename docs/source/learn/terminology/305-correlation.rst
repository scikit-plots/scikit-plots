:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-correlation:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Correlation</b></div>`

=============
Correlation
=============

*The strength and direction of a linear relationship between variables.*

What it is
----------

**Correlation** measures the **strength and direction** of the relationship between two variables. The
**Pearson coefficient** — **r** in a sample, **ρ** in the population — runs from **−1 to +1**: −1 a perfect
**negative** line, +1 a perfect **positive** line, and **0** no **linear** relationship.

How to read it
--------------

The **sign** gives direction, the **magnitude** gives strength; squaring it yields **r²**, the share of one
variable's **variance explained** by the other. Rough effect-size guides call 0.1 small, 0.3 medium, 0.5
large — but a statistically significant r can still be **trivially** small in a large sample.

Its limits
----------

Correlation captures only **linear** association, so it can **miss** strong nonlinear patterns; it is **not
robust** to **outliers**, which can inflate or hide it; and, crucially, **correlation is not causation** — two
variables can move together because a **third** drives both. Use rank correlation (Spearman) for monotonic,
non-linear ties.

----

**Mind map — connected ideas**

   :doc:`R² (R-squared) <259-r2-r-squared>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`Statistical Tests <328-statistical-tests>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Outlier <307-outlier>`

----

**More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Correlation <https://insightful-data-lab.com/2025/08/21/correlation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: beginner

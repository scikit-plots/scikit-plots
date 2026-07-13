:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-median:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Median</b></div>`

========
Median
========

*The middle value separating the higher and lower halves of data.*

What it is
----------

The **median** is the **middle value** of the data once sorted — for an odd count the single center value,
for an even count the **average of the two** middle values. It splits the data into two **equal halves** and
is exactly the **0.5 quantile**.

Its strength
------------

The median is **robust to outliers** — extreme values change **which** point sits in the middle only
slightly, and never by their magnitude, so its **breakdown point is 50%**. That resilience makes it the
right summary for **skewed** data (income, house prices) and the value that **minimizes absolute error**.

The trade-off
-------------

The median ignores the **magnitudes** of all but the central values, so it discards information the mean
uses, and it is **harder to manipulate** algebraically. It needs **orderable** data — it works on ordinal,
not nominal, values.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Mean <316-mean>` · :doc:`Outlier <307-outlier>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Regression Models <309-regression-models>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Median <https://insightful-data-lab.com/2025/08/21/median/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner

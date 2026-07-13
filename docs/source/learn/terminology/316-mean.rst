:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-mean:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Mean</b></div>`

======
Mean
======

*The arithmetic average of a set of values.*

What it is
----------

The **(arithmetic) mean** is the **average** — the sum of all values divided by their count:

.. math::

   \mu = \frac{1}{n}\sum_{i=1}^{n} x_i.

It uses **every** data point, which makes it the natural summary of a **symmetric** distribution and the
value that **minimizes squared error**.

Its weakness
------------

Because it incorporates every value, the mean is **sensitive to outliers** and **skew** — a single extreme
value drags it toward the tail. Statisticians say its **breakdown point is 0%**: one bad point can move it
arbitrarily. On **skewed** data like income, the mean **overstates** where most values sit.

When to use it
--------------

Prefer the mean for **roughly symmetric**, outlier-free numeric data, where it is efficient and
mathematically convenient (it feeds variance, standard error, regression). For skewed or outlier-heavy
data, reach for the **median**. It is undefined for **categorical** data.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Median <315-median>` · :doc:`Outlier <307-outlier>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Regression Models <309-regression-models>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Mean <https://insightful-data-lab.com/2025/08/21/mean/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner

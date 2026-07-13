:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-z-score:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Z-Score</b></div>`

=========
Z-Score
=========

*How many standard deviations a value lies from the mean.*

What it is
----------

A **z-score** (or **standard score**) says **how many standard deviations a value sits
from the mean** of its distribution. It re-expresses raw numbers on a common, unit-free
scale so values from different datasets can be compared directly.

The formula
-----------

For a population,

.. math::

   z = \frac{x - \mu}{\sigma},

with raw value :math:`x`, mean :math:`\mu` and standard deviation :math:`\sigma`. From a
**sample**, use :math:`z = (x - \bar{x})/s`.

Reading it
----------

:math:`z = 0` is exactly at the mean; :math:`z = +1` is one SD above; :math:`z = -2` two
SD below. Large magnitudes (:math:`|z| > 3`) flag likely **outliers**.

Examples
--------

An exam score of 85 with mean 70 and SD 10 gives :math:`z = (85 - 70)/10 = 1.5` — 1.5 SD
above average. A height of 150 cm with mean 170 and SD 8 gives
:math:`z = (150 - 170)/8 = -2.5`.

Why it's useful
---------------

Three things at once: **standardisation** (compare maths and English scores on different
scales), **probability** (in the standard normal — mean 0, SD 1 — a z maps to a tail area,
e.g. :math:`z = 1.96` bounds the central 95%), and **test statistics** (z-, t- and
χ²-tests all compare an observed value to its expected spread in z-like units). A
**z-score** standardises one data point; a **z-test** uses that machinery to test a
hypothesis about a mean or proportion.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Critical Value <087-critical-value>` · :doc:`Sample Standard Deviation <088-sample-standard-deviation>` · :doc:`True Mean (Population Mean) <085-true-mean-population-mean>` · :doc:`Two-Proportion Z-Test <098-two-proportion-z-test>` · :doc:`Statistical Significance <096-statistical-significance>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Z-Score <https://insightful-data-lab.com/2025/08/24/z-score/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner

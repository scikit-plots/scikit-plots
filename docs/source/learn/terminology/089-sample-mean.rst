:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-sample-mean:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Sample Mean</b></div>`

=============
Sample Mean
=============

*The arithmetic average of a sample, used to estimate the population mean.*

What it is
----------

The **sample mean** :math:`\bar{x}` is the **arithmetic average of a sample** — the sum of
the observations divided by their count. It is the **statistic** used to estimate the
**true population mean** :math:`\mu`:

.. math::

   \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.

Examples
--------

Test scores :math:`\{80, 85, 90, 95, 100\}` give :math:`\bar{x} = 450/5 = 90`. Ten people
with total height 1,720 cm give :math:`\bar{x} = 172` cm.

Its three key properties
------------------------

- **Unbiased** — :math:`\mathbb{E}[\bar{x}] = \mu`; on average the sample mean equals the
  population mean.
- **Sampling distribution (CLT)** — for large :math:`n`, the Central Limit Theorem makes
  :math:`\bar{x}` approximately normal,

  .. math::

     \bar{x} \sim N\!\left(\mu, \frac{\sigma^2}{n}\right),

  with **standard error of the mean** :math:`SE = \sigma/\sqrt{n}` — so its variability
  shrinks as :math:`n` grows.
- **Outlier-sensitive** — being a sum, the mean is pulled by extreme values (unlike the
  median).

Where it shows up
-----------------

The sample mean is everywhere: **descriptive** summaries, **estimating** :math:`\mu`,
**hypothesis tests** (the one-sample t-test), and **confidence intervals**. It is the
**best unbiased estimator** of the population mean.

----

**Mind map — connected ideas**

   :doc:`Sample Standard Deviation <088-sample-standard-deviation>` · :doc:`True Mean (Population Mean) <085-true-mean-population-mean>` · :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Frequentist <059-frequentist>` · :doc:`Regression Coefficient <090-regression-coefficient>`

----

**More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Sample Mean <https://insightful-data-lab.com/2025/08/25/sample-mean/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: beginner

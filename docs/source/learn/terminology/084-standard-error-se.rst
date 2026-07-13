:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-standard-error-se:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Standard Error (SE)</b></div>`

=====================
Standard Error (SE)
=====================

*The standard deviation of a statistic's sampling distribution — how much an estimate varies across samples.*

What it is
----------

The **standard error (SE)** measures **how much a sample statistic — a mean or a
proportion — would vary across many random samples** from the same population. It is the
**uncertainty of the estimate**: how far the sample value is likely to fall from the true
population value.

The formulas
------------

For a **mean** (population SD :math:`\sigma`, or sample SD :math:`s` when :math:`\sigma`
is unknown):

.. math::

   SE_{\bar{x}} = \frac{\sigma}{\sqrt{n}} \quad\left(\text{or } \frac{s}{\sqrt{n}}\right).

For a **proportion**:

.. math::

   SE_{\hat{p}} = \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}.

Key properties
--------------

**Bigger samples shrink it** (:math:`SE \propto 1/\sqrt{n}`, so estimates get more
precise), **more spread inflates it** (larger :math:`\sigma`), and it is the bridge to the
**Central Limit Theorem**: for large :math:`n` the sample mean is approximately normal
with the SE as its spread.

Examples
--------

With :math:`\mu = 100, \sigma = 20, n = 25`, :math:`SE = 20/\sqrt{25} = 4` — sample means
vary about ±4 around the truth. For a proportion, 520 "yes" out of 1,000 gives
:math:`\hat{p} = 0.52` and :math:`SE = \sqrt{0.52 \times 0.48 / 1000} \approx 0.016`, a
±1.6% margin.

SE vs standard deviation
------------------------

They are easy to confuse: the **standard deviation** describes the spread of *individual
data points*, while the **standard error** describes the spread of the *estimate*. SD is
variability in the data; SE is variability in the statistic — and SE shrinks with
:math:`n` while SD does not.

Where it shows up
-----------------

The SE is the denominator of inference: **confidence intervals**
(:math:`\text{estimate} \pm z_{\alpha/2}\, SE`), **test statistics** (the z- and
t-tests), and **A/B comparisons** of conversion rates all run on it.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`True Conversion Rate <083-true-conversion-rate>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Frequentist <059-frequentist>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`True Mean (Population Mean) <085-true-mean-population-mean>` · :doc:`A/B Testing <380-a-b-testing>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Standard Error (SE) <https://insightful-data-lab.com/2025/08/25/standard-error-se/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner

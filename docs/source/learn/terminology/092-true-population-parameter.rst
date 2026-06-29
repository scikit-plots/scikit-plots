:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-true-population-parameter:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>True Population Parameter</b></div>`

===========================
True Population Parameter
===========================

*The fixed, usually unknown quantity (mean, proportion, coefficient) that estimation targets.*

What it is
----------

A **true population parameter** is a **fixed (but usually unknown) number that describes a
whole population**. The word *true* stresses that the value exists even though we seldom
observe it. The familiar ones are written with Greek letters: :math:`\mu` (mean),
:math:`\sigma` (standard deviation), :math:`p` (proportion) and :math:`\rho`
(correlation).

Parameter vs statistic
----------------------

Because we rarely measure an entire population, we draw a **sample** and compute a
**statistic** to estimate each parameter:

- mean — parameter :math:`\mu`, statistic :math:`\bar{x}`
- standard deviation — :math:`\sigma`, statistic :math:`s`
- proportion — :math:`p`, statistic :math:`\hat{p}`
- correlation — :math:`\rho`, statistic :math:`r`

The **statistic is random** (it depends on which sample you draw); the **parameter is
fixed but unknown**. This single distinction underlies every "true vs sample" page in
statistics.

Example
-------

The parameter :math:`p` might be the real fraction of all U.S. voters who support
candidate A; a survey of 1,000 gives the statistic :math:`\hat{p} = 0.52`, an **estimate**
of that unknown :math:`p`.

Inference is about parameters
-----------------------------

We never know a parameter exactly without a census, so we **infer** it: a **confidence
interval** brackets it ("95% confident the true parameter lies in this range"), and a
**hypothesis test** evaluates a claim about it (:math:`H_0 : \mu = 100`). Estimators,
standard errors, intervals and tests all exist to pin down population parameters from
sample data.

----

**Mind map — connected ideas**

   :doc:`True Mean (Population Mean) <085-true-mean-population-mean>` · :doc:`True Conversion Rate <083-true-conversion-rate>` · :doc:`Sample Mean <089-sample-mean>` · :doc:`Sample Standard Deviation <088-sample-standard-deviation>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Proportion <091-proportion>`

----

**More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `True Population Parameter <https://insightful-data-lab.com/2025/08/25/true-population-parameter/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: beginner

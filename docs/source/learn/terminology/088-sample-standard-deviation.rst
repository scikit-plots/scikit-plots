:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-sample-standard-deviation:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Sample Standard Deviation</b></div>`

===========================
Sample Standard Deviation
===========================

*An estimate of population spread from a sample, using n - 1 in the denominator.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

The **sample standard deviation** :math:`s` measures **how spread out the values in a
sample are around the sample mean** :math:`\bar{x}`. It is the **square root of the sample
variance** — roughly, the average distance of a data point from the mean.

The formula and Bessel's correction
-----------------------------------

.. math::

   s = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1}}.

The denominator is :math:`n - 1`, not :math:`n` — **Bessel's correction**. Dividing by
:math:`n` would *underestimate* the true spread, because the deviations are taken from the
sample mean (which is itself fitted to the data); using :math:`n - 1` makes :math:`s^2` an
**unbiased estimator** of the population variance :math:`\sigma^2`.

Worked example
--------------

For :math:`\{5, 7, 9\}`: the mean is :math:`\bar{x} = 7`; deviations are
:math:`-2, 0, +2`; squared, :math:`4, 0, 4`, summing to 8; divide by :math:`n - 1 = 2` to
get 4; the square root is :math:`s = 2`.

Reading it
----------

A **small** :math:`s` means points cluster near the mean (low variability); a **large**
:math:`s` means they're spread out; :math:`s = 0` means every value is identical. The
**population** standard deviation :math:`\sigma` uses :math:`N` in the denominator (the
whole population); the **sample** version uses :math:`n - 1` (an estimate of
:math:`\sigma`).

Where it shows up
-----------------

:math:`s` is the raw material of inference: it feeds the **standard error**
(:math:`SE = s/\sqrt{n}`), the **t-test**, **ANOVA** and **regression**, and the
**confidence intervals** built around sample means.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Sample Mean <089-sample-mean>` · :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`True Mean (Population Mean) <085-true-mean-population-mean>` · :doc:`Frequentist <059-frequentist>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Sample Standard Deviation <https://insightful-data-lab.com/2025/08/25/sample-standard-deviation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner

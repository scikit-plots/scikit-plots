:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-critical-value:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Critical Value</b></div>`

================
Critical Value
================

*The cutoff from a reference distribution that a test statistic must pass to reject the null.*

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

A **critical value** is the **cutoff on a distribution that marks the rejection boundary**
for a hypothesis test. It separates the **acceptance region** (fail to reject :math:`H_0`)
from the **rejection region** (reject :math:`H_0`), and depends on three things: the
**significance level** :math:`\alpha`, whether the test is **one- or two-tailed**, and the
**distribution** used (Z, t, :math:`\chi^2`, F).

The decision rule
-----------------

If the **test statistic exceeds the critical value in magnitude**, reject :math:`H_0`; if
it falls inside, fail to reject.

Values by distribution
----------------------

- **Z** (large :math:`n`, known :math:`\sigma`) — two-tailed :math:`\alpha = 0.05` gives
  :math:`\pm 1.96`; one-tailed, :math:`1.645`.
- **t** (small :math:`n`, unknown :math:`\sigma`) — two-tailed, :math:`\alpha = 0.05`,
  :math:`df = 10` gives :math:`\pm 2.228`; as :math:`n` grows, :math:`t \to z`.
- **χ²** (goodness-of-fit, independence) — :math:`\alpha = 0.05, df = 4` gives
  :math:`\approx 9.49`.
- **F** (ANOVA) — :math:`\alpha = 0.05, df_1 = 3, df_2 = 20` gives :math:`\approx 3.10`.

Two roles
---------

The same critical value drives both **confidence intervals** —
:math:`\text{estimate} \pm (\text{critical value}) \times \text{SE}` (a 95% z-interval for
a mean of 100 with :math:`\text{SE} = 2` is :math:`[96.08, 103.92]`) — and **hypothesis
tests**.

Worked test
-----------

Test :math:`H_0 : \mu = 50` vs :math:`H_1 : \mu \neq 50` with sample mean 53,
:math:`\sigma = 10, n = 100`:

.. math::

   z = \frac{53 - 50}{10/\sqrt{100}} = \frac{3}{1} = 3.0.

At :math:`\alpha = 0.05` two-tailed the critical value is :math:`\pm 1.96`; since
:math:`3.0 > 1.96`, **reject** :math:`H_0`.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Type I Error <080-type-i-error>` · :doc:`Frequentist <059-frequentist>` · :doc:`True Mean (Population Mean) <085-true-mean-population-mean>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Critical Value <https://insightful-data-lab.com/2025/08/25/critical-value/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner

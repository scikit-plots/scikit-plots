:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-frequentist:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Frequentist</b></div>`

=============
Frequentist
=============

*The school of statistics treating probability as long-run frequency and parameters as fixed unknowns.*

What it is
----------

The **frequentist** approach defines probability as the **long-run frequency** of an
event over infinitely many repeated trials. Parameters — a population mean, a
conversion rate, a treatment effect — are treated as **fixed but unknown constants**,
and all the uncertainty comes from the **randomness of the data (the sample)**, never
from the parameter itself.

Key principles
--------------

- **Probability = frequency** — "P(heads) = 0.5" means about half of many flips land
  heads.
- **Parameters fixed, data random** — the true mean :math:`\mu` is a fixed constant;
  the sample mean :math:`\bar{x}` varies from sample to sample.
- **Inference via repeated sampling** — p-values, confidence intervals and tests are
  all defined by *what would happen if the experiment were repeated many times*.

The toolkit
-----------

Point estimates (the sample mean :math:`\bar{x}` for :math:`\mu`), **confidence
intervals** (an interval procedure that captures the true parameter in, say, 95% of
repeated experiments), **hypothesis tests** (a null :math:`H_0` and a p-value — the
probability of data at least as extreme as observed, *assuming* :math:`H_0`), and
**maximum likelihood estimation**.

A subtlety worth stating
------------------------

A 95% confidence interval does **not** mean "95% probability the true value is in this
interval" — under frequentism the parameter is fixed, so it's either in or out. It means
*the procedure* covers the truth 95% of the time across repetitions. The "95%
probability the mean lies here" reading is the **Bayesian credible interval**.

Example — an A/B test
------------------------

To ask whether variant B beats A, treat the true rates :math:`p_A, p_B` as fixed,
estimate them with sample proportions :math:`\hat{p}_A, \hat{p}_B`, run a
**two-proportion z-test**, and reject :math:`H_0` if the p-value is below 0.05.

Frequentist vs Bayesian
-----------------------

The two paradigms differ along clear lines: probability as **long-run frequency** vs
**degree of belief**; parameters as **fixed constants** vs **random variables**;
uncertainty from **data only** vs **prior + data**; inference via **p-values and
confidence intervals** vs **posteriors and credible intervals**; and on sequential
data, frequentist peeking **inflates error** (needs corrections) while Bayesian updating
is **continuously valid**. Frequentist methods remain dominant in medicine, the social
sciences and classic A/B testing.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Posterior <063-posterior>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Frequentist <https://insightful-data-lab.com/2025/08/28/frequentist/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner

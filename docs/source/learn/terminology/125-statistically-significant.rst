:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-statistically-significant:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Statistically Significant</b></div>`

===========================
Statistically Significant
===========================

*Describing a result unlikely under the null hypothesis at the chosen level.*

What it is
----------

A result is **statistically significant** when the **observed effect is unlikely to have
arisen by chance** under the null hypothesis, judged against a chosen significance level
:math:`\alpha`. Operationally it means there is **enough evidence to reject** :math:`H_0` —
and crucially, "significant" here means *statistical evidence*, **not real-world
importance**.

The decision rule
-----------------

It comes down to comparing the **p-value** to the threshold: if :math:`p \le \alpha`, the
result **is** statistically significant and you reject :math:`H_0`; if :math:`p > \alpha`, it
is **not**, and you fail to reject. The usual :math:`\alpha` is 0.05.

Examples
--------

A drug trial with :math:`p = 0.01` against :math:`\alpha = 0.05` **is** significant —
evidence the drug beats placebo. An A/B test where a new button lifts clicks 3% but returns
:math:`p = 0.2` is **not** significant — the lift could be noise.

The cautions
------------

Three matter. **Significance is not importance**: with a large enough dataset a trivial 0.5%
effect can clear the bar yet mean nothing. It is **sample-size dependent**: bigger samples
make significance easier to reach. And it is vulnerable to **p-hacking** — running many tests
or slicing data until something crosses :math:`\alpha`. A significant result is a starting
point for judgement, read alongside effect size and context, not a verdict on its own.

----

**Mind map — connected ideas**

   :doc:`Statistical Significance <096-statistical-significance>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Significance Level (α) <105-significance-level>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Type I Error <080-type-i-error>` · :doc:`Effect Size (δ) <106-effect-size>`

----

**More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Statistically Significant <https://insightful-data-lab.com/2025/08/24/statistically-significant/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: beginner

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-beta-distribution:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Beta Distribution</b></div>`

===================
Beta Distribution
===================

*A continuous distribution on [0, 1]; the conjugate prior for a binomial probability.*

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

The **Beta distribution** is a continuous distribution on the interval :math:`[0, 1]`,
which makes it the natural way to model a **probability or proportion**. It has two
**shape parameters**, :math:`\alpha` and :math:`\beta`, and density

.. math::

   f(p \mid \alpha, \beta) = \frac{1}{B(\alpha, \beta)}\, p^{\alpha - 1} (1 - p)^{\beta - 1},
   \qquad 0 \le p \le 1,

where :math:`B(\alpha, \beta)` is the Beta function, the normalising constant.

Summary statistics
------------------

.. math::

   \mathbb{E}[p] = \frac{\alpha}{\alpha + \beta}, \qquad
   \operatorname{Var}(p) = \frac{\alpha\beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)},

and, when :math:`\alpha, \beta > 1`, the mode is
:math:`(\alpha - 1)/(\alpha + \beta - 2)`. A larger :math:`\alpha + \beta` means a more
concentrated (lower-variance) belief.

The shape gallery
-----------------

The two parameters give the family enormous flexibility:

- :math:`\alpha = \beta = 1` — **uniform** on :math:`[0, 1]`.
- :math:`\alpha > \beta` — skewed toward 1; :math:`\alpha < \beta` — skewed toward 0.
- :math:`\alpha, \beta > 1` — unimodal (a single bump).
- :math:`\alpha, \beta < 1` — U-shaped (mass piling up near 0 and 1).

Why it's central: conjugacy
---------------------------

The Beta is the **conjugate prior** for the **Binomial likelihood**. With a
:math:`\text{Beta}(\alpha, \beta)` prior and :math:`k` successes in :math:`n` trials,
the posterior is again Beta:

.. math::

   p \mid \text{data} \sim \text{Beta}(\alpha + k,\; \beta + n - k).

So Bayesian updating is just *adding successes to* :math:`\alpha` *and failures to*
:math:`\beta` — no integration required.

Examples
--------

- :math:`\text{Beta}(1,1)` — a flat prior, no preference before data.
- :math:`\text{Beta}(20, 20)` — sharply peaked at 0.5, a strong "fair coin" belief.
- Start from :math:`\text{Beta}(1,1)`, observe 7 heads and 3 tails →
  :math:`\text{Beta}(8, 4)`, mean :math:`8/12 \approx 0.67`.

Where it shows up
-----------------

Bayesian inference over probabilities, **A/B testing** (belief about conversion rates),
reliability (success/failure rates), and any proportion modelling. Its multivariate
generalisation, the **Dirichlet distribution**, plays the same conjugate role for the
categorical/multinomial likelihood. In code, ``scipy.stats.beta`` provides the density,
sampling and quantiles.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>` · :doc:`Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB)) <050-thompson-sampling-ts-in-bandits-multi-armed-band>` · :doc:`Bayes' Theorem <066-bayes-theorem>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Beta Distribution <https://insightful-data-lab.com/2025/08/28/beta-distribution/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner

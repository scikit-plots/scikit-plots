:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-parameter-s-of-interest:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Parameter(s) of Interest</b></div>`

==========================
Parameter(s) of Interest
==========================

*The unknown quantities an analysis sets out to estimate.*

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

The **parameter(s) of interest** are the **population characteristics you set out to
estimate or test** — usually unknown population values like a true mean, a true
proportion, a difference in means, or regression coefficients. The whole study or
experiment is designed to learn about them. In short: *the parameter of interest is the
thing you actually care about measuring.*

Parameter vs statistic
----------------------

The crucial distinction: the **parameter** is the fixed (usually unknown) population
value; the **statistic** is the sample quantity you compute to *estimate* it.

- Population mean :math:`\mu` — estimated by the sample mean :math:`\bar{x}`.
- Population proportion :math:`p` — estimated by :math:`\hat{p}`.
- Difference in means :math:`\mu_1 - \mu_2` — estimated by the difference in sample
  means.
- Regression coefficients :math:`\beta` — estimated by :math:`\hat{\beta}`.

Examples
--------

- **A/B test** — the true conversion rates :math:`p_A` and :math:`p_B`, or their
  difference :math:`p_B - p_A`.
- **Medical trial** — the average treatment effect,

  .. math::

     \text{ATE} = P(\text{recovery} \mid \text{drug}) - P(\text{recovery} \mid \text{placebo}).

- **Regression** — the coefficients :math:`\beta_1, \beta_2, \dots` linking predictors
  to the outcome.

Why it matters
--------------

Naming the parameter of interest is the **first step** in designing any study: every
estimator, hypothesis test (:math:`H_0` vs :math:`H_1`) and confidence (or credible)
interval is built to make inferences about it. Frequentists treat it as a fixed unknown
constant; Bayesians put a posterior distribution over it — but in both frameworks it is
the *target* of the analysis.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Frequentist <059-frequentist>` · :doc:`Posterior <063-posterior>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`True Conversion Rate <083-true-conversion-rate>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Parameter(s) of Interest <https://insightful-data-lab.com/2025/08/28/parameters-of-interest/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

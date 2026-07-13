:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-posterior:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Posterior</b></div>`

===========
Posterior
===========

*The distribution of parameters given the data; the central object of Bayesian inference.*

What it is
----------

The **posterior** is the **updated probability distribution of a parameter after
observing data** — the new belief formed by combining the **prior** (what you thought
before) with the **likelihood** (what the data say). The one-line version: *posterior =
prior updated with evidence*.

Bayes' theorem
--------------

.. math::

   P(\theta \mid D) = \frac{P(D \mid \theta)\, P(\theta)}{P(D)}
   \;\propto\; \underbrace{P(D \mid \theta)}_{\text{likelihood}} \times \underbrace{P(\theta)}_{\text{prior}},

with the marginal likelihood :math:`P(D)` as the normalising constant. The
**proportional** form is what you actually work with: the posterior *shape* is just
prior times likelihood.

What you get from it
--------------------

The posterior is a **distribution**, so it supports direct probability statements — "a
95% probability the conversion rate is between 4% and 6%" — which a frequentist
confidence interval cannot make. Point summaries (the posterior **mean** or
**mode/MAP**) and **credible intervals** are all read off it.

Example — coin toss
-------------------

Uniform prior :math:`\text{Beta}(1,1)`, a Binomial likelihood, and 7 heads in 10 tosses
give

.. math::

   P(p \mid \text{data}) \propto p^7 (1-p)^3 \cdot 1 = \text{Beta}(8, 4),

a posterior centred near 0.67.

The prior washes out
--------------------

A defining property: **the more data you collect, the more the likelihood dominates and
the less the prior matters**. With small samples the prior shapes the answer; with large
ones the posterior is driven almost entirely by the data — which is why honest priors
are cheap insurance, not permanent bias. The posterior is the object *all* Bayesian
decisions are based on.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`Beta Distribution <099-beta-distribution>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Posterior <https://insightful-data-lab.com/2025/08/28/posterior/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

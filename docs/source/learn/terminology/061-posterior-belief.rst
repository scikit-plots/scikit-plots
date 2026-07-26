:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-posterior-belief:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Posterior belief</b></div>`

==================
Posterior belief
==================

*Updated belief about a parameter after combining prior and data through Bayes' theorem.*

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

**Posterior belief** is your **updated belief about a parameter after seeing data** —
represented, in Bayesian statistics, by the **posterior distribution**. It fuses two
ingredients: the **prior** (what you believed beforehand) and the **likelihood** (what
the data say). In a sentence: *posterior = prior updated by evidence*.

Bayes' theorem
--------------

.. math::

   P(\theta \mid D) = \frac{P(D \mid \theta)\, P(\theta)}{P(D)},

with prior :math:`P(\theta)`, likelihood :math:`P(D \mid \theta)`, evidence
:math:`P(D)`, and posterior :math:`P(\theta \mid D)`.

It's a distribution, not a number
---------------------------------

The posterior is a **whole distribution** over parameter values, showing how plausible
each value is *after* the data — and from it you read off summaries (mean, mode,
intervals) or specific **posterior probabilities** of events.

Example — coin toss
-------------------

A uniform prior :math:`\text{Beta}(1,1)`, updated with 7 heads in 10 tosses, gives the
posterior :math:`\text{Beta}(8, 4)`, centred near 0.67 — the updated belief that
:math:`p` is most likely around two-thirds.

Posterior belief vs posterior probability
-----------------------------------------

A useful distinction: the **posterior belief** is the entire distribution
(:math:`p \sim \text{Beta}(8, 4)`), while a **posterior probability** is a single number
pulled from it — for instance :math:`P(p > 0.5 \mid \text{data}) = 0.9`. The latter is
one summary of the former.

Where it shows up
-----------------

A/B testing (belief about a conversion-rate difference), clinical trials (belief about
treatment effect), and any Bayesian model that updates parameter distributions as data
arrive.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>` · :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Binomial Likelihood <060-binomial-likelihood>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Posterior belief <https://insightful-data-lab.com/2025/08/28/posterior-belief/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-bayes-theorem:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Bayes' Theorem</b></div>`

================
Bayes' Theorem
================

*The rule that turns a prior into a posterior using the likelihood: posterior is proportional to likelihood x prior.*

What it is
----------

**Bayes' theorem** is the rule for **updating a belief when new evidence arrives**. It
is the single equation that ties together the four quantities of Bayesian inference —
prior, likelihood, evidence and posterior.

The formula
-----------

For a hypothesis :math:`H` and data :math:`D`,

.. math::

   P(H \mid D) = \frac{P(D \mid H)\, P(H)}{P(D)},

where :math:`P(H)` is the **prior**, :math:`P(D \mid H)` the **likelihood**,
:math:`P(D)` the **marginal likelihood** (evidence / normaliser) and
:math:`P(H \mid D)` the **posterior**. Stripped to its working form,

.. math::

   \text{Posterior} \propto \text{Prior} \times \text{Likelihood}.

Worked example — the base-rate trap
-----------------------------------

A disease affects 1% of people; a test is 99% sensitive and has a 5% false-positive
rate. A patient tests positive — how likely are they to be sick? The evidence is

.. math::

   P(+) = \underbrace{0.99 \times 0.01}_{\text{true positive}} + \underbrace{0.05 \times 0.99}_{\text{false positive}} = 0.0099 + 0.0495 = 0.0594,

so

.. math::

   P(\text{disease} \mid +) = \frac{0.0099}{0.0594} \approx 0.167.

Despite a positive result on a "99% accurate" test, the chance of disease is only about
**16.7%** — because the disease is rare, false positives swamp the true ones. This
**base-rate fallacy** is exactly what Bayes' theorem corrects: the rare prior pulls the
posterior far below the test's sensitivity.

Where it shows up
-----------------

Bayesian inference and updating, the **Naive Bayes** classifier and Bayesian networks,
medical diagnostics, **A/B testing** (Bayesian sequential testing, posterior probability
of uplift), fraud and risk estimation, and everyday belief revision under new
information.

----

**Mind map — connected ideas**

   :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>` · :doc:`Posterior <063-posterior>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Posterior Probability <073-posterior-probability>`

----

**More in Bayesian Inference**

   :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Bayes' Theorem <https://insightful-data-lab.com/2025/08/28/bayes-theorem/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-posterior-probability:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Posterior Probability</b></div>`

=======================
Posterior Probability
=======================

*The probability of a hypothesis or event evaluated under the posterior distribution.*

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

The **posterior probability** is the **probability of a hypothesis (or parameter value)
after observing data**, computed with Bayes' theorem by combining the **prior** (belief
before) with the **likelihood** (evidence from the data):

.. math::

   P(H \mid \text{data}) = \frac{P(\text{data} \mid H)\, P(H)}{P(\text{data})}.

What it answers
---------------

It answers a directly useful question — *"given the data, how probable is this
hypothesis?"* — which is **not** what a frequentist p-value answers (*"if* :math:`H_0`
*were true, how likely is data this extreme?"*). The posterior is the quantity people
usually *think* a p-value gives them.

Worked example
--------------

Is a coin biased toward heads? Take two point hypotheses — :math:`H_0: p = 0.5` and
:math:`H_1: p = 0.7` — with equal priors, and observe 7 heads in 10 tosses. The
likelihoods are

.. math::

   P(\text{data} \mid H_0) = \binom{10}{7}(0.5)^{10} \approx 0.117, \qquad
   P(\text{data} \mid H_1) = \binom{10}{7}(0.7)^7 (0.3)^3 \approx 0.266,

so the posterior for the biased hypothesis is

.. math::

   P(H_1 \mid \text{data}) = \frac{0.266 \times 0.5}{0.266 \times 0.5 + 0.117 \times 0.5} \approx 0.69.

After 7 heads in 10, there's about a **69% probability** the coin is biased.

Parameter posteriors
--------------------

When the unknown is a continuous parameter rather than a discrete hypothesis, the
posterior is a whole **distribution**,
:math:`P(\theta \mid \text{data}) \propto P(\text{data} \mid \theta)\, P(\theta)` — the
object used to estimate means, regression coefficients or conversion rates. (A single
**posterior probability** is then one number read off that distribution.)

Where it shows up
-----------------

Bayesian sequential testing (stop once :math:`P(H_1 \mid \text{data}) > 0.95`), Bayesian
estimation, ML models (Naive Bayes, Bayesian nets), and medical diagnosis (probability of
disease given a test result).

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Posterior <063-posterior>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Frequentist <059-frequentist>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Posterior Probability <https://insightful-data-lab.com/2025/08/25/posterior-probability/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

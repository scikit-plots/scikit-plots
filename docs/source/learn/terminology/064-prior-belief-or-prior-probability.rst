:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-prior-belief-or-prior-probability:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Prior Belief (or Prior Probability)</b></div>`

=====================================
Prior Belief (or Prior Probability)
=====================================

*Belief about a parameter before observing the current data.*

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

A **prior belief** (or **prior probability**) is your **initial belief about a parameter
before seeing any new data**, encoded as a **prior distribution**. It captures what is
plausible from past studies, domain expertise or reasonable assumptions: *what you think
before you see the evidence*.

Where it sits
-------------

In Bayes' theorem the prior :math:`P(H)` is the term that the likelihood multiplies and
the data update:

.. math::

   P(H \mid D) = \frac{P(D \mid H)\, P(H)}{P(D)}.

Kinds of prior
--------------

- **Informative** — built on strong prior knowledge (a drug's effect is likely 5–10%,
  from past trials).
- **Non-informative / flat** — expresses near-ignorance (every value equally likely,
  e.g. :math:`\text{Beta}(1,1)`).
- **Weakly informative** — adds gentle, realistic bounds to rule out absurd values
  (conversion rates almost never exceed 50%), which stabilises inference without forcing
  a conclusion.

Example — coin toss
-------------------

For the probability of heads :math:`p`: a dogmatic "fair coin" prior fixes
:math:`p = 0.5`; total uncertainty is :math:`\text{Beta}(1,1)` (uniform); "probably fair
but unsure" is :math:`\text{Beta}(20, 20)`, peaked at 0.5. Each is updated by the data
into a posterior.

In A/B testing
--------------

A sensible prior on the conversion-rate difference — centred at 0 with small variance,
reflecting that new features rarely move conversion more than a few points — keeps early
results from over-reacting to noise; the experiment then updates it into a posterior on
the uplift.

The key caveat
--------------

The **choice of prior matters most with small datasets** and fades as data grows (the
likelihood takes over). A weakly informative prior is usually the safe default: enough
structure to regularise, not so much that it overrides the evidence.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Posterior <063-posterior>` · :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Binomial Likelihood <060-binomial-likelihood>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Prior Belief (or Prior Probability) <https://insightful-data-lab.com/2025/08/28/prior-belief-or-prior-probability/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

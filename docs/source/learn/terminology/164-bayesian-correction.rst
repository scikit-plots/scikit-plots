:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-bayesian-correction:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Bayesian Correction</b></div>`

=====================
Bayesian Correction
=====================

*Adjusting estimates using prior information within a Bayesian framework.*

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

**Bayesian correction** uses **Bayes' theorem** to **adjust raw probabilities, predictions or test
results** when the observed data is biased, noisy or incomplete. It is, in essence, "correcting"
outputs by **Bayesian updating** — folding in what we already know.

Where it appears
----------------

Several places. **Base-rate adjustment**: a model trained as if classes were balanced can be
corrected toward the **true prior** — a 90% rare-disease score shrinks sharply once low prevalence
is accounted for, via :math:`P(y=1 \mid x) \propto P(x \mid y=1)\, P(y=1)`. **Diagnostic tests**:
combining sensitivity and specificity with prevalence gives the true posterior,
:math:`P(\text{disease} \mid +) = \frac{P(+ \mid \text{disease})\, P(\text{disease})}{P(+)}`.
**Label-noise correction** estimates a noise transition matrix (e.g. for crowdsourced labels), and
**Bayesian calibration** updates scores much like Platt scaling but through Bayesian inference.

An example
----------

A spam classifier scores an email **0.8**, but the true spam base rate is only **10%**. After
Bayesian correction with that prior, the calibrated probability might fall to **0.4** — preventing
**overconfidence** when the prior is low.

Why it's useful
---------------

The method handles **class imbalance**, corrects for **measurement noise**, and yields
**better-calibrated probabilities** with **principled uncertainty** — exactly what risk-sensitive
decisions need. It is the Bayesian sibling of recalibration: where Platt scaling fits a curve,
Bayesian correction reasons from priors.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Recalibration <159-recalibration>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Posterior <063-posterior>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>` · :doc:`Recalibrate Thresholds <165-recalibrate-thresholds>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Bayesian Correction <https://insightful-data-lab.com/2025/08/23/bayesian-correction/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

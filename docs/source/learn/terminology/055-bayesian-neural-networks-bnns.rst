:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-bayesian-neural-networks-bnns:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Bayesian Neural Networks (BNNs)</b></div>`

=================================
Bayesian Neural Networks (BNNs)
=================================

*Neural networks with distributions over their weights, producing predictive uncertainty.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

From point estimates to distributions
-------------------------------------

A standard neural network fixes its weights :math:`W` after training — gradient descent
finds a single best **point estimate**. That captures no **uncertainty**: on limited or
noisy data the network can be confidently wrong. A **Bayesian neural network (BNN)**
fixes this by treating the weights as **random variables** with distributions.

The Bayesian view of weights
----------------------------

Instead of one weight vector, infer a **posterior** over weights:

.. math::

   p(W \mid D) \propto p(D \mid W)\, p(W),

with prior :math:`p(W)` (often a zero-mean Gaussian), likelihood :math:`p(D \mid W)`
(how well the weights explain the data) and posterior :math:`p(W \mid D)`. Training
therefore yields a **whole distribution** of plausible networks, not one.

Prediction integrates over weights
----------------------------------

Predictions average over every weight setting, weighted by posterior probability:

.. math::

   p(y \mid x, D) = \int p(y \mid x, W)\, p(W \mid D)\, dW.

This separates two kinds of uncertainty: **aleatoric** (noise in the labels) and
**epistemic** (not enough data to pin down the weights) — and the output is a
*distribution* over predictions, not a point.

Why it's hard, and how it's done
--------------------------------

That integral is **intractable**, so BNNs rely on approximation:

- **Variational inference** — approximate the posterior with a simpler :math:`q(W)`.
- **MCMC** — sample weights from the posterior (accurate but slow).
- **MC Dropout** (Gal & Ghahramani, 2016) — keeping dropout *on at test time*
  approximates sampling from a weight distribution, a cheap practical trick.

Why it matters
--------------

A BNN **knows when it doesn't know**: it resists overconfident errors and generalises
better, which is decisive in medicine, self-driving and finance where the *cost of an
uncertain prediction* matters. The price is heavier training and inference, sensitivity
to the approximation chosen, and less mainstream tooling than standard nets.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Variational Inference (VI) <056-variational-inference-vi>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Posterior <063-posterior>` · :doc:`Bayesian Inference. <375-bayesian-inference>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Bayesian Neural Networks (BNNs) <https://insightful-data-lab.com/2025/08/29/bayesian-neural-networks-bnns/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-bayesian-decision-theory-bdt:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Bayesian Decision Theory (BDT)</b></div>`

================================
Bayesian Decision Theory (BDT)
================================

*Choosing the action that minimises expected loss under the posterior distribution.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Core idea
---------

Bayesian inference hands you a **posterior** :math:`p(\theta \mid D)` over unknown
parameters :math:`\theta` given data :math:`D` — but in practice you don't just want
probabilities, you need to **act**: classify the email, approve the loan, treat the
patient. **Bayesian Decision Theory (BDT)** turns the posterior into an *optimal
decision* under uncertainty.

The three ingredients
---------------------

- **Actions** :math:`a` — the choices available (label spam vs not-spam).
- **States of nature** :math:`\theta` — the unknown truth (the email really is or isn't
  spam).
- **Loss function** :math:`L(a, \theta)` — the cost of taking action :math:`a` when the
  truth is :math:`\theta` (flagging real mail as spam may cost far more than missing a
  spam).

Bayes risk and the optimal rule
-------------------------------

The **Bayes risk** of an action is its posterior expected loss,

.. math::

   R(a \mid D) = \mathbb{E}_{\theta \sim p(\theta \mid D)}\big[L(a, \theta)\big],

and the **Bayes action** is the one that minimises it:

.. math::

   a^*(D) = \arg\min_a R(a \mid D).

By construction this is the choice that does best on average under everything the
posterior knows.

The classification special case
-------------------------------

Let :math:`\theta \in \{C_1, \dots, C_k\}` with posterior class probabilities
:math:`p(C_i \mid x)`. Under **0–1 loss** (0 if correct, 1 if wrong), the Bayes-optimal
classifier reduces to

.. math::

   a^*(x) = \arg\max_i \; p(C_i \mid x),

which is exactly the **MAP (maximum a posteriori)** classifier — pick the most probable
class.

Asymmetric loss shifts the boundary
-----------------------------------

When errors cost differently, BDT moves the decision threshold rather than the 0.5
default. In a medical test where a missed disease (false negative) is worse than a
false alarm, the optimal rule classifies as positive at a **lower** posterior
probability — trading more false alarms for fewer missed cases.

Where it shows up
-----------------

BDT is the formal backbone of decision-making under uncertainty across **ML**
(classification, regression, model selection), **medicine** (treat vs not), **finance**
(portfolio choice under risk) and **engineering / reinforcement learning** — including
bandit strategies like Thompson Sampling, which is BDT applied sequentially with the
posterior updated as rewards arrive.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB)) <050-thompson-sampling-ts-in-bandits-multi-armed-band>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Posterior <063-posterior>` · :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Bandit Algorithms <113-bandit-algorithms>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Bayesian Decision Theory (BDT) <https://insightful-data-lab.com/2025/08/29/bayesian-decision-theory-bdt/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

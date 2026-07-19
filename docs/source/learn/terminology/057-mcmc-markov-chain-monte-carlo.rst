:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-mcmc-markov-chain-monte-carlo:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>MCMC (Markov Chain Monte Carlo)</b></div>`

=================================
MCMC (Markov Chain Monte Carlo)
=================================

*Sampling from a posterior via a Markov chain whose stationary distribution is that posterior.*

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

**MCMC (Markov Chain Monte Carlo)** is a family of algorithms for **drawing samples
from complex probability distributions** — above all the **posterior** in Bayesian
inference, which usually has no closed form. Rather than computing the distribution
analytically, MCMC generates a long **sequence of samples** that approximate it: it
lets you approximate posteriors by simulation when the math is intractable.

The two ideas in the name
-------------------------

- **Markov chain** — a sequence of states where the next depends only on the current
  one (not the full history); this structure is what lets the sampler *walk* through
  the distribution.
- **Monte Carlo** — approximating integrals and expectations by repeated random
  sampling (the classic toy is estimating :math:`\pi` from random points in a square).

Put together: build a Markov chain whose **stationary distribution is the posterior**,
run it long enough, and its samples approximate that posterior.

The estimator
-------------

Most Bayesian quantities are posterior expectations,

.. math::

   \mathbb{E}[f(\theta) \mid \text{data}] = \int f(\theta)\, p(\theta \mid \text{data})\, d\theta,

intractable in high dimensions — so MCMC approximates it by the sample average over the
chain,

.. math::

   \mathbb{E}[f(\theta) \mid \text{data}] \approx \frac{1}{N} \sum_{i=1}^N f(\theta_i),
   \qquad \theta_i \sim \text{MCMC}.

The algorithms
--------------

- **Metropolis–Hastings** — propose a move, then accept or reject it by a ratio of
  posterior densities.
- **Gibbs sampling** — a special case that updates each parameter from its full
  conditional; natural for hierarchical models.
- **Hamiltonian Monte Carlo (HMC)** — uses gradients to propose distant,
  high-acceptance moves; the engine of **Stan**.
- **NUTS (No-U-Turn Sampler)** — an adaptive HMC that tunes step size and trajectory
  length automatically; the default in **PyMC** and **Stan**.

Example
-------

Coin toss with 7 heads, 3 tails and a :math:`\text{Beta}(2,2)` prior gives a
:math:`\text{Beta}(9,5)` posterior. Here it's closed-form, but if it weren't, MCMC
would draw thousands of samples of :math:`p` to estimate the posterior mean, variance
and credible interval.

Using it well
-------------

MCMC handles very complex, high-dimensional posteriors and yields **full samples** (any
posterior summary is then just a function of them). The costs: it is **computationally
expensive**, needs **tuning** (burn-in, thinning, proposal scale), and requires
**convergence diagnostics** — discard a **burn-in**, then check the
potential-scale-reduction :math:`\hat{R}` (near 1.0) and the effective sample size to
confirm the chains have mixed. Versus **variational inference**, MCMC is asymptotically
exact but slower; VI is faster but approximate.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Variational Inference (VI) <056-variational-inference-vi>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Posterior <063-posterior>` · :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `MCMC (Markov Chain Monte Carlo) <https://insightful-data-lab.com/2025/08/28/mcmc-markov-chain-monte-carlo/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

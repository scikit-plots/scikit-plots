:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-marginal-likelihood-also-called-the-model-evidence-or-integrated-likelihood:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Marginal Likelihood (also called The Model Evidence or Integrated Likelihood)</b></div>`

===============================================================================
Marginal Likelihood (also called The Model Evidence or Integrated Likelihood)
===============================================================================

*The probability of the data averaged over the prior — the normaliser and a model-comparison score.*

What it is
----------

The **marginal likelihood** — also called the **model evidence** or **integrated
likelihood** — is the probability of the observed data under a model, **averaged over
all possible parameter values**. It is the **denominator** in Bayes' theorem:

.. math::

   P(\theta \mid D) = \frac{P(D \mid \theta)\, P(\theta)}{P(D)},
   \qquad
   P(D) = \int P(D \mid \theta)\, P(\theta)\, d\theta.

In words, :math:`P(D)` is the **prior-weighted average** of the likelihood — the
overall probability of the data, considering every parameter value the prior allows.

Two jobs
--------

**1. Normalisation.** Dividing by :math:`P(D)` is what makes the posterior integrate to
1 — without it, prior × likelihood is only *proportional* to the posterior.

**2. Model comparison.** Because it scores how well an *entire model* (not one parameter
setting) predicts the data, the marginal likelihood is the basis of the **Bayes
factor**,

.. math::

   \text{BF} = \frac{P(D \mid M_1)}{P(D \mid M_2)},

which weighs two competing models — a higher marginal likelihood means the model
explains the data better, with a built-in Occam penalty for needless complexity.

Example — coin toss
-------------------

With a uniform prior :math:`\text{Beta}(1,1)` and 7 heads in 10 tosses,

.. math::

   P(D) = \int_0^1 \binom{10}{7} p^7 (1-p)^3 \, dp,

the overall probability of "7 heads out of 10" averaged across all plausible :math:`p`.

Why it's hard
-------------

That integral is usually **intractable** in real models, which is the whole reason
approximate methods exist: the **Laplace approximation**, **variational inference**
(whose ELBO is a *lower bound* on :math:`\log P(D)`), and **MCMC**-based estimators all
exist to approximate the evidence for posterior computation and model selection.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Posterior <063-posterior>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>` · :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Variational Inference (VI) <056-variational-inference-vi>` · :doc:`Binomial Likelihood <060-binomial-likelihood>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <https://insightful-data-lab.com/2025/08/28/marginal-likelihood-also-called-the-model-evidence-or-integrated-likelihood/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

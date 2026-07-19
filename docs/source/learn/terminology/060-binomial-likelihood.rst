:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-binomial-likelihood:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Binomial Likelihood</b></div>`

=====================
Binomial Likelihood
=====================

*The probability model for the number of successes in a fixed number of independent Bernoulli trials.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The setup
---------

Model :math:`n` independent Bernoulli (success/failure) trials with success probability
:math:`p`, observing :math:`k` successes. The **binomial PMF** gives the probability of
exactly :math:`k` successes:

.. math::

   P(X = k \mid p) = \binom{n}{k} p^k (1 - p)^{n - k}.

Likelihood vs PMF
-----------------

The **likelihood** is the *same expression read the other way*: with the data
:math:`(k, n)` fixed, it is a function of the unknown :math:`p`,

.. math::

   L(p \mid k, n) = \binom{n}{k} p^k (1 - p)^{n - k} \;\propto\; p^k (1 - p)^{n - k},

dropping the binomial coefficient because it doesn't depend on :math:`p`. The PMF asks
"given :math:`p`, how probable is this data?"; the likelihood asks "given this data, how
well does each :math:`p` explain it?".

Maximum likelihood
------------------

Maximising the log-likelihood,

.. math::

   \ell(p) = k \ln p + (n - k)\ln(1 - p),

and setting the derivative to zero gives the intuitive estimate

.. math::

   \hat{p}_{\text{MLE}} = \frac{k}{n},

the observed success proportion. For :math:`n = 10,\ k = 7`, the likelihood
:math:`\propto p^7 (1-p)^3` peaks at :math:`\hat{p} = 0.7`.

The Bayesian half
-----------------

Pair the binomial likelihood with a **Beta prior** and conjugacy makes the posterior
Beta as well:

.. math::

   \text{Beta}(\alpha, \beta) \times \text{Binomial}(k, n) \;\Rightarrow\;
   \text{Beta}(\alpha + k,\; \beta + n - k).

So the binomial likelihood sits at the centre of **both** worlds — its MLE is the
frequentist sample proportion, and with a Beta prior it gives the classic Beta–Binomial
Bayesian model.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Frequentist <059-frequentist>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Binomial Likelihood <https://insightful-data-lab.com/2025/08/28/binomial-likelihood/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

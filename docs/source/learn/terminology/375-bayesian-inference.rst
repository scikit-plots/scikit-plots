:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-bayesian-inference:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Bayesian Inference.</b></div>`

=====================
Bayesian Inference.
=====================

*Updating beliefs about parameters using priors and observed data.*

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

**Bayesian inference** updates **beliefs** in light of evidence using **Bayes' theorem** — it combines a
**prior** (what you believed before) with the **likelihood** (how probable the data are under each hypothesis)
to produce a **posterior** (what you believe after):

.. math::

   \text{posterior} \propto \text{prior} \times \text{likelihood}.

Parameters are treated as **random variables** with distributions, not fixed points.

What makes it distinctive
-------------------------

Because it yields a **full posterior distribution**, Bayesian inference quantifies **uncertainty** directly — a
**credible interval** says there's a 95% probability the parameter lies inside it — and it naturally
**incorporates prior knowledge** and **updates sequentially** as data arrive. This contrasts with the
**frequentist** view of fixed parameters and p-values.

The catch and the tools
-----------------------

Posteriors are usually **intractable**, so they're approximated with **Markov Chain Monte Carlo** or
variational methods via tools like **Stan**, **PyMC**, or NumPyro. Bayesian inference underlies **Bayesian A/B
testing**, **Bayesian optimization**, and the **causal** tree models above.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Causal Trees <301-causal-trees>` · :doc:`Likelihood <304-likelihood>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Statistical Tests <328-statistical-tests>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Uplift Random Forests <302-uplift-random-forests>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Bayesian Inference. <https://insightful-data-lab.com/2025/08/19/bayesian-inference/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

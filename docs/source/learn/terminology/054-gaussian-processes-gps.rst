:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-gaussian-processes-gps:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Gaussian Processes (GPs)</b></div>`

==========================
Gaussian Processes (GPs)
==========================

*A nonparametric Bayesian prior over functions that yields predictions with calibrated uncertainty.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The big idea
------------

A **Gaussian process (GP)** is a **distribution over functions**. Just as a Gaussian
over numbers is fixed by a mean and variance, a GP over functions is fixed by a **mean
function** and a **covariance function (kernel)**. Rather than positing parameters
(like neural-network weights) and fitting them, a GP says directly: *here is the family
of functions I believe in, with uncertainty around it.*

Formal definition
-----------------

.. math::

   f(x) \sim \mathcal{GP}\big(m(x),\, k(x, x')\big),

with mean :math:`m(x) = \mathbb{E}[f(x)]` and kernel
:math:`k(x, x') = \operatorname{Cov}(f(x), f(x'))`. The defining property: **any**
finite set of inputs has a joint multivariate Gaussian,

.. math::

   (f(x_1), \dots, f(x_n)) \sim \mathcal{N}(\mathbf{m}, K),

where :math:`K_{ij} = k(x_i, x_j)` is built from the kernel.

Kernels encode your assumptions
-------------------------------

The kernel decides how correlated nearby inputs are:

- **RBF / squared-exponential** —
  :math:`k(x, x') = \exp\!\big(-\lVert x - x' \rVert^2 / 2\ell^2\big)` — smooth
  functions, with the length-scale :math:`\ell` setting how fast correlation decays.
- **Linear** — straight-line trends.
- **Periodic** — repeating patterns.

Kernels **compose** (sum or product) to build structure like trend + seasonality.

Posterior inference
-------------------

Given training data :math:`(X, y)` and test inputs :math:`X_*`, the GP prior makes
:math:`y` and the test values :math:`f_*` jointly Gaussian; **conditioning** on
:math:`y` yields a Gaussian posterior over :math:`f_*` with both a **mean prediction**
and a **variance**. So a GP returns a smooth curve *and* a calibrated confidence band —
uncertainty for free.

Cost and trade-offs
-------------------

The catch is computation: conditioning inverts an :math:`n \times n` covariance matrix
at :math:`O(n^3)` cost, so exact GPs suit **small-to-medium** data and need sparse /
inducing-point approximations to scale. Kernel choice is critical — the wrong kernel
gives poor predictions. In scikit-learn, ``GaussianProcessRegressor`` implements this.

GPs and neural networks
-----------------------

A GP places uncertainty over **functions**; a **Bayesian neural network** places it
over **weights**. The two meet at a famous limit: an **infinitely wide** neural network
with random weights *converges to a Gaussian process*.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Variational Inference (VI) <056-variational-inference-vi>` · :doc:`Posterior <063-posterior>` · :doc:`Bayesian Inference. <375-bayesian-inference>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Gaussian Processes (GPs) <https://insightful-data-lab.com/2025/08/29/gaussian-processes-gps/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-posterior-probability-of-uplift:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Posterior probability of uplift</b></div>`

=================================
Posterior probability of uplift
=================================

*The posterior probability that a treatment's effect exceeds zero (or a chosen threshold).*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The question it answers
-----------------------

In an experiment with a binary outcome (say conversion), the **uplift** is the
difference in success rates between treatment and control,

.. math::

   u = p_T - p_C,

and the decision-relevant quantity is not a p-value but the **posterior probability
that the uplift is positive** (or beats a business threshold :math:`\tau`):

.. math::

   \Pr(u > 0 \mid \text{data}) \quad\text{or}\quad \Pr(u > \tau \mid \text{data}).

Beta–Binomial model (binary outcomes)
-------------------------------------

With :math:`s_T` successes in :math:`n_T` trials (treatment) and :math:`s_C` in
:math:`n_C` (control), conjugate Beta priors give Beta posteriors:

.. math::

   p_T \mid \text{data} \sim \text{Beta}(\alpha_T + s_T,\; \beta_T + n_T - s_T),
   \qquad
   p_C \mid \text{data} \sim \text{Beta}(\alpha_C + s_C,\; \beta_C + n_C - s_C).

The difference :math:`u = p_T - p_C` has no closed-form CDF, so estimate it by **Monte
Carlo**: draw :math:`p_T^{(m)}` and :math:`p_C^{(m)}` from their posteriors, form
:math:`u^{(m)} = p_T^{(m)} - p_C^{(m)}`, and approximate

.. math::

   \Pr(u > \tau \mid \text{data}) \approx \frac{1}{M}\sum_{m=1}^M \mathbf{1}\{u^{(m)} > \tau\}.

The empirical quantiles of :math:`\{u^{(m)}\}` give a **credible interval** for the
uplift.

.. code-block:: python

   import numpy as np
   pT = np.random.beta(1 + sT, 1 + nT - sT, size=200_000)
   pC = np.random.beta(1 + sC, 1 + nC - sC, size=200_000)
   u  = pT - pC
   prob_uplift = np.mean(u > tau)          # Pr(u > tau | data)
   ci = np.quantile(u, [0.025, 0.975])     # 95% credible interval

Continuous outcomes (revenue)
-----------------------------

For approximately Normal outcomes with unknown means, a Normal–Inverse-Gamma posterior
makes the mean difference :math:`d = \mu_T - \mu_C` (approximately) Normal, so the
uplift probability is closed-form:

.. math::

   \Pr(d > \tau \mid \text{data}) = 1 - \Phi\!\left(\frac{\tau - \hat{d}}{\text{SE}_{\text{post}}}\right).

Bayesian logistic / hierarchical regression
-------------------------------------------

With covariates and a treatment indicator,

.. math::

   \Pr(Y=1 \mid x, T) = \text{logit}^{-1}\!\big(\beta_0 + \beta^\top x + \gamma T + \delta^\top (x \cdot T)\big),

per-draw segment uplift is computed from posterior coefficient samples, and
**hierarchical priors** borrow strength across small segments to reduce false
positives.

What to report, and how to decide
---------------------------------

Decision-friendly outputs: the posterior probability :math:`\Pr(u > \tau)`, the
**expected uplift** :math:`\mathbb{E}[u \mid \text{data}]`, a 95% credible interval,
and a **risk-aware rule** — "ship if :math:`\Pr(u > \tau) \ge q`" (e.g. :math:`q =
0.9`). Practical guidance: use weakly informative priors (Beta(1,1) or Beta(0.5,0.5))
for small samples, prefer :math:`\Pr(u > \tau)` over point estimates when a bad launch
is costly, model many arms hierarchically, and — unlike fixed-horizon tests — Bayesian
posteriors support **always-valid** monitoring without naive peeking penalties.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Posterior <063-posterior>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Incremental Conversions <394-incremental-conversions>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Posterior probability of uplift <https://insightful-data-lab.com/2025/08/29/posterior-probability-of-uplift/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

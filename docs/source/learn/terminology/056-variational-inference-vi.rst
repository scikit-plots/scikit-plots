:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-variational-inference-vi:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔁&nbsp;&nbsp;<b>Variational Inference (VI)</b></div>`

============================
Variational Inference (VI)
============================

*Approximating an intractable posterior by optimising a simpler distribution to be close to it.*

The problem it solves
---------------------

Bayesian inference wants the **posterior**

.. math::

   p(\theta \mid x) = \frac{p(x \mid \theta)\, p(\theta)}{p(x)},

but the **evidence** in the denominator,
:math:`p(x) = \int p(x \mid \theta)\, p(\theta)\, d\theta`, is an integral over all
parameters that is **intractable** in most real models. MCMC tackles this by sampling;
**variational inference (VI)** tackles it by **optimisation**.

The core idea
-------------

Replace the hard posterior with the closest member of a **simpler family**
:math:`q(\theta)` (say, Gaussians):

.. math::

   q^*(\theta) = \arg\min_q \; \operatorname{KL}\!\big(q(\theta) \,\Vert\, p(\theta \mid x)\big),

measuring closeness by **Kullback–Leibler divergence**. Inference becomes a search for
the best-fitting approximation.

The ELBO
--------

We can't minimise that KL directly (it contains the unknown posterior), so VI instead
**maximises the Evidence Lower Bound**:

.. math::

   \mathcal{L}(q) = \mathbb{E}_{q(\theta)}\big[\log p(x, \theta) - \log q(\theta)\big].

Maximising the ELBO is *equivalent* to minimising the KL, and the ELBO is a genuine
**lower bound** on :math:`\log p(x)` — so gradient-based optimisation of
:math:`\mathcal{L}` drives :math:`q` toward the posterior.

How it's done in practice
-------------------------

Pick a **variational family** :math:`q(\theta; \phi)` (e.g. a Gaussian's mean and
variance), then optimise :math:`\phi` to maximise the ELBO and use :math:`q` as the
posterior. Common machinery:

- **Mean-field** — assume independence, :math:`q(\theta) = \prod_i q_i(\theta_i)`.
- **CAVI** — coordinate ascent, updating each factor in turn.
- **Stochastic VI** — mini-batch stochastic optimisation for large data.
- **Reparameterisation trick** — write :math:`\theta = g(\phi, \epsilon)` with noise
  :math:`\epsilon` so gradients flow through samples (the engine of **variational
  autoencoders**).

Where it shows up
-----------------

Topic models (latent Dirichlet allocation), variational autoencoders, Bayesian neural
networks and probabilistic graphical models. The trade-off vs **MCMC**: VI is **faster
and scalable** but gives a *biased* approximation (only as good as the family), whereas
MCMC is asymptotically exact but slower.

----

*Theme:* :ref:`Bayesian Inference <term-theme-bayes>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Posterior <063-posterior>` · :doc:`Bayesian Inference. <375-bayesian-inference>`

----

.. hint::
   **More in Bayesian Inference**

   :doc:`Bayes' Theorem <066-bayes-theorem>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Posterior <063-posterior>` · :doc:`Posterior belief <061-posterior-belief>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Variational Inference (VI) <https://insightful-data-lab.com/2025/08/29/variational-inference-vi/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

.. _bda-normal-distribution-with-known-variance:

========================================================================
Normal Distribution with Known Variance
========================================================================

**Part 1 · Stage 2 · 📍 Single-Parameter Models & Priors**  ·  Lesson 015 of 144  ·  *beginner*

:doc:`◀ Previous · Informative Prior Distributions <014-informative-prior-distributions>`   ·   :doc:`Next · Other Standard Single-Parameter Models ▶ <016-other-standard-single-parameter-models>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The second workhorse
----------------------

After the Beta–Binomial, the **normal model with known variance** is the most useful closed-form case,
and the template for nearly every hierarchical and regression result later in the course. Take
:math:`n` observations :math:`y_i \sim \mathrm{N}(\theta, \sigma^2)` with :math:`\sigma^2` **known**,
and a conjugate prior :math:`\theta \sim \mathrm{N}(\mu_0, \tau_0^2)`.

Precision adds
----------------

The algebra is cleanest in terms of **precision** — the reciprocal of variance, :math:`1/\sigma^2` —
because precisions simply **add**. The posterior is normal,

.. math::

   \theta \mid y \;\sim\; \mathrm{N}(\mu_n, \tau_n^2), \qquad
   \frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2},

and its mean is the **precision-weighted average** of the prior mean and the sample mean:

.. math::

   \mu_n = \frac{\dfrac{1}{\tau_0^2}\,\mu_0 + \dfrac{n}{\sigma^2}\,\bar{y}}
                {\dfrac{1}{\tau_0^2} + \dfrac{n}{\sigma^2}} .

This is the compromise property again, now with **precision** as the weight. Data contribute precision
:math:`n / \sigma^2`; the prior contributes :math:`1/\tau_0^2`, equivalent to
:math:`\sigma^2 / \tau_0^2` observations. Each new observation **increases** posterior precision, so
the posterior variance can only shrink.

In code
---------

.. code-block:: python

   import numpy as np
   from scipy import stats

   mu0, tau0, sigma = 0.0, 2.0, 1.0        # prior N(0, 2^2); known sd = 1
   y = np.array([1.2, 0.8, 1.5, 0.9])
   n, ybar = len(y), y.mean()

   prec = 1 / tau0**2 + n / sigma**2
   mun = (mu0 / tau0**2 + n * ybar / sigma**2) / prec
   post = stats.norm(mun, np.sqrt(1 / prec))
   post.mean(), post.interval(0.95)         # ≈ 1.04, (0.61, 1.47)

Where it leads
----------------

Two consequences echo through the rest of the course. As :math:`n \to \infty` the prior's precision
becomes negligible and :math:`\mu_n \to \bar{y}` — data win, as always. And a **flat** prior
(:math:`\tau_0 \to \infty`) gives exactly :math:`\theta \mid y \sim \mathrm{N}(\bar{y}, \sigma^2/n)`,
the familiar sampling-theory result reappearing as a Bayesian posterior. That coincidence, and its
limits, is the subject of Stage 4.

.. hint::

   **Related lessons:** :doc:`Estimating a Probability from Binomial Data <011-estimating-a-probability-from-binomial-data>`  ·  :doc:`Normal Data with a Conjugate Prior Distribution <022-normal-data-with-a-conjugate-prior-distribution>`  ·  :doc:`Other Standard Single-Parameter Models <016-other-standard-single-parameter-models>`  ·  :doc:`Normal model with exchangeable parameters <036-normal-model-with-exchangeable-parameters>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/normal-distribution-with-known-variance/ <https://insightful-data-lab.com/2025/11/09/normal-distribution-with-known-variance/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

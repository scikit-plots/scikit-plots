.. _bda-normal-data-with-a-conjugate-prior-distribution:

========================================================================
Normal Data with a Conjugate Prior Distribution
========================================================================

**Part 1 · Stage 3 · 🧮 Multiparameter Models**  ·  Lesson 022 of 144  ·  *beginner*

:doc:`◀ Previous · Normal Data with a Noninformative Prior Distribution <021-normal-data-with-a-noninformative-prior-distribution>`   ·   :doc:`Next · Multinomial Model for Categorical Data ▶ <023-multinomial-model-for-categorical-data>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Prior information on both
---------------------------

To bring real prior knowledge to the normal model, the conjugate choice is the
**normal--inverse-**:math:`\chi^2` family, specified hierarchically — the prior for the mean is scaled
by the very variance it accompanies:

.. math::

   \mu \mid \sigma^2 \sim \mathrm{N}\!\left(\mu_0,\, \frac{\sigma^2}{\kappa_0}\right),
   \qquad
   \sigma^2 \sim \text{Inv-}\chi^2(\nu_0,\, \sigma_0^2).

Four hyperparameters carry four meanings: :math:`\mu_0` a prior mean, :math:`\kappa_0` how many
observations that mean is worth; :math:`\sigma_0^2` a prior variance, :math:`\nu_0` how many
observations *it* is worth.

The update adds counts
------------------------

Conjugacy delivers a posterior in the same family, with hyperparameters that update by **addition**:

.. math::

   \kappa_n = \kappa_0 + n, \qquad \nu_n = \nu_0 + n, \qquad
   \mu_n = \frac{\kappa_0\, \mu_0 + n\, \bar{y}}{\kappa_0 + n},

so :math:`\mu_n` is the familiar weighted average, with :math:`\kappa_0` acting as an **equivalent
sample size** for the prior mean. The posterior scale :math:`\sigma_n^2` combines three ingredients:
the prior sum of squares, the sample sum of squares, and a term
:math:`\frac{\kappa_0 n}{\kappa_0 + n}(\bar{y} - \mu_0)^2` that **inflates the variance when the data
and the prior disagree** — a built-in conflict detector.

Marginals, and the limit
--------------------------

Marginalising :math:`\sigma^2` again yields a :math:`t`, now centred between prior and data:

.. math::

   \mu \mid y \;\sim\; t_{\,\nu_n}\!\left(\mu_n,\; \frac{\sigma_n^2}{\kappa_n}\right).

Setting :math:`\kappa_0 \to 0` and :math:`\nu_0 \to -1` recovers the noninformative
:math:`p(\mu, \sigma^2) \propto 1/\sigma^2` of the previous lesson: the flat prior is the **limiting
case** of the conjugate one, carrying zero pseudo-observations.

.. code-block:: python

   mu0, k0, nu0, s0sq = 5.0, 2, 2, 0.25       # prior: mean worth 2 obs, var worth 2 obs
   n, ybar, s2 = len(y), y.mean(), y.var(ddof=1)
   kn, nun = k0 + n, nu0 + n
   mun = (k0 * mu0 + n * ybar) / kn
   snsq = (nu0 * s0sq + (n - 1) * s2 + k0 * n / kn * (ybar - mu0) ** 2) / nun

Convenient, but coupled
-------------------------

The tidy algebra has a cost worth naming: this prior **couples** :math:`\mu` and :math:`\sigma^2` — a
belief about the mean is expressed in units of the unknown variance. If that dependence is not
something you actually believe, the conjugate family is buying convenience at the price of realism, and
a non-conjugate prior fitted by MCMC is the more honest route.

.. hint::

   **Related lessons:** :doc:`Normal Data with a Noninformative Prior Distribution <021-normal-data-with-a-noninformative-prior-distribution>`  ·  :doc:`Informative Prior Distributions <014-informative-prior-distributions>`  ·  :doc:`Normal Distribution with Known Variance <015-normal-distribution-with-known-variance>`  ·  :doc:`Multivariate Normal with Unknown Mean and Variance <025-multivariate-normal-with-unknown-mean-and-variance>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/normal-data-with-a-conjugate-prior-distribution/ <https://insightful-data-lab.com/2025/11/09/normal-data-with-a-conjugate-prior-distribution/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

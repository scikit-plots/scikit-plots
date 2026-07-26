.. _bda-normal-data-with-a-noninformative-prior-distribution:

========================================================================
Normal Data with a Noninformative Prior Distribution
========================================================================

**Part 1 · Stage 3 · 🧮 Multiparameter Models**  ·  Lesson 021 of 144  ·  *beginner*

:doc:`◀ Previous · Averaging Over Nuisance Parameters <020-averaging-over-nuisance-parameters>`   ·   :doc:`Next · Normal Data with a Conjugate Prior Distribution ▶ <022-normal-data-with-a-conjugate-prior-distribution>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Two unknowns at last
----------------------

The normal model with **both** :math:`\mu` and :math:`\sigma^2` unknown is the first genuinely
multiparameter problem, and it delivers a classical result as a by-product. Take
:math:`y_i \sim \mathrm{N}(\mu, \sigma^2)` with the standard noninformative prior

.. math::

   p(\mu, \sigma^2) \;\propto\; \frac{1}{\sigma^2},

which is flat in :math:`\mu` and in :math:`\log \sigma` — the location/scale answer from the Jeffreys
lesson.

Factor the joint posterior
----------------------------

The joint posterior factors conveniently as :math:`p(\mu, \sigma^2 \mid y) = p(\mu \mid \sigma^2, y)
\, p(\sigma^2 \mid y)`. The conditional is the known-variance result from Stage 2, and the marginal for
the variance is a scaled inverse-:math:`\chi^2`:

.. math::

   \mu \mid \sigma^2, y \sim \mathrm{N}\!\left(\bar{y},\, \frac{\sigma^2}{n}\right),
   \qquad
   \sigma^2 \mid y \sim \text{Inv-}\chi^2\!\left(n - 1,\, s^2\right),

with :math:`s^2 = \frac{1}{n-1}\sum_i (y_i - \bar{y})^2`. Draw :math:`\sigma^2` first, then
:math:`\mu` given it, and you have exact joint draws — no MCMC needed.

The t emerges
---------------

Now **average over the nuisance** :math:`\sigma^2`, exactly as the last lesson prescribed. The mixture
of normals, weighted by the inverse-:math:`\chi^2`, is a **Student :math:`t`**:

.. math::

   \mu \mid y \;\sim\; t_{\,n-1}\!\left(\bar{y},\; \frac{s^2}{n}\right).

The heavy tails are not an assumption — they are the **price of not knowing** :math:`\sigma^2`, the
second term of the variance decomposition made visible. And note the coincidence: the resulting 95%
interval is numerically the classical :math:`t` confidence interval, though its interpretation as a
**probability statement about** :math:`\mu` is available only here.

.. code-block:: python

   import numpy as np
   from scipy import stats
   y = np.array([5.1, 4.8, 5.6, 5.0, 4.7]); n = len(y)
   ybar, s2 = y.mean(), y.var(ddof=1)
   # exact joint draws: sigma^2 first, then mu | sigma^2
   sigma2 = (n - 1) * s2 / stats.chi2(n - 1).rvs(10_000)
   mu = stats.norm(ybar, np.sqrt(sigma2 / n)).rvs()
   np.percentile(mu, [2.5, 97.5])          # matches the t_{n-1} interval

One caveat: this prior is **improper**, and with :math:`n = 1` the posterior for :math:`\mu` fails to
integrate — a reminder to check propriety rather than trust the algebra.

.. hint::

   **Related lessons:** :doc:`Averaging Over Nuisance Parameters <020-averaging-over-nuisance-parameters>`  ·  :doc:`Normal Data with a Conjugate Prior Distribution <022-normal-data-with-a-conjugate-prior-distribution>`  ·  :doc:`Noninformative Prior Distributions <018-noninformative-prior-distributions>`  ·  :doc:`Normal Distribution with Known Variance <015-normal-distribution-with-known-variance>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/normal-data-with-a-noninformative-prior-distribution/ <https://insightful-data-lab.com/2025/11/09/normal-data-with-a-noninformative-prior-distribution/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

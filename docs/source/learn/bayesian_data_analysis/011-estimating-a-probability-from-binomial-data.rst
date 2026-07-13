.. _bda-estimating-a-probability-from-binomial-data:

========================================================================
Estimating a Probability from Binomial Data
========================================================================

**Part 1 · Stage 2 · 📍 Single-Parameter Models & Priors**  ·  Lesson 011 of 144  ·  *beginner*

:doc:`◀ Previous · Bayesian Inference in Applied Statistics <010-bayesian-inference-in-applied-statistics>`   ·   :doc:`Next · Posterior as a Compromise Between Data and Prior Information ▶ <012-posterior-as-a-compromise-between-data-and-prior-information>`   ·   :doc:`↑ Section <index>`


The workhorse model
---------------------

The simplest interesting Bayesian problem: estimate an unknown probability :math:`\theta` — a
conversion rate, a survival rate, the chance of a female birth — from :math:`y` successes in :math:`n`
independent trials. The likelihood is **binomial**,

.. math::

   p(y \mid \theta) = \binom{n}{y} \theta^{y} (1 - \theta)^{n - y}
   \;\;\propto\;\; \theta^{y} (1 - \theta)^{n - y},

and the whole of single-parameter Bayesian inference can be seen in miniature here.

A Beta prior
--------------

For a parameter confined to :math:`[0, 1]`, the natural prior is a **Beta** distribution,
:math:`\theta \sim \mathrm{Beta}(\alpha, \beta)`, whose density is proportional to
:math:`\theta^{\alpha - 1} (1 - \theta)^{\beta - 1}`. Note the shape: it is *the same functional form*
as the likelihood. That is not a coincidence but the definition of **conjugacy**, and it makes the
update exact.

The update is addition
------------------------

Multiply prior by likelihood and read off the kernel:

.. math::

   p(\theta \mid y) \;\propto\; \theta^{y}(1-\theta)^{n-y} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}
   = \theta^{\alpha + y - 1} (1 - \theta)^{\beta + n - y - 1},

so

.. math::

   \theta \mid y \;\sim\; \mathrm{Beta}(\alpha + y,\; \beta + n - y).

Bayesian updating here is nothing more than **counting**: add successes to :math:`\alpha`, failures to
:math:`\beta`. This licenses reading :math:`\alpha` and :math:`\beta` as **prior successes and
failures**, with :math:`\alpha + \beta` a **prior sample size**.

In code
---------

With :math:`\mathrm{Beta}(1,1)` (uniform) and 8 successes in 10 trials:

.. code-block:: python

   from scipy import stats
   post = stats.beta(1 + 8, 1 + 10 - 8)          # Beta(9, 3)
   post.mean()                                    # 0.75
   post.interval(0.95)                            # 95% credible interval
   1 - post.cdf(0.5)                              # P(theta > 0.5 | y) ≈ 0.981

The MLE is :math:`8/10 = 0.80`; the posterior mean is :math:`0.75`, pulled toward the prior mean of
:math:`0.5`. That pull — its size, and its fate as :math:`n` grows — is the subject of the next lesson.

.. hint::

   **Related lessons:** :doc:`Posterior as a Compromise Between Data and Prior Information <012-posterior-as-a-compromise-between-data-and-prior-information>`  ·  :doc:`Summarizing Posterior Inference <013-summarizing-posterior-inference>`  ·  :doc:`Informative Prior Distributions <014-informative-prior-distributions>`  ·  :doc:`Discrete Bayesian Examples – Genetics and Spell Checking (with θ) <004-discrete-bayesian-examples-genetics-and-spell-checking-with>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/estimating-a-probability-from-binomial-data/ <https://insightful-data-lab.com/2025/11/09/estimating-a-probability-from-binomial-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

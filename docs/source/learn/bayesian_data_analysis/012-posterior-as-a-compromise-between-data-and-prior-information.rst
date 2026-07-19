.. _bda-posterior-as-a-compromise-between-data-and-prior-information:

========================================================================
Posterior as a Compromise Between Data and Prior Information
========================================================================

**Part 1 · Stage 2 · 📍 Single-Parameter Models & Priors**  ·  Lesson 012 of 144  ·  *beginner*

:doc:`◀ Previous · Estimating a Probability from Binomial Data <011-estimating-a-probability-from-binomial-data>`   ·   :doc:`Next · Summarizing Posterior Inference ▶ <013-summarizing-posterior-inference>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Between two answers
---------------------

The posterior always lies **between** the prior and the data. The Beta–Binomial makes this exact rather
than metaphorical: the posterior mean is a **weighted average** of the prior mean and the sample
proportion, and the weights are interpretable.

The weighted average
----------------------

With :math:`\theta \mid y \sim \mathrm{Beta}(\alpha + y,\ \beta + n - y)`, the posterior mean is

.. math::

   \mathrm{E}[\theta \mid y] = \frac{\alpha + y}{\alpha + \beta + n}
   = \underbrace{\frac{\alpha + \beta}{\alpha + \beta + n}}_{\text{prior weight}}
     \cdot \frac{\alpha}{\alpha + \beta}
   \;+\;
     \underbrace{\frac{n}{\alpha + \beta + n}}_{\text{data weight}}
     \cdot \frac{y}{n} .

The prior mean is :math:`\alpha / (\alpha+\beta)`; the data's answer is the sample proportion
:math:`y/n`. Each is weighted by its **effective sample size** — :math:`\alpha + \beta` for the prior,
:math:`n` for the data. A :math:`\mathrm{Beta}(2, 8)` prior carries the weight of **ten** prior
observations; against :math:`n = 10` it contributes about half the answer, against :math:`n = 200`,
about five per cent.

The data wins, eventually
---------------------------

The data weight :math:`n / (n + \alpha + \beta)` climbs to **1** as :math:`n \to \infty`. So the
posterior mean converges to the sample proportion, and the posterior itself concentrates — a preview of
the large-sample theory in Stage 4. Two analysts with different (reasonable) priors are **driven to
agreement** by enough data. The prior matters most exactly where it should: when data are **sparse**.

.. code-block:: python

   from scipy import stats
   for n in (10, 100, 1000):
       y = int(0.8 * n)                       # same proportion, growing n
       a, b = 2, 8                            # prior mean 0.2, prior "sample size" 10
       print(n, round(stats.beta(a + y, b + n - y).mean(), 3))
   # 10 -> 0.500   100 -> 0.745   1000 -> 0.792   (approaching 0.8)

Shrinkage, and its price
--------------------------

Pulling the estimate toward the prior mean is **shrinkage**, and it is a feature: it regularises noisy
small-sample estimates and prevents the absurdity of a 0% or 100% rate from three trials. The price is
that a **badly chosen** informative prior biases the answer, most damagingly when :math:`n` is small
and the pull is strongest. Hence the discipline: state the prior, justify it, and **check the
sensitivity** of conclusions to reasonable alternatives.

.. hint::

   **Related lessons:** :doc:`Estimating a Probability from Binomial Data <011-estimating-a-probability-from-binomial-data>`  ·  :doc:`Informative Prior Distributions <014-informative-prior-distributions>`  ·  :doc:`Noninformative Prior Distributions <018-noninformative-prior-distributions>`  ·  :doc:`Large-Sample Theory <029-large-sample-theory>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/posterior-as-a-compromise-between-data-and-prior-information/ <https://insightful-data-lab.com/2025/11/09/posterior-as-a-compromise-between-data-and-prior-information/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

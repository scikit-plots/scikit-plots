.. _bda-noninformative-prior-distributions:

========================================================================
Noninformative Prior Distributions
========================================================================

**Part 1 · Stage 2 · 📍 Single-Parameter Models & Priors**  ·  Lesson 018 of 144  ·  *beginner*

:doc:`◀ Previous · Informative Prior Distribution for Cancer Rates <017-informative-prior-distribution-for-cancer-rates>`   ·   :doc:`Next · Weakly Informative Prior Distributions ▶ <019-weakly-informative-prior-distributions>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Letting the data speak
------------------------

Sometimes you want the prior to contribute as **little** as possible — to report what *these* data say,
with minimal external input. A **noninformative** (or *reference*, or *vague*) prior aims at that. The
ambition is honourable and the execution is subtler than it looks, because "knowing nothing" turns out
not to be a well-defined state.

Flat is not neutral
---------------------

The obvious candidate is a **uniform** prior, :math:`p(\theta) \propto 1`. But flatness is **not
invariant** under reparameterisation: if :math:`\theta` is uniform, then :math:`\log \theta` is not —
the Jacobian from the probability-theory lesson intervenes. A prior that claims ignorance about a rate
therefore asserts something quite specific about the **log**-rate. "Noninformative" always means
noninformative *on some scale*, and that scale is a choice.

Jeffreys' prior
-----------------

Harold **Jeffreys** solved the invariance problem by tying the prior to the model itself. The
**Jeffreys prior** is proportional to the square root of the **Fisher information**:

.. math::

   p_J(\theta) \;\propto\; \sqrt{\det I(\theta)}, \qquad
   I(\theta) = -\,\mathrm{E}\!\left[\frac{\partial^2}{\partial\theta^2} \log p(y \mid \theta)\right] .

Because the Fisher information transforms with exactly the Jacobian that densities need, this prior is
**invariant under smooth reparameterisation** — the same beliefs, whatever coordinates you use. It
recovers the intuitive answers: **flat** :math:`p(\theta) \propto 1` for a location parameter, and
:math:`p(\theta) \propto 1/\theta` for a scale parameter. For the binomial it gives
:math:`\mathrm{Beta}(\tfrac12, \tfrac12)`.

Improper priors
-----------------

Many noninformative priors are **improper**: they do not integrate to a finite value (a flat prior on
the whole real line; :math:`1/\sigma` on :math:`(0, \infty)`). This is tolerable **only if the
resulting posterior is proper**, which must be checked, not assumed — for hierarchical variance
parameters especially, a natural-looking improper prior can yield a posterior that does not exist,
while the sampler reports numbers regardless.

.. code-block:: python

   from scipy import stats
   y, n = 8, 10
   flat     = stats.beta(1 + y, 1 + n - y)          # Beta(1,1): uniform
   jeffreys = stats.beta(0.5 + y, 0.5 + n - y)      # Beta(1/2,1/2)
   flat.mean(), jeffreys.mean()                     # 0.750, 0.773

Modern practice has largely moved on: rather than chase an unattainable neutrality, use a
**weakly informative** prior that rules out the absurd while letting the data dominate.

.. hint::

   **Related lessons:** :doc:`Weakly Informative Prior Distributions <019-weakly-informative-prior-distributions>`  ·  :doc:`Informative Prior Distributions <014-informative-prior-distributions>`  ·  :doc:`Some Useful Results from Probability Theory <008-some-useful-results-from-probability-theory>`  ·  :doc:`Normal Data with a Noninformative Prior Distribution <021-normal-data-with-a-noninformative-prior-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/noninformative-prior-distributions/ <https://insightful-data-lab.com/2025/11/09/noninformative-prior-distributions/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

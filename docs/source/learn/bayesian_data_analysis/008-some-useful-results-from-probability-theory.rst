.. _bda-some-useful-results-from-probability-theory:

========================================================================
Some Useful Results from Probability Theory
========================================================================

**Part 1 · Stage 1 · 🎲 The Bayesian Idea**  ·  Lesson 008 of 144  ·  *beginner*

:doc:`◀ Previous · Example — Calibration for Record Linkage <007-example-calibration-for-record-linkage>`   ·   :doc:`Next · Computation and Software ▶ <009-computation-and-software>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The toolkit
-------------

A handful of identities do nearly all the work in Bayesian derivations. They are elementary, but worth
stating once, precisely, because every later chapter leans on them.

Conditioning and marginalising
--------------------------------

Two operations turn joint distributions into the ones you want. **Marginalising** integrates a variable
away; **conditioning** fixes it. Together they give the law of total probability:

.. math::

   p(y) = \int p(y, \theta) \, d\theta = \int p(y \mid \theta) \, p(\theta) \, d\theta .

This is exactly the evidence in Bayes' rule, and the same averaging produces the **posterior
predictive** distribution :math:`p(\tilde{y} \mid y) = \int p(\tilde{y} \mid \theta) p(\theta \mid y)
\, d\theta` — a mixture of predictions, weighted by posterior plausibility.

Iterated expectation and variance
-----------------------------------

Summaries of a marginal can be built from summaries of conditionals, without doing the integral:

.. math::

   \mathrm{E}(u) = \mathrm{E}\bigl(\mathrm{E}(u \mid v)\bigr), \qquad
   \mathrm{var}(u) = \mathrm{E}\bigl(\mathrm{var}(u \mid v)\bigr)
                   + \mathrm{var}\bigl(\mathrm{E}(u \mid v)\bigr).

The variance decomposition is the more revealing: total uncertainty splits into the **average
conditional uncertainty** plus the **uncertainty in the conditional mean**. Averaging over a parameter
you have not learned always *adds* variance — which is precisely why point estimates understate
uncertainty, and why hierarchical models must propagate it.

Transformation of variables
-----------------------------

Reparameterising (to :math:`\log \sigma`, to :math:`\mathrm{logit}\,\theta`) is routine, and densities
do not simply carry over: they pick up a **Jacobian**,

.. math::

   p_{\phi}(\phi) = p_{\theta}(\theta) \left| \frac{d\theta}{d\phi} \right| .

Forgetting the Jacobian is a classic bug — a "noninformative" flat prior on :math:`\theta` is **not**
flat on :math:`\log \theta`, a subtlety the noninformative-prior lesson confronts directly. Simulation
sidesteps the algebra entirely: transform the **draws** and summarise them.

.. code-block:: python

   import numpy as np
   theta = ...                       # posterior draws
   phi = np.log(theta)               # any function of the draws is itself posterior draws
   phi.mean(), np.percentile(phi, [2.5, 97.5])

That last property — that a function of posterior draws gives the posterior of that function, with no
Jacobian and no delta method — is one of the quiet advantages of the simulation-based workflow this
course adopts.

.. hint::

   **Related lessons:** :doc:`General Notation for Statistical Inference <002-general-notation-for-statistical-inference>`  ·  :doc:`Bayesian Inference <003-bayesian-inference>`  ·  :doc:`Averaging Over Nuisance Parameters <020-averaging-over-nuisance-parameters>`  ·  :doc:`Noninformative Prior Distributions <018-noninformative-prior-distributions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/08/some-useful-results-from-probability-theory/ <https://insightful-data-lab.com/2025/11/08/some-useful-results-from-probability-theory/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

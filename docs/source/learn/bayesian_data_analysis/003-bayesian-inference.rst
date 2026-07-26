.. _bda-bayesian-inference:

========================================================================
Bayesian Inference
========================================================================

**Part 1 · Stage 1 · 🎲 The Bayesian Idea**  ·  Lesson 003 of 144  ·  *beginner*

:doc:`◀ Previous · General Notation for Statistical Inference <002-general-notation-for-statistical-inference>`   ·   :doc:`Next · Discrete Bayesian Examples – Genetics and Spell Checking (with θ) ▶ <004-discrete-bayesian-examples-genetics-and-spell-checking-with>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Conclusions as probabilities
------------------------------

**Bayesian inference** is the process of drawing conclusions about unknown quantities as **probability
statements conditional on the observed data**. Not "the estimate is 0.58 ± 0.05", but "given these
data, there is a 93% probability that :math:`\theta` exceeds 0.5". Every conclusion is read off the
posterior distribution, and every conclusion carries its uncertainty with it.

Bayes' rule, again
--------------------

The machinery is one line. Starting from the joint model :math:`p(\theta, y) = p(\theta) p(y \mid
\theta)` and conditioning on :math:`y`:

.. math::

   p(\theta \mid y) = \frac{p(\theta)\, p(y \mid \theta)}{p(y)},
   \qquad p(y) = \int p(\theta)\, p(y \mid \theta) \, d\theta .

The denominator — the **marginal likelihood** or *evidence* — does not depend on :math:`\theta`, so for
inference it merely normalises. This is why the unnormalised form
:math:`p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta)` is the working equation, and why samplers
need only the numerator.

Reading a posterior
---------------------

Once you have the posterior (analytically or as samples), every question is answered by summarising it:

.. code-block:: python

   import numpy as np
   post = ...                                   # draws from p(theta | y)
   post.mean(), np.median(post)                 # point summaries
   np.percentile(post, [2.5, 97.5])             # 95% credible interval
   (post > 0.5).mean()                          # P(theta > 0.5 | y), directly

That last line is the Bayesian signature: a probability of a hypothesis, computed by counting draws.

What differs, and why
-----------------------

Bayesian and frequentist conclusions often agree in simple problems with plenty of data. They diverge
where **conditioning** matters: small samples, many parameters, hierarchical structure, or genuine
prior information. The cost is that you must state a prior; the benefit is that the answer is a
distribution you may interpret directly, and that uncertainty propagates automatically into any
derived quantity.

.. hint::

   **Related lessons:** :doc:`The three steps of Bayesian data analysis <001-the-three-steps-of-bayesian-data-analysis>`  ·  :doc:`General Notation for Statistical Inference <002-general-notation-for-statistical-inference>`  ·  :doc:`Discrete Bayesian Examples – Genetics and Spell Checking (with θ) <004-discrete-bayesian-examples-genetics-and-spell-checking-with>`  ·  :doc:`Probability as a Measure of Uncertainty <005-probability-as-a-measure-of-uncertainty>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/08/bayesian-inference-2/ <https://insightful-data-lab.com/2025/11/08/bayesian-inference-2/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

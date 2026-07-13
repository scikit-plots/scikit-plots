.. _bda-discrete-bayesian-examples-genetics-and-spell-checking-with:

========================================================================
Discrete Bayesian Examples – Genetics and Spell Checking (with θ)
========================================================================

**Part 1 · Stage 1 · 🎲 The Bayesian Idea**  ·  Lesson 004 of 144  ·  *beginner*

:doc:`◀ Previous · Bayesian Inference <003-bayesian-inference>`   ·   :doc:`Next · Probability as a Measure of Uncertainty ▶ <005-probability-as-a-measure-of-uncertainty>`   ·   :doc:`↑ Section <index>`


Bayes' rule on a discrete unknown
-----------------------------------

The clearest way to see Bayes' rule work is when :math:`\theta` takes only a **few discrete values**.
No integrals, no sampling — just arithmetic on probabilities. Two classic examples, both in Gelman's
first chapter, show the whole logic of updating.

Genetics: the carrier problem
-------------------------------

Hemophilia is X-linked recessive. A woman whose brother is affected has a carrier status
:math:`\theta \in \{0, 1\}` with **prior** :math:`\Pr(\theta = 1) = \tfrac{1}{2}`. Suppose she then
has two **unaffected** sons, :math:`y = (0, 0)`. A carrier passes the gene with probability
:math:`\tfrac{1}{2}` per son, so the likelihoods are
:math:`p(y \mid \theta = 1) = (\tfrac{1}{2})^2 = \tfrac14` and :math:`p(y \mid \theta = 0) = 1`.
Bayes' rule updates her carrier probability:

.. math::

   \Pr(\theta = 1 \mid y)
   = \frac{\tfrac14 \cdot \tfrac12}{\tfrac14 \cdot \tfrac12 + 1 \cdot \tfrac12}
   = \frac{1}{5} = 0.20 .

Two healthy sons drop the probability from 0.50 to 0.20 — **evidence, not proof**. Each further
unaffected son multiplies the odds again; the update is **sequential**, and today's posterior is
tomorrow's prior.

Spell checking: which word was meant?
---------------------------------------

The same rule powers a spell checker. Seeing the typed string ``radom``, the candidate corrections
:math:`\theta \in \{\text{random}, \text{radon}, \text{radom}\}` are scored by
:math:`p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta)`: the **prior** is how common each word is
in the language, and the **likelihood** is how likely that typo is given the intended word.

.. code-block:: python

   prior = {"random": 7.6e-5, "radon": 6.1e-6, "radom": 1.2e-7}   # word frequencies
   like  = {"random": 0.00193, "radon": 0.000143, "radom": 0.975} # typo model
   post  = {w: prior[w] * like[w] for w in prior}
   Z = sum(post.values())
   {w: p / Z for w, p in post.items()}    # normalised posterior over intended words

A rare word needs a *much* better typo-likelihood to win. Both examples make the same point: the prior
supplies context, the likelihood supplies evidence, and the posterior is their **compromise** — the
theme of the next stage.

.. hint::

   **Related lessons:** :doc:`Bayesian Inference <003-bayesian-inference>`  ·  :doc:`Probability as a Measure of Uncertainty <005-probability-as-a-measure-of-uncertainty>`  ·  :doc:`Posterior as a Compromise Between Data and Prior Information <012-posterior-as-a-compromise-between-data-and-prior-information>`  ·  :doc:`Estimating a Probability from Binomial Data <011-estimating-a-probability-from-binomial-data>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/08/discrete-bayesian-examples-genetics-and-spell-checking-with-%ce%b8/ <https://insightful-data-lab.com/2025/11/08/discrete-bayesian-examples-genetics-and-spell-checking-with-%ce%b8/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

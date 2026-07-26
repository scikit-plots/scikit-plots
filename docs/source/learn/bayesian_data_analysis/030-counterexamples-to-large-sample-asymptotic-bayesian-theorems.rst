.. _bda-counterexamples-to-large-sample-asymptotic-bayesian-theorems:

========================================================================
Counterexamples to large-sample (asymptotic) Bayesian theorems
========================================================================

**Part 1 · Stage 4 · 📏 Asymptotics & Frequentist Ties**  ·  Lesson 030 of 144  ·  *beginner*

:doc:`◀ Previous · Large-Sample Theory <029-large-sample-theory>`   ·   :doc:`Next · Frequency Evaluations of Bayesian Inferences ▶ <031-frequency-evaluations-of-bayesian-inferences>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

When the guarantees fail
--------------------------

Large-sample theory is a promise with fine print, and the fine print matters. Each regularity condition
can fail in practice, and when it does, the reassuring picture — concentration, normality, correct
coverage — can fail with it.

At the boundary
-----------------

If the true value sits on the **edge** of the parameter space, the posterior cannot be normal there: it
has nowhere to put mass on one side. A hierarchical variance :math:`\tau^2` whose true value is
**zero** (groups genuinely identical) produces a posterior heaped against the boundary — asymmetric,
non-normal, and badly summarised by a mode and curvature. The same happens for a probability at 0 or 1.
The normal approximation on a **transformed** scale (:math:`\log \tau`) helps, but the boundary case is
inherently awkward.

Not identified, or growing
----------------------------

**Unidentified** parameters — those the likelihood cannot distinguish — never concentrate; the
posterior in that direction remains the prior, however much data arrive. **Label switching** in
mixtures (Stage 16) is a benign instance; a multimodal likelihood is a harsher one. And when the
number of parameters **grows with** :math:`n` — one per observation, as in the classic
Neyman–Scott problem — consistency for the parameters of interest can fail outright, since new data
bring new unknowns.

Infinite dimensions
---------------------

Most striking are the **nonparametric** counterexamples of Diaconis and Freedman: with an
infinite-dimensional parameter, seemingly innocuous priors yield posteriors that **converge to the
wrong answer**. In some of their examples, the posterior mean and density converge on a false value
while the posterior **mode** remains consistent — a warning that in infinite dimensions, intuition
built on finite parameter counts is not merely imprecise but wrong. The nonparametric models of Part V
must therefore be chosen with care, not adopted casually.

Misspecification, the everyday case
-------------------------------------

The likeliest failure is the mundane one: **the model is wrong**. The posterior then concentrates on
the parameter minimising Kullback–Leibler divergence from the truth — the "best available lie" — and
becomes normal around it, but with a **sandwich** variance rather than the inverse Fisher information.
Credible intervals shrink like :math:`1/\sqrt{n}` around a value that is not the truth, growing more
confident and no less wrong.

.. code-block:: python

   # symptom, not proof: posterior concentrating away from any sensible value
   # while posterior predictive checks fail -> suspect misspecification, not sample size

The moral is not that asymptotics are useless but that they are **conditional**. They justify
approximations in regular, correctly specified, fixed-dimension problems — and they justify **model
checking** everywhere else.

.. hint::

   **Related lessons:** :doc:`Large-Sample Theory <029-large-sample-theory>`  ·  :doc:`Normal Approximations to the Posterior Distribution <028-normal-approximations-to-the-posterior-distribution>`  ·  :doc:`Dirichlet process prior distributions <140-dirichlet-process-prior-distributions>`  ·  :doc:`Label switching and posterior computation <136-label-switching-and-posterior-computation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/counterexamples-to-large-sample-asymptotic-bayesian-theorems/ <https://insightful-data-lab.com/2025/11/09/counterexamples-to-large-sample-asymptotic-bayesian-theorems/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

.. _bda-probability-as-a-measure-of-uncertainty:

========================================================================
Probability as a Measure of Uncertainty
========================================================================

**Part 1 · Stage 1 · 🎲 The Bayesian Idea**  ·  Lesson 005 of 144  ·  *beginner*

:doc:`◀ Previous · Discrete Bayesian Examples – Genetics and Spell Checking (with θ) <004-discrete-bayesian-examples-genetics-and-spell-checking-with>`   ·   :doc:`Next · Example — Probabilities from Football Point Spreads ▶ <006-example-probabilities-from-football-point-spreads>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Uncertainty, not just frequency
---------------------------------

Bayesian analysis uses **probability to quantify uncertainty** — about anything, not merely about
repeatable experiments. "The probability this coin lands heads is 0.5" and "the probability this
patient carries the gene is 0.2" are, in this framework, the same kind of statement: a **numerical
measure of how uncertain you are**, given what you know.

Where the numbers come from
-----------------------------

Three standard justifications are offered for assigning a probability:

* **Symmetry** — if :math:`n` outcomes are indistinguishable in every relevant respect, each gets
  :math:`1/n` (a fair die, a shuffled deck).
* **Frequency** — the long-run proportion in a sequence of similar trials (the rate of a disease).
* **Subjective assessment** — a considered degree of belief, when neither symmetry nor a reference
  sequence is available.

Look closely and all three lean on **judgement**. Symmetry requires deciding *which* respects are
relevant; frequency requires choosing *which* trials count as "similar". The choice of a **reference
set** is unavoidably a modelling decision, so even "objective" probabilities are conditional on
context.

Conditional on what you know
------------------------------

This is why Bayesian probabilities are always written with a **conditioning bar**: they are
:math:`p(\theta \mid \text{information})`. The probability that the patient carries the gene changes —
legitimately, not fickly — when a son is born unaffected. Probability is not a property of the
parameter; it is a property of your **state of knowledge** about it.

Subjective, but disciplined
-----------------------------

The obvious objection is that subjective probabilities are arbitrary. Two disciplines answer it.
**Coherence**: degrees of belief must obey the probability axioms, or you can be led into a set of bets
you are certain to lose (the Dutch-book argument). And **calibration**: of all the events to which you
assign probability 0.7, about 70% should actually occur — an empirically checkable standard, and the
subject of the next two lessons. Bayesian analysis does not demand that priors match anyone's inner
convictions; it demands that assumptions be **stated clearly** and their implications **checked**.

.. hint::

   **Related lessons:** :doc:`Bayesian Inference <003-bayesian-inference>`  ·  :doc:`Example — Probabilities from Football Point Spreads <006-example-probabilities-from-football-point-spreads>`  ·  :doc:`Example — Calibration for Record Linkage <007-example-calibration-for-record-linkage>`  ·  :doc:`Noninformative Prior Distributions <018-noninformative-prior-distributions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/08/probability-as-a-measure-of-uncertainty/ <https://insightful-data-lab.com/2025/11/08/probability-as-a-measure-of-uncertainty/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

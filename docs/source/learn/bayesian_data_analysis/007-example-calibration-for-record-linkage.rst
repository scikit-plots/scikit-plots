.. _bda-example-calibration-for-record-linkage:

========================================================================
Example — Calibration for Record Linkage
========================================================================

**Part 1 · Stage 1 · 🎲 The Bayesian Idea**  ·  Lesson 007 of 144  ·  *beginner*

:doc:`◀ Previous · Example — Probabilities from Football Point Spreads <006-example-probabilities-from-football-point-spreads>`   ·   :doc:`Next · Some Useful Results from Probability Theory ▶ <008-some-useful-results-from-probability-theory>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Are two records the same person?
----------------------------------

**Record linkage** is the problem of deciding which records in two files — a census and a survey, two
hospital databases — refer to the **same individual**, when there is no shared unique key and the
fields are noisy: misspelled names, transposed digits, missing values. Each candidate pair gets a
**score** from comparing its fields, and the question is what that score *means*.

From score to probability
---------------------------

A score is only useful if it can be turned into :math:`\Pr(\text{match} \mid \text{score})`. Bayes'
rule supplies the conversion: with a **prior** probability that a random pair matches, and the
**likelihood** of the observed score under matching and non-matching pairs,

.. math::

   \Pr(\text{match} \mid \text{score})
   = \frac{p(\text{score} \mid \text{match}) \; \pi}
          {p(\text{score} \mid \text{match}) \; \pi
         + p(\text{score} \mid \text{non-match}) \, (1 - \pi)} .

The prior :math:`\pi` matters enormously: comparing two files of size :math:`n` produces :math:`n^2`
pairs but at most :math:`n` true matches, so **most pairs are non-matches** and :math:`\pi` is tiny. A
score that looks convincing can still leave the pair more likely a coincidence.

Calibration
-------------

The point of the example is **calibration**: among all pairs assigned probability 0.9, roughly 90%
should truly be matches. Calibration is **checkable**, by holding out pairs whose truth is known:

.. code-block:: python

   import numpy as np
   # bin predicted match probabilities; compare bin mean to observed match rate
   bins = np.linspace(0, 1, 11)
   idx = np.digitize(p_match, bins) - 1
   for b in range(10):
       m = idx == b
       if m.sum():
           print(f"{bins[b]:.1f}-{bins[b+1]:.1f}: predicted {p_match[m].mean():.2f}"
                 f"  observed {is_match[m].mean():.2f}  (n={m.sum()})")

A model whose stated probabilities survive this test can be **trusted downstream**; one that does not
will quietly corrupt every analysis built on the linked file. The mismatch between fitted and observed
probabilities is a first taste of the **posterior predictive check** in Part II — and the decision of
*which* pairs to declare matched is a decision problem, with its own costs for false links and missed
ones.

.. hint::

   **Related lessons:** :doc:`Probability as a Measure of Uncertainty <005-probability-as-a-measure-of-uncertainty>`  ·  :doc:`Example — Probabilities from Football Point Spreads <006-example-probabilities-from-football-point-spreads>`  ·  :doc:`Posterior predictive checking <042-posterior-predictive-checking>`  ·  :doc:`Bayesian decision theory in diﬀerent contexts <057-bayesian-decision-theory-in-different-contexts>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/08/example-calibration-for-record-linkage/ <https://insightful-data-lab.com/2025/11/08/example-calibration-for-record-linkage/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

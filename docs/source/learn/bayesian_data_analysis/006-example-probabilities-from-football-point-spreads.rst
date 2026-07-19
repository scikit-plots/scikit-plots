.. _bda-example-probabilities-from-football-point-spreads:

========================================================================
Example — Probabilities from Football Point Spreads
========================================================================

**Part 1 · Stage 1 · 🎲 The Bayesian Idea**  ·  Lesson 006 of 144  ·  *beginner*

:doc:`◀ Previous · Probability as a Measure of Uncertainty <005-probability-as-a-measure-of-uncertainty>`   ·   :doc:`Next · Example — Calibration for Record Linkage ▶ <007-example-calibration-for-record-linkage>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Assignment, not inference
---------------------------

This example illustrates **probability assignment** — how to arrive at a number — rather than Bayesian
inference itself. Its subject is the American-football **point spread**: the bookmakers' published
prediction of the margin by which the favourite will win. Given a spread, what is the probability the
favourite actually covers it, or simply wins?

Three routes to a number
--------------------------

The same question is approached three ways, matching the three justifications of the previous lesson:

* **Subjective** — an informed fan states a probability directly.
* **Empirical** — count outcomes in a database of games. Across **672** professional games, one can
  simply tabulate how often favourites at a given spread won.
* **Parametric** — build a probability **model** for the outcome and read the probability off it.

The parametric model
----------------------

The empirical route runs out of data at any particular spread, so the model earns its keep. Plotting
:math:`d = (\text{actual outcome}) - (\text{point spread})` against the spread shows the differences
are roughly **centred at zero** with a spread of about **14 points**, and largely **independent of the
spread itself**. That suggests

.. math::

   d \sim \mathrm{N}(0,\; 14^2),

so the favourite (spread :math:`s`) wins when the actual margin exceeds 0, i.e. when :math:`d > -s`:

.. code-block:: python

   from scipy.stats import norm
   s = 3.5                                  # point spread
   p_win = 1 - norm.cdf(-s, loc=0, scale=14)   # P(favourite wins)  ≈ 0.60
   p_cover = 1 - norm.cdf(0, loc=0, scale=14)  # P(covers spread)   = 0.50

The lessons
-------------

Two. First, the **model smooths and extrapolates**: it gives a probability at spreads where few games
were ever played, which raw counts cannot. Second, the model is **checked against data** — the
zero-centred, constant-variance normal is adopted *because* the scatterplot supports it, not because it
is convenient. Probability assignment, done honestly, already involves the third of the three steps.

.. hint::

   **Related lessons:** :doc:`Probability as a Measure of Uncertainty <005-probability-as-a-measure-of-uncertainty>`  ·  :doc:`Example — Calibration for Record Linkage <007-example-calibration-for-record-linkage>`  ·  :doc:`Some Useful Results from Probability Theory <008-some-useful-results-from-probability-theory>`  ·  :doc:`Normal Distribution with Known Variance <015-normal-distribution-with-known-variance>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/08/example-probabilities-from-football-point-spreads/ <https://insightful-data-lab.com/2025/11/08/example-probabilities-from-football-point-spreads/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner

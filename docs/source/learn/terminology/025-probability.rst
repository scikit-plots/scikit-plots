:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-probability:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Probability</b></div>`

=============
Probability
=============

*A number in [0, 1] quantifying how likely an event is.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Probability** is a number between 0 and 1 that expresses how likely an event is:
0 means impossible, 1 means certain, and values in between are degrees of
likelihood. A fair coin landing heads has probability 0.5.

The basic formula
-----------------

When every outcome is equally likely, probability is the ratio of favourable to
total outcomes:

.. math::

   P(E) = \frac{\text{number of favourable outcomes}}{\text{total number of outcomes}}.

Rolling a 4 on a fair die: :math:`P = 1/6 \approx 0.167`.

Three ways to assign it
-----------------------

- **Theoretical** — from the structure of the problem (a fair die, a fair coin).
- **Experimental (frequentist)** — from observed frequencies (flip a coin 100 times
  and count heads); it converges to the theoretical value as trials grow.
- **Subjective (Bayesian)** — a degree of belief (a forecaster's "70% chance of
  rain").

The core rules
--------------

- **Complement** — :math:`P(\text{not } A) = 1 - P(A)`.
- **Addition (OR)** — in general
  :math:`P(A \cup B) = P(A) + P(B) - P(A \cap B)`, which reduces to
  :math:`P(A) + P(B)` only when the events are mutually exclusive.
- **Multiplication (AND)** — in general
  :math:`P(A \cap B) = P(A)\,P(B \mid A)`, which reduces to :math:`P(A)\,P(B)` only
  when the events are independent. Rolling a 2 then a 5 on two dice:
  :math:`\tfrac{1}{6} \times \tfrac{1}{6} = \tfrac{1}{36}`.

Pitfalls and edge cases
-----------------------

- **Mutually exclusive is not independent** — exclusive events *cannot* co-occur
  (and are strongly dependent); independent events have no influence on each other.
  Confusing the two is the classic probability error.
- **Conditional probability** — :math:`P(B \mid A) = P(A \cap B)/P(A)` underlies the
  general AND rule and, ultimately, Bayes' theorem.
- **Normalisation** — probabilities over all mutually exclusive outcomes must sum to
  exactly 1.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Probability <https://insightful-data-lab.com/2025/08/30/probability/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner

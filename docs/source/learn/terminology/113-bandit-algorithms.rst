:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-bandit-algorithms:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎰&nbsp;&nbsp;<b>Bandit Algorithms</b></div>`

===================
Bandit Algorithms
===================

*Strategies that allocate trials to options to maximise reward while learning.*

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

**Bandit algorithms** solve the **exploration–exploitation trade-off** in sequential
decisions. Cast as a **multi-armed bandit (MAB)** — slot machines with several arms — the
goal is to **maximise cumulative reward** by choosing which arm to pull, balancing
**exploration** (trying options to learn their payoff) against **exploitation** (playing the
best-known option now).

Why it matters
--------------

An A/B test holds a **fixed split** until significance, spending traffic on losing variants
the whole time. A bandit **reallocates traffic dynamically** toward what's winning, so better
options get more users **sooner** — cutting the cost of learning. It powers ad selection,
recommendations, pricing and adaptive trials.

The core tension
----------------

Suppose three arms pay out at 5%, 7% and 6%. Always playing the current best (7%) risks
**never discovering** the 6% arm is actually better with more data; always exploring
**wastes** reward. Good bandits trade these off automatically.

The main algorithms
-------------------

- **ε-greedy**: with probability :math:`\varepsilon` explore a random arm, otherwise exploit
  the best — simple and effective.
- **UCB** (upper confidence bound): play the arm with the highest **optimistic** estimate,
  exploring when uncertainty is high and settling as data accrues.
- **Thompson sampling** (Bayesian): keep a posterior per arm (a Beta distribution for
  click/no-click), **sample** from each, and play the highest draw — a natural, highly
  effective balance.
- **Softmax / Boltzmann**: choose arms with probability rising in estimated reward, tuned by
  a temperature.

Trade-offs vs A/B testing
-------------------------

Bandits **maximise reward during the test** (not just identify a winner after it) and
typically need **fewer samples**, at the cost of **more complexity** and less clean
significance reporting. They shine when rewards are **immediate and measurable** and the
environment is **stable**; rapidly shifting preferences or delayed rewards weaken them.

----

*Theme:* :ref:`Sequential Methods & Bandits <term-theme-bandits>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB)) <050-thompson-sampling-ts-in-bandits-multi-armed-band>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Beta Distribution <099-beta-distribution>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>`

----

.. hint::
   **More in Sequential Methods & Bandits**

   :doc:`O'Brien–Fleming (OBF) Method <078-o-brienfleming-obf-method>` · :doc:`Pocock Method <077-pocock-method>` · :doc:`Sequential Probability Ratio Test (SPRT) <076-sequential-probability-ratio-test-sprt>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB)) <050-thompson-sampling-ts-in-bandits-multi-armed-band>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Bandit Algorithms <https://insightful-data-lab.com/2025/08/24/bandit-algorithms/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

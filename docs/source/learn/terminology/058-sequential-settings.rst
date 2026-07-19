:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-sequential-settings:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎰&nbsp;&nbsp;<b>Sequential Settings</b></div>`

=====================
Sequential Settings
=====================

*Decision problems where data arrive over time and choices adapt as evidence accrues.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

A **sequential setting** is any framework where **data arrives over time** and you can
make decisions — stop or continue, accept or reject, reallocate traffic — **as the data
accumulates**, rather than only at a single fixed end point. It is the opposite of a
**fixed-horizon** design, where you pre-commit a sample size and analyse once at the
end.

What makes it different
-----------------------

Data comes one observation (or small batch) at a time, decisions can be **adaptive**,
and — crucially — naive repeated testing **inflates the Type I error rate**: every time
you "peek" at a frequentist p-value and consider stopping, you get extra chances to hit
significance by chance. Sequential settings therefore demand **specialised methods**.

Where it shows up
-----------------

- **Clinical trials** — interim analyses with stopping rules for efficacy, harm or
  futility (continuing a harmful treatment is unethical).
- **A/B testing** — users arrive over time and teams want to peek mid-experiment;
  valid sequential methods let them.
- **Online learning / bandits** — algorithms learn as data streams, shifting traffic
  to better variants.

The three families of methods
-----------------------------

- **Frequentist** — the **SPRT (sequential probability ratio test)** checks a
  likelihood ratio continuously; **group-sequential** designs (O'Brien–Fleming,
  Pocock) analyse at pre-planned checkpoints with **α-spending** to keep the overall
  error controlled.
- **Bayesian** — naturally sequential: the **posterior updates continuously** as data
  arrive, and you stop when a posterior probability crosses a threshold (say
  :math:`\Pr(\text{uplift} > 0) > 0.95`), with no peeking penalty.
- **Adaptive / online** — multi-armed bandits (:math:`\epsilon`-greedy, Thompson
  sampling) and reinforcement learning, built for sequential decisions.

Why it matters
--------------

Stopping early saves **time and cost**, avoids **ethically** continuing a clearly
harmful trial, lets businesses **adapt quickly**, and matches how many ML algorithms
actually receive data — provided the right method preserves statistical validity.

----

*Theme:* :ref:`Sequential Methods & Bandits <term-theme-bandits>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB)) <050-thompson-sampling-ts-in-bandits-multi-armed-band>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Frequentist <059-frequentist>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>`

----

.. hint::
   **More in Sequential Methods & Bandits**

   :doc:`Bandit Algorithms <113-bandit-algorithms>` · :doc:`O'Brien–Fleming (OBF) Method <078-o-brienfleming-obf-method>` · :doc:`Pocock Method <077-pocock-method>` · :doc:`Sequential Probability Ratio Test (SPRT) <076-sequential-probability-ratio-test-sprt>` · :doc:`Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB)) <050-thompson-sampling-ts-in-bandits-multi-armed-band>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Sequential Settings <https://insightful-data-lab.com/2025/08/28/sequential-settings/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced

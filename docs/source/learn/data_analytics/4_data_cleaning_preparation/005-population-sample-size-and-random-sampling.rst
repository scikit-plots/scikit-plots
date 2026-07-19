:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-005:
.. _da-4-cleaning-cleaning-005:

========================================================================
Population, Sample Size, and Random Sampling
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🧱 Data Integrity & Sampling` :bdg-info:`Lesson 005`

◀ :doc:`Previous <004-handling-insufficient-data-in-data-analysis>` · :doc:`Next <006-statistical-power-in-data-analysis>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


The whole and the part
------------------------

The insufficient-data lesson raised the question of *how much* data is enough;
answering it needs vocabulary. The **population** is the entire group you want to
understand — every customer, every transaction, every citizen. A **sample** is
the subset you actually examine, and the **sample size** is how many members it
contains. Almost all analysis works from samples, because examining the whole
population is usually impossible or impractical — which makes *how you sample*
one of the most consequential decisions in the whole process.

Why we sample
--------------

Studying an entire population is often infeasible: too large, too expensive, too
slow, or simply inaccessible (you cannot survey every possible future customer).
A well-chosen sample lets you draw reliable conclusions about the whole from a
manageable part — the core bargain of statistics. The bargain only holds,
though, when the sample **represents** the population, which is exactly where
sampling method matters.

Random sampling and its relatives
-----------------------------------

**Random sampling** selects members so that every member of the population has an
**equal chance** of being chosen. Its purpose is representativeness *by design*:
because selection does not depend on any characteristic, the sample tends to
mirror the population's mix, and the systematic lean that defines sampling bias
is removed. Common sampling approaches:

- **Simple random sampling** — every member equally likely; the baseline.
- **Stratified sampling** — divide the population into groups (strata) and sample
  from each, ensuring every group is represented in proportion — useful when some
  subgroups are small but important.
- **Systematic sampling** — select every *n*-th member from an ordered list; a
  practical approximation of random when the list has no hidden pattern.
- **Cluster sampling** — divide into clusters, randomly select whole clusters;
  efficient when the population is naturally grouped (e.g. by location).

The contrast case remains the biased methods from earlier — convenience and
self-selection — which do *not* give everyone an equal chance and therefore skew.

The role of sample size
-------------------------

Given a fair method, **size** governs *precision*: larger samples yield estimates
closer to the population truth, with less variability from the luck of the draw.
But size cannot fix bias — a large biased sample is confidently wrong, as the
1936-poll lesson showed. The two requirements are separate and both necessary: a
sample must be **representative** (right method) *and* **large enough** (right
size) to support reliable conclusions. The next lessons quantify "large enough."

The caveat
------------

Random sampling is the ideal, but genuinely random samples are hard to achieve
in practice — the sampling frame (the list you draw from) may itself omit part of
the population, reintroducing bias no amount of randomness within the frame can
fix. A "random" sample of phone numbers still misses people without phones.
The professional habit is to scrutinise not just *how* members were selected but
*what population the frame actually covers* — and to disclose the gap. The next
lesson turns from representativeness to a different sufficiency question: the
power to detect a real effect.

.. hint::

   - :doc:`Handling Insufficient Data in Data Analysis <004-handling-insufficient-data-in-data-analysis>`
   - :doc:`Sampling Bias and Unbiased Data <../3_data_preparation/009-sampling-bias-and-unbiased-data>`
   - :doc:`Statistical Power in Data Analysis <006-statistical-power-in-data-analysis>`
   - :doc:`Sample Size and Data Integrity <007-sample-size-and-data-integrity>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/10/31/population-sample-size-and-random-sampling/ <https://insightful-data-lab.com/2023/10/31/population-sample-size-and-random-sampling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: integrity

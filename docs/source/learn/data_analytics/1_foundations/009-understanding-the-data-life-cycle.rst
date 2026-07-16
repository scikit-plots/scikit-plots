:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-foundations-009:
.. _da-1-foundations-foundations-009:

========================================================================
Understanding the Data Life Cycle
========================================================================

:bdg-primary:`🌱 Foundations` :bdg-secondary:`🔄 The Analysis Process & Data Life Cycle` :bdg-info:`Lesson 009`

◀ :doc:`Previous <008-understanding-the-data-analysis-process-and-the-data-life-cycle>` · :doc:`Next <010-a-review-of-the-six-stages-of-the-data-life-cycle>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The data's own biography
--------------------------

The previous lesson separated the analyst's process from the **data life
cycle** — the journey the data itself travels. This lesson walks that journey
stage by stage, because an analyst who knows where data comes from and where it
is going works faster and trusts the right things.

The six stages, in depth
--------------------------

1. **Plan.** The stage most people never see, and the one that decides
   everything after. Before collection, someone chooses *what* data is needed,
   *how* it will be managed, *who* is responsible, and under what privacy and
   retention rules. A well-planned dataset arrives with definitions and owners;
   a badly planned one arrives as a mystery.
2. **Capture.** The data comes into existence or into the organisation:
   recorded by transactions and sensors, typed into forms, imported from
   external providers or public sources. Capture choices — what fields, what
   granularity, what validation at entry — set the ceiling on later quality.
3. **Manage.** The custodial stage: storing the data, securing it, organising
   it so it can be found, backing it up, and controlling who may access it.
   Most of the ecosystem lesson's storage layer lives here.
4. **Analyze.** The data is put to work answering questions — the stage where
   this whole course happens.
5. **Archive.** Data no longer in active use moves to long-term storage: out of
   the way, cheaper to keep, but retrievable when an audit or a historical
   comparison needs it.
6. **Destroy.** The deliberate end. When retention schedules or privacy
   obligations require it, data is securely deleted — a governed act, not
   housekeeping neglect.

Variation is normal
---------------------

Companies and industries carve the cycle differently — a hospital inserts
compliance reviews, a bank extends retention for regulators, a startup may
barely formalise it at all. The stage *names* matter less than the underlying
questions each stage answers: is this data planned, captured, kept, used,
parked, or gone?

Why the analyst should walk the cycle
---------------------------------------

Each stage upstream of *Analyze* is a place your data could have been shaped or
damaged: a Plan that never defined "active customer", a Capture form that made
the field optional, a Manage migration that truncated text. When a number looks
wrong, the life cycle is your checklist of where to look. And downstream, Archive
and Destroy explain the gaps: the missing 2019 data may not be lost — it may
have been destroyed on schedule, which is an answer, not a dead end.

.. hint::

   - :doc:`Understanding the Data Analysis Process and the Data Life Cycle <008-understanding-the-data-analysis-process-and-the-data-life-cycle>`
   - :doc:`A Review of the Six Stages of the Data Life Cycle <010-a-review-of-the-six-stages-of-the-data-life-cycle>`
   - :doc:`Understanding the Data Ecosystem <007-understanding-the-data-ecosystem>`
   - :doc:`Data Privacy in Data Ethics <../3_data_preparation/014-data-privacy-in-data-ethics>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/07/30/understanding-the-data-life-cycle/ <https://insightful-data-lab.com/2023/07/30/understanding-the-data-life-cycle/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: foundations, topic: process

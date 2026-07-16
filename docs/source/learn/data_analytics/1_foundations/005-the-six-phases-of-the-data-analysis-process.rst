:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-foundations-005:
.. _da-1-foundations-foundations-005:

========================================================================
The Six Phases of the Data Analysis Process
========================================================================

:bdg-primary:`🌱 Foundations` :bdg-secondary:`🔄 The Analysis Process & Data Life Cycle` :bdg-info:`Lesson 005`

◀ :doc:`Previous <004-detectives-and-data-analysts>` · :doc:`Next <006-the-origins-of-data-analysis-and-the-many-ways-to-structure-it>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


A map for every project
-------------------------

Every analysis in this course — and most you will do professionally — follows the
same six-phase path: **Ask, Prepare, Process, Analyze, Share, Act**. The phases
turn a vague request ("figure out why sales dipped") into a sequence of concrete,
manageable steps, which is the essence of **structured thinking**.

The six phases
----------------

1. **Ask.** Define the problem and the question. Who are the stakeholders, what
   decision hinges on the answer, and what would a useful answer look like? You
   cannot solve a problem you have not stated.
2. **Prepare.** Decide what data can answer the question, then find or collect
   it — identifying sources, checking credibility, and organising it for use.
3. **Process.** Clean the data: remove duplicates and errors, handle missing
   values, fix inconsistencies, and document every change. Clean data is the
   foundation everything after stands on.
4. **Analyze.** Organise, aggregate, and compute until the pattern that answers
   the question is visible — the sorting, formatting, pivoting, and querying of
   the analysis section.
5. **Share.** Communicate the finding to the people who asked, with visuals and
   narrative matched to the audience.
6. **Act.** Put the insight to work: recommendations, decisions, changes — and
   the feedback from acting becomes input for the next Ask.

A worked thread
-----------------

A bike-share company wants more annual members. *Ask:* how do casual riders and
members use the service differently? *Prepare:* twelve months of trip records.
*Process:* remove test rides and corrupted rows, standardise timestamps.
*Analyze:* compare ride length and day-of-week patterns by rider type. *Share:*
a short deck showing casual riders concentrate on long weekend rides. *Act:*
marketing targets weekend riders with a membership offer. One question, six
phases, a decision at the end — this exact case structure recurs in published
walkthroughs of the framework.

Not a straight line
---------------------

The phases are a map, not a railway. Real projects loop: analysis exposes a data
problem that sends you back to Process; sharing raises a follow-up that restarts
Ask. The map's value is knowing **where you are** and what the current phase owes
the next one — not forbidding movement between them.

This course, in phase order
-----------------------------

The course sections mirror the phases: asking and deciding (Section 2), preparing
data (Section 3), processing and cleaning (Section 4), analysing (Sections 5 and
7), and sharing (Section 6). Keep the six-phase map in mind and every technique
that follows has an obvious home.

.. hint::

   - :doc:`Why Data Analytics Matters Today <001-why-data-analytics-matters-today>`
   - :doc:`Understanding the Data Analysis Process and the Data Life Cycle <008-understanding-the-data-analysis-process-and-the-data-life-cycle>`
   - :doc:`The Stages of the Data Analysis Process and Their Roles <011-the-stages-of-the-data-analysis-process-and-their-roles>`
   - :doc:`Practical Application of the Data Analysis Process <012-practical-application-of-the-data-analysis-process>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/07/30/the-six-phases-of-the-data-analysis-process/ <https://insightful-data-lab.com/2023/07/30/the-six-phases-of-the-data-analysis-process/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: foundations, topic: process

:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-foundations-012:
.. _da-1-foundations-foundations-012:

========================================================================
Practical Application of the Data Analysis Process
========================================================================

:bdg-primary:`🌱 Foundations` :bdg-secondary:`🔄 The Analysis Process & Data Life Cycle` :bdg-info:`Lesson 012`

◀ :doc:`Previous <011-the-stages-of-the-data-analysis-process-and-their-roles>` · :doc:`Next <013-analytical-skills-and-their-core-components>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


The process, end to end
-------------------------

Frameworks earn their keep only in use. This lesson runs the six phases on one
realistic case from start to finish — the same shape as published walkthroughs
of the framework, including a well-known bike-share analysis that follows
exactly these steps.

The case
----------

A city bike-share company earns more from **annual members** than from
**casual riders** (single-ride and day passes). Marketing believes converting
casual riders to members is the cheapest growth available, and asks the
analytics team to help.

Running the phases
--------------------

**Ask.** The business question is sharpened to something data can answer: *how
do members and casual riders use the service differently, and what do those
differences suggest for converting casuals?* Stakeholders: marketing (will act),
finance (cares about revenue), the analytics lead (reviews the work). Success:
findings concrete enough to shape a campaign.

**Prepare.** Twelve months of trip records are identified as the evidence —
rider type, start and end times, stations. Credibility check: the data is the
company's own system of record, current, and complete; a known limitation is
that privacy rules prevent linking trips to individual riders, so the analysis
must work at the trip level.

**Process.** Cleaning finds what cleaning always finds: test rides from staff,
a handful of negative durations from clock errors, inconsistent station names
after a renaming. Rules are applied — drop rides under one minute, standardise
names — and every rule is documented so the counts are reproducible.

**Analyze.** Aggregation by rider type reveals the story: members ride briefly
and steadily on weekdays (commutes); casual riders take **longer rides,
concentrated on weekends** and afternoons (leisure). A simple pivot of average
duration and ride count by day-of-week and rider type makes the contrast vivid.

**Share.** A short deck leads with the one chart that carries the finding —
weekday-versus-weekend usage by rider type — and states the implication in
plain language: casual riders are leisure users, so membership pitches framed
around commuting will miss them.

**Act.** Marketing pilots a weekend-oriented membership offer at the busiest
leisure stations, with sign-ups tracked. The measured result — whatever it turns
out to be — becomes the data behind the *next* Ask.

What the walkthrough teaches
------------------------------

Three things worth carrying to your own projects. Most of the elapsed effort
sat in **Ask through Process**, exactly as the previous lesson predicted. The
Analyze step was, computationally, a modest aggregation — the value came from
asking a sharp question of clean data, not from sophisticated math. And the
process did not end at the insight: it ended at an **action with a measurement
attached**, which is what makes the loop a loop.

.. hint::

   - :doc:`The Stages of the Data Analysis Process and Their Roles <011-the-stages-of-the-data-analysis-process-and-their-roles>`
   - :doc:`The Six Phases of the Data Analysis Process <005-the-six-phases-of-the-data-analysis-process>`
   - :doc:`Using Data Analysis to Choose the Right Advertising Strategy <../2_data_driven_decisions/001-using-data-analysis-to-choose-the-right-advertising-strategy>`
   - :doc:`Understanding Data Analysis <../5_analyze_data/001-understanding-data-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/07/30/practical-application-of-the-data-analysis-process/ <https://insightful-data-lab.com/2023/07/30/practical-application-of-the-data-analysis-process/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: foundations, topic: process

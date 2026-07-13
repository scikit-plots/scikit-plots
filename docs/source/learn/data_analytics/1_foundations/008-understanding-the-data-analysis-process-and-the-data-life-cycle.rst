:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-foundations-008:
.. _data-analytics-foundations-008:
.. _da-foundations-foundations-008:
.. _da-decisions-foundations-008:
.. _da-prep-foundations-008:
.. _da-cleaning-foundations-008:
.. _da-analyze-foundations-008:
.. _da-viz-foundations-008:
.. _da-python-foundations-008:
.. _da-jobsearch-foundations-008:

========================================================================
Understanding the Data Analysis Process and the Data Life Cycle
========================================================================

:bdg-primary:`🌱 Foundations` :bdg-secondary:`🔄 The Analysis Process & Data Life Cycle` :bdg-info:`Lesson 008`

◀ :doc:`Previous <007-understanding-the-data-ecosystem>` · :doc:`Next <009-understanding-the-data-life-cycle>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Two journeys, often confused
------------------------------

Two sequences run through every data project, and they are not the same. The
**data analysis process** (Ask, Prepare, Process, Analyze, Share, Act) describes
what the **analyst** does to answer a question. The **data life cycle** describes
what happens to the **data itself**, from the moment someone decides to collect
it to the day it is deleted. One follows the worker; the other follows the
material. Keeping them straight prevents a common muddle — and interviewers like
asking about exactly this distinction.

The data life cycle
---------------------

A widely taught version has six stages:

1. **Plan.** Before any collection: decide what data is needed, how it will be
   managed, who is responsible for it, and under what rules.
2. **Capture.** Bring the data into existence or into the organisation —
   collecting from sources, sensors, forms, or external providers.
3. **Manage.** Store, secure, organise, and maintain it so it stays usable:
   where it lives, how it is backed up, who may access it.
4. **Analyze.** Use it — the stage where the entire analysis *process* happens.
5. **Archive.** Move data no longer in active use into long-term storage, still
   retrievable if needed.
6. **Destroy.** Delete it — securely and deliberately — when retention rules or
   privacy obligations say its time is up.

The exact stages and names vary by company and industry; regulated sectors add
compliance checkpoints. The shape, though — from planned birth to deliberate
death — is universal.

How the two interlock
-----------------------

The whole six-phase analysis process lives **inside** one stage of the life
cycle: *Analyze*. Conversely, the analyst constantly depends on the other
stages. Good **planning** upstream determines whether the data you need even
exists; good **management** determines whether you can find and trust it;
**archive** and **destroy** determine whether last year's comparison data is
still there — or legally must not be. When a lesson later in this course says
"check where the data came from," it is sending you back up the life cycle.

Why analysts should care about the whole cycle
------------------------------------------------

Because the biggest analysis problems are usually born outside the analysis. A
question that cannot be answered often traces to a Plan stage that never
anticipated it; dirty data traces to Capture; a missing year traces to Destroy.
Analysts who understand the life cycle diagnose these quickly — and, when
consulted early, help design collection so the *next* question is answerable.

The caveat
------------

Life-cycle diagrams look tidier than reality: data gets copied, forked into
spreadsheets, and half-archived, so the same record can sit at several stages at
once. Treat the cycle as the intended governance path, and expect to do some
detective work about where a given dataset really is on it.

.. hint::

   - :doc:`The Six Phases of the Data Analysis Process <005-the-six-phases-of-the-data-analysis-process>`
   - :doc:`Understanding the Data Life Cycle <009-understanding-the-data-life-cycle>`
   - :doc:`A Review of the Six Stages of the Data Life Cycle <010-a-review-of-the-six-stages-of-the-data-life-cycle>`
   - :doc:`Why Asking the Right Questions Matters in Data Analytics <../2_data_driven_decisions/004-why-asking-the-right-questions-matters-in-data-analytics>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/07/30/understanding-the-data-analysis-process-and-the-data-life-cycle/ <https://insightful-data-lab.com/2023/07/30/understanding-the-data-analysis-process-and-the-data-life-cycle/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: foundations, topic: process

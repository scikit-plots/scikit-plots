:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-foundations-025:
.. _da-1-foundations-foundations-025:

========================================================================
The Role of Business Tasks in Data Analysis
========================================================================

:bdg-primary:`🌱 Foundations` :bdg-secondary:`🧰 Tools, Applications & Ethics` :bdg-info:`Lesson 025`

◀ :doc:`Previous <024-industries-where-data-analysts-work-and-how-data-is-used>` · :doc:`Next <026-fairness-in-data-analysis>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The question behind the work
------------------------------

A **business task** is the question or problem that a data analysis answers for
a business. It is the anchor of the entire six-phase process: the *Ask* phase
exists to define it, every later phase is judged against it, and an analysis
without one is activity without purpose. This short lesson makes the concept
precise, because "we analysed the data" means nothing until you can say *which
task the analysis served*.

From situation to task
------------------------

Business tasks rarely arrive well-formed. They arrive as situations — "sales
are down", "customers are complaining", "we're launching in a new city" — and
the analyst's first job is converting the situation into a task: specific,
answerable with data, and attached to a decision.

- Situation: *afternoon sales feel weak.* Task: *identify which stores'
  2–5 pm sales fall below their own trailing average, and what distinguishes
  them* — feeding a staffing and promotion decision.
- Situation: *we want more members.* Task: *determine how casual riders and
  members use the service differently* — feeding a marketing decision.
- Situation: *the warehouse seems slow.* Task: *measure where time is spent
  between order and shipment* — feeding a process decision.

The pattern: a task names the **comparison or measurement** to perform and the
**decision** waiting on it. If you cannot state both, the Ask phase is not
finished.

Why the task governs everything
---------------------------------

Each phase consults the task. *Prepare* asks: what data bears on **this**
question? *Process* asks: clean to what standard **this** decision needs?
*Analyze* asks: does this computation move **this** question forward?
*Share* asks: what does the decision-maker need to understand? Scope creep,
rabbit holes, and beautiful-but-irrelevant charts are all, at bottom, moments
when work detached from the task. The cheapest project-management tool an
analyst owns is rereading the task statement.

The caveat
------------

Tasks can be wrong. Sometimes honest work reveals that the stated task
misdiagnoses the situation — sales are not "down", they moved channels — and
the analyst's duty is to surface that, not to answer the broken question
faithfully. A business task is the anchor of the analysis, not a gag order on
what the data actually shows.

.. hint::

   - :doc:`Why Asking the Right Questions Matters in Data Analytics <../2_data_driven_decisions/004-why-asking-the-right-questions-matters-in-data-analytics>`
   - :doc:`Data-Driven Decision-Making <003-data-driven-decision-making>`
   - :doc:`Defining the Problem Domain <../2_data_driven_decisions/017-defining-the-problem-domain>`
   - :doc:`Industries Where Data Analysts Work and How Data Is Used <024-industries-where-data-analysts-work-and-how-data-is-used>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/07/30/the-role-of-business-tasks-in-data-analysis/ <https://insightful-data-lab.com/2023/07/30/the-role-of-business-tasks-in-data-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: foundations, topic: tools

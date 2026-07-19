:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-ddd-008:

========================================================================
The Difference Between Data and Metrics, and the Role of Metrics
========================================================================

:bdg-primary:`🎯 Data-Driven Decisions` :bdg-secondary:`📐 Metrics & Dashboards` :bdg-info:`Lesson 008`

◀ :doc:`Previous <007-data-creates-value-only-when-it-is-communicated>` · :doc:`Next <009-dashboards>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


From facts to yardsticks
--------------------------

**Data** is the raw material: the collection of facts an organisation records
— every transaction, click, and timestamp. A **metric** is something more
deliberate: a **single, quantifiable type of data used when setting and
evaluating goals**. Revenue rows are data; *monthly recurring revenue* is a
metric. Ride records are data; *average ride length by rider type* is a
metric. The difference is purpose: a metric is data that has been given a job.

What turning data into a metric involves
------------------------------------------

Three decisions convert raw facts into a yardstick, and each is a judgement:

1. **A definition** — precisely which records count. Is an "active user"
   anyone who logged in this month? Performed an action? The metric's meaning
   lives in this choice.
2. **A calculation** — the formula applied. **Customer retention rate**: of
   the customers present at the period's start, what fraction remain at its
   end. **Return on investment (ROI)**: the profit an investment produced
   relative to its cost. Same data, different formulas, different stories.
3. **A comparison basis** — against what: last quarter, a target, a
   competitor, the metric's own history. A number without a comparison is a
   fact; with one, it is a signal.

Why metrics matter
--------------------

Metrics are how goals become **checkable**. "Improve customer loyalty" is a
wish; "raise 90-day retention from 78% to 85% by Q4" is a metric-defined goal
the whole organisation can steer by, measure progress against, and honestly
declare met or missed. Metrics are also the shared vocabulary between analysts
and stakeholders: when both sides agree what *retention* means and how it is
computed, a whole class of talking-past-each-other disappears.

Choosing them well
--------------------

Good metrics share three properties: they **track the actual goal** (not a
convenient proxy for it), they are **hard to game** (a metric people can
inflate without improving anything will be), and they are **few** — a handful
watched seriously beats a wall of numbers watched by no one. The foundations'
warning recurs with force here: what gets measured gets managed, *including
when the measure is wrong*.

The caveat
------------

Every metric compresses, and compression discards. Retention rate says
nothing about *which* customers stayed; average ride length hides the
distribution's shape. Treat metrics as instruments on a dashboard —
indispensable for steering, and always an invitation to look underneath when
one moves strangely. The next lesson is about exactly that dashboard.

.. hint::

   - :doc:`Data Creates Value Only When It Is Communicated <007-data-creates-value-only-when-it-is-communicated>`
   - :doc:`Dashboards <009-dashboards>`
   - :doc:`Mathematical Thinking <010-mathematical-thinking>`
   - :doc:`The Relationship Between Data and Decision-Making <005-the-relationship-between-data-and-decision-making>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/08/31/the-difference-between-data-and-metrics-and-the-role-of-metrics/ <https://insightful-data-lab.com/2023/08/31/the-difference-between-data-and-metrics-and-the-role-of-metrics/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: ddd, topic: metrics

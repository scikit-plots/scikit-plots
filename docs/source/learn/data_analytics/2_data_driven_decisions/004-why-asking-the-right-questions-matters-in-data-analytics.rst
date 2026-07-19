:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-ddd-004:

========================================================================
Why Asking the Right Questions Matters in Data Analytics
========================================================================

:bdg-primary:`🎯 Data-Driven Decisions` :bdg-secondary:`🧭 Framing the Problem` :bdg-info:`Lesson 004`

◀ :doc:`Previous <003-applying-data-analytics-problem-types-in-real-business-scenarios>` · :doc:`Next <005-the-relationship-between-data-and-decision-making>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


The question is the steering wheel
------------------------------------

Every failure mode met so far — answering the wrong problem, gathering
irrelevant data, precise answers nobody can act on — traces back to the
question asked at the start. Questions are the steering of the entire process,
and the standard framing gives them a memorable quality bar: effective
questions are **SMART**.

SMART questions
-----------------

- **Specific** — simple, significant, and focused on a single topic or a few
  closely related ideas. Not "how are sales?" but "how did repeat-customer
  revenue change after the March price update?"
- **Measurable** — answerable with something you can quantify and assess. "Do
  customers like us?" becomes "what fraction of survey respondents rate us 8+,
  and how has it moved?"
- **Action-oriented** — framed to encourage change: "what features would make
  the timesheet page easier to complete by Friday noon?" invites answers you
  can act on, where "why don't people do timesheets?" invites blame.
- **Relevant** — mattering to the problem at hand; a fascinating question that
  no pending decision needs is a detour.
- **Time-bound** — specifying the period under study, because "recently" is
  not a date range and every comparison needs one.

Run a draft question through the five letters and its weaknesses announce
themselves — usually a missing measure or a missing time window.

The questions to avoid
------------------------

Three anti-patterns corrupt answers before any data arrives:

- **Leading questions** presuppose their answer ("how great was the new
  design?") and harvest agreement, not information.
- **Closed-ended questions** ("did you like it?") collect yes/no where the
  value lives in elaboration; they have their place in structured surveys but
  starve discovery.
- **Assumption-laden questions** build on unproven premises ("why did the
  discount drive the sales spike?" — did it? was there a spike?), smuggling a
  conclusion into the framing.

The fairness thread from the foundations applies here too: questions should
not create or reinforce bias — who a question is asked *of*, and who it
silently excludes, shapes the answer as much as its wording.

A worked sharpening
---------------------

Stakeholder: *"Can you look into whether marketing is working?"* SMART pass:
**Specific** — which campaigns, which products? **Measurable** — define
"working": cost per acquisition? revenue lift? **Action-oriented** — what
decision waits: reallocate budget? **Relevant** — is the summer campaign
review the actual occasion? **Time-bound** — compare which quarters? Ten
minutes of this produces: *"For Q1–Q2, what was cost per new customer by
channel, and which channels should receive Q3 budget?"* — a question the rest
of the process can actually serve.

The caveat
------------

SMART is a quality filter, not a source of curiosity. The sharpest analysts
still begin with open, exploratory wondering — then *refine* the promising
wonder into SMART form before committing the project to it. Filter too early
and you only ever ask what is easy to measure.

.. hint::

   - :doc:`Applying Data Analytics Problem Types in Real Business Scenarios <003-applying-data-analytics-problem-types-in-real-business-scenarios>`
   - :doc:`The Role of Business Tasks in Data Analysis <../1_foundations/025-the-role-of-business-tasks-in-data-analysis>`
   - :doc:`The Relationship Between Data and Decision-Making <005-the-relationship-between-data-and-decision-making>`
   - :doc:`Data Creates Value Only When It Is Communicated <007-data-creates-value-only-when-it-is-communicated>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/08/31/why-asking-the-right-questions-matters-in-data-analytics/ <https://insightful-data-lab.com/2023/08/31/why-asking-the-right-questions-matters-in-data-analytics/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: ddd, topic: framing

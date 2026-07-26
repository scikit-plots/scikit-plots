:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-ddd-011:

========================================================================
Spreadsheets in Data Analysis
========================================================================

:bdg-primary:`🎯 Data-Driven Decisions` :bdg-secondary:`📗 Spreadsheets for Analysis` :bdg-info:`Lesson 011`

◀ :doc:`Previous <010-mathematical-thinking>` · :doc:`Next <012-building-and-organizing-a-spreadsheet>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


The spreadsheet's seat at the decision table
----------------------------------------------

The foundations introduced spreadsheets as a tool; this stage places them in
the **decision workflow**. For data-driven decisions of everyday scale — the
small-data tier of the previous lesson — the spreadsheet is frequently the
entire pipeline: where the data lands, where the metric is computed, and what
the stakeholder opens. Knowing *where in the process* it serves keeps its use
deliberate rather than habitual.

Phase by phase
----------------

Mapped onto the six phases, the spreadsheet contributes to each:

- **Ask** — a scratchpad for the decomposition: sketching *revenue =
  customers × frequency × value* in cells makes the question concrete.
- **Prepare** — the landing zone for extracts and collected data, where a
  first eyeball happens.
- **Process** — first-pass cleaning: spotting blanks, fixing obvious typos,
  standardising labels (the cleaning section deepens this greatly).
- **Analyze** — sorting, filtering, formulas, and pivot tables that turn rows
  into the comparison the decision needs.
- **Share** — a labelled summary table or chart any stakeholder can open, no
  special tools required.
- **Act** — often the humble tracker where the decision's follow-through is
  recorded, closing the loop.

Its comparative advantage
---------------------------

Against the rest of the toolkit, the spreadsheet's edge is **transparency plus
ubiquity**: every intermediate value is visible and clickable, and every
collaborator already has the tool. That makes it unbeatable for work that must
be *inspected* by non-analysts — a budget model a manager will poke at, a
shared tracker a team maintains. The corresponding costs are the familiar
ones: scale limits, silent overtyping, and no audit trail — which is why the
same workflow graduates to SQL and Python as data grows and repetition sets
in.

The caveat
------------

The spreadsheet's danger is precisely its convenience: analyses that deserved
a durable, reviewable pipeline live for years as ``final_v3_REAL.xlsx``. A
useful rule: the *first* pass at almost anything belongs in a spreadsheet; the
*tenth* pass belongs in something rerunnable. The next lessons make the first
pass excellent — starting with the layout decisions that determine whether a
sheet helps or fights you.

.. hint::

   - :doc:`The Role of Spreadsheets in Data Analysis and Basic Concepts <../1_foundations/021-the-role-of-spreadsheets-in-data-analysis-and-basic-concepts>`
   - :doc:`Building and Organizing a Spreadsheet <012-building-and-organizing-a-spreadsheet>`
   - :doc:`How Data Analysts Use Spreadsheets <013-how-data-analysts-use-spreadsheets>`
   - :doc:`Mathematical Thinking <010-mathematical-thinking>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/08/31/spreadsheets-in-data-analysis/ <https://insightful-data-lab.com/2023/08/31/spreadsheets-in-data-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: ddd, topic: spreadsheets

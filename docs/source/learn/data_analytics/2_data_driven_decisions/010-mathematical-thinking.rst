:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-ddd-010:

========================================================================
Mathematical Thinking
========================================================================

:bdg-primary:`🎯 Data-Driven Decisions` :bdg-secondary:`📐 Metrics & Dashboards` :bdg-info:`Lesson 010`

◀ :doc:`Previous <009-dashboards>` · :doc:`Next <011-spreadsheets-in-data-analysis>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Thinking in structure and size
--------------------------------

**Mathematical thinking** here does not mean advanced mathematics. It means
looking at a problem and **breaking it down step by step**, so relationships
and patterns become visible — the same decomposition habit as the technical
mindset, applied to quantities. Two everyday forms of it do most of the work
in analytics: decomposing calculations, and reasoning about size.

Decomposition: the arithmetic of questions
--------------------------------------------

Most business quantities are compositions of simpler ones, and writing the
composition down is often the analysis. *Revenue = customers × orders per
customer × average order value.* Revenue fell 12%? The decomposition converts
one mystery into three checks — which factor moved? *Cost per acquisition =
spend ÷ new customers*; *retention = remaining ÷ starting*. Each formula is a
small model of the business, and disagreements about "why the number moved"
become inspections of named terms instead of debates.

Orders of magnitude: the sanity check
---------------------------------------

The second form is estimating **roughly how big** things should be before
computing exactly. If the dashboard says yesterday's revenue was 4.2 million
and quick mental math says ~40 thousand customers × ~$10 average order ≈
$400k, something is off by 10× — a units error, a duplicated join, a decimal
slip. Rough estimation catches these instantly, which is why experienced
analysts approximate *first* and compute *second*. The habit also scopes
work: an effect worth at most $5k a year does not merit a month of analysis.

Matching data size to decision size
-------------------------------------

Mathematical thinking includes choosing the *scale* of data a decision needs.
**Small data** — a small number of specific metrics over a short period — is
effective for day-to-day decisions: this week's staffing, this campaign's
pacing. **Big data** — larger, broader, less immediately specific — serves
more substantial decisions: market strategy, long-term trends. Reaching for
big-data machinery to answer a small-data question wastes time; deciding a
strategic question from one week's numbers mistakes noise for signal.

The caveat
------------

Decompositions are models, and models simplify: *customers × frequency ×
value* hides that the three factors interact (a discount raises frequency and
lowers value at once). Use the decomposition to locate where to look, then
look at the real data — the formula points, the data answers. Next, the tool
in which most of this thinking first happens: the spreadsheet.

.. hint::

   - :doc:`Analytical Thinking and Its Core Components <../1_foundations/015-analytical-thinking-and-its-core-components>`
   - :doc:`The Difference Between Data and Metrics, and the Role of Metrics <008-the-difference-between-data-and-metrics-and-the-role-of-metrics>`
   - :doc:`Dashboards <009-dashboards>`
   - :doc:`Spreadsheets in Data Analysis <011-spreadsheets-in-data-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/08/31/mathematical-thinking/ <https://insightful-data-lab.com/2023/08/31/mathematical-thinking/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: ddd, topic: metrics

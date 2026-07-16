:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-ddd-002:

========================================================================
Understanding Common Problem Types in Data Analytics
========================================================================

:bdg-primary:`🎯 Data-Driven Decisions` :bdg-secondary:`🧭 Framing the Problem` :bdg-info:`Lesson 002`

◀ :doc:`Previous <001-using-data-analysis-to-choose-the-right-advertising-strategy>` · :doc:`Next <003-applying-data-analytics-problem-types-in-real-business-scenarios>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Six shapes of problems
------------------------

Analyst problems feel endlessly varied, but the standard framing observes that
they overwhelmingly take **six recurring shapes**. Recognising the shape early
matters: each type suggests its own data, methods, and outputs, so naming the
type is the first act of structuring the work.

The six types
---------------

- **Making predictions.** Using data to make informed decisions about how
  things may be in the future — the advertising-channel choice of the previous
  lesson. Past data cannot guarantee outcomes, only inform them.
- **Categorizing things.** Assigning items to groups based on common
  features: classifying customer-service calls by keywords or scores, tagging
  products into price tiers, grading suppliers.
- **Spotting something unusual.** Identifying data outside the norm: a smart
  watch flagging an abnormal heart rate, a sudden dip in daily sign-ups, one
  region's numbers that stopped moving (often a broken pipeline, not a calm
  market).
- **Identifying themes.** Taking categorisation a step further by grouping
  categories into **broader concepts**: hundreds of differently-worded survey
  comments becoming the themes "pricing", "reliability", "support tone".
- **Discovering connections.** Finding how different entities' problems
  relate, and combining data to solve them: a logistics firm analysing wait
  times at shipping hubs to change schedules and lift on-time deliveries.
- **Finding patterns.** Using historical data to understand what tends to
  happen and is therefore likely to happen again: maintenance records showing
  most machine failures occur when regular servicing slips past a certain
  window.

The pair people confuse
-------------------------

**Categorizing** versus **identifying themes**: categorising groups *the same
kind* of thing together (all calls scored 1–3); themes group *similar but
different* things under a broader concept (many distinct complaints, one
theme of "delivery frustration"). Categories are boxes; themes are the ideas
the boxes suggest.

Using the typology
--------------------

When a request lands, ask: *which type is this?* "Why did churn spike in
March?" is spotting something unusual, then finding patterns. "Which customers
should get the premium pitch?" is categorizing, in service of a prediction.
Many real projects **chain types** — unusual → pattern → prediction is the
classic investigation arc — and recognising the chain tells you the order of
work. The next lesson practises exactly this recognition on realistic
scenarios.

.. hint::

   - :doc:`Using Data Analysis to Choose the Right Advertising Strategy <001-using-data-analysis-to-choose-the-right-advertising-strategy>`
   - :doc:`Applying Data Analytics Problem Types in Real Business Scenarios <003-applying-data-analytics-problem-types-in-real-business-scenarios>`
   - :doc:`The Role of Business Tasks in Data Analysis <../1_foundations/025-the-role-of-business-tasks-in-data-analysis>`
   - :doc:`Analytical Thinking and Questions for Problem Solving <../1_foundations/016-analytical-thinking-and-questions-for-problem-solving>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/08/01/understanding-common-problem-types-in-data-analytics/ <https://insightful-data-lab.com/2023/08/01/understanding-common-problem-types-in-data-analytics/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: ddd, topic: framing

:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-viz-014:
.. _da-6-viz-viz-014:

========================================================================
Linking Multiple Datasets in Tableau Public
========================================================================

:bdg-primary:`🎨 Data Visualization` :bdg-secondary:`📊 Tableau` :bdg-info:`Lesson 014`

◀ :doc:`Previous <013-using-creativity-in-tableau>` · :doc:`Next <015-data-storytelling-giving-numbers-a-clear-and-convincing-voice>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Combining sources for richer views
------------------------------------

Real analysis often draws on several datasets, and Tableau can *combine* them —
linking multiple data sources so a single visualization draws on more than one table.
This lesson covers linking datasets in Tableau Public, applying the data-combining
concepts from the analysis section within the visualization tool.

How Tableau links data
------------------------

Tableau offers a few ways to combine data sources, mirroring the combining techniques
already learned:

- **Joins** — combining tables on a matching key, exactly the SQL JOIN from the
  analysis section, configured visually in Tableau. Tables are joined on a shared
  field (a key), producing a combined dataset with the join-type choices (inner, left,
  right, full) the JOIN lesson covered.
- **Relationships** — Tableau's flexible way to relate tables without a rigid
  up-front join, letting Tableau determine how to combine them per visualization.
- **Blending** — combining data from *different sources* (say a spreadsheet and a
  database) at the visualization level, aggregating each and linking on a common
  field.
- **Unions** — stacking tables with the same structure (appending rows), for
  combining like datasets (this month's and last month's data).

The concepts are the ones from the analysis section — joining on keys, combining
sources — now performed inside Tableau to feed richer visualizations.

Why link datasets
-------------------

Linking data lets a visualization draw on information spread across tables — sales
data joined to product details joined to regional information, visualized together.
Just as the analysis section combined tables to answer richer questions, linking
datasets in Tableau enables richer *visualizations*, showing relationships across data
that no single table holds. It brings the relational-combine power into the
visualization layer.

The caveat
------------

Combining data in Tableau carries exactly the hazards the analysis section flagged for
joins, now one step removed and thus easier to get wrong unnoticed. Joining on a
**non-unique key** multiplies rows and inflates the aggregates Tableau computes — the
fan-out problem, now hidden inside a chart where the wrong numbers look authoritative.
**Mismatched keys** silently drop data; **blending** aggregates before combining,
which can produce subtly different results than a join. The discipline is the same:
understand the relationship between the tables (one-to-one, one-to-many), verify that
combined visualizations show the row counts and totals you expect, and treat a chart
built on linked data with the same row-count skepticism as a SQL join. A visualization
of wrongly-combined data misleads with a confident, polished face. The next lessons
turn from building charts to telling stories with them.

.. hint::

   - :doc:`Using Creativity in Tableau <013-using-creativity-in-tableau>`
   - :doc:`Getting Started with Tableau Public <010-getting-started-with-tableau-public>`
   - :doc:`Effective vs. Ineffective Data Visualizations in Tableau <012-effective-vs-ineffective-data-visualizations-in-tableau>`
   - :doc:`Data Storytelling: Giving Numbers a Clear and Convincing Voice <015-data-storytelling-giving-numbers-a-clear-and-convincing-voice>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/linking-multiple-datasets-in-tableau-public/ <https://insightful-data-lab.com/2023/11/26/linking-multiple-datasets-in-tableau-public/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: viz, topic: tableau

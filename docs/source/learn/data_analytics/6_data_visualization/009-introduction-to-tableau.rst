:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-viz-009:
.. _da-6-viz-viz-009:

========================================================================
Introduction to Tableau
========================================================================

:bdg-primary:`🎨 Data Visualization` :bdg-secondary:`📊 Tableau` :bdg-info:`Lesson 009`

◀ :doc:`Previous <008-accessibility-in-data-visualization-designing-for-everyone>` · :doc:`Next <010-getting-started-with-tableau-public>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


A tool built for visualization
--------------------------------

While spreadsheets can chart and SQL can query, **Tableau** is a tool built
specifically for *visual analytics* — creating interactive visualizations and
dashboards from data, with little or no code. Opening the Tableau stage, this lesson
introduces what Tableau is, what it does, and why it is one of the most widely used
visualization tools in data analytics.

What Tableau is
----------------

Tableau is a visual-analytics platform that connects to data sources and lets you
build visualizations by *dragging and dropping* fields rather than writing code. Its
core idea is *direct manipulation*: you place a field on an axis, a colour, or a
filter, and Tableau draws the chart, letting you explore and refine visually. This
makes sophisticated, interactive visualization accessible to analysts who are not
programmers, which is much of its appeal.

What Tableau does well
------------------------

Tableau's strengths align with the visualization principles the section established:

- **Interactive visualizations** — the dynamic charts and dashboards from the
  static-versus-dynamic lesson, letting viewers filter, hover, and drill down.
- **Rapid exploration** — drag-and-drop makes trying different views fast, supporting
  the exploratory purpose of finding patterns.
- **Connecting to many data sources** — spreadsheets, databases (via SQL), and files,
  bringing data in without manual export where possible.
- **Dashboards** — combining multiple visualizations into a single interactive view
  for monitoring and self-service (a later lesson's subject).
- **Polished, shareable output** — professional visualizations suitable for
  communicating findings to stakeholders.

Where Tableau fits
--------------------

Tableau occupies the *visualization and dashboard* niche in the analyst's toolkit,
downstream of the data preparation and analysis the earlier sections covered.
Typically, data is prepared and analysed (in spreadsheets, SQL, or code), then Tableau
visualizes the results — it is a presentation-and-exploration layer, not primarily a
cleaning or heavy-computation tool. It complements rather than replaces the other
tools: SQL or a spreadsheet shapes the data, Tableau makes it visible and interactive.

The caveat
------------

Tableau is powerful but is one option among several, and it is not always the right
one. It is a specialised (and, in its full version, commercial) tool with its own
learning curve, and for a simple static chart a spreadsheet may be faster and more
universally accessible. Tableau also does not remove the need for the *principles* —
its ease of making charts makes it just as easy to make *bad* charts (the wrong type,
misleading scales, chartjunk), so the design judgement from the principles stage
matters as much in Tableau as anywhere. The tool accelerates visualization; it does
not substitute for knowing what makes a visualization good. The next lesson gets hands
on with the free version, Tableau Public.

.. hint::

   - :doc:`Data Visualization <001-data-visualization>`
   - :doc:`Getting Started with Tableau Public <010-getting-started-with-tableau-public>`
   - :doc:`Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity <004-static-vs-dynamic-data-visualizations-design-tradeoffs-control-and-interactivity>`
   - :doc:`Effective vs. Ineffective Data Visualizations in Tableau <012-effective-vs-ineffective-data-visualizations-in-tableau>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/introduction-to-tableau/ <https://insightful-data-lab.com/2023/11/26/introduction-to-tableau/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: viz, topic: tableau

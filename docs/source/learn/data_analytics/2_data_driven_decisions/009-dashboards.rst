:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-ddd-009:

========================================================================
Dashboards
========================================================================

:bdg-primary:`🎯 Data-Driven Decisions` :bdg-secondary:`📐 Metrics & Dashboards` :bdg-info:`Lesson 009`

◀ :doc:`Previous <008-the-difference-between-data-and-metrics-and-the-role-of-metrics>` · :doc:`Next <010-mathematical-thinking>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Metrics, made ambient
-----------------------

The previous lesson turned data into metrics; a **dashboard** is where metrics
live. The standard definition draws the key contrast: a dashboard **monitors
live, incoming data** from multiple datasets, organised in one central
location — while a **report** is a **static collection of data** delivered
periodically. A report is a photograph; a dashboard is a window.

What a dashboard is for
-------------------------

Dashboards serve the *ongoing* relationship with a metric: is retention
holding, are today's orders on pace, did the error rate move after the deploy?
Their value is threefold. **Currency** — the numbers are now, not last month.
**Centralisation** — the handful of metrics that matter, together, instead of
scattered across systems. **Shared truth** — everyone steering by the same
instruments, the workplace benefit from the foundations made literal.

Dashboard or report?
----------------------

The choice follows the decision's tempo. **Continuous decisions** (operations,
monitoring, campaigns in flight) want a dashboard: the question recurs, so the
answer should stand ready. **Periodic or one-off decisions** (the quarterly
review, the pricing study) want a report: a curated, stable snapshot with
narrative, where numbers do not shift under the reader mid-discussion. Teams
misfire in both directions — dashboards nobody opens standing in for analysis
that was never done, and hand-built weekly reports that a dashboard would
automate.

Designing one that earns its screen
-------------------------------------

Four habits separate working dashboards from decoration. **Few metrics,
chosen** — the handful from the metrics lesson, not everything measurable.
**Comparison built in** — each number against its target or history, since a
lone value is not a signal. **Hierarchy** — the decision-critical figure large
and first; supporting detail below. **A named audience** — one dashboard per
decision-making group beats one dashboard for everyone. (The visualization
section later covers the craft of the charts themselves; a dedicated lesson
there returns to dashboards as products.)

The caveat
------------

A dashboard shows *what* is happening, never *why* — it is a smoke detector,
not an investigation. Its glanceability also breeds false confidence: a
metric can be green while its definition has quietly rotted. Treat every
surprising dashboard movement as the start of an analysis, and audit the
definitions behind the tiles on a schedule, because screens age faster than
they look.

.. hint::

   - :doc:`The Difference Between Data and Metrics, and the Role of Metrics <008-the-difference-between-data-and-metrics-and-the-role-of-metrics>`
   - :doc:`Data Dashboards: Organizing Insight for Real-Time Decision Making <../6_data_visualization/017-data-dashboards-organizing-insight-for-real-time-decision-making>`
   - :doc:`The Role and Importance of Data Visualization <../1_foundations/023-the-role-and-importance-of-data-visualization>`
   - :doc:`Mathematical Thinking <010-mathematical-thinking>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/08/31/dashboards/ <https://insightful-data-lab.com/2023/08/31/dashboards/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: ddd, topic: metrics

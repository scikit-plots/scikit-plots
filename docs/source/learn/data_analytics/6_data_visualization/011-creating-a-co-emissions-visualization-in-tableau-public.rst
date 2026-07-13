:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-viz-011:
.. _data-analytics-viz-011:
.. _da-foundations-viz-011:
.. _da-decisions-viz-011:
.. _da-prep-viz-011:
.. _da-cleaning-viz-011:
.. _da-analyze-viz-011:
.. _da-viz-viz-011:
.. _da-python-viz-011:
.. _da-jobsearch-viz-011:

========================================================================
Creating a CO₂ Emissions Visualization in Tableau Public
========================================================================

:bdg-primary:`🎨 Data Visualization` :bdg-secondary:`📊 Tableau` :bdg-info:`Lesson 011`

◀ :doc:`Previous <010-getting-started-with-tableau-public>` · :doc:`Next <012-effective-vs-ineffective-data-visualizations-in-tableau>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


A worked example
-----------------

Principles and workflow come together in a concrete build. This lesson walks through
creating a visualization of **CO\u2082 emissions** data in Tableau Public — a
real-world environmental dataset — showing how the abstract steps become an actual
chart, and how the design principles guide the choices along the way.

The data and the question
---------------------------

An emissions dataset typically holds, per row, a place and time and an emissions
figure — for example country, year, and CO\u2082 emissions (in tonnes or per capita).
Before building, the *question* comes first (the purpose principle): are we showing
emissions *over time* (a trend), *across countries* (a comparison), or *by region on a
map* (a geographic pattern)? The question determines everything downstream — the chart
type, the fields used, the emphasis. Suppose the question is how emissions have changed
over time for a set of countries: a *trend comparison*.

Building the visualization
----------------------------

With the question set, the Tableau workflow builds the answer:

- **Connect** the emissions dataset (a CSV). Tableau classifies ``country`` and
  ``year`` as dimensions and ``emissions`` as a measure.
- **Build the view** — for an over-time trend, place ``year`` on columns and
  ``emissions`` on rows, producing a line; place ``country`` on colour to draw one line
  per country, enabling comparison. This is a multi-series line chart — the right
  structure for a trend comparison (position over time, the precisely-read encoding).
- **Refine** — apply the principles: a clear title stating the message, a limited and
  accessible colour palette (not colour alone — label the lines directly where
  possible), appropriate axis scaling that does not distort, and removal of chartjunk.
  Focus the chart on its one message: how these countries' emissions have diverged or
  converged over time.
- **Publish** to Tableau Public as an interactive chart viewers can hover and filter.

Principles in the concrete
----------------------------

The example makes the abstract principles tangible. The **purpose** (the emissions
question) drove the **chart choice** (a line for a trend); the **encoding** used the
eye's precise channel (position over time); **colour** distinguished countries but was
kept accessible and paired with labels; and **focus** kept the chart to its single
message. A CO\u2082 visualization built this way communicates a real environmental
insight clearly — the whole section's principles applied to one concrete, meaningful
dataset.

The caveat
------------

Emissions data — like much real-world data — carries interpretation hazards the
visualization must respect. *Total* versus *per-capita* emissions tell very different
stories (a populous country leads on total but may be low per person), and choosing
which to show is a framing decision with real consequences for what the chart implies;
showing one while implying the other misleads. Time ranges, the set of countries
included, and absolute-versus-relative framing similarly shape the message. The
honest-visualization obligation is acute for consequential public data like emissions:
the chart should represent what the data genuinely says, with its framing made clear,
not shade a complex picture toward a preferred narrative. The next lesson contrasts
effective and ineffective visualizations directly.

.. hint::

   - :doc:`Getting Started with Tableau Public <010-getting-started-with-tableau-public>`
   - :doc:`Introduction to Tableau <009-introduction-to-tableau>`
   - :doc:`Effective vs. Ineffective Data Visualizations in Tableau <012-effective-vs-ineffective-data-visualizations-in-tableau>`
   - :doc:`Choosing the Right Visualization: Audience-Centered Design and Chart Selection <006-choosing-the-right-visualization-audience-centered-design-and-chart-selection>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/creating-a-co%e2%82%82-emissions-visualization-in-tableau-public/ <https://insightful-data-lab.com/2023/11/26/creating-a-co%e2%82%82-emissions-visualization-in-tableau-public/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: viz, topic: tableau

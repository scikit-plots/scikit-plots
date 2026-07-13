:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-viz-006:
.. _data-analytics-viz-006:
.. _da-foundations-viz-006:
.. _da-decisions-viz-006:
.. _da-prep-viz-006:
.. _da-cleaning-viz-006:
.. _da-analyze-viz-006:
.. _da-viz-viz-006:
.. _da-python-viz-006:
.. _da-jobsearch-viz-006:

================================================================================
Choosing the Right Visualization: Audience-Centered Design and Chart Selection
================================================================================

:bdg-primary:`🎨 Data Visualization` :bdg-secondary:`🎨 Visualization Principles & Design` :bdg-info:`Lesson 006`

◀ :doc:`Previous <005-elements-of-art-in-data-visualization-line-shape-color-space-and-movement>` · :doc:`Next <007-design-thinking-in-data-visualization-a-user-centered-framework>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Matching chart to message
---------------------------

With the principles and elements established, a concrete question recurs constantly:
*which chart type* for this data and this message? **Choosing the right
visualization** is the skill of matching chart type to what the data is and what you
want to say — and doing so with the *audience* in mind. The wrong chart type can
obscure a clear finding; the right one makes it obvious.

Chart types and what they show
--------------------------------

Common chart types each suit particular data and messages:

- **Bar chart** — comparing quantities across categories. The eye reads bar lengths
  accurately, making bars the reliable default for "compare these amounts" (sales by
  region, counts by type).
- **Line chart** — showing change over time or a continuous trend. Position along a
  path reads clearly, so lines are the standard for time series.
- **Pie chart** — showing parts of a whole, *sparingly*. Because the eye reads angles
  and areas poorly, pie charts work only for a few slices where proportions are
  roughly comparable; a bar chart is usually clearer.
- **Scatter plot** — showing the relationship between two numeric variables.
  Position (the most precise channel) reveals correlation, clusters, and outliers.
- **Histogram** — showing the distribution of one numeric variable — its shape,
  centre, and spread.
- **Heat map** — showing magnitude across two dimensions via colour intensity, for a
  grid of values.

The rule: **let the data and the message choose the chart.** A comparison wants bars;
a trend wants a line; a relationship wants a scatter plot; a distribution wants a
histogram. Forcing data into the wrong chart type (a pie chart for a comparison, a
line for unrelated categories) fights the reader.

Audience-centered selection
-----------------------------

Chart choice also depends on the *audience*. A technical audience reads a scatter plot
or box plot fluently; a general audience may need a simpler bar or line chart. A
familiar chart type communicates faster than a sophisticated one the audience must
decode — the best chart is not the most advanced but the one *this* audience reads
most easily. Matching the chart to the audience's visual literacy is as important as
matching it to the data.

Selection as a decision
-------------------------

Choosing a chart is a *deliberate decision*, not a default. The process: identify what
the data *is* (categories, a time series, two numeric variables, a distribution),
identify the *message* (a comparison, a trend, a relationship, a shape), consider the
*audience* (what they read easily), and pick the chart type that fits all three. This
turns chart selection from habit ("always a bar chart") into a reasoned choice that
serves the specific data, message, and reader.

The caveat
------------

There is rarely a single "correct" chart — often several would work, and the choice
involves judgement and trade-offs (a stacked bar versus grouped bars, a line versus a
bar for few time points). The failure to avoid is not picking the *theoretically
optimal* chart but picking an *actively wrong* one — a pie chart with fifteen slices,
a 3D chart that distorts, a chart type that mismatches the data. Aim for a chart that
clearly and honestly conveys the message to the audience; among the several that do,
the differences are refinements. The next lesson brings a structured design process
to these choices.

.. hint::

   - :doc:`Connecting Data and Images <002-connecting-data-and-images>`
   - :doc:`Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement <005-elements-of-art-in-data-visualization-line-shape-color-space-and-movement>`
   - :doc:`Design Thinking in Data Visualization: A User-Centered Framework <007-design-thinking-in-data-visualization-a-user-centered-framework>`
   - :doc:`Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose <003-creating-powerful-data-visualizations-focus-structure-and-analytical-purpose>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/choosing-the-right-visualization-audience-centered-design-and-chart-selection/ <https://insightful-data-lab.com/2023/11/26/choosing-the-right-visualization-audience-centered-design-and-chart-selection/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: viz, topic: principles

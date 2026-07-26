:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-viz-002:
.. _da-6-viz-viz-002:

========================================================================
Connecting Data and Images
========================================================================

:bdg-primary:`🎨 Data Visualization` :bdg-secondary:`🎨 Visualization Principles & Design` :bdg-info:`Lesson 002`

◀ :doc:`Previous <001-data-visualization>` · :doc:`Next <003-creating-powerful-data-visualizations-focus-structure-and-analytical-purpose>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


From numbers to visual properties
-----------------------------------

A visualization works by *mapping* data to visual properties — turning a number into
a bar's height, a category into a colour, a pair of values into a point's position.
Understanding this mapping, the connection between **data and images**, is what lets
you read visualizations critically and build them deliberately, rather than treating
charts as magic.

Visual encodings
-----------------

Every visualization *encodes* data values as visual attributes. The common encodings:

- **Position** — where a mark sits (a point's place on an axis, a dot in a scatter
  plot). Position is the most precisely readable encoding — the eye judges position
  extremely well, which is why scatter plots and dot plots are so effective.
- **Length** — how long a mark is (a bar's height or length). Also read accurately,
  which is why bar charts are excellent for comparing quantities.
- **Angle and area** — as in a pie chart's slices. Read *less* accurately by the eye
  — people struggle to compare angles and areas precisely, which is a known weakness
  of pie charts.
- **Colour** — hue (different categories) or intensity (a value's magnitude, as in a
  heat map). Effective for categories and for showing a third dimension, but read
  less precisely for exact quantities.
- **Shape and size** — distinguishing categories or encoding a value in a mark's
  size.

The choice of encoding determines how *accurately and easily* the data can be read —
some encodings the eye judges precisely, others only roughly.

Why the encoding matters
--------------------------

Because encodings differ in how accurately they are read, the *same data* is more or
less legible depending on how it is encoded. A quantity comparison is clear as bar
lengths (accurately read) but muddy as pie slices (poorly read); a trend is clear as
position over time (a line) but lost in a table. Choosing an encoding the eye reads
well for the message you want is the foundation of effective visualization — a chart
that encodes its key comparison in a hard-to-read attribute fights its own reader.

Reading visualizations critically
------------------------------------

Understanding encodings also makes you a *critical* reader of charts. Knowing that
the eye reads position and length accurately but angle and area poorly, you can spot
when a chart's design obscures rather than reveals — a pie chart where a bar chart
would be clearer, a 3D effect that distorts areas, an encoding chosen for
decoration over legibility. The connection between data and image is where both good
design and misleading design live.

The caveat
------------

The mapping from data to image is where *distortion* enters, deliberately or
carelessly. The same encoding can mislead if its scale is manipulated (a bar chart
with a truncated axis exaggerates differences), if area is used to encode a value
people will read as one-dimensional (doubling a circle's radius quadruples its area,
overstating the value fourfold), or if colour implies an order that the data does not
have. Every visualization is a set of encoding *choices*, and those choices can
serve clarity or distort it — reading and making charts both require attention to
whether the encoding represents the data faithfully. The next lesson turns to what
makes a visualization genuinely powerful.

.. hint::

   - :doc:`Data Visualization <001-data-visualization>`
   - :doc:`Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose <003-creating-powerful-data-visualizations-focus-structure-and-analytical-purpose>`
   - :doc:`Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement <005-elements-of-art-in-data-visualization-line-shape-color-space-and-movement>`
   - :doc:`Choosing the Right Visualization: Audience-Centered Design and Chart Selection <006-choosing-the-right-visualization-audience-centered-design-and-chart-selection>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/connecting-data-and-images/ <https://insightful-data-lab.com/2023/11/26/connecting-data-and-images/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: viz, topic: principles

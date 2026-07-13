:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-viz-008:
.. _data-analytics-viz-008:
.. _da-foundations-viz-008:
.. _da-decisions-viz-008:
.. _da-prep-viz-008:
.. _da-cleaning-viz-008:
.. _da-analyze-viz-008:
.. _da-viz-viz-008:
.. _da-python-viz-008:
.. _da-jobsearch-viz-008:

========================================================================
Accessibility in Data Visualization: Designing for Everyone
========================================================================

:bdg-primary:`🎨 Data Visualization` :bdg-secondary:`🎨 Visualization Principles & Design` :bdg-info:`Lesson 008`

◀ :doc:`Previous <007-design-thinking-in-data-visualization-a-user-centered-framework>` · :doc:`Next <009-introduction-to-tableau>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Visualizations for everyone
-----------------------------

A visualization only communicates to those who can *perceive* it — and a meaningful
fraction of any audience perceives colour, contrast, or detail differently.
**Accessibility** in data visualization means designing charts that everyone can read,
including people with colour vision deficiency, low vision, or other differences.
Closing the principles stage, this lesson makes clear that accessible design is not a
niche concern but part of communicating to your *whole* audience.

Why accessibility matters
---------------------------

Roughly one in twelve men and one in two hundred women has some form of colour vision
deficiency, and many viewers have low vision or read charts on small or poor screens.
A visualization that relies on distinctions these viewers cannot perceive simply fails
for them — the insight does not land. Since the purpose of an explanatory
visualization is to communicate to an audience, a chart that excludes part of that
audience has partially failed at its one job. Accessibility is effectiveness for
everyone.

Practices for accessible visualization
-----------------------------------------

Concrete practices make charts widely readable:

- **Do not rely on colour alone.** Because colour vision varies, never encode
  meaning *only* in colour — add a second cue: labels, patterns, shapes, or direct
  text. If red-versus-green is the only difference between two lines, colour-blind
  viewers cannot tell them apart; adding labels or distinct markers fixes it.
- **Use colour-blind-friendly palettes.** Certain colour combinations (notably
  red/green) are problematic; palettes designed for colour vision deficiency (and
  distinguishable by brightness, not just hue) work for far more viewers.
- **Ensure sufficient contrast.** Text and marks must contrast enough with the
  background to be readable, especially for low-vision viewers and poor displays.
- **Label directly and clearly.** Direct labels on data (rather than a legend the
  viewer must cross-reference) and legible text sizes help everyone, and are
  essential for some.
- **Provide text alternatives.** A caption or description conveying the chart's key
  insight in words serves viewers using screen readers and reinforces the message for
  all.

Accessibility as good design
------------------------------

Accessible design generally *improves* a visualization for everyone, not only for
those who need it. Not relying on colour alone, ensuring contrast, and labelling
directly make a chart clearer for *all* viewers — the same practices that serve
colour-blind or low-vision readers reduce ambiguity and effort for everyone. Designing
for accessibility is, in large part, simply designing *well*: the constraints push
toward the clarity that good visualization aims at regardless.

The caveat
------------

Accessibility is a spectrum, not a checkbox — no single design serves every possible
need perfectly, and there are trade-offs (a palette optimised for one form of colour
vision deficiency may not be ideal for another). The goal is *reasonable* inclusion:
apply the well-established practices (not colour alone, adequate contrast, clear
labels, text alternatives) that serve the large majority, rather than either ignoring
accessibility or being paralysed by the impossibility of perfection. Some
accessibility is vastly better than none, and the core practices are low-cost and
broadly beneficial. This completes the principles of visualization; the next lessons
turn to a specific tool for building them — Tableau.

.. hint::

   - :doc:`Design Thinking in Data Visualization: A User-Centered Framework <007-design-thinking-in-data-visualization-a-user-centered-framework>`
   - :doc:`Choosing the Right Visualization: Audience-Centered Design and Chart Selection <006-choosing-the-right-visualization-audience-centered-design-and-chart-selection>`
   - :doc:`Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement <005-elements-of-art-in-data-visualization-line-shape-color-space-and-movement>`
   - :doc:`Introduction to Tableau <009-introduction-to-tableau>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/accessibility-in-data-visualization-designing-for-everyone/ <https://insightful-data-lab.com/2023/11/26/accessibility-in-data-visualization-designing-for-everyone/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: viz, topic: principles

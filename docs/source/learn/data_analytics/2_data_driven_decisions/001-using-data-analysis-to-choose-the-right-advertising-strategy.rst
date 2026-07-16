:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-ddd-001:

========================================================================
Using Data Analysis to Choose the Right Advertising Strategy
========================================================================

:bdg-primary:`🎯 Data-Driven Decisions` :bdg-secondary:`🧭 Framing the Problem` :bdg-info:`Lesson 001`

:doc:`Next <002-understanding-common-problem-types-in-data-analytics>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


A decision every business faces
---------------------------------

Where should the next advertising money go? Radio, social media, search ads,
billboards, a sponsorship? Every organisation that markets faces this choice,
and it is the canonical example of a **prediction problem**: use data about
what past campaigns produced to make an informed decision about what a future
one will produce. This section opens with it because it shows the entire
decision-making arc of the coming lessons in one familiar case.

What the data can say
-----------------------

Suppose the company has records of past campaigns: the **medium**, the
**location or audience targeted**, the **spend**, and the **number of new
customers acquired** in each campaign window. Analysis can then compare like
with like:

- cost per new customer, by medium — which channel acquires cheapest;
- performance by location or audience — where each medium works best;
- trend over time — whether a channel's performance is rising or fading.

A simple table of *cost per acquisition by channel and region* is often the
whole deliverable: it converts "which advertising do we believe in?" into
"which row of this table is smallest, and do we trust why?"

What the data cannot say
--------------------------

Past performance cannot **guarantee** future results — the honest phrasing is
that analysis helps *predict the best placement to reach the target audience*,
not that it certifies an outcome. Markets shift, competitors react, and a
channel that worked at small spend may saturate at large spend. Two
disciplines keep the prediction honest: state the assumption (next quarter
resembles last), and **instrument the decision** — run the chosen strategy in
a way that measures its result, so the next round of this same decision has
fresher evidence. That is the decision loop from the foundations, operating.

Why this case opens the section
---------------------------------

Notice what the analysis required *before* any computation: someone had to
frame the vague worry ("is our advertising working?") as a specific,
answerable, decision-attached question ("which channel acquires customers
cheapest, for whom, and is that stable?"). The rest of this stage teaches that
framing skill in general — the problem types that recur, and the questions
that unlock them.

.. hint::

   - :doc:`Understanding Common Problem Types in Data Analytics <002-understanding-common-problem-types-in-data-analytics>`
   - :doc:`Data-Driven Decision-Making <../1_foundations/003-data-driven-decision-making>`
   - :doc:`Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making <../1_foundations/019-case-studies-in-data-analysis-and-the-practical-impact-of-data-driven-decision-making>`
   - :doc:`The Relationship Between Data and Decision-Making <005-the-relationship-between-data-and-decision-making>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/08/01/using-data-analysis-to-choose-the-right-advertising-strategy/ <https://insightful-data-lab.com/2023/08/01/using-data-analysis-to-choose-the-right-advertising-strategy/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: ddd, topic: framing

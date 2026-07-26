.. _dpa-measuring-associations-in-data:

========================================================================
Measuring Associations in Data
========================================================================

**Stage 2 · 🔗 Associations & Correlation**  ·  Lesson 10 of 56  ·  *beginner*

:doc:`◀ Previous · Objective Selection of the Bin Width for a Time Histogram <09-objective-selection-of-the-bin-width-for-a-time-histogram>`   ·   :doc:`Next · Measuring Associations Between Two Continuous Variables ▶ <11-measuring-associations-between-two-continuous-variables>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

One idea, many measures
-------------------------

To move from "these two variables seem related" to a **number**, you need an **association measure** —
a single value capturing how strongly, and often in which direction, two variables move together.
There is no one measure for all cases; the right choice depends on **what kind of variables** you have.

It depends on the types
-------------------------

Variables come in two broad flavours — **continuous** (numbers on a scale, like fare or distance) and
**categorical** (labels, like payment type or company). The pairing decides the tool: comparing two
numbers is a different problem from comparing two labels, or a number against a label.

The taxonomy
--------------

The map for this stage:

* **continuous ↔ continuous** — **correlation** (Pearson, Spearman, Kendall);
* **categorical ↔ categorical** — the **chi-square** test and **Cramér's V**;
* **continuous ↔ categorical** — **ANOVA** and its effect size **eta-squared** (:math:`\eta^2`).

The lessons ahead take these in turn.

Strength and direction
------------------------

Two properties matter. **Strength** — how tightly the variables track, usually scaled so that 0 means
"no association" and 1 (or :math:`\pm 1`) means "perfect"; and **direction** — whether they rise
together or move oppositely, which only makes sense for **ordered** variables. A good measure reports
strength on a comparable scale, so associations across different variable pairs can be ranked.

.. hint::

   **Related lessons:** :doc:`Measuring Associations Between Two Continuous Variables <11-measuring-associations-between-two-continuous-variables>`  ·  :doc:`Correlation Coefficients in Python (Pearson, Spearman, Kendall) <12-correlation-coefficients-in-python-pearson-spearman-kendall>`  ·  :doc:`What Are Statistical Tests? <15-what-are-statistical-tests>`  ·  :doc:`Eta Squared (η²): Effect Size in ANOVA <16-eta-squared-2-effect-size-in-anova>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/measuring-associations-in-data/ <https://insightful-data-lab.com/2026/01/14/measuring-associations-in-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: beginner

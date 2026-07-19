.. _dpa-discovering-associations-through-data-from-everyday-patterns-to-chicago-taxi-trips-september-2022:

======================================================================================================
Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022)
======================================================================================================

**Stage 2 · 🔗 Associations & Correlation**  ·  Lesson 07 of 56  ·  *beginner*

:doc:`◀ Previous · IEEE 754 Floating-Point Standard <06-ieee-754-floating-point-standard>`   ·   :doc:`Next · Taxi Trips – 2022 dataset from the City of Chicago open data portal ▶ <08-taxi-trips-2022-dataset-from-the-city-of-chicago-open-data-portal>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Things that move together
---------------------------

Much of data analysis begins with a simple question: **do two things move together?** When ice-cream
sales rise with temperature, or umbrella sales with rainfall, we sense an **association** — a tendency
for two variables to vary in step. Making that intuition precise, and measuring how strong it is, is
the work of this stage.

Why it's useful
-----------------

Associations are useful in two ways. They **explain** — revealing which factors relate to an outcome
you care about — and they **predict** — if two variables move together, knowing one helps you guess
the other. Spotting the right associations is often the first real insight a dataset yields, and it
points to which variables are worth modelling later.

The running example
---------------------

This stage grounds the idea in a real dataset used throughout: the **City of Chicago taxi trips** from
September 2022. It is an intuitive place to look for associations — does the **fare** rise with the
**distance** travelled? with the **time** taken? The next lesson introduces the dataset in detail; for
now, the point is that a familiar, everyday relationship becomes a measurable one once it is in data.

Association is not cause
--------------------------

One caution to carry from the start: **association is not causation**. Two variables can move together
because one drives the other, because a third factor drives both, or by pure coincidence. Measuring a
strong association tells you the variables are **related**, not **why** — a distinction that matters
the moment anyone tries to act on the finding.

.. hint::

   **Related lessons:** :doc:`Taxi Trips – 2022 dataset from the City of Chicago open data portal <08-taxi-trips-2022-dataset-from-the-city-of-chicago-open-data-portal>`  ·  :doc:`Measuring Associations in Data <10-measuring-associations-in-data>`  ·  :doc:`Measuring Associations Between Two Continuous Variables <11-measuring-associations-between-two-continuous-variables>`  ·  :doc:`Correlation Coefficients in Python (Pearson, Spearman, Kendall) <12-correlation-coefficients-in-python-pearson-spearman-kendall>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/discovering-associations-through-data-from-everyday-patterns-to-chicago-taxi-trips-september-2022/ <https://insightful-data-lab.com/2026/01/14/discovering-associations-through-data-from-everyday-patterns-to-chicago-taxi-trips-september-2022/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: beginner

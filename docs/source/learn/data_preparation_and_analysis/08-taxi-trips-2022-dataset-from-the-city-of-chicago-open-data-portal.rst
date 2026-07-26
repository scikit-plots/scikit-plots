.. _dpa-taxi-trips-2022-dataset-from-the-city-of-chicago-open-data-portal:

========================================================================
Taxi Trips – 2022 dataset from the City of Chicago open data portal
========================================================================

**Stage 2 · 🔗 Associations & Correlation**  ·  Lesson 08 of 56  ·  *beginner*

:doc:`◀ Previous · Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022) <07-discovering-associations-through-data-from-everyday-patterns-to-chicago-taxi-trips-september-2022>`   ·   :doc:`Next · Objective Selection of the Bin Width for a Time Histogram ▶ <09-objective-selection-of-the-bin-width-for-a-time-histogram>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

An open, real dataset
-----------------------

The running dataset is the **Taxi Trips** file from the **City of Chicago open data portal** — a real,
public record of taxi journeys the city collects as a regulator. It is a favourite teaching set
because it is large, genuinely messy, and full of intuitive relationships to explore.

What's in a row
-----------------

Each **row is one trip**, with a unique ID and a timestamp, described by around two dozen fields. The
most useful for association work are the numeric ones — **Trip Miles** (distance), **Trip Seconds**
(duration), **Fare**, **Tips**, **Tolls** and **Trip Total** — alongside categorical fields like
**Payment Type** and **Company**, and location fields (**pickup / dropoff community area**, census
tract, and centroid latitude / longitude). Most columns load as floats or text.

Scale and loading
-------------------

The full record runs to **hundreds of millions** of trips over the years; a single week of September
2022 is already a workable subset of tens of thousands. It is published through the city's **Socrata**
portal, so you can download a CSV and read it with ``pandas.read_csv`` (or pull a filtered slice via
the Socrata API) rather than loading everything at once.

It needs cleaning
-------------------

Being real, it needs the **preparation** this course is about. Records appear with trip end **before**
start, absent durations, impossible distances (over 100 miles), fares below the city's base charge,
and missing community areas for trips outside Chicago. Filtering these out is a prerequisite before
any association or model is trustworthy — a concrete instance of why data preparation dominates the
workflow.

.. hint::

   **Related lessons:** :doc:`Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022) <07-discovering-associations-through-data-from-everyday-patterns-to-chicago-taxi-trips-september-2022>`  ·  :doc:`The First Step in Knowing Your Data <05-the-first-step-in-knowing-your-data>`  ·  :doc:`Measuring Associations Between Two Continuous Variables <11-measuring-associations-between-two-continuous-variables>`  ·  :doc:`Objective Selection of the Bin Width for a Time Histogram <09-objective-selection-of-the-bin-width-for-a-time-histogram>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/taxi-trips-2022-dataset-from-the-city-of-chicago-open-data-portal/ <https://insightful-data-lab.com/2026/01/14/taxi-trips-2022-dataset-from-the-city-of-chicago-open-data-portal/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: beginner

:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-017:
.. _da-3-prep-prep-017:

========================================================================
Metadata in Databases
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🗄️ Databases & Data Sources` :bdg-info:`Lesson 017`

◀ :doc:`Previous <016-databases-and-relational-database-concepts>` · :doc:`Next <018-metadata-repositories-and-data-governance>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Data about data
-----------------

A dataset alone is often not enough to use safely; you also need to know *what
it is* — where it came from, what each field means, when it was last updated.
That describing information is **metadata**: data about data. It is what turns an
anonymous grid of values into something an analyst can trust and interpret
correctly, and it is a first-class concern of the Prepare phase.

The kinds of metadata
-----------------------

Metadata is conventionally grouped into three kinds:

- **Descriptive metadata** — identifies and describes the data: a dataset's
  title, author, a column's meaning, keywords. It answers *what is this?* — the
  difference between a column called ``val`` and knowing it holds "monthly
  revenue in USD, net of refunds."
- **Structural metadata** — describes how the data is organised and how its
  parts relate: the tables, their columns and types, the keys linking them,
  the order and grouping of elements. It answers *how is this put together?*
- **Administrative metadata** — describes management and provenance: who created
  the data and when, how and when it was last updated, who may access it, and
  usage or licensing terms. It answers *where did this come from and who owns
  it?*

Together these supply exactly the **context** the bias lessons demanded —
metadata is context, made explicit and stored alongside the data.

Why metadata is indispensable
-------------------------------

Without metadata, data is ambiguous and dangerous. A column of numbers with no
description invites the wrong interpretation; a dataset with no update date
might be dangerously stale; data with no provenance cannot be evaluated for
ROCCC. Good metadata makes data **findable** (you can locate the right dataset),
**understandable** (you interpret fields correctly), **trustworthy** (you can
assess its source and currency), and **usable** (you know how it is structured
and what you may do with it). Much of the confusion and error in real analysis
traces to metadata that was missing, wrong, or ignored.

Metadata in the analyst's workflow
------------------------------------

Practically, metadata is the *first thing to read* about an unfamiliar dataset:
the data dictionary defining each column, the documentation of how it was
collected, the last-updated timestamp. Reading it first prevents a whole class
of downstream mistakes — averaging a field that turns out to be a code,
comparing two columns that measure subtly different things, trusting a stale
extract. And as a *producer*, leaving good metadata (clear column definitions,
provenance notes, update dates) is what makes your work usable by others and by
your future self — the chain-of-custody discipline from the foundations.

The caveat
------------

Metadata is only as good as its upkeep: documentation drifts out of date as data
changes, and stale or wrong metadata can mislead more confidently than none —
a data dictionary that no longer matches the table sends you wrong with an air
of authority. Treat metadata as valuable but verify it against the data when
stakes are high, and when you maintain data, keep its metadata current as part
of the work. The next lesson turns to where metadata is catalogued
organisation-wide, and the governance around it.

.. hint::

   - :doc:`Databases and Relational Database Concepts <016-databases-and-relational-database-concepts>`
   - :doc:`Metadata Repositories and Data Governance <018-metadata-repositories-and-data-governance>`
   - :doc:`Identifying Good Data Sources (ROCCC Framework) <011-identifying-good-data-sources-roccc-framework>`
   - :doc:`Structured Data and Data Models <004-structured-data-and-data-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/metadata-in-databases/ <https://insightful-data-lab.com/2023/09/04/metadata-in-databases/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: sources

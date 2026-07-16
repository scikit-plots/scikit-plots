:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-017:
.. _da-4-cleaning-cleaning-017:

========================================================================
Data Mapping and the Big Picture of Clean Data
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🧹 Dirty Data & Spreadsheet Cleaning` :bdg-info:`Lesson 017`

◀ :doc:`Previous <016-viewing-data-differently-for-more-effective-data-cleaning>` · :doc:`Next <018-introduction-to-sql>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Making sources speak the same language
----------------------------------------

When data comes from several sources, cleaning each one is not enough — the
sources must be made to *fit together*. **Data mapping** is the process of
matching fields from one data source to the corresponding fields in another, so
that data from different systems can be combined coherently. It is the "big
picture" of clean data: not just tidying values within a table, but ensuring the
tables *mean the same things* so a merge produces sense rather than nonsense.

What data mapping does
------------------------

Two systems rarely describe the same reality identically. One calls a field
``customer_name``, another ``client``; one stores state as ``"NY"``, another as
``"New York"``; one records dates as ``MM/DD/YYYY``, another as ISO. Data mapping
is the explicit specification of *which field corresponds to which*, and *how
their values translate*:

- **Field mapping** — ``source.client`` ↔ ``target.customer_name``: recording
  that these differently-named columns hold the same thing.
- **Value mapping** — ``"NY"`` ↔ ``"New York"``: recording how one source's codes
  translate to another's.
- **Format mapping** — ``MM/DD/YYYY`` → ISO dates: recording the transformation
  each field needs to align.

The map is a *plan* for integration, made before combining, so the merge is
deliberate rather than improvised.

Why the big picture matters
-----------------------------

Cleaning field-by-field without the big picture produces tidy tables that still
will not combine — each internally consistent, but mutually incompatible. Data
mapping is what prevents that: it forces you to see how all the pieces relate
*before* merging, catching mismatches (a field with no counterpart, two fields
that seem to match but mean subtly different things) while they are cheap to fix.
It is the difference between cleaning as isolated janitorial work and cleaning as
preparing a *coherent whole* ready for analysis — the same big-picture-first
discipline the foundations urged, applied to data integration.

Mapping and documentation
---------------------------

A data map is also **documentation** — a record of how sources relate that
outlives the immediate merge. When next month's data arrives in the same
sources, the map tells you exactly how to combine it again; when a colleague
inherits the work, the map explains how the pieces fit. This connects data
mapping to the metadata and chain-of-custody disciplines: the map is metadata
about *relationships between* datasets, and keeping it is what makes a
multi-source pipeline reproducible.

The caveat
------------

Data mapping surfaces a hard truth: sometimes fields that *look* like they
correspond do not quite — two "revenue" columns computed under different
definitions, two "customer" fields counting different populations. Mapping them
as if identical silently corrupts the merged data. The discipline is to map on
*meaning*, not just name similarity — verifying that matched fields genuinely
measure the same thing, using the metadata and definitions from the prep
section. This completes the spreadsheet-and-concepts side of cleaning; the next
lessons scale cleaning up to SQL, where these operations run on data far too
large for a sheet.

.. hint::

   - :doc:`Cleaning and Merging Multiple Datasets <013-cleaning-and-merging-multiple-datasets>`
   - :doc:`Viewing Data Differently for More Effective Data Cleaning <016-viewing-data-differently-for-more-effective-data-cleaning>`
   - :doc:`Structured Data and Data Models <../3_data_preparation/004-structured-data-and-data-models>`
   - :doc:`Introduction to SQL <018-introduction-to-sql>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/data-mapping-and-the-big-picture-of-clean-data/ <https://insightful-data-lab.com/2023/11/01/data-mapping-and-the-big-picture-of-clean-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: dirty

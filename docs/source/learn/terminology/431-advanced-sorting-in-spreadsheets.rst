:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-advanced-sorting-in-spreadsheets:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Advanced Sorting in Spreadsheets</b></div>`

==================================
Advanced Sorting in Spreadsheets
==================================

*Ordering and arranging tabular data, including with dynamic sort functions.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Advanced sorting** in spreadsheets means ordering data by more than a single column A-to-Z — **multi-level**
sorts that break ties across several keys, **custom** orders, and sorts by **color** or **format**. It goes
well beyond the one-click sort button.

How it works
------------

The core tool is a **multi-key** sort — order by one column, then by a second **within** ties, then a third —
plus **custom lists** (sorting Low, Medium, High in **logical** rather than alphabetical order), case-sensitive
sorting, and sorting **left-to-right** by rows. Excel's Sort dialog and Google Sheets both expose these, and
functions like **SORT / SORTBY** do it dynamically.

Why it matters
--------------

Sorting is a foundational step in **exploring** and **preparing** tabular data — grouping records, surfacing
extremes, and readying data for analysis. The key **caution**: always extend the sort to **all related
columns**, or you'll shuffle one field out of alignment with the rest and silently **corrupt** the rows.

----

*Theme:* :ref:`Data Preparation & Features <term-theme-features>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Encode (in Feature Engineering) <318-encode-in-feature-engineering>` · :doc:`Normalize (in Feature Engineering) <319-normalize-in-feature-engineering>` · :doc:`Sensitivity in Feature Engineering <317-sensitivity-in-feature-engineering>` · :doc:`Outlier <307-outlier>` · :doc:`Correlation <305-correlation>` · :doc:`Median <315-median>`

----

.. hint::
   **More in Data Preparation & Features**

   :doc:`Encode (in Feature Engineering) <318-encode-in-feature-engineering>` · :doc:`Normalize (in Feature Engineering) <319-normalize-in-feature-engineering>` · :doc:`Sensitivity in Feature Engineering <317-sensitivity-in-feature-engineering>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Advanced Sorting in Spreadsheets <https://insightful-data-lab.com/2023/11/26/advanced-sorting-in-spreadsheets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate

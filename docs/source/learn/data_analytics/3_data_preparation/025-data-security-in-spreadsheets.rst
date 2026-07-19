:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-025:
.. _da-3-prep-prep-025:

========================================================================
Data Security in Spreadsheets
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🔢 Spreadsheets, SQL & Organization` :bdg-info:`Lesson 025`

◀ :doc:`Previous <024-organizing-data-for-personal-and-work-projects>` · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Protecting the data you hold
------------------------------

Preparing and organising data comes with an obligation to **protect** it —
especially when it is shared, and doubly so when it contains the personal
information the ethics and privacy lessons covered. Spreadsheets are among the
most shared data tools, which makes their **security** features worth knowing:
they are the practical controls that keep shared data from being seen by the
wrong people or changed by accident. This lesson closes the section, and the
preparation phase, on that protective note.

Why spreadsheet security matters
----------------------------------

Spreadsheets are easily shared, copied, and emailed — which is their
convenience and their risk. A sheet with customer data forwarded to the wrong
recipient is a privacy breach; a shared budget model where anyone can overtype a
formula is an accident waiting to happen; a public share link on sensitive data
exposes it to anyone who finds the URL. Security features exist to manage exactly
these risks.

The core protections
----------------------

- **Access control (sharing permissions).** The first line of defence: control
  *who* can open a sheet and *what* they can do — view only, comment, or edit.
  Grant the minimum access each person needs (the *minimise* principle from
  privacy), and prefer sharing with named people over open "anyone with the
  link" access for sensitive data.
- **Protected ranges and sheets.** Lock specific cells, ranges, or whole tabs so
  they cannot be edited, while leaving others open. This guards formulas and
  reference data from accidental change — the antidote to the silently-overtyped
  cell from the spreadsheet-errors lesson — while still letting collaborators
  enter data where intended.
- **Hiding versus protecting.** Hiding a sheet or column removes it from view but
  does **not** secure it — anyone can unhide it. Hiding is for tidiness;
  protection and permissions are for security. Confusing the two is a common and
  dangerous mistake.
- **Careful sharing of sensitive data.** For personal or confidential data,
  consider whether a spreadsheet is the right vessel at all, remove or mask
  identifiers where possible (aggregation and anonymisation from the privacy
  lesson), and be deliberate about link sharing and download permissions.

Security as part of preparation
---------------------------------

Security is not a separate concern bolted on at the end — it is part of handling
data responsibly throughout. Setting appropriate permissions when you share,
protecting the cells that must not change, and being careful with sensitive data
are habits that belong in every project, and they connect directly to the ethics
and privacy obligations that opened this section: protecting people's data is
both an ethical duty and a practical safeguard against costly mistakes.

The caveat
------------

Spreadsheet security has real limits: protected ranges deter accidents but are
not strong security against a determined actor, and once someone can *view*
data they can usually copy it, screenshot it, or download it regardless of other
restrictions. For genuinely sensitive data, spreadsheet features are a layer, not
a fortress — proper databases with real access controls, and organisational data
governance, are the stronger protections. Match the protection to the
sensitivity, and never assume a spreadsheet's controls make truly confidential
data safe to share widely.

This completes the Data Preparation section. You have moved from where data comes
from, through its types, structure, bias, ethics, and sources, to accessing,
querying, organising, and securing it. The data is now understood and in hand —
and the next section confronts the reality that it is almost never clean.

.. hint::

   - :doc:`Data Privacy in Data Ethics <014-data-privacy-in-data-ethics>`
   - :doc:`Organizing Data for Personal and Work Projects <024-organizing-data-for-personal-and-work-projects>`
   - :doc:`Building and Organizing a Spreadsheet <../2_data_driven_decisions/012-building-and-organizing-a-spreadsheet>`
   - :doc:`Data Ethics in Data Analysis <013-data-ethics-in-data-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/15236/ <https://insightful-data-lab.com/2023/09/04/15236/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: spreadsheets_sql

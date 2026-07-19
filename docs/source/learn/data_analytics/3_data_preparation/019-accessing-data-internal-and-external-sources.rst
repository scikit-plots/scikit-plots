:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-019:
.. _da-3-prep-prep-019:

========================================================================
Accessing Data: Internal and External Sources
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🗄️ Databases & Data Sources` :bdg-info:`Lesson 019`

◀ :doc:`Previous <018-metadata-repositories-and-data-governance>` · :doc:`Next <020-importing-data-into-spreadsheets>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Getting to the data
----------------------

Knowing what data you need is one thing; *getting* it is another. Data an
analyst uses comes from two broad places — **inside** the organisation and
**outside** it — and each has its own access routes, trust characteristics, and
pitfalls. Recognising which kind you are dealing with shapes how you obtain it
and how much you can trust it.

Internal (first-party) data
-----------------------------

**Internal data** is data the organisation collected and holds itself — the
first-party data from earlier. It lives in the organisation's own systems:

- **Databases** — the relational systems from the previous stage, queried with
  SQL.
- **Data warehouses** — large stores consolidating data from many internal
  systems for analysis.
- **Application and operational systems** — the CRM, the sales platform, the
  support tool, each holding its own records.
- **Internal files** — spreadsheets, exports, and documents on shared drives.

Internal data's advantages are relevance and trust: it is about your own
organisation, and you can (in principle) understand and verify how it was
collected. Its challenges are practical — knowing it exists (the metadata
repository's job), getting *access* to it (permissions and data owners), and its
frequent scattering across disconnected systems.

External (second- and third-party) data
------------------------------------------

**External data** comes from outside the organisation — the second- and
third-party data from earlier:

- **Public and open data** — government statistics, open datasets, public
  research (the open-data lesson's territory).
- **Purchased data** — datasets bought from vendors and data providers.
- **Partner data** — shared directly by another organisation.
- **APIs** — programmatic interfaces that deliver external data on request,
  often as JSON (the semi-structured format from earlier).

External data extends what internal data alone can answer — market context,
benchmarks, demographics — but demands more scrutiny: you did not collect it, so
its provenance, quality, and currency must be evaluated with full ROCCC rigour,
and its licensing and privacy terms must be respected.

Choosing and combining sources
--------------------------------

Real analysis often *combines* both: internal sales data enriched with external
demographic or economic data, joined on a common key. The internal data grounds
the analysis in your own reality; the external data supplies context your own
systems do not hold. The judgement is matching source to question — and
remembering that combining sources multiplies not only insight but also the
provenance, quality, and privacy questions you must answer for *each* source.

The caveat
------------

Access is also a *permission and ethics* matter, not merely a technical one.
Just because data is reachable does not mean you are authorised to use it, or
that using it is ethical — internal data carries access controls for reasons,
and external data carries licences and privacy obligations. Confirm you are
*permitted* to use data for your purpose, not just *able* to obtain it — the
governance and ethics threads from the previous lessons, applied at the moment
of access. The next lesson gets concrete about pulling external data into the
analyst's most common tool: the spreadsheet.

.. hint::

   - :doc:`How Data Is Generated and Collected <001-how-data-is-generated-and-collected>`
   - :doc:`Importing Data into Spreadsheets <020-importing-data-into-spreadsheets>`
   - :doc:`Metadata Repositories and Data Governance <018-metadata-repositories-and-data-governance>`
   - :doc:`Identifying Good Data Sources (ROCCC Framework) <011-identifying-good-data-sources-roccc-framework>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/accessing-data-internal-and-external-sources/ <https://insightful-data-lab.com/2023/09/04/accessing-data-internal-and-external-sources/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: sources

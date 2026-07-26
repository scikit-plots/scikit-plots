:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-018:
.. _da-3-prep-prep-018:

========================================================================
Metadata Repositories and Data Governance
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🗄️ Databases & Data Sources` :bdg-info:`Lesson 018`

◀ :doc:`Previous <017-metadata-in-databases>` · :doc:`Next <019-accessing-data-internal-and-external-sources>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Managing metadata at scale
----------------------------

One dataset's metadata can live in a README; an organisation's cannot. As data
multiplies across many systems, the metadata describing it needs its own
organised home — a **metadata repository** — and the whole practice of keeping
data trustworthy and well-managed becomes **data governance**. Both matter to
analysts because they determine whether the data you need can be *found*,
*understood*, and *trusted* across an organisation.

Metadata repositories
------------------------

A **metadata repository** is a central store that describes an organisation's
data — a catalogue of what datasets exist, where they live, what their fields
mean, how they relate, and how current and trustworthy they are. Sometimes
called a data catalogue, it is the "card catalogue" for an organisation's data:
instead of asking around to discover whether some dataset exists and what its
columns mean, an analyst consults the repository. It brings the descriptive,
structural, and administrative metadata of the previous lesson together in one
searchable place, and it is what makes data *discoverable* at organisational
scale rather than tribal knowledge held by whoever has been there longest.

Data governance
-----------------

**Data governance** is the set of processes, roles, policies, and standards that
ensure an organisation's data is accurate, secure, usable, and handled
responsibly throughout its life cycle. It is the management framework around the
data-life-cycle stages from the foundations, and it typically covers:

- **Quality standards** — definitions and rules for what "correct" data looks
  like, so the same term means the same thing everywhere (the reconciliation
  problem from the life-cycle lessons, solved at the policy level).
- **Access and security** — who may see and change what data, enforcing the
  privacy and security obligations from earlier.
- **Ownership and stewardship** — named people (data owners, data stewards)
  accountable for specific datasets, so responsibility is not diffuse.
- **Compliance** — ensuring data handling meets legal and regulatory
  requirements, especially for personal data.

A common governance role is the **data steward** — someone responsible for the
quality, definitions, and proper use of a particular data domain.

Why analysts should care
--------------------------

Governance and repositories shape an analyst's daily reality. Good governance
means the data you pull is defined consistently, its quality is known, and you
can find it via the catalogue — enormous time saved and errors avoided. Weak
governance means the opposite: undocumented datasets, the same metric computed
three incompatible ways, and hours lost discovering what data even exists.
Analysts are also governance *participants* — following its standards, using
agreed definitions, and often helping improve the metadata and documentation
they consume.

The caveat
------------

Governance is a means, not an end — and it can fail in both directions.
Too little, and data becomes an untrustworthy free-for-all; too much, and
bureaucratic process strangles the agility analysis needs, with catalogues so
cumbersome nobody maintains them. The goal is *enough* governance to make data
trustworthy and findable without making it inaccessible — a balance every
data-mature organisation is perpetually tuning. This closes the sources stage;
the next turns from finding and describing data to actually *getting* it, and
bringing it into the tools where analysis happens.

.. hint::

   - :doc:`Metadata in Databases <017-metadata-in-databases>`
   - :doc:`Data Ethics in Data Analysis <013-data-ethics-in-data-analysis>`
   - :doc:`Accessing Data: Internal and External Sources <019-accessing-data-internal-and-external-sources>`
   - :doc:`Databases and Relational Database Concepts <016-databases-and-relational-database-concepts>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/metadata-repositories-and-data-governance/ <https://insightful-data-lab.com/2023/09/04/metadata-repositories-and-data-governance/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: sources

:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-014:
.. _da-3-prep-prep-014:

========================================================================
Data Privacy in Data Ethics
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`⚖️ Bias & Data Ethics` :bdg-info:`Lesson 014`

◀ :doc:`Previous <013-data-ethics-in-data-analysis>` · :doc:`Next <015-open-data-and-openness-in-data-ethics>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Privacy as protection
-----------------------

**Data privacy** is the aspect of data ethics concerned with protecting a
person's information — the right to control how their personal data is
collected, used, shared, and retained. Where ownership and consent establish
*whose* data it is and *whether* it may be used, privacy governs *how it is
protected* once entrusted, and it is often the most legally regulated part of
an analyst's obligations.

What privacy requires
-----------------------

Several concrete duties follow from the privacy principle:

- **Protect personal data** — secure it against unauthorised access, with the
  access controls, encryption, and careful handling the data-security lesson
  addresses. A breach is a privacy failure regardless of intent.
- **Limit use to purpose** — use personal data only for the purpose it was
  collected and consented for. Repurposing data for a new use people never
  agreed to violates privacy even when the data is held securely.
- **Minimise** — collect and retain only the personal data actually needed, and
  no longer than needed. Data you do not hold cannot be breached or misused; the
  life cycle's *destroy* stage is a privacy control.
- **Guard identity** — recognise that combining datasets or retaining detail can
  make supposedly anonymous data re-identifiable.

Personally identifiable information (PII)
-------------------------------------------

The heart of privacy is **PII** — information that can identify a specific
individual, directly (name, government ID, email) or indirectly (a combination
of attributes that together single someone out). Analysts handle PII with
particular care: accessing it only when necessary, and preferring to work with
data from which identifiers have been removed.

**Anonymisation** (removing identifying information so individuals cannot be
recognised) is the standard protective technique — but it is harder than it
looks. The recurring lesson from real re-identification cases is that stripping
obvious identifiers is often *insufficient*: a combination of "anonymous"
attributes — a postal code, a birth date, a gender — can uniquely identify a
person even with names removed. Genuine anonymisation must consider what the
remaining fields, and any dataset they could be joined with, reveal in
combination.

Privacy in the analyst's workflow
-----------------------------------

Practically: access the minimum PII the task requires; work with anonymised or
aggregated data by default; never join datasets in ways that re-identify people
without authorisation; and treat "could this analysis expose an individual?" as
a question asked *before* running it. Aggregation — reporting groups rather than
individuals — is a workhorse privacy technique, though even aggregates can leak
when a group is small enough to point at one person.

The caveat
------------

Privacy exists in tension with analytical usefulness: the same detail that makes
data powerful can make it privacy-invasive, and stronger protection often means
coarser data. There is no universal setting — the right balance depends on the
data's sensitivity, the purpose, applicable law, and what people consented to.
The obligation is to weigh usefulness against protection deliberately, default
toward protecting people, and know the legal requirements that apply. The next
lesson turns to the opposite impulse: data that *should* be open.

.. hint::

   - :doc:`Data Ethics in Data Analysis <013-data-ethics-in-data-analysis>`
   - :doc:`Open Data and Openness in Data Ethics <015-open-data-and-openness-in-data-ethics>`
   - :doc:`Data Security in Spreadsheets <025-data-security-in-spreadsheets>`
   - :doc:`Context and Bias in Data Analysis <../2_data_driven_decisions/018-context-and-bias-in-data-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/data-privacy-in-data-ethics/ <https://insightful-data-lab.com/2023/09/04/data-privacy-in-data-ethics/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: bias_ethics

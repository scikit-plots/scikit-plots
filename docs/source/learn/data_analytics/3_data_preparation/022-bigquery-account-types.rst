:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-022:
.. _data-analytics-prep-022:
.. _da-foundations-prep-022:
.. _da-decisions-prep-022:
.. _da-prep-prep-022:
.. _da-cleaning-prep-022:
.. _da-analyze-prep-022:
.. _da-viz-prep-022:
.. _da-python-prep-022:
.. _da-jobsearch-prep-022:

========================================================================
BigQuery Account Types
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🔢 Spreadsheets, SQL & Organization` :bdg-info:`Lesson 022`

◀ :doc:`Previous <021-sorting-and-filtering-data-in-spreadsheets>` · :doc:`Next <023-querying-data-with-sql>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


A place to practise SQL at scale
----------------------------------

To practise SQL on realistic, large datasets, you need access to a database
system — and a widely used, beginner-friendly option is **Google BigQuery**, a
cloud-based, serverless data warehouse that runs SQL queries over large datasets
without any local setup. Because BigQuery is a paid cloud service, understanding
its **account types** matters: they determine how you get started and whether you
risk any charge while learning.

The account types
-------------------

- **Sandbox** — the entry point for learners. The BigQuery **sandbox** lets you
  use BigQuery *without providing a credit card or creating a billing account*.
  It grants the same free-tier usage limits — around **1 TB of query processing
  and 10 GB of storage per month** — and lets you query BigQuery's library of
  **public datasets** immediately with just a Google account. Its main
  restriction is that tables automatically **expire after 60 days**, and some
  features (streaming inserts, DML statements, the Data Transfer Service) are
  unavailable. For learning SQL, it is ideal: zero cost, zero billing risk.
- **Free tier** — available to any account that *has* set up a billing account.
  It offers the same monthly free allowances (1 TB queries, 10 GB storage) but,
  because billing is enabled, allows **permanent storage** (no 60-day expiration)
  and the full feature set — with charges beginning only if you exceed the free
  limits.
- **Paid (full) account** — a billing account with no reliance on the free
  allowances, for production and heavier workloads; you pay for query processing
  and storage beyond the free tier.

Separately, new Google Cloud customers are typically offered a **$300 credit**
across Google Cloud products, which requires a credit card and is distinct from
the no-card sandbox.

Which to use, and why it matters
----------------------------------

For following this course and practising SQL, the **sandbox** is the right
choice: it removes the single biggest barrier for beginners — the fear of an
unexpected bill — while giving genuine access to the same query engine and real
public datasets professionals use. The 60-day expiration and monthly limits are
generous for learning and function as helpful guardrails toward efficient
queries. You can upgrade later, by attaching billing, if a real project needs
permanent storage or more capacity.

Why a cloud warehouse at all
------------------------------

BigQuery illustrates where much modern data lives: not on a laptop but in a
**cloud data warehouse** that scales to enormous datasets and is queried with
standard SQL. Learning on it means the SQL you practise is the SQL you would use
professionally, on infrastructure of the kind employers actually run — the
"learn the logic on real tools" principle from the foundations.

The caveat
------------

Cloud service details — limits, tiers, interface, even the exact free
allowances — change over time and vary by provider, so treat specific numbers as
current-at-writing rather than permanent, and check the provider's documentation
for the latest. BigQuery is one convenient environment among several (other cloud
warehouses and local databases work too); what transfers is the **SQL and the
warehouse concept**, not the particular vendor's current packaging. The next
lesson puts the environment to use with actual queries.

.. hint::

   - :doc:`Databases and Relational Database Concepts <016-databases-and-relational-database-concepts>`
   - :doc:`Querying Data with SQL <023-querying-data-with-sql>`
   - :doc:`Introduction to SQL <../4_data_cleaning_preparation/018-introduction-to-sql>`
   - :doc:`Accessing Data: Internal and External Sources <019-accessing-data-internal-and-external-sources>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/bigquery-account-types/ <https://insightful-data-lab.com/2023/09/04/bigquery-account-types/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: spreadsheets_sql

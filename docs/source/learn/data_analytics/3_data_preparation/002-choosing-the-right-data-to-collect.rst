:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-002:
.. _da-3-prep-prep-002:

========================================================================
Choosing the Right Data to Collect
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🧬 Data Types & Structure` :bdg-info:`Lesson 002`

◀ :doc:`Previous <001-how-data-is-generated-and-collected>` · :doc:`Next <003-understanding-data-types-and-data-formats>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Not all data is worth collecting
----------------------------------

Once you know where data can come from, the next decision is *which* to collect
or acquire. More data is not automatically better: the right data answers the
question at acceptable cost, while the wrong data — however voluminous — wastes
storage, effort, and attention, and can actively mislead. Choosing well is a
core Prepare-phase judgement, and it flows directly from the question the
framing section taught you to sharpen.

What "the right data" means
-----------------------------

Three properties decide whether candidate data is worth having:

- **Relevance** — does it actually bear on the question? Data that does not
  inform the decision is clutter, no matter how interesting or easy to obtain.
- **Coverage** — does it represent the whole population or period the question
  concerns? Data that covers only part (one region, one season, one channel)
  answers only a partial question, and mistaking it for the whole is how
  sampling bias enters.
- **Quality and granularity** — is it accurate, and recorded at a fine enough
  level to answer the question? Monthly totals cannot answer a question about
  daily patterns; a coarse category cannot answer one about specifics.

Balanced against these benefits is **cost**: collection effort, money, time,
and — for personal data — privacy and ethical obligation. The right choice
maximises relevance and coverage against what the decision can justify
spending.

Sample versus census
----------------------

A recurring choice is whether to collect *everything* (a census) or a
representative *sample*. A well-chosen sample is often faster, cheaper, and
entirely sufficient — the whole logic of sampling is that a representative part
can answer questions about the whole. The critical word is *representative*:
a large but skewed sample is worse than a small balanced one, because size
lends false confidence to a biased picture. (The cleaning section's sampling
lessons develop this with the mathematics.)

Choosing in practice
----------------------

Work backward from the question. State what an answer requires, list the data
that would supply it, then check each candidate for relevance, coverage, and
affordable quality. This backward pass routinely reveals two things: that some
eagerly-collected data is irrelevant to the actual decision, and that some
genuinely needed data is missing and must be collected or acquired. Better to
learn both at the Prepare stage than after weeks of analysis.

The caveat
------------

Choosing data introduces the analyst's own judgement — and therefore the
analyst's own bias — into the dataset before analysis begins. Deciding what to
collect is deciding what the analysis *can* see; excluding a source excludes a
possible finding. The discipline is to choose deliberately and *document the
choice*, so the boundaries of the data are visible to anyone who later relies
on the conclusion.

.. hint::

   - :doc:`How Data Is Generated and Collected <001-how-data-is-generated-and-collected>`
   - :doc:`Understanding Data Types and Data Formats <003-understanding-data-types-and-data-formats>`
   - :doc:`Understanding Bias in Data Analysis <008-understanding-bias-in-data-analysis>`
   - :doc:`The Relationship Between Data and Decision-Making <../2_data_driven_decisions/005-the-relationship-between-data-and-decision-making>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/choosing-the-right-data-to-collect/ <https://insightful-data-lab.com/2023/09/04/choosing-the-right-data-to-collect/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: types

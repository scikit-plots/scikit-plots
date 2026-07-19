:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-011:
.. _da-3-prep-prep-011:

========================================================================
Identifying Good Data Sources (ROCCC Framework)
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`⚖️ Bias & Data Ethics` :bdg-info:`Lesson 011`

◀ :doc:`Previous <010-common-types-of-data-bias>` · :doc:`Next <012-identifying-bad-data-sources-when-data-does-not-roccc>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


A checklist for trust
-----------------------

Before building on a dataset, an analyst must judge whether the *source* is
trustworthy — and a memorable checklist captures what "trustworthy" means. Good
data sources are **ROCCC**: **R**eliable, **O**riginal, **C**omprehensive,
**C**urrent, and **C**ited. Running a candidate source through the five letters
turns a vague sense of trust into specific, checkable questions.

The five criteria
-------------------

- **Reliable** — is the data accurate, complete, and unbiased, from a source
  with a track record of getting things right? Reliable data comes from
  well-regarded, vetted origins, not anonymous or casual ones.
- **Original** — can you get to the *source* of the data, rather than a copy of
  a copy? Original data comes from the party that actually collected it; each
  hand it passes through is a chance for distortion, so validating against the
  origin matters. (This echoes the first/second/third-party gradient from
  earlier.)
- **Comprehensive** — does the data contain all the information you need to
  answer the question? A source missing critical fields or covering only part
  of the population, however accurate on what it has, cannot fully answer the
  question.
- **Current** — is the data recent enough to be relevant? Data ages; a source
  from years ago may no longer reflect present reality, and the right recency
  depends on how fast the subject changes.
- **Cited** — is the source of the data documented — who created it, when,
  where, and how? Cited data can be traced and vetted; uncited data asks you to
  trust it blindly.

A source that satisfies all five is a strong foundation; the more it satisfies,
the more weight your conclusions can bear.

Using ROCCC in practice
-------------------------

ROCCC is a pre-flight check for the Prepare phase: before committing to a
dataset, walk the five letters and note where it is strong and weak. A source
need not be perfect on every criterion — few are — but the framework makes the
trade-offs *explicit*. "This data is comprehensive and current but not fully
original, since it's a third-party aggregation" is a clear-eyed basis for
proceeding with appropriate caution, and for stating that caution in the final
analysis.

Why a checklist beats a gut feeling
-------------------------------------

The value of ROCCC is that it makes source evaluation *systematic* rather than
intuitive. Under time pressure, it is tempting to grab whatever data is at hand
and start analysing — and the resulting problems (unreliable, stale, or
uncited data) surface only much later, after conclusions have been built on
sand. Five questions at the start are cheap insurance against that.

The caveat
------------

ROCCC evaluates the *source*, not the *fit to your specific question*. Data can
be impeccably reliable, original, comprehensive, current, and cited — and still
be the wrong data for your problem, or carry sampling bias the framework does
not directly test. Use ROCCC together with the relevance and coverage judgements
from the choosing-data lesson: source quality and question-fit are separate
checks, and a good analysis passes both. The next lesson applies ROCCC in
reverse, to recognise the sources you should *not* trust.

.. hint::

   - :doc:`Identifying Bad Data Sources (When Data Does Not ROCCC) <012-identifying-bad-data-sources-when-data-does-not-roccc>`
   - :doc:`Common Types of Data Bias <010-common-types-of-data-bias>`
   - :doc:`How Data Is Generated and Collected <001-how-data-is-generated-and-collected>`
   - :doc:`Accessing Data: Internal and External Sources <019-accessing-data-internal-and-external-sources>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/identifying-good-data-sources-roccc-framework/ <https://insightful-data-lab.com/2023/09/04/identifying-good-data-sources-roccc-framework/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: bias_ethics

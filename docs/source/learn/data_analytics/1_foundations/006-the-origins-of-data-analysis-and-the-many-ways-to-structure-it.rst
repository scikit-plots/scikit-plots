:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-foundations-006:
.. _da-1-foundations-foundations-006:

========================================================================
The Origins of Data Analysis and the Many Ways to Structure It
========================================================================

:bdg-primary:`🌱 Foundations` :bdg-secondary:`🔄 The Analysis Process & Data Life Cycle` :bdg-info:`Lesson 006`

◀ :doc:`Previous <005-the-six-phases-of-the-data-analysis-process>` · :doc:`Next <007-understanding-the-data-ecosystem>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


An old craft with new names
-----------------------------

Analysing data to answer questions long predates the job title. States counted
people and harvests for millennia; the statistics of the 18th and 19th centuries
formalised inference from samples; the 20th century added machine computation.
Two more recent turns shaped the modern craft. In 1977 John Tukey's
*Exploratory Data Analysis* argued that analysts should **look at data first** —
plot it, summarise it, let it suggest hypotheses — rather than jumping straight
to confirming a preconceived model. And from the 1990s, industry codified the
workflow itself: **CRISP-DM** (the Cross-Industry Standard Process for Data
Mining, published in 1999) became the most widely used formal process model for
analytics projects.

Many framings, one process
----------------------------

Different communities carve the same journey differently:

- **Ask–Prepare–Process–Analyze–Share–Act** — the six-phase framing this course
  uses, oriented around the analyst's tasks.
- **CRISP-DM** — Business Understanding, Data Understanding, Data Preparation,
  Modeling, Evaluation, Deployment; explicitly **cyclical**, with movement back
  and forth between phases expected rather than exceptional.
- Compressed or expanded variants — some teams merge exploration and modelling
  into five steps; others split deployment into more.

Lay them side by side and the correspondence is plain: *Ask* is business
understanding; *Prepare* and *Process* are data understanding and preparation;
*Analyze* spans exploration and modelling; *Share* is evaluation and
communication; *Act* is deployment. Learning one framing well makes you fluent
in all of them — the vocabulary changes, the discipline does not.

Why structure at all
----------------------

The recurring failure of unstructured analysis is starting in the middle:
grabbing data and computing before the question is clear, then discovering the
data cannot answer what was actually needed. Every framework above exists to
prevent that — each front-loads *understanding the problem* and treats cleaning
as a first-class phase, because those are the steps that unstructured work
skips. Structure is also what makes work **repeatable and reviewable**: a
colleague can pick up a CRISP-DM project and know where things stand.

The caveat
------------

Frameworks describe; they do not think. Following the phases mechanically, with
a weak question or credulous data, produces well-organised nonsense. Treat the
structure as scaffolding for judgement — the judgement itself is what the rest
of this course builds.

.. hint::

   - :doc:`The Six Phases of the Data Analysis Process <005-the-six-phases-of-the-data-analysis-process>`
   - :doc:`Understanding the Data Ecosystem <007-understanding-the-data-ecosystem>`
   - :doc:`Data-Driven Decision-Making <003-data-driven-decision-making>`
   - :doc:`The Stages of the Data Analysis Process and Their Roles <011-the-stages-of-the-data-analysis-process-and-their-roles>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/07/30/the-origins-of-data-analysis-and-the-many-ways-to-structure-it/ <https://insightful-data-lab.com/2023/07/30/the-origins-of-data-analysis-and-the-many-ways-to-structure-it/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: foundations, topic: process

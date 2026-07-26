:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-ddd-018:

========================================================================
Context and Bias in Data Analysis
========================================================================

:bdg-primary:`🎯 Data-Driven Decisions` :bdg-secondary:`🗣 Stakeholders, Communication & Execution` :bdg-info:`Lesson 018`

◀ :doc:`Previous <017-defining-the-problem-domain>` · :doc:`Next <019-stakeholder-expectations-in-data-analysis>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Context makes data mean something
-----------------------------------

**Context** is the condition in which data exists — the who, what, when, where,
and how of its creation. The foundations showed context is what makes a value
meaningful; this lesson shows the sharper edge: **unexamined context is where
bias enters an analysis**. The same data yields opposite conclusions depending
on context the analyst did or did not establish.

The questions that establish context
---------------------------------------

Before trusting any dataset, five questions locate it:

- **Who** collected it, and who is *in* it — which populations are represented,
  and which are silently missing?
- **What** exactly does each field measure, in what units, under what
  definition?
- **When** was it collected — and is that period representative, or a holiday,
  an outage, a boom?
- **Where** did it come from — which systems, regions, channels?
- **How** was it gathered — self-reported, sensor-logged, sampled how?

A satisfaction score of 4.5 means one thing from a representative survey and
another from one answered only by the delighted and the furious. Context is not
background; it is half the meaning.

Where context failures become bias
------------------------------------

**Bias** is a preference in favour of or against a thing, person, or group,
and unexamined context is its commonest doorway. The specific mechanisms (which
the prep section dissects) all trace to a context question skipped:

- **Sampling bias** — the data over-represents some groups (the *who* went
  unasked).
- **Historical bias** — the data faithfully records a biased past and the
  analysis projects it forward (the Amazon recruiting case from the
  foundations).
- **Selection and survivorship bias** — only certain cases made it into the
  data (the *what got captured* went unasked).
- **Confirmation bias** — the analyst's own preference, steering which
  questions get asked and which results get scrutinised.

Notice the last one is about the *analyst*, not the data: context includes
your own position and expectations, which shape the analysis as surely as the
data's origin does.

Establishing context in practice
----------------------------------

Two cheap habits. **Interrogate provenance** on arrival — walk the data-life-
cycle backward (who planned, captured, managed it?) before computing.
**Disaggregate and compare** — check results across relevant groups and time
periods, because bias hides in aggregates and surfaces in breakdowns. Neither
needs special tools; both need the decision that context is part of the job.

The caveat
------------

Perfect context is unattainable — you rarely know everything about how data
was made. The professional standard is not omniscience but **honesty about the
limits**: stating what context you established, what you could not, and how
that bounds the conclusion. An analysis that names its context gaps is trusted
far longer than one that hides them, which is exactly the fairness obligation
from the foundations, operating at the project level.

.. hint::

   - :doc:`Fairness in Data Analysis <../1_foundations/026-fairness-in-data-analysis>`
   - :doc:`Understanding Bias in Data Analysis <../3_data_preparation/008-understanding-bias-in-data-analysis>`
   - :doc:`Common Types of Data Bias <../3_data_preparation/010-common-types-of-data-bias>`
   - :doc:`Defining the Problem Domain <017-defining-the-problem-domain>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/08/31/context-and-bias-in-data-analysis/ <https://insightful-data-lab.com/2023/08/31/context-and-bias-in-data-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: ddd, topic: execution

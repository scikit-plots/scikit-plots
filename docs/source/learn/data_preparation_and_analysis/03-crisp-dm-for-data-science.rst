.. _dpa-crisp-dm-for-data-science:

========================================================================
CRISP-DM for Data Science
========================================================================

**Stage 1 · 📋 Foundations**  ·  Lesson 03 of 56  ·  *beginner*

:doc:`◀ Previous · The Process of Data Analysis <02-the-process-of-data-analysis>`   ·   :doc:`Next · Big Data: Definition, Characteristics, Evolution, and Business Impact ▶ <04-big-data-definition-characteristics-evolution-and-business-impact>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

A shared blueprint
--------------------

**CRISP-DM** — the **Cross-Industry Standard Process for Data Mining** — is the most widely used
framework for structuring a data project. Introduced in the late 1990s and deliberately
**industry-independent**, it turns the loose workflow of the last lesson into six named phases that
any team can share as a common plan.

The six phases
----------------

The phases run:

1. **Business Understanding** — pin down the objective and turn it into a data-mining goal;
2. **Data Understanding** — collect the data, describe it, explore it, and check its quality;
3. **Data Preparation** — select, clean, construct, integrate and format the modelling dataset;
4. **Modeling** — choose techniques, build models, and tune them;
5. **Evaluation** — judge the results **against the business objective**, not just the metrics;
6. **Deployment** — put the model to work, with monitoring and a plan to maintain it.

Where the time goes
---------------------

One phase dominates the calendar: **Data Preparation** is generally the **most time-consuming** part
of the whole project — routinely cited as the bulk of the effort. It is also, not coincidentally, the
subject this course is named for. Getting the data right is most of the work, and most of the payoff.

A cycle, not a line
---------------------

CRISP-DM is drawn as a **cycle**, not a checklist. Arrows link the phases both ways — modelling sends
you back to preparation, evaluation back to business understanding — and an outer loop returns the
finished project to the start as new questions emerge. The framework's real lesson is that data work
is **iterative**: you revisit earlier phases as later ones teach you what you missed.

.. hint::

   **Related lessons:** :doc:`The Process of Data Analysis <02-the-process-of-data-analysis>`  ·  :doc:`Why Do We Analyze Data? <01-why-do-we-analyze-data>`  ·  :doc:`The First Step in Knowing Your Data <05-the-first-step-in-knowing-your-data>`  ·  :doc:`Partitioning Observations to Train Objective Models <25-partitioning-observations-to-train-objective-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/crisp-dm-for-data-science/ <https://insightful-data-lab.com/2026/01/14/crisp-dm-for-data-science/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: beginner

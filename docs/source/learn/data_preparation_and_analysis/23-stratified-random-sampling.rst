.. _dpa-stratified-random-sampling:

========================================================================
Stratified Random Sampling
========================================================================

**Stage 4 · 🧩 Sampling, Partitioning & Segmentation**  ·  Lesson 23 of 56  ·  *intermediate*

:doc:`◀ Previous · Cross-Selling <22-cross-selling>`   ·   :doc:`Next · Linear Congruential Random Number Generator (LCG) ▶ <24-linear-congruential-random-number-generator-lcg>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Representative by design
--------------------------

A sample is only useful if it **resembles** the population it is drawn from. **Stratified random
sampling** guarantees that resemblance for the characteristics you care about, by sampling **within**
subgroups rather than trusting chance to balance them. It is a workhorse of survey design and, in this
course, of splitting data for modelling.

Strata
--------

The method starts by dividing the population into **strata** — mutually exclusive, exhaustive
subgroups that share a characteristic (gender, region, customer type). Strata are chosen to be
**internally homogeneous**: alike within, different between. A **simple random sample** is then drawn
independently from **each** stratum, and the pieces combined. Under **proportional allocation**, each
stratum contributes in proportion to its share of the population, so the sample mirrors the whole.

Why not simple random?
------------------------

Plain random sampling can, by luck, **under-represent** a small but important group — draw 100
customers at random and a rare segment might barely appear. Stratifying **removes** that luck: every
subgroup is present by construction, in the right proportion. The result is **greater precision**
(lower sampling variability) than a simple random sample of the same size, especially when the strata
differ from one another.

In machine learning
---------------------

The same idea is essential when splitting data. A **stratified** train/test split keeps the **class
proportions** identical in both parts — vital for **imbalanced** problems, where a naive split might
leave too few positive cases in the test set. In scikit-learn it is one argument:
``train_test_split(..., stratify=y)``, or ``StratifiedKFold`` for cross-validation. The next lessons
need this discipline, because honest model evaluation depends on representative partitions.

.. hint::

   **Related lessons:** :doc:`Partitioning Observations to Train Objective Models <25-partitioning-observations-to-train-objective-models>`  ·  :doc:`Linear Congruential Random Number Generator (LCG) <24-linear-congruential-random-number-generator-lcg>`  ·  :doc:`Creating Segments of Observations for Business Reasons (RFM) <30-creating-segments-of-observations-for-business-reasons-rfm>`  ·  :doc:`Assessing the Quality of Prediction Models <50-assessing-the-quality-of-prediction-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/stratified-random-sampling/ <https://insightful-data-lab.com/2026/01/14/stratified-random-sampling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate

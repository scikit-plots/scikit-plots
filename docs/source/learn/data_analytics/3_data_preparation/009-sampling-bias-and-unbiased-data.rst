:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-009:
.. _data-analytics-prep-009:
.. _da-foundations-prep-009:
.. _da-decisions-prep-009:
.. _da-prep-prep-009:
.. _da-cleaning-prep-009:
.. _da-analyze-prep-009:
.. _da-viz-prep-009:
.. _da-python-prep-009:
.. _da-jobsearch-prep-009:

========================================================================
Sampling Bias and Unbiased Data
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`⚖️ Bias & Data Ethics` :bdg-info:`Lesson 009`

◀ :doc:`Previous <008-understanding-bias-in-data-analysis>` · :doc:`Next <010-common-types-of-data-bias>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The sample stands in for the whole
------------------------------------

Analysis rarely examines an entire population; it examines a **sample** and
generalises. That move is only valid when the sample fairly represents the
whole — and when it does not, the result is **sampling bias**: a sample in
which some members of the population are systematically over- or
under-represented, so conclusions drawn from it do not hold for the population.
It is the most common and most consequential bias analysts face.

How sampling bias happens
---------------------------

The classic mechanism is a sampling method that reaches some groups more than
others:

- **Convenience sampling** — surveying whoever is easiest to reach (your own
  customers, people who answer the phone at 2 pm), which skews toward whoever
  that happens to be.
- **Self-selection** — letting people opt in (online reviews, voluntary
  surveys), which over-represents those with strong opinions and the time to
  express them.
- **Coverage gaps** — a sampling frame that silently excludes part of the
  population (a phone survey missing people without phones).

A famous historical example: a 1936 US presidential poll drew millions of
responses from telephone and automobile-registration lists, predicted the wrong
winner, and failed precisely because — in that era — phone and car owners were
wealthier and unrepresentative. Size did not save it; the sample was biased,
and more of a biased thing is still biased.

Unbiased data as the goal
---------------------------

**Unbiased data** is a sample representative of the whole population — one where
every relevant group appears in roughly its true proportion. The principal tool
for achieving it is **random sampling**: selecting members so that each has an
equal chance of being chosen, which lets representativeness arise by chance
rather than depending on a method that might lean. (The next stage's lesson on
population and sample size develops the mechanics.) Random selection does not
*guarantee* a representative sample in any single draw, but it removes the
*systematic* lean that defines bias.

Detecting it in practice
--------------------------

Two habits catch most sampling bias. **Ask how the data was collected** — the
provenance question from earlier — because the collection method is where the
bias lives: who could have been included, and who could not? **Compare the
sample to known population facts** — if your survey respondents are 70% one
demographic but the population is 50%, the skew is visible and quantifiable.
When you cannot fix a skewed sample, you can at least *state* its skew, so
conclusions are read with the right caution.

The caveat
------------

Perfectly unbiased data is largely unattainable — every real sample departs
from the population somehow. The professional standard is not perfection but
*awareness and disclosure*: knowing which groups your data over- and
under-represents, adjusting where you can, and being explicit about the limits
where you cannot. A modest analysis honest about its sampling beats an ambitious
one that assumes representativeness it never checked.

.. hint::

   - :doc:`Understanding Bias in Data Analysis <008-understanding-bias-in-data-analysis>`
   - :doc:`Common Types of Data Bias <010-common-types-of-data-bias>`
   - :doc:`Population, Sample Size, and Random Sampling <../4_data_cleaning_preparation/005-population-sample-size-and-random-sampling>`
   - :doc:`Choosing the Right Data to Collect <002-choosing-the-right-data-to-collect>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/sampling-bias-and-unbiased-data/ <https://insightful-data-lab.com/2023/09/04/sampling-bias-and-unbiased-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: bias_ethics

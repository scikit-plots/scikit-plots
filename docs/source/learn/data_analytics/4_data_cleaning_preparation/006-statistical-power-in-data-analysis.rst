:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-006:
.. _data-analytics-cleaning-006:
.. _da-foundations-cleaning-006:
.. _da-decisions-cleaning-006:
.. _da-prep-cleaning-006:
.. _da-cleaning-cleaning-006:
.. _da-analyze-cleaning-006:
.. _da-viz-cleaning-006:
.. _da-python-cleaning-006:
.. _da-jobsearch-cleaning-006:

========================================================================
Statistical Power in Data Analysis
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🧱 Data Integrity & Sampling` :bdg-info:`Lesson 006`

◀ :doc:`Previous <005-population-sample-size-and-random-sampling>` · :doc:`Next <007-sample-size-and-data-integrity>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The chance of seeing what is there
------------------------------------

Having enough representative data is not only about precision — it is also about
**sensitivity**: the ability to *detect a real effect when one exists*. That
sensitivity has a name. **Statistical power** is the probability that an analysis
will correctly identify a real effect (a genuine difference or relationship) when
it is truly present. Low power means real effects go undetected — the analysis
concludes "nothing here" when something was there — which is one of the quieter
and more dangerous failures in data work.

What power means, concretely
------------------------------

Imagine testing whether a new checkout design genuinely improves conversion. If
the improvement is real but your sample is too small, the difference can be
swamped by random noise, and the test comes back inconclusive — a **false
negative**. Statistical power is the probability of *avoiding* that outcome:
of detecting the improvement given that it is real. Power ranges from 0 to 1, and
the widely used convention is a target of **0.8 (80%)** — meaning an 80% chance
of detecting a true effect of the size you care about, and correspondingly a 20%
chance of missing it. Higher power is better but costs more data.

What power depends on
-----------------------

Four things move statistical power, and they trade off against each other:

- **Sample size** — the lever analysts most directly control. Larger samples
  give higher power; this is the main reason "enough data" matters for detecting
  effects, not just for precision.
- **Effect size** — how large the real effect is. Big effects are easy to detect
  with modest samples; subtle effects need large ones. Detecting a 0.2%
  conversion lift takes far more data than detecting a 20% lift.
- **Significance threshold** — how strict you are about false positives; stricter
  thresholds lower power for a given sample.
- **Variability** — noisier data lowers power, which is another reason clean,
  consistent data matters.

The practical upshot: to detect a small effect reliably, you need a large sample,
and if you cannot get one, you must accept that small real effects may remain
invisible to your analysis.

Why analysts should care
--------------------------

Power connects directly to the insufficient-data lesson. A common, invisible
error is running an analysis with too little data, finding "no significant
effect," and concluding there *is* no effect — when in truth the study simply
lacked the power to see it. **Absence of evidence is not evidence of absence**:
an underpowered "no effect" result means "we could not detect one," not "there is
none." Knowing roughly how much data an effect of a given size requires — before
running the analysis — is what separates a genuine null result from an
underpowered one.

The caveat
------------

Power analysis involves assumptions — chiefly a guess at the effect size you are
looking for, which you often do not know in advance — so power calculations are
estimates, not guarantees, and the conventional 0.8 target is a norm, not a law
of nature. The value of thinking about power is less the exact number than the
discipline it enforces: asking, *before* collecting or analysing, whether the
data could plausibly reveal the effect you care about. The next lesson ties
sample size directly to data integrity.

.. hint::

   - :doc:`Population, Sample Size, and Random Sampling <005-population-sample-size-and-random-sampling>`
   - :doc:`Sample Size and Data Integrity <007-sample-size-and-data-integrity>`
   - :doc:`Margin of Error <008-margin-of-error>`
   - :doc:`Handling Insufficient Data in Data Analysis <004-handling-insufficient-data-in-data-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/10/31/statistical-power-in-data-analysis/ <https://insightful-data-lab.com/2023/10/31/statistical-power-in-data-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: integrity

:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-007:
.. _da-4-cleaning-cleaning-007:

========================================================================
Sample Size and Data Integrity
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🧱 Data Integrity & Sampling` :bdg-info:`Lesson 007`

◀ :doc:`Previous <006-statistical-power-in-data-analysis>` · :doc:`Next <008-margin-of-error>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


How much is enough?
--------------------

The recurring question of this stage — *how much data is enough?* — is itself a
matter of **data integrity**, because a conclusion drawn from too small a sample
is untrustworthy no matter how clean each record is. Sample size determines how
much confidence you can place in a result, and choosing it well is part of
ensuring the analysis is sound rather than merely precise-looking.

Sample size and confidence
-----------------------------

The core relationship: **larger samples give more reliable estimates** of the
population, because the random variation from the luck of the draw shrinks as the
sample grows. A tiny sample can land far from the population truth by chance
alone; a large one is pulled close. This is why a result from five customers is
suggestive at best while a result from five thousand can be trusted — the same
question, very different weight of evidence.

But — and this echoes through the whole section — **more data reduces random
error, not bias**. Increasing the size of a *biased* sample makes the estimate
more precisely wrong; it converges confidently on the wrong answer. Size and
representativeness are independent requirements: integrity needs both a fair
method *and* an adequate size.

What determines the size you need
-----------------------------------

The sample size required for a trustworthy conclusion depends on several factors:

- **The confidence you need** — higher confidence in the result requires more
  data.
- **The precision you need** — a tighter margin of error (the next lesson)
  requires a larger sample.
- **The population's variability** — more variable populations need larger
  samples to pin down.
- **The effect size** — detecting small effects requires more data than
  detecting large ones (the power lesson).
- **The stakes** — higher-stakes decisions justify the cost of collecting more.

These combine so that "enough" is not a single number but a judgement tied to the
question. Statistical tools and sample-size calculators formalise the
relationship; the analyst's job is to recognise that the question *has* an answer
and that ignoring it — grabbing whatever sample is convenient — risks conclusions
the data cannot support.

Sample size in the cleaning phase
-----------------------------------

Why does this live in a cleaning section? Because judging sample sufficiency is
part of assessing whether data is fit to analyse — the integrity check that sits
alongside checking for errors and inconsistencies. A dataset can be spotlessly
clean and still too small to answer its question reliably; recognising that
before analysis, and either gathering more or scoping the conclusion accordingly,
is part of preparing data responsibly.

The caveat
------------

Bigger is not limitlessly better: beyond the size needed for the required
confidence and precision, additional data yields diminishing returns while
costing time and money, and very large samples can make *trivial*, unimportant
differences appear "statistically significant" — significance is not the same as
importance. The goal is the *right* size for the question and its stakes — large
enough to trust, not so large that you over-invest or mistake a tiny effect for a
meaningful one. The next lesson makes the precision side of this concrete with
the margin of error.

.. hint::

   - :doc:`Population, Sample Size, and Random Sampling <005-population-sample-size-and-random-sampling>`
   - :doc:`Statistical Power in Data Analysis <006-statistical-power-in-data-analysis>`
   - :doc:`Margin of Error <008-margin-of-error>`
   - :doc:`Data Integrity and Its Risks in Data Analysis <002-data-integrity-and-its-risks-in-data-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/10/31/sample-size-and-data-integrity/ <https://insightful-data-lab.com/2023/10/31/sample-size-and-data-integrity/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: integrity

:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-012:
.. _data-analytics-prep-012:
.. _da-foundations-prep-012:
.. _da-decisions-prep-012:
.. _da-prep-prep-012:
.. _da-cleaning-prep-012:
.. _da-analyze-prep-012:
.. _da-viz-prep-012:
.. _da-python-prep-012:
.. _da-jobsearch-prep-012:

========================================================================
Identifying Bad Data Sources (When Data Does Not ROCCC)
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`⚖️ Bias & Data Ethics` :bdg-info:`Lesson 012`

◀ :doc:`Previous <011-identifying-good-data-sources-roccc-framework>` · :doc:`Next <013-data-ethics-in-data-analysis>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


ROCCC in reverse
------------------

The ROCCC framework doubles as a detector for *bad* data: a source that fails
one or more criteria is one to distrust, or at least to use with open eyes. Just
as good data "ROCCCs", bad data fails to — and learning to read each failure is
how an analyst avoids building on unsound ground. This lesson runs the five
criteria backward.

The five failures
-------------------

- **Not Reliable** — from an unvetted or low-quality origin, known for errors,
  or visibly incomplete and inconsistent. Red flags: no clear methodology, a
  source with no reputation to protect, obvious internal contradictions.
- **Not Original** — a copy of a copy, with no path back to who actually
  collected it. Red flags: data quoted from a secondary article that cites no
  primary source; aggregations whose components cannot be traced. Every removed
  hand is unverifiable distortion.
- **Not Comprehensive** — missing fields, populations, or time spans the
  question requires. Red flags: gaps in coverage, suspiciously round or sparse
  data, a scope narrower than the question needs.
- **Not Current** — too old to reflect present reality, in a domain that has
  moved on. Red flags: no collection date given, or a date that predates
  relevant changes in the subject.
- **Not Cited** — no documentation of who created it, when, where, or how. Red
  flags: data with no attribution, no methodology, no provenance — asking for
  blind trust.

A source failing several of these is not merely imperfect; it is a liability,
and analysis built on it inherits every weakness.

The seductive-but-bad source
------------------------------

The dangerous case is data that is *convenient* — free, large, immediately
available — but fails ROCCC quietly. A big, easily-downloaded dataset with no
methodology and no date is tempting under deadline, and its problems do not
announce themselves: the analysis runs, the charts render, and the unreliability
surfaces only when a decision built on it goes wrong. Convenience is not a ROCCC
criterion; weigh it against the five that are.

What to do with a flawed source
---------------------------------

Failing ROCCC is not always disqualifying — sometimes flawed data is the only
data available. The professional responses, in order of preference: **find a
better source** if one exists; **supplement** the weak source with others to
cover its gaps; **validate** what you can against independent data; and, at
minimum, **disclose** the limitation prominently, so anyone relying on the
conclusion knows the ground it stands on. What is never acceptable is using
known-bad data *silently*, letting others assume a soundness you know is absent.

The caveat
------------

Source evaluation is a judgement, not a pass/fail gate — real data almost always
fails *some* criterion to *some* degree, and rejecting everything imperfect
would leave you with nothing to analyse. The skill is proportionality: matching
the required source quality to the stakes of the decision (the speed-versus-
accuracy trade-off, applied to data), and being honest about where the data
falls short of what the question deserves. This completes the bias and source-
quality core; the next lessons turn to the ethics and privacy obligations that
govern how data may be used at all.

.. hint::

   - :doc:`Identifying Good Data Sources (ROCCC Framework) <011-identifying-good-data-sources-roccc-framework>`
   - :doc:`Sampling Bias and Unbiased Data <009-sampling-bias-and-unbiased-data>`
   - :doc:`Data Ethics in Data Analysis <013-data-ethics-in-data-analysis>`
   - :doc:`Choosing the Right Data to Collect <002-choosing-the-right-data-to-collect>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/identifying-bad-data-sources-when-data-does-not-roccc/ <https://insightful-data-lab.com/2023/09/04/identifying-bad-data-sources-when-data-does-not-roccc/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: bias_ethics

:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-foundations-004:
.. _da-1-foundations-foundations-004:

========================================================================
Detectives and Data Analysts
========================================================================

:bdg-primary:`🌱 Foundations` :bdg-secondary:`🌟 Why Data Analytics` :bdg-info:`Lesson 004`

◀ :doc:`Previous <003-data-driven-decision-making>` · :doc:`Next <005-the-six-phases-of-the-data-analysis-process>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Two jobs, one method
----------------------

A detective and a data analyst do strikingly similar work. Both start with a
question (*who did it? why are sales falling?*), gather evidence, test
explanations against that evidence, discard the ones that fail, and present a
conclusion that has to **hold up under scrutiny** — to a court in one case, to
stakeholders in the other. The analogy is worth taking seriously because it
captures the *mindset* this course trains, before any tool.

What transfers
----------------

- **Questions first.** A detective does not collect fingerprints at random; the
  investigation is shaped by what needs answering. Likewise, analysis begins
  with a sharp question — the subject of a whole later stage — because data
  gathered without one is just clutter.
- **Evidence over assumption.** Detectives distrust the obvious suspect;
  analysts distrust the obvious explanation. Both ask: *what does the evidence
  actually show?* — and let it overrule a comfortable story.
- **Multiple hypotheses.** Good investigators hold several explanations at once
  and look for the evidence that separates them. "Sales fell because of price"
  and "sales fell because a competitor launched" predict different patterns;
  the data can say which.
- **The chain of custody.** A conclusion is only as credible as the trail behind
  it. Documenting where data came from and what was done to it — a major theme
  of the cleaning section — is the analyst's chain of custody.
- **Presenting the case.** Neither job ends at the private "aha". The finding
  must be assembled into a case that a non-specialist can follow and believe,
  which is what the visualization and presentation section teaches.

Where the analogy bends
-------------------------

One difference matters. A detective usually seeks a single past fact — who did
it. An analyst often characterises an **ongoing pattern** (what drives churn,
which segment is growing) where there is no one culprit and the answer is a
distribution, a trend, or a trade-off. Analytical conclusions are therefore
usually **probabilistic** — "strong evidence that", not "proof beyond doubt" —
and stating that uncertainty honestly is part of the job, not a weakness in it.

The takeaway
--------------

Tools change; the investigator's discipline does not. Ask a precise question,
gather evidence deliberately, test explanations rather than defend them, keep
the trail, and present a case that survives cross-examination. Every technique
in the coming sections — spreadsheets, SQL, cleaning, charts, Python — is in
service of that discipline.

.. hint::

   - :doc:`Data-Driven Decision-Making <003-data-driven-decision-making>`
   - :doc:`Analytical Skills and Their Core Components <013-analytical-skills-and-their-core-components>`
   - :doc:`Analytical Thinking and Questions for Problem Solving <016-analytical-thinking-and-questions-for-problem-solving>`
   - :doc:`Why Asking the Right Questions Matters in Data Analytics <../2_data_driven_decisions/004-why-asking-the-right-questions-matters-in-data-analytics>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/07/30/detectives-and-data-analysts/ <https://insightful-data-lab.com/2023/07/30/detectives-and-data-analysts/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: foundations, topic: why

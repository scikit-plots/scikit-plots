:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-foundations-026:
.. _da-1-foundations-foundations-026:

========================================================================
Fairness in Data Analysis
========================================================================

:bdg-primary:`🌱 Foundations` :bdg-secondary:`🧰 Tools, Applications & Ethics` :bdg-info:`Lesson 026`

◀ :doc:`Previous <025-the-role-of-business-tasks-in-data-analysis>` · :doc:`Next <027-key-factors-to-consider-when-choosing-a-data-analytics-role>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


What fairness means here
--------------------------

**Fairness** in data analysis means ensuring that the analysis does not create
or reinforce bias — that its conclusions and the decisions built on them do not
systematically disadvantage groups of people. It belongs in the foundations of
this course, not an appendix, because unfair analysis is usually not malicious:
it is ordinary work done on unexamined data, and every analyst is one
unexamined dataset away from producing it.

A famous failure, instructive in every detail
-----------------------------------------------

The canonical case: as reported by Reuters in 2018, Amazon built an
experimental tool to score job applicants' resumes, training it on roughly ten
years of resumes the company had received. Because the historical applicant
pool was heavily male, the tool **taught itself that male candidates were
preferable** — penalising resumes containing the word "women's" (as in
"women's chess club captain") and downgrading graduates of all-women's
colleges. Engineers tried to correct it, could not guarantee neutrality, and
the tool was scrapped.

Every step generalises. The data faithfully recorded a biased history; the
analysis faithfully learned the bias; the output would have faithfully
projected it into future decisions. Nothing was "wrong" technically — which is
precisely the warning. As one civil-liberties commentary put it, such systems
do not remove human bias; they **launder it through software**: ask a model to
find candidates who resemble past successes, and reproducing the demographics
of the past workforce is virtually guaranteed.

Where unfairness enters ordinary analysis
-------------------------------------------

You do not need machine learning to repeat the pattern. The entry points are
mundane:

- **Unrepresentative data** — a customer survey answered mostly by one
  demographic, treated as "what customers think".
- **Historical bias in the target** — measuring "success" by past outcomes
  that themselves reflect unequal treatment.
- **Proxy variables** — a neutral-looking field (postal code, college name)
  that stands in for a protected characteristic.
- **Aggregation that hides harm** — an average that improves while a subgroup
  worsens.

The prep section's bias lessons dissect these mechanisms; here the point is
that each is *detectable* if someone looks.

The analyst's fairness habits
-------------------------------

Four habits, cheap and repeatable. **Ask who is in the data** — and who is
missing — before trusting it. **Disaggregate**: check results across relevant
groups, not just overall. **Interrogate proxies**: for any influential
variable, ask what it might be standing in for. **Trace the decision**: follow
the analysis to the people it will affect, and ask who bears the cost of its
errors. None requires special tools; all require deciding that fairness is
part of the job — which, in this course, it is.

The caveat
------------

Fairness is not a checkbox with a formula; reasonable definitions can even
conflict, and judgement is unavoidable. But the failure mode to fear is not
subtle philosophy — it is nobody having looked at all. The habits above are the
looking.

.. hint::

   - :doc:`Understanding Bias in Data Analysis <../3_data_preparation/008-understanding-bias-in-data-analysis>`
   - :doc:`Common Types of Data Bias <../3_data_preparation/010-common-types-of-data-bias>`
   - :doc:`Data Ethics in Data Analysis <../3_data_preparation/013-data-ethics-in-data-analysis>`
   - :doc:`Context and Bias in Data Analysis <../2_data_driven_decisions/018-context-and-bias-in-data-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/07/30/fairness-in-data-analysis/ <https://insightful-data-lab.com/2023/07/30/fairness-in-data-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: foundations, topic: tools

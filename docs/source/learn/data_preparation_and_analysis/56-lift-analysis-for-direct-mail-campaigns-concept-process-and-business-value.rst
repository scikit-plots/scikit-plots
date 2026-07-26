.. _dpa-lift-analysis-for-direct-mail-campaigns-concept-process-and-business-value:

===============================================================================
Lift Analysis for Direct Mail Campaigns: Concept, Process, and Business Value
===============================================================================

**Stage 8 · 📊 Model Evaluation**  ·  Lesson 56 of 56  ·  *advanced*

:doc:`◀ Previous · AUC–ROC Curve: Evaluating Classification Model Performance <55-auc-roc-curve-evaluating-classification-model-performance>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

A budget, not a threshold
---------------------------

The course closes with the evaluation tool born in **direct mail** — the corner of marketing where
every contact costs real money. A campaign can afford to mail, say, only **20%** of the customer base.
The question is not "which cases are positive?" but "**whom should we contact first?**" — a **ranking**
problem with a budget. **Lift analysis** measures how much better a model's ranking is than mailing at
**random**.

Score, sort, slice
--------------------

The procedure is simple. **Score** every prospect with the model (their probability of responding),
**sort** the list from highest to lowest score, then **slice** it into ten equal groups — **deciles** —
so decile 1 holds the top 10% of prospects by model score. Mailing a past campaign's data through this
lens shows the **response rate per decile**: if the model ranks well, decile 1 responds far above
average, decile 2 next, and the bottom deciles barely respond at all.

Gains and lift
----------------

Two curves summarise the table. The **cumulative gains** curve shows, for each targeting depth, the
share of **all responders** captured: for example, the top decile alone might contain **28%** of all
responders, and the top two deciles together over half. The **baseline** is random targeting — contact
X% of customers, reach X% of responders. **Lift** is the ratio of the two:

.. math::

   \text{lift at depth } X =
   \frac{\%\ \text{of responders captured in the top } X\%}{X\,\%}.

A lift of **1** means the model adds nothing; a lift of **2** at 20% depth means the model finds
**twice** the responders random mailing would. The **further** the gains curve rises above the
baseline, the more valuable the model — and scikit-plots draws both the cumulative-gain and lift
curves directly from scores.

The business payoff
---------------------

Lift converts model quality into **money**. If the top four deciles capture most responders, the
campaign mails **40%** of the base, captures the bulk of the responses, and saves the cost of the
other 60% — a concrete, defensible decision drawn straight from the chart. It is a fitting end to the
course: the journey that began with *why we analyse data* ends with a model, honestly evaluated on
held-out data, telling a business **exactly what to do** — the prescriptive payoff the first lesson
promised.

.. hint::

   **Related lessons:** :doc:`AUC–ROC Curve: Evaluating Classification Model Performance <55-auc-roc-curve-evaluating-classification-model-performance>`  ·  :doc:`Binary Classification Model Evaluation and Threshold Optimization <53-binary-classification-model-evaluation-and-threshold-optimization>`  ·  :doc:`Recency, Frequency, and Monetary Value (RFM) <28-recency-frequency-and-monetary-value-rfm>`  ·  :doc:`Why Do We Analyze Data? <01-why-do-we-analyze-data>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/lift-analysis-for-direct-mail-campaigns-concept-process-and-business-value/ <https://insightful-data-lab.com/2026/01/16/lift-analysis-for-direct-mail-campaigns-concept-process-and-business-value/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: advanced

:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-011:
.. _da-5-analyze-analyze-011:

========================================================================
Problem-Solving and Seeking Help in Data Analysis
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🔗 Problem-Solving & Combining Data` :bdg-info:`Lesson 011`

◀ :doc:`Previous <010-working-with-strings-in-spreadsheets-len-left-right-find>` · :doc:`Next <012-how-to-effectively-search-for-solutions-online-as-a-data-analyst>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Getting unstuck is a skill
----------------------------

Every analyst gets stuck — a formula that will not work, a query that errors, a
result that makes no sense, a technique they have never used. What separates
effective analysts is not avoiding these moments but *handling* them well:
**problem-solving** (systematically working toward a solution) and **seeking help**
(knowing when and how to ask). This lesson, opening the combine stage, is about the
debugging mindset that the hands-on techniques ahead will repeatedly demand.

The problem-solving mindset
-----------------------------

Effective problem-solving in analysis follows a rough method, echoing the root-
cause discipline from the foundations:

- **Understand the problem precisely.** What exactly is wrong? What did you expect
  versus what happened? A vague "it's broken" cannot be solved; "this query
  returns 0 rows when it should return hundreds" can.
- **Isolate it.** Narrow down where the problem is — which part of the formula,
  which clause of the query, which step of the process. Simplify until you find
  the smallest thing that reproduces the problem.
- **Form and test hypotheses.** Guess what might be causing it, then test that
  guess — change one thing, see if it helps. Systematic, one-variable-at-a-time
  testing beats random flailing.
- **Read the error.** Error messages usually say what is wrong (the spreadsheet
  error codes, the SQL error text); reading them carefully is often the whole
  solution.
- **Check your assumptions.** Frequently the problem is an assumption that turned
  out false — the data was not the type you thought, the column meant something
  different than expected.

This method — understand, isolate, hypothesise, test — is the analyst's debugging
loop, and it works on formulas, queries, and confusing results alike.

Seeking help effectively
--------------------------

When your own problem-solving stalls, seeking help is not failure — it is
efficient, provided you ask well:

- **Ask a specific, well-framed question.** "How do I do X" is weak; "I'm trying to
  do X, I tried Y, I expected A but got B — what am I missing?" gives a helper what
  they need to help.
- **Show what you tried.** Your attempt, the error, the relevant data shape — the
  context that lets someone diagnose rather than guess.
- **Ask the right source.** A colleague for domain or organisational context;
  online resources and communities for technical problems (the next lesson);
  documentation for how a tool works.

A good question often solves itself in the asking — articulating the problem
clearly frequently reveals the answer, the "rubber duck" effect.

The caveat
------------

Both halves have a failure mode. Problem-solving can become *stubbornness* —
grinding on a problem far past the point where asking would be faster, wasting
hours to avoid a five-minute question. Seeking help can become *dependence* —
asking before making any real attempt, which neither solves the immediate problem
efficiently nor builds your own capability. The balance is a reasonable attempt
first (understand, isolate, try), then ask when genuinely stuck — and always in a
way that shows your work, so the help teaches you rather than just unblocking you.
The next lesson develops the most common help-seeking skill: searching online
effectively.

.. hint::

   - :doc:`How to Effectively Search for Solutions Online as a Data Analyst <012-how-to-effectively-search-for-solutions-online-as-a-data-analyst>`
   - :doc:`Analytical Thinking and Questions for Problem Solving <../1_foundations/016-analytical-thinking-and-questions-for-problem-solving>`
   - :doc:`Choosing the Right Tool in Data Analysis <013-choosing-the-right-tool-in-data-analysis>`
   - :doc:`Troubleshooting VLOOKUP and Building a Problem-Solving Framework <016-troubleshooting-vlookup-and-building-a-problem-solving-framework>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/problem-solving-and-seeking-help-in-data-analysis/ <https://insightful-data-lab.com/2023/11/26/problem-solving-and-seeking-help-in-data-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: combine

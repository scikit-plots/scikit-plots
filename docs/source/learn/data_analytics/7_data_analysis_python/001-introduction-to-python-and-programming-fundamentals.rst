:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-001:
.. _data-analytics-python-001:
.. _da-foundations-python-001:
.. _da-decisions-python-001:
.. _da-prep-python-001:
.. _da-cleaning-python-001:
.. _da-analyze-python-001:
.. _da-viz-python-001:
.. _da-python-python-001:
.. _da-jobsearch-python-001:

========================================================================
Introduction to Python and Programming Fundamentals
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐍 Python Fundamentals` :bdg-info:`Lesson 001`

:doc:`Next <002-python-fundamentals>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Why Python
-----------

Spreadsheets and SQL take an analyst far, but a *programming language* takes them
further — automating work, handling any data size, and expressing analyses too complex
for a query. **Python** is the language most data analysts learn, and this section
builds the skill of using it for data analysis. Opening the Python section, this lesson
introduces why Python matters and the programming fundamentals underneath it.

Why Python for data analysis
------------------------------

Python has become the dominant language in data analysis for concrete reasons:

- **Readable and approachable** — Python's syntax is clean and close to plain English,
  making it one of the easier languages to learn, which matters for analysts who are not
  primarily programmers.
- **Powerful data libraries** — Python's ecosystem includes purpose-built data tools:
  ``pandas`` for tabular data, ``numpy`` for numerical computing, ``matplotlib`` and
  others for visualization (the later lessons). These do the heavy lifting.
- **Automation and reproducibility** — a Python script runs the same analysis
  identically every time and can be automated, the reproducibility that manual
  spreadsheet work lacks (a theme since the cleaning section).
- **Scale and flexibility** — Python handles data sizes and analytical complexity
  beyond spreadsheets, and can do anything the other tools can plus much they cannot.

Python is where the analyst's workflow is *automated and scaled* — the tool that ties
preparation, analysis, and visualization into repeatable pipelines.

Programming fundamentals
--------------------------

Beneath Python lie ideas common to all programming, worth naming before the syntax:

- **A program is a sequence of instructions** — code tells the computer precisely what
  to do, step by step, executed in order.
- **Data and operations** — programs hold *data* (values: numbers, text) and perform
  *operations* on it (calculations, transformations, comparisons).
- **Variables** — named places to store data for reuse (the next lessons).
- **Control flow** — deciding *which* instructions run and *how many times* (conditions
  and loops, the control stage).
- **Functions** — named, reusable blocks of instructions (a later lesson) — the
  abstraction-and-reuse principle from the foundations, in code.

These concepts underlie every language; Python is one readable way to express them.

Python in the analyst's toolkit
----------------------------------

Python does not replace spreadsheets and SQL so much as *complete* the toolkit. The
tool-choice lessons apply: spreadsheets for quick visual work, SQL for querying
databases, and Python for automation, complex transformation, and anything that must
run repeatably at scale. In practice Python often *orchestrates* the others — pulling
data via SQL, transforming it, and producing visualizations — making it the glue of a
mature analytical workflow. It is the most powerful and most general of the analyst's
tools, and correspondingly the most involved to learn.

The caveat
------------

Python's power comes with a steeper learning curve than spreadsheets, and it is not the
right tool for *every* task — a quick look at a small dataset is faster in a spreadsheet
than in code, and forcing simple work into Python is over-engineering. Learning Python
is a genuine investment, and its payoff is in automation, scale, and complexity, not in
one-off simple tasks. The judgement from the tool-choice lessons holds: reach for
Python when its strengths (reproducibility, scale, complex logic, automation) are worth
the effort, and use the simpler tools when they suffice. Python expands what you *can*
do; it does not make itself the right choice for everything. The next lesson begins with
Python's fundamentals.

.. hint::

   - :doc:`Choosing the Right Tool in Data Analysis <../5_analyze_data/013-choosing-the-right-tool-in-data-analysis>`
   - :doc:`Python Fundamentals <002-python-fundamentals>`
   - :doc:`Jupyter Notebook and Coding Environments <003-jupyter-notebook-and-coding-environments>`
   - :doc:`Spreadsheets vs. SQL <../4_data_cleaning_preparation/019-spreadsheets-vs-sql>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/introduction-to-python-and-programming-fundamentals/ <https://insightful-data-lab.com/2023/12/06/introduction-to-python-and-programming-fundamentals/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: basics

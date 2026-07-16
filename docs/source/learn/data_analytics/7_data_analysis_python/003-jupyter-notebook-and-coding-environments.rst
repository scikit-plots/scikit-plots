:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-003:
.. _da-7-python-python-003:

========================================================================
Jupyter Notebook and Coding Environments
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐍 Python Fundamentals` :bdg-info:`Lesson 003`

◀ :doc:`Previous <002-python-fundamentals>` · :doc:`Next <004-object-oriented-programming-oop-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Where analysts write Python
-----------------------------

Code needs somewhere to be written and run, and the *environment* an analyst uses shapes
how they work. For data analysis, the **Jupyter Notebook** is the most common
environment, alongside other coding tools. This lesson covers where and how analysts
write Python, and why the notebook in particular suits data work.

What a coding environment provides
------------------------------------

An environment for writing and running code offers, at minimum, a place to write code,
a way to run it, and a way to see the results. The main options for data analysts:

- **Jupyter Notebook** — an interactive, cell-based environment (below), dominant in
  data analysis.
- **IDEs (Integrated Development Environments)** — full-featured tools like VS Code or
  PyCharm, better for larger programs and software development.
- **The interactive interpreter / scripts** — running Python directly or as ``.py``
  script files, for automation and production code.

Each suits different work; for exploratory data analysis, the notebook is usually the
starting point.

The Jupyter Notebook
---------------------

Jupyter Notebook is an interactive environment that organises code into **cells** —
blocks of code you run individually, seeing each cell's output immediately below it:

- **Cells run independently** — write code in a cell, run it, see the result, then write
  the next cell building on it. This *interactive, incremental* workflow suits
  exploration: try something, see what happens, adjust.
- **Output appears inline** — results, tables, and (crucially) *visualizations* display
  right below the cell that produced them, keeping code and results together.
- **Mixes code and narrative** — notebooks combine code cells with text (Markdown)
  cells, so an analysis can be documented alongside the code that produces it — code,
  results, and explanation in one document.

This cell-based, inline-output, documented style is why Jupyter dominates data
analysis: it matches how analysis is actually done — exploring incrementally, seeing
results immediately, and documenting the reasoning.

Why the environment matters for analysis
------------------------------------------

The notebook's fit for analysis is not incidental. Exploratory analysis is iterative
(the analysis-process theme) — you try a transformation, examine the result, try
another — and the notebook's run-a-cell-see-the-result loop matches that rhythm exactly.
Inline visualizations mean charts appear where you make them; mixed narrative means the
analysis is self-documenting (the documentation discipline). The environment shapes the
work, and Jupyter shapes it toward the interactive, visual, documented style good
analysis wants.

The caveat
------------

The notebook's strengths carry matching weaknesses. Because cells can be run in *any
order*, a notebook can reach a state that is not reproducible from top to bottom — you
ran cells out of sequence, and re-running fresh gives different results, a subtle
reproducibility trap exactly opposite to the reproducibility Python promises. Notebooks
also suit exploration better than production: automated, scheduled, or large software is
usually better as ``.py`` scripts. The disciplines that address this — periodically
restarting and running the notebook top to bottom to confirm it reproduces, and moving
mature code into scripts — matter because the notebook's flexibility can quietly
undermine the reproducibility that is Python's point. Use the notebook for its
interactive strengths, but guard the reproducibility it can erode. The next lesson turns
to a model underlying Python itself: objects.

.. hint::

   - :doc:`Python Fundamentals <002-python-fundamentals>`
   - :doc:`Introduction to Python and Programming Fundamentals <001-introduction-to-python-and-programming-fundamentals>`
   - :doc:`Object-Oriented Programming (OOP) in Python <004-object-oriented-programming-oop-in-python>`
   - :doc:`Comments, Algorithms, and Docstrings in Python <010-comments-algorithms-and-docstrings-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/jupyter-notebook-and-coding-environments/ <https://insightful-data-lab.com/2023/12/06/jupyter-notebook-and-coding-environments/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: basics

:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-009:
.. _data-analytics-python-009:
.. _da-foundations-python-009:
.. _da-decisions-python-009:
.. _da-prep-python-009:
.. _da-cleaning-python-009:
.. _da-analyze-python-009:
.. _da-viz-python-009:
.. _da-python-python-009:
.. _da-jobsearch-python-009:

========================================================================
Code Reusability, Modularity, and Clean Code in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐍 Python Fundamentals` :bdg-info:`Lesson 009`

◀ :doc:`Previous <008-functions-in-python>` · :doc:`Next <010-comments-algorithms-and-docstrings-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Writing code that lasts
-------------------------

Code that merely *works* is not enough; code that is **reusable, modular, and clean**
lasts — it can be understood, maintained, and built upon. These qualities, drawn straight
from the foundations' engineering principles, apply directly to Python, and this lesson
covers writing code that embodies them, closing the basics stage.

Reusability
------------

**Reusable** code is written once and used many times, rather than copied. Functions
(the previous lesson) are the primary tool: a well-designed function captures a piece of
logic that any part of a program can call. The test of reusability is the
duplication-elimination principle — if you find yourself copying code, that code should
become a function. Reusable code means a change to the logic happens in *one* place,
which is both less work and less error-prone than editing many copies.

Modularity
-----------

**Modular** code is organised into distinct, self-contained pieces — functions, and (as
programs grow) *modules* (separate files) — each responsible for one thing. The
single-responsibility principle from the foundations applies: each function or module
does one well-defined job, and the pieces combine into the whole. Modularity makes code
easier to understand (each piece is small and focused), test (verify pieces
independently), and change (modify one piece without disturbing others) — the
break-into-components principle, in code.

Clean code
-----------

**Clean** code is readable and clear — code written for humans to understand, not just
for the computer to run. Its marks are the ones this course has stressed throughout:

- **Meaningful names** — variables and functions named for what they are and do (the
  naming lesson).
- **Small, focused functions** — each doing one thing, named for it.
- **Consistent style** — following conventions (PEP 8) so code looks familiar.
- **Clarity over cleverness** — a straightforward solution over a clever, cryptic one
  (the foundations' principle, stated for code).
- **Appropriate documentation** — comments and docstrings where they aid understanding
  (the next lesson).

Clean code is a discipline of writing for the *next reader* — often your future self.

Why these matter
-----------------

Reusability, modularity, and cleanliness are what separate code that can *evolve* from
code that becomes a liability. Analysis code is rarely written once and discarded — it is
rerun, adapted, and extended, often by someone other than its author. Code with these
qualities supports that; code without them (duplicated, tangled, cryptic) resists it,
accumulating the technical debt the foundations warned against. The small disciplines —
factor out duplication, keep pieces focused, name and structure for clarity — compound
into code that remains workable as it grows.

The caveat
------------

These principles can be *over*-applied, the over-engineering the foundations cautioned
against. Factoring every two-line snippet into a function, splitting a simple script into
many modules, or adding abstraction for flexibility never needed makes code *harder* to
follow, not easier — the indirection costs more than the duplication it removes. The
judgement is proportionate: apply reusability and modularity where they genuinely aid a
program of real size and complexity, and keep a simple script simple. Clean code is not
maximally abstracted code; it is code that is as clear and simple as the problem allows,
which for a small analysis may be a single straightforward script. Match the engineering
to the scale. The next lesson covers documenting code.

.. hint::

   - :doc:`Functions in Python <008-functions-in-python>`
   - :doc:`Comments, Algorithms, and Docstrings in Python <010-comments-algorithms-and-docstrings-in-python>`
   - :doc:`Naming Conventions and Restrictions in Python <006-naming-conventions-and-restrictions-in-python>`
   - :doc:`Data Types and Type Conversion in Python <007-data-types-and-type-conversion-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/code-reusability-modularity-and-clean-code-in-python/ <https://insightful-data-lab.com/2023/12/06/code-reusability-modularity-and-clean-code-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: basics

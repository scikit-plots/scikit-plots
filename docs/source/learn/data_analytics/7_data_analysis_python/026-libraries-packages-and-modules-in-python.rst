:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-026:
.. _data-analytics-python-026:
.. _da-foundations-python-026:
.. _da-decisions-python-026:
.. _da-prep-python-026:
.. _da-cleaning-python-026:
.. _da-analyze-python-026:
.. _da-viz-python-026:
.. _da-python-python-026:
.. _da-jobsearch-python-026:

========================================================================
Libraries, Packages, and Modules in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐼 NumPy & pandas` :bdg-info:`Lesson 026`

◀ :doc:`Previous <025-sets-in-python>` · :doc:`Next <027-introduction-to-numpy-and-vectorization>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Standing on others' code
--------------------------

Python's true power for data analysis comes not from the language alone but from its vast
ecosystem of **libraries** — reusable code others have written, imported and used rather
than reinvented. Opening the final stage, this lesson covers libraries, packages, and
modules: what they are, how to import them, and why they make Python the dominant data tool.

Modules, packages, libraries
------------------------------

The terms nest:

- A **module** is a single file of Python code (functions, classes, values) that can be
  imported and used elsewhere.
- A **package** is a collection of modules organised together (a directory of related
  modules).
- A **library** is, loosely, a body of reusable code — a package or set of packages —
  providing functionality for some purpose (numerical computing, data analysis,
  visualization).

The distinctions are technical; in practice "library" is used broadly for importable code
you use rather than write. The point is *reuse at scale* — leveraging code the community has
built, tested, and maintained.

Importing
---------

Code from a library is brought in with ``import``:

.. code-block:: python

   import math                     # import a module
   math.sqrt(16)                   # use it with the module name: 4.0

   import numpy as np              # import with an alias (convention)
   np.array([1, 2, 3])             # use via the alias

   from statistics import mean     # import a specific name
   mean([1, 2, 3])                 # use it directly: 2

``import numpy as np`` brings in the NumPy library under the short alias ``np`` (a universal
convention); ``from ... import ...`` brings a specific name into direct use. These import
forms are how every library in the coming lessons is accessed — ``import numpy as np``,
``import pandas as pd`` are the standard incantations.

The data ecosystem
-------------------

Python's data-analysis dominance rests on its libraries, the key ones being:

- **NumPy** — fast numerical computing on arrays (the next lessons).
- **pandas** — tabular data analysis, the DataFrame (the lessons after).
- **matplotlib** / others — visualization.
- and a vast constellation for statistics, machine learning, and more.

These libraries do the heavy lifting — decades of optimised, tested code — so an analyst
composes existing powerful tools rather than building from scratch. This is the reuse
principle at the ecosystem scale: the reason Python is *the* data language is largely this
library ecosystem.

The caveat
------------

Libraries are indispensable but bring their own considerations. **Dependencies** — the
libraries your code relies on — must be installed and, importantly, *versioned*: code
written for one version of a library may behave differently or break on another, so managing
which versions are used matters for reproducibility (the reproducibility theme, at the
dependency level). Importing many libraries also has a cost, and reaching for a heavy library
for a task the standard tools handle is its own over-engineering. The disciplines: use
well-established, maintained libraries; be aware of version compatibility; and import what the
task genuinely needs. The ecosystem is Python's great strength, used deliberately. The next
lessons dive into the first essential library: NumPy.

.. hint::

   - :doc:`Introduction to NumPy and Vectorization <027-introduction-to-numpy-and-vectorization>`
   - :doc:`Object-Oriented Programming (OOP) in Python <004-object-oriented-programming-oop-in-python>`
   - :doc:`Introduction to Pandas (Data Analysis Library) <029-introduction-to-pandas-data-analysis-library>`
   - :doc:`Advanced Dictionary Usage in Python <024-advanced-dictionary-usage-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/libraries-packages-and-modules-in-python/ <https://insightful-data-lab.com/2023/12/06/libraries-packages-and-modules-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: libraries

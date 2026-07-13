:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-006:
.. _data-analytics-python-006:
.. _da-foundations-python-006:
.. _da-decisions-python-006:
.. _da-prep-python-006:
.. _da-cleaning-python-006:
.. _da-analyze-python-006:
.. _da-viz-python-006:
.. _da-python-python-006:
.. _da-jobsearch-python-006:

========================================================================
Naming Conventions and Restrictions in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐍 Python Fundamentals` :bdg-info:`Lesson 006`

◀ :doc:`Previous <005-variables-in-python>` · :doc:`Next <007-data-types-and-type-conversion-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Naming well and legally
-------------------------

Variable names are subject to *rules* (what Python allows) and *conventions* (what good
practice recommends), and both matter — the rules because breaking them is an error, the
conventions because they make code readable. This lesson covers Python's naming
**restrictions** and **conventions**, applying the "good names" principle to code.

The restrictions (what Python requires)
-----------------------------------------

Python enforces rules for valid names:

- **Allowed characters** — names may contain letters, digits, and underscores
  (``_``), and must *start* with a letter or underscore, not a digit. ``sales_2024`` is
  valid; ``2024_sales`` is not.
- **No spaces** — names cannot contain spaces (use underscores: ``tax_rate``, not
  ``tax rate``).
- **Case-sensitive** — ``sales``, ``Sales``, and ``SALES`` are three different names.
- **No reserved keywords** — Python's keywords (``if``, ``for``, ``class``, ``def``,
  ``True``, and so on) cannot be used as names, since they have special meaning.

Breaking these rules is a *syntax error* — the code will not run — so the restrictions
are non-negotiable.

The conventions (what good practice recommends)
-------------------------------------------------

Beyond legality, Python has widely-followed conventions (codified in the style guide
**PEP 8**) that make code readable:

- **snake_case for variables and functions** — lowercase words joined by underscores:
  ``total_sales``, ``tax_rate``, ``customer_count``. This is the standard Python style.
- **Descriptive names** — ``total_sales`` over ``ts`` or ``x``; a name should say what
  the value *is*. Clarity over brevity (the foundations' principle).
- **UPPER_CASE for constants** — values meant not to change (``TAX_RATE = 0.08``) by
  convention use all caps.
- **Avoid confusing names** — not single letters like ``l`` (looks like ``1``), not
  names that shadow built-ins (``list``, ``sum``, ``type``), which causes subtle bugs.

Conventions are not enforced by Python — code violating them still runs — but following
them makes code readable to yourself and others, which is why they are near-universal.

Why naming matters
-------------------

Good names are documentation that cannot go out of date. ``monthly_revenue`` needs no
comment to explain it; ``mr`` or ``x`` does. As programs grow, the difference between
well-named and cryptically-named code is the difference between maintainable and
baffling — the exact parallel to well-named versus cryptic data columns. Naming is a
small, constant discipline with a large cumulative payoff in readability, which is why it
is worth doing deliberately from the start.

The caveat
------------

The two failure modes are opposite. *Under-naming* — cryptic abbreviations, single
letters, meaningless names — makes code unreadable. But *over-naming* — names so long
they clutter (``the_total_amount_of_sales_for_the_northern_region_this_year``) — harms
readability too, and can tempt breaking the no-spaces rule. The balance is names
*descriptive enough to be clear, concise enough to read* — the same judgement as naming
anything. And a name that shadows a built-in (``sum = 5`` overwriting the ``sum``
function) is legal but causes confusing bugs, so convention rightly warns against it.
Name for the reader, within the rules. The next lesson covers the types of values names
can hold.

.. hint::

   - :doc:`Variables in Python <005-variables-in-python>`
   - :doc:`Data Types and Type Conversion in Python <007-data-types-and-type-conversion-in-python>`
   - :doc:`Code Reusability, Modularity, and Clean Code in Python <009-code-reusability-modularity-and-clean-code-in-python>`
   - :doc:`Python Fundamentals <002-python-fundamentals>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/naming-conventions-and-restrictions-in-python/ <https://insightful-data-lab.com/2023/12/06/naming-conventions-and-restrictions-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: basics

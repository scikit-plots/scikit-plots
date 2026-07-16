:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-018:
.. _da-7-python-python-018:

========================================================================
String Formatting with .format() in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`📚 Strings & Data Structures` :bdg-info:`Lesson 018`

◀ :doc:`Previous <017-string-indexing-and-slicing-in-python>` · :doc:`Next <019-data-types-vs-data-structures-and-introduction-to-lists>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Building strings from values
------------------------------

Analysts constantly build strings that *incorporate values* — a label with a number, a
message with a name, a formatted report line. Python's **string formatting** does this
cleanly, and this lesson covers the ``.format()`` method and the modern **f-string**, the
tools for inserting values into text without clumsy concatenation.

Why not just concatenate
--------------------------

You *can* build strings with ``+``, but it gets awkward, especially with non-string values:

.. code-block:: python

   region = "North"
   sales = 1000
   # clumsy: requires converting the number, easy to mangle spacing
   msg = "Region " + region + " had " + str(sales) + " in sales"

The conversions (``str(sales)``) and spacing make this error-prone and hard to read.
Formatting solves it.

The .format() method
---------------------

``.format()`` inserts values into *placeholders* (``{}``) in a template string:

.. code-block:: python

   msg = "Region {} had {} in sales".format(region, sales)
   # "Region North had 1000 in sales"

   # placeholders can be numbered or named for clarity:
   msg = "Region {0} had {1} in sales".format(region, sales)
   msg = "Region {r} had {s} in sales".format(r=region, s=sales)

Each ``{}`` is filled, in order, by the arguments to ``.format()`` — which handles the
type conversion automatically (the number becomes text without an explicit ``str()``). This
is cleaner and clearer than concatenation.

f-strings: the modern way
--------------------------

Modern Python offers **f-strings** — the most concise formatting, marking the string with
``f`` and putting expressions directly in the braces:

.. code-block:: python

   msg = f"Region {region} had {sales} in sales"
   # "Region North had 1000 in sales"

   total = 1234.5678
   msg = f"Total: {total:.2f}"     # "Total: 1234.57" — formatted to 2 decimals

The f-string embeds the variable (or any expression) right in the braces — the most
readable option, now the common Python idiom. It also supports *format specifiers*
(``:.2f`` for two decimal places), which control how numbers display — decimal places,
thousands separators, percentages — the presentation formatting from earlier sections,
in code.

Why formatting matters
-----------------------

Producing readable, correctly-formatted text output is constant in data work — labels for
charts, messages, report lines, formatted numbers. Formatting (especially f-strings) makes
this clean and reliable, handling conversions and controlling number display without the
fragility of manual concatenation. It is a small skill used constantly, and f-strings in
particular are worth adopting as the default.

The caveat
------------

Formatting is straightforward but has minor pitfalls. Forgetting the ``f`` prefix on an
f-string leaves the braces as *literal text* (``"{region}"`` appears verbatim rather than
the value) — a common confusion. Format specifiers (``:.2f`` and the like) have their own
small syntax to learn, and misapplying one can misformat output. And formatting controls
*display*, not the underlying value — ``f"{total:.2f}"`` shows two decimals but does not
round the stored ``total`` (the display-versus-value distinction from the spreadsheet
formatting lesson). Use formatting for clean output, remembering it shapes appearance, not
data. The next lesson turns to the first structure for holding many values: the list.

.. hint::

   - :doc:`String Indexing and Slicing in Python <017-string-indexing-and-slicing-in-python>`
   - :doc:`Strings in Python <016-strings-in-python>`
   - :doc:`Data Types and Type Conversion in Python <007-data-types-and-type-conversion-in-python>`
   - :doc:`Data Types vs Data Structures & Introduction to Lists <019-data-types-vs-data-structures-and-introduction-to-lists>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/string-formatting-with-format-in-python/ <https://insightful-data-lab.com/2023/12/06/string-formatting-with-format-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: structures

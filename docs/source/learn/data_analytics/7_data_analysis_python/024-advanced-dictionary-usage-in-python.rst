:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-024:
.. _da-7-python-python-024:

========================================================================
Advanced Dictionary Usage in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`📚 Strings & Data Structures` :bdg-info:`Lesson 024`

◀ :doc:`Previous <023-dictionaries-in-python>` · :doc:`Next <025-sets-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Dictionaries for real data
----------------------------

Basic dictionaries store and retrieve by key; *real* data work needs more — iterating over a
dictionary's contents, handling missing keys safely, and nesting dictionaries for structured
data. This lesson covers advanced dictionary usage, the techniques that make dictionaries
practical for actual data, closing the core structures.

Iterating dictionaries
-----------------------

Dictionaries are iterated by keys, values, or both:

.. code-block:: python

   customer = {"name": "Jane", "region": "North", "sales": 1000}

   for key in customer:                 # iterate keys
       print(key)

   for key, value in customer.items():  # iterate key-value pairs
       print(key, value)

   customer.keys()          # the keys
   customer.values()        # the values
   customer.items()         # the key-value pairs

``.items()`` is the common way to loop over a dictionary's contents — each iteration
unpacking a key and its value (tuple unpacking again). This is how you process every field of
a record, or every entry of a mapping.

Safe access with .get()
-------------------------

The ``KeyError`` from accessing a missing key is avoided with ``.get()``, which returns a
default instead of erroring:

.. code-block:: python

   customer["phone"]              # KeyError if 'phone' is missing — stops the program
   customer.get("phone")          # returns None if missing — safe
   customer.get("phone", "N/A")   # returns "N/A" if missing — safe with a default

``.get(key, default)`` is the *safe* way to read a dictionary when a key might be absent —
exactly the missing-data handling that ``COALESCE`` provided in SQL, in dictionary form.
Using ``.get()`` where fields may be missing prevents the crash that direct access would
cause.

Nested dictionaries
-------------------

Dictionaries can contain dictionaries (and lists), representing *structured*, hierarchical
data:

.. code-block:: python

   data = {
       "north": {"sales": 1000, "customers": 50},
       "south": {"sales": 800,  "customers": 40},
   }
   data["north"]["sales"]         # 1000 — access nested by chaining keys

Nesting represents data with structure — regions each holding their own metrics — and is
exactly the shape of JSON and much real-world data. Accessing nested data chains the keys
(``data["north"]["sales"]``), reaching down through the levels.

Why advanced usage matters
----------------------------

These techniques are what make dictionaries usable for *real* data rather than toy examples.
Iterating processes records field by field; ``.get()`` handles the missing fields real data
always has; nesting represents the hierarchical structure real data often takes (especially
data from web APIs and JSON). Together they turn the dictionary from a simple lookup into a
practical tool for structured data — a bridge toward the DataFrames of pandas, which
generalise these key-value, record-oriented ideas to full tables.

The caveat
------------

Advanced dictionary use concentrates the earlier pitfalls plus new ones. Nested access
*multiplies* the ``KeyError`` risk — ``data["west"]["sales"]`` fails if *either* "west" or
"sales" is missing, so deep access into possibly-incomplete data needs ``.get()`` at each
level (or careful checking), lest a single missing key crash the program. Deeply nested
dictionaries also grow *hard to navigate* — many levels of keys become as tangled as deeply
nested anything, and at that point a more structured representation (or a DataFrame) is often
better. Use ``.get()`` for anything that might be missing, keep nesting to what the data
genuinely requires, and reach for pandas when dictionary-of-dictionaries starts to strain.
The next lesson covers the last core structure, the set.

.. hint::

   - :doc:`Dictionaries in Python <023-dictionaries-in-python>`
   - :doc:`Sets in Python <025-sets-in-python>`
   - :doc:`Libraries, Packages, and Modules in Python <026-libraries-packages-and-modules-in-python>`
   - :doc:`For Loops in Python <014-for-loops-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/advanced-dictionary-usage-in-python/ <https://insightful-data-lab.com/2023/12/06/advanced-dictionary-usage-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: structures

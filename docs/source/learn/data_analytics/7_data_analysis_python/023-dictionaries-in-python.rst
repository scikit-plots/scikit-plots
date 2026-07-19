:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-023:
.. _da-7-python-python-023:

========================================================================
Dictionaries in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`📚 Strings & Data Structures` :bdg-info:`Lesson 023`

◀ :doc:`Previous <022-advanced-use-of-loops-lists-tuples-and-list-comprehension>` · :doc:`Next <024-advanced-dictionary-usage-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Looking values up by key
--------------------------

Lists and tuples hold values in *order*, accessed by position. But often you want to look a
value up by a *meaningful key* — a customer's name, a product code, a field label — rather
than a numeric position. The **dictionary** is Python's key-value structure, and it is one
of the most important for data work. This lesson covers dictionaries.

What a dictionary is
---------------------

A **dictionary** stores **key-value pairs** — each value is associated with a key that
identifies it, written with curly braces:

.. code-block:: python

   customer = {
       "name": "Jane Smith",
       "region": "North",
       "sales": 1000,
   }

Values are looked up *by key*, not position:

.. code-block:: python

   customer["name"]          # "Jane Smith"
   customer["sales"]         # 1000

Where a list answers "what is at position 2?", a dictionary answers "what is the value for
'name'?" — access by meaningful key rather than numeric index. This makes dictionaries ideal
for representing *records* with named fields.

Modifying dictionaries
-----------------------

Dictionaries are mutable — pairs can be added, changed, and removed:

.. code-block:: python

   customer["email"] = "jane@example.com"    # add a new key-value pair
   customer["sales"] = 1200                    # update an existing value
   del customer["region"]                      # remove a pair

Assigning to a key either adds it (if new) or updates it (if it exists); ``del`` removes a
pair. Dictionaries grow and change like lists, but keyed rather than ordered.

Why dictionaries matter
------------------------

Dictionaries are fundamental to data work for several reasons. They represent *records*
naturally — a row of data as field-name-to-value pairs (``{"name": ..., "sales": ...}``),
which is exactly how structured data is often held. They enable *fast lookup* by key (far
faster than searching a list). And they are the structure behind much of Python's data
ecosystem — JSON data is dictionaries, pandas DataFrames can be built from them, and
configuration and mappings are dictionaries. Understanding key-value access is understanding
a core pattern of representing and retrieving structured data.

The caveat
------------

Dictionaries have specific rules and pitfalls. **Keys must be unique** — assigning to an
existing key *overwrites* its value rather than adding a second, so duplicate keys silently
lose data. **Keys must be immutable** — strings, numbers, and tuples can be keys, but lists
cannot (their mutability would break the dictionary's lookup), which is one reason tuples
exist. And the classic error: **accessing a key that does not exist** (``customer["phone"]``
when there is no phone) raises a ``KeyError`` and stops the program — a frequent bug when
data may be missing a field. The next lesson covers handling this and other advanced
dictionary usage safely. Keys unique and immutable, and access defensively.

.. hint::

   - :doc:`Advanced Dictionary Usage in Python <024-advanced-dictionary-usage-in-python>`
   - :doc:`Data Types vs Data Structures & Introduction to Lists <019-data-types-vs-data-structures-and-introduction-to-lists>`
   - :doc:`Tuples in Python <021-tuples-in-python>`
   - :doc:`Sets in Python <025-sets-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/dictionaries-in-python/ <https://insightful-data-lab.com/2023/12/06/dictionaries-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: structures

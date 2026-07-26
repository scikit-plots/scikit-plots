:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-021:
.. _da-7-python-python-021:

========================================================================
Tuples in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`📚 Strings & Data Structures` :bdg-info:`Lesson 021`

◀ :doc:`Previous <020-modifying-lists-in-python>` · :doc:`Next <022-advanced-use-of-loops-lists-tuples-and-list-comprehension>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Ordered and unchangeable
--------------------------

A **tuple** is like a list — an ordered collection of values — but with one crucial
difference: it is **immutable**, unchangeable after creation. Tuples serve where a
collection should be fixed, and understanding them (and why immutability is sometimes
wanted) rounds out the sequence structures. This lesson covers tuples.

Creating and using tuples
---------------------------

A tuple is written with parentheses (or just commas):

.. code-block:: python

   point = (3, 5)
   rgb = (255, 128, 0)
   record = ("North", 1000, 2024)     # a mixed tuple

Tuples are accessed exactly like lists — by index, with slicing:

.. code-block:: python

   point[0]          # 3
   record[1]         # 1000
   record[-1]        # 2024
   len(record)       # 3

Everything about *reading* a tuple mirrors a list; the difference is entirely in
*changing* it.

Immutability
------------

A tuple *cannot be changed* after creation — no adding, removing, or updating elements:

.. code-block:: python

   point = (3, 5)
   point[0] = 10     # ERROR — tuples do not support item assignment

This immutability is the tuple's defining property, and it is the same immutability strings
have. Where a list is a *changeable* ordered collection, a tuple is a *fixed* one — the two
are otherwise similar.

Why use a tuple
----------------

If tuples are just unchangeable lists, why use them? Immutability is a *feature* in the
right situations:

- **Fixed data that should not change** — coordinates, RGB colours, a fixed record — where
  accidental modification would be a bug. The immutability *protects* the data.
- **Meaning and intent** — using a tuple signals "this collection is fixed," documenting
  intent to readers.
- **Dictionary keys** — tuples can serve as dictionary keys (the next lessons) where lists
  cannot, precisely because they are immutable and stable.
- **Multiple return values** — functions often return several values as a tuple
  (``return x, y``), a common Python idiom.

The choice between list and tuple is the choice between *changeable* and *fixed*: use a
list when the collection will change, a tuple when it should not.

The caveat
------------

The tuple-versus-list choice is easy to get wrong in either direction: using a tuple for
data that *does* need to change forces awkward workarounds (you cannot modify it), while
using a list for data that should be *fixed* forgoes the protection immutability gives. The
guidance is intent-based — will this collection change during its life? Changeable → list;
fixed → tuple. And a subtle trap: a tuple's *immutability is shallow* — a tuple cannot be
reassigned, but if it *contains* a mutable object (a list inside a tuple), that inner object
can still change. For the flat collections of typical data work this rarely bites, but it is
worth knowing that immutability applies to the tuple's own structure, not necessarily to
everything within it. The next lesson combines the structures with loops and introduces list
comprehension.

.. hint::

   - :doc:`Modifying Lists in Python <020-modifying-lists-in-python>`
   - :doc:`Data Types vs Data Structures & Introduction to Lists <019-data-types-vs-data-structures-and-introduction-to-lists>`
   - :doc:`Advanced Use of Loops, Lists, Tuples & List Comprehension <022-advanced-use-of-loops-lists-tuples-and-list-comprehension>`
   - :doc:`Strings in Python <016-strings-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/tuples-in-python/ <https://insightful-data-lab.com/2023/12/06/tuples-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: structures

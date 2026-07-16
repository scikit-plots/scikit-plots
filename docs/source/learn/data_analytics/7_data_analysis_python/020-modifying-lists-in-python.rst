:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-020:
.. _da-7-python-python-020:

========================================================================
Modifying Lists in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`📚 Strings & Data Structures` :bdg-info:`Lesson 020`

◀ :doc:`Previous <019-data-types-vs-data-structures-and-introduction-to-lists>` · :doc:`Next <021-tuples-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Changing lists in place
-------------------------

Unlike strings, lists are **mutable** — they can be *changed in place* after creation:
items added, removed, or updated. This mutability makes lists the flexible, dynamic
structure they are, and this lesson covers the operations that modify them — the everyday
tools for building and updating collections.

Adding elements
---------------

Several methods add to a list:

.. code-block:: python

   sales = [100, 250]
   sales.append(300)         # add one item to the end: [100, 250, 300]
   sales.insert(0, 50)       # insert at a position: [50, 100, 250, 300]
   sales.extend([400, 500])  # add multiple items: [50, 100, 250, 300, 400, 500]

``append`` adds a single item to the end (the most common); ``insert`` places an item at a
given index; ``extend`` appends all items of another list. These grow a list as data
arrives — the accumulate pattern from the for-loop lesson often uses ``append``.

Removing elements
-----------------

Methods remove items:

.. code-block:: python

   sales.remove(50)          # remove the first matching value
   popped = sales.pop()      # remove and return the last item
   popped = sales.pop(0)     # remove and return the item at an index
   del sales[0]              # delete the item at an index

``remove`` deletes by *value* (the first match); ``pop`` deletes by *position* and returns
the removed item; ``del`` deletes by position. These shrink a list as items are consumed
or filtered.

Updating and other operations
-------------------------------

Items are updated by assigning to an index, and lists have further useful methods:

.. code-block:: python

   sales[0] = 999            # update the item at index 0
   sales.sort()              # sort the list in place (ascending)
   sales.reverse()           # reverse the order in place
   count = sales.count(250)  # count occurrences of a value

Assigning to ``sales[0]`` changes that element; ``sort`` and ``reverse`` reorder the list
*in place* (modifying the original, not returning a new list). These operations make lists
dynamic — reorderable, updatable collections.

Mutability: the key property
------------------------------

The defining feature is that these operations change the list *in place* — unlike string
methods, which return new strings. ``sales.append(300)`` modifies ``sales`` directly (no
reassignment needed); ``sales.sort()`` reorders ``sales`` itself. This in-place mutability
is what makes lists efficient for building and updating collections, and it is the direct
contrast to strings' immutability from the earlier lesson — a distinction worth holding
clearly, because it changes how you use each.

The caveat
------------

Mutability is powerful and *hazardous*, in ways that catch even experienced programmers.
Because a list is changed in place, if two variables refer to the *same* list (``b = a``
makes ``b`` another name for ``a``'s list, not a copy), modifying one changes the other —
the "shared reference" surprise, a classic source of baffling bugs. To get an independent
copy, you must explicitly copy the list (``b = a.copy()``). Also, in-place methods like
``sort()`` return ``None``, not the sorted list, so ``sales = sales.sort()`` mistakenly sets
``sales`` to ``None`` — the opposite mistake to strings (where you *must* reassign). The
disciplines: copy a list when you need an independent one, and remember in-place methods
modify rather than return. The next lessons cover tuples and further structures.

.. hint::

   - :doc:`Data Types vs Data Structures & Introduction to Lists <019-data-types-vs-data-structures-and-introduction-to-lists>`
   - :doc:`Advanced Use of Loops, Lists, Tuples & List Comprehension <022-advanced-use-of-loops-lists-tuples-and-list-comprehension>`
   - :doc:`Tuples in Python <021-tuples-in-python>`
   - :doc:`Strings in Python <016-strings-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/modifying-lists-in-python/ <https://insightful-data-lab.com/2023/12/06/modifying-lists-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: structures

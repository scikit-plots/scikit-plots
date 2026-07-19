.. _Writing in RST - Task 4:

********************
Task 4: Nested Lists
********************


.. contents:: Table of Contents

RST Nested Lists
================

Lists are `simple to create <https://sublime-and-sphinx-guide.readthedocs.io/en/latest/lists.html>`_.

- Create an unordered list (bullet list) using ``*``, ``-``, or ``+``.
- Create an enumerated list with numbers using ``#.`` or start the list at
  a specific value, such as 8 using ``8.``

  x. Create `other types of enumerated lists
     <https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#enumerated-lists>`_
     using common item designators, such as ``A.`` or ``a.``.
  #. Sphinx automatically enumerates the value using ``#.``
  #. Unfortunately, RST rendering engines `do not always render
     <https://github.com/github/markup/issues/515>`_
     the parenthesis or Roman Numerals correctly.

You can nest list or add additional content by indenting element.

    .. note:: Add a blank line above and below each level

    .. code-block:: RST

        - A bullet list

          #. Nested number list.
          #. Nested item 2.

        - A bullet list item 2.

          Paragraph 2 of item 2.

          * Nested bullet list.
          * Nested item 2.

            A. Third level staring with 'A'
            #. Third level item B.

          * Nested item 3.


References
----------

- https://sublime-and-sphinx-guide.readthedocs.io/en/latest/lists.html
- https://docutils.sourceforge.io/docs/user/rst/quickref.html#bullet-lists

----


Task
============

**Task 4**: Add installation instructions to learn how to use nested lists.

* Use `ordered or unordered lists <https://sphinx-rtd-theme.readthedocs.io/en/stable/demo/lists_tables.html>`_
* Add a ``caution`` box not to install both versions, but to chose one.
* Use ``monospace text`` for the installation commands.

----


Text to Add
============

    .. code-block:: text

        You can choose to install the Sphinx version (1.7.x) using the apt package
        manager or you install the latest version (2.x) using from PyPI using.

        Caution: Installing both versions could result in a conflict.

        1. Update your repository and upgrade any out-of-date packages using
           apt-get update && apt upgrade.
        2. Install a version of Sphinx:

          - Sphinx version 1.x

            a. Install Sphinx version 1: apt-get -y install python3-sphinx

          - Sphinx version 2.x

            a. Install pip3: ``apt-get -y install python3-pip``
            b. Install Sphinx version 2: ``pip3 install Sphinx``

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/workshops/rst/writing-rst-4.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.

.. -*- coding: utf-8 -*-

=====================
Markdown Cheat Sheet
=====================

:Info:    Markdown essentials — quick, copy/paste-friendly reminders.
:Sources: John-CD Markdown cheatsheet + referenced Markdown/GFM links.

.. contents:: Table of contents
   :depth: 2
   :local:


Resources
=========

* Markdown main site: https://daringfireball.net/projects/markdown/
* GitHub Flavored Markdown (GFM) guide:
  https://guides.github.com/features/mastering-markdown/
* John-CD cheatsheet page:
  https://john-cd.com/cheatsheets/Markup_and_Documentation/Markdown/


Basics
======

Paragraphs
----------

A paragraph is one or more consecutive lines of text separated by one or more
blank lines. A blank line contains nothing but spaces or tabs.

Do **not** indent normal paragraphs with spaces or tabs.

Line breaks
-----------

Most Markdown renderers wrap lines inside a paragraph. To force a hard line
break, many flavors require two trailing spaces at line end (or an explicit
``<br>`` in HTML-capable renderers).

Common “starter” syntax
-----------------------

::

   # Header 1
   ## Header 2
   ### Header 3

   - Bulleted
   - List

   1. Numbered
   2. List

   **Bold** and _Italic_ and `Code` text

   [Link](url) and ![Image](src)


Emphasis
========

::

   *single asterisks*
   _single underscores_
   **double asterisks**
   __double underscores__

Emphasis can be used in the mi\*dd\*le of a word.


Headers
=======

ATX-style headers
-----------------

::

   # H1
   ## H2
   ### H3
   #### H4
   ##### H5
   ###### H6

Setext-style headers
--------------------

::

   Alt-H1
   ======

   Alt-H2
   ------


Lists
=====

Bulleted lists
--------------

::

   - Item
   - Item
     - Nested item (indent to nest)
   - Item

Numbered lists
--------------

::

   1. First
   2. Second
      1. Nested number (some renderers require re-numbering)
   3. Third


Links and images
================

Inline link
-----------

::

   [Text for the link](URL)

Reference-style link
--------------------

::

   This is [an example][id] reference-style link.
   [id]: https://example.com/  "Optional Title Here"

Image
-----

::

   ![Alt text](/path/to/img.jpg "Optional title")


Code
====

Inline code span
----------------

Use backticks:

::

   `inline code`

Indented code block
-------------------

Indent at least **4 spaces** (or a tab):

::

       def hello(name):
           return f"Hello, {name}"

Fenced code block (syntax highlighted, common in GFM)
-----------------------------------------------------

Use triple backticks; many renderers support an optional language tag:

::

   ```python
   def wiki_rocks(text):
       formatter = lambda t: "funky" + t
       return formatter(text)
   ```

Displayed as:

::

   def wiki_rocks(text):
       formatter = lambda t: "funky" + t
       return formatter(text)


Blockquotes
===========

::

   > This is a blockquote with two paragraphs.
   >
   > Second paragraph.


Escaping special characters
===========================

Use a backslash to escape Markdown punctuation when needed:

::

   \*literal asterisk\*
   \_literal underscore\_
   \`literal backtick\`


GitHub Pages (note)
===================

GitHub Pages uses the layout and styles from the Jekyll theme selected in your
repository settings. The theme name is stored in the Jekyll ``_config.yml``
configuration file.


Bitbucket (note)
================

Bitbucket often runs Markdown in a restricted/safe mode (HTML may be removed or
sanitized). Replace, remove, or escape HTML tags appropriately.

Some Bitbucket/Python-Markdown setups use ``:::{language}`` for code blocks:

::

   :::python
   friends = ['john', 'pat', 'gary', 'michael']
   for i, name in enumerate(friends):
       print("iteration {iteration} is {name}".format(iteration=i, name=name))


Cloning a Bitbucket Wiki
========================

::

   git clone https://bitbucket.org/MY_USER/MY_REPO/wiki

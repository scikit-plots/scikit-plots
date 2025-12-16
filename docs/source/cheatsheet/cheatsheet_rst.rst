.. -*- coding: utf-8 -*-

============================
reStructuredText Cheat Sheet
============================

:Info:    A practical, copy/paste-friendly reStructuredText (reST) cheat sheet.
:Sources: John-CD cheatsheet + Docutils quickref/cheatsheet (see links below).

.. contents:: Table of contents
   :depth: 3
   :local:


Resources
=========

* reStructuredText (Docutils): https://docutils.sourceforge.io/rst.html
* Quick reference: https://docutils.sourceforge.io/docs/user/rst/quickref.html
* Cheat sheet (Docutils): https://docutils.sourceforge.io/docs/user/rst/cheatsheet.html
* John-CD cheatsheet page:
  https://john-cd.com/cheatsheets/Markup_and_Documentation/reStructuredText/


reST short overview
===================

Formatting conventions
----------------------

* Indentation is significant; keep indentation consistent within a block.
* Many style guides recommend a max line length of ~80 characters for prose
  (tables, long links, and deeply-indented code may exceed that).
* No tabs.

Paragraphs
----------

Paragraphs are chunks of text separated by one or more blank lines.

Section headers
---------------

Section headers are created by underlining (and optionally overlining) the
section title with punctuation characters, at least as long as the title:

::

   =================
   This is a heading
   =================

A common “adornment ladder” (choose one consistent scheme per project):

* ``#`` with overline, for parts
* ``*`` with overline, for chapters
* ``=`` for sections
* ``-`` for subsections
* ``^`` for subsubsections
* ``"`` for paragraphs


Inline text styles
------------------

* One asterisk: ``*text*`` for emphasis (italics)
* Two asterisks: ``**text**`` for strong emphasis (bold)
* Double backquotes: ````text```` for inline code/literals
* Escape markup characters with a backslash: ``\*literal asterisk*``


Lists
-----

Bulleted list:

* This is a bulleted list.
* It has two items, the second
  item uses two lines.

Numbered list:

1. This is a numbered list.
2. It has two items too.

Auto-numbered list:

#. This is a numbered list.
#. It has two items too.

Nested lists are possible, but keep a blank line between a parent list item
and a nested list.


Source code blocks
==================

Literal blocks via double colon
-------------------------------

A paragraph ending with ``::`` introduces a literal block:

::

   This is a normal text paragraph. The next paragraph is a code sample::

       It is not processed in any way, except
       that the indentation is removed.

       It can span multiple lines.

   This is a normal text paragraph again.

Sphinx-friendly code blocks
---------------------------

Sphinx supports syntax highlighting with ``code-block``:

.. code-block:: python

   def hello(name: str) -> str:
       return f"Hello, {name}!"


Links
=====

Inline external link:

`Link text <https://target>`_

Named references:

See reStructuredText_.

.. _reStructuredText: https://docutils.sourceforge.io/rst.html

Anonymous reference:

`anonymous reference`__

__ https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html


Definitions
===========

Definition lists:

::

   term (up to a line of text)
      Definition of the term, which must be indented and
      can even consist of multiple paragraphs

   next term
      Description.


Footnotes and citations
=======================

Footnote references:

::

   Lorem ipsum [#]_ dolor sit amet ... [#]_

Footnote bodies (explicit markup):

.. [#] Text of the first footnote.
.. [#] Text of the second footnote.


Use of reST in Python docstrings
================================

Sphinx/ReST docstring fields (common in Python projects):

::

   :param volume_id: The ID of the EBS volume to be attached.
   :type volume_id: str

   :param instance_id: The ID of the EC2 instance.
   :type instance_id: str

   :return: `Reverse geocoder return value`_ dictionary giving closest
      address(es) to ``(lat, lng)``
   :rtype: dict
   :raises GoogleMapsError: If the coordinates could not be reverse geocoded.

   Keyword arguments and return value are identical to those of :meth:`geocode()`.

   .. _`Reverse geocoder return value`:
      https://code.google.com/apis/maps/documentation/geocoding/index.html#ReverseGeocoding

Notes:

* Normal docstring conventions apply (see PEP 257).
* Identifier references go in ``backticks``.
* ``>>>`` starts a doctest and is formatted as code.
* Link to other objects with roles like :py:meth:`mymethod`, :func:`myfunc`,
  :class:`myclass`, and :mod:`mymodule`.


Explicit markup
===============

An explicit markup block begins with a line starting with ``..`` followed by
whitespace, and ends at the next paragraph with the same indentation level.
Keep a blank line between explicit markup and normal paragraphs.

Examples:

::

   .. sectionauthor:: Guido van Rossum <guido@python.org>

   .. rubric:: Footnotes

   .. [#] Text of the first footnote.
   .. [#] Text of the second footnote.

   :mod:`parrot` -- Dead parrot access
   ===================================

   .. module:: parrot
      :platform: Unix, Windows
      :synopsis: Analyze and reanimate dead parrots.

   .. moduleauthor:: Eric Cleese <eric@python.invalid>
   .. moduleauthor:: John Idle <john@python.invalid>

   .. function:: repeat([repeat=3[, number=1000000]])
      repeat(y, z)
      :bar: no

      Return a line of text input from the user.

   .. class:: Spam

      Description of the class.

      .. data:: ham

         Description of the attribute.


Inline markup roles
===================

General role forms:

* ``:rolename:`content```  (e.g., ``:class:`dict``)
* ``:role:`title <target>```  (custom link text)

Shortened method display:

``:meth:`~Queue.Queue.get``` refers to ``Queue.Queue.get`` but displays only
``get`` as the link text.

Object roles (common)
---------------------

mod
   The name of a module (a dotted name may be used). Also used for packages.

func
   The name of a Python function; dotted names may be used. The role text
   should not include trailing parentheses (they are stripped when searching
   for identifiers).

data
   The name of a module-level variable or constant.

const
   The name of a “defined” constant (e.g., C ``#define`` or a constant-like
   Python value).

class
   A class name; a dotted name may be used.

meth
   A method name. Role text should include the type name and the method name.
   A dotted name may be used.

attr
   The name of a data attribute.

exc
   The name of an exception; a dotted name may be used.


Official Docutils cheat sheet (excerpt)
=======================================

The Docutils cheat sheet is a compact “syntax reminders” page, with examples
for tables, block quotes, inline markup, directives, and roles. For the
canonical version, see the Docutils page linked in *Resources* above.

Below is a copy/paste-friendly excerpt (as commonly shared):

::

   =====================================================
    The reStructuredText_ Cheat Sheet: Syntax Reminders
   =====================================================
   :Info: See <https://docutils.sf.net/rst.html> for introductory docs.
   :Author: David Goodger <goodger@python.org>
   :Date: $Date: 2013-02-20 01:10:53 +0000 (Wed, 20 Feb 2013) $
   :Revision: $Revision: 7612 $
   :Description: This is a "docinfo block", or bibliographic field list

   .. NOTE:: If you are reading this as HTML, please read
      `<cheatsheet.txt>`_ instead to see the input syntax examples!

   Section Structure
   =================
   Section titles are underlined or overlined & underlined.

   Body Elements
   =============
   Grid table:

   +--------------------------------+-----------------------------------+
   | Paragraphs are flush-left,     | Literal block, preceded by "::":: |
   | separated by blank lines.      |                                   |
   |                                |     Indented                      |
   |     Block quotes are indented. |                                   |
   +--------------------------------+ or::                              |
   | >>> print 'Doctest block'      |                                   |
   | Doctest block                  | > Quoted                          |
   +--------------------------------+-----------------------------------+
   | | Line blocks preserve line breaks & indents. [new in 0.3.6]       |
   | |     Useful for addresses, verse, and adornment-free lists; long  |
   |       lines can be wrapped with continuation lines.                |
   +--------------------------------------------------------------------+

   Simple tables:

   ================  ============================================================
   List Type         Examples (syntax in the `text source <cheatsheet.txt>`_)
   ================  ============================================================
   Bullet list       * items begin with "-", "+", or "*"
   Enumerated list   1. items use any variation of "1.", "A)", and "(i)"
         #. also auto-enumerated
   Definition list   Term is flush-left : optional classifier
         Definition is indented, no blank line between
   Field list        :field name: field body
   Option list       -o  at least 2 spaces between option & description
   ================  ============================================================

   Inline Markup
   =============
   *emphasis*; **strong emphasis**; `interpreted text`; `interpreted text
   with role`:emphasis:; ``inline literal text``; standalone hyperlink,
   https://docutils.sourceforge.net; named reference, reStructuredText_;
   `anonymous reference`__; footnote reference, [1]_; citation reference,
   [CIT2002]_; |substitution|; _`inline internal target`.

   Directive Quick Reference
   =========================
   See <https://docutils.sf.net/docs/ref/rst/directives.html> for full info.

   Interpreted Text Role Quick Reference
   =====================================
   See <https://docutils.sf.net/docs/ref/rst/roles.html> for full info.

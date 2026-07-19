.. _Writing in RST - Task 7:

******************************
Task 7: Menus and Navigation
******************************

.. contents:: Table of Contents

Sphinx can automatically build a site navigation menu (called the
*TOC Tree*) and a table of contents on each page. These use different
directives.

`A StackOverflow user <https://stackoverflow.com/a/39770866>`_ describes
the difference between the two TOCs as:

    `.. contents <http://docutils.sourceforge.net/docs/ref/rst/directives.html#table-of-contents>`_
    is a doctutils directive (the underlying library which defines ReST
    and associated utilities) and automatically generates a table of
    contents from headlines within the current topic.

    `.. toctree <http://www.sphinx-doc.org/en/stable/markup/toctree.html>`_
    is a Sphinx-defined directive in which you explicitly list
    documents whose TOCs will be listed out.


TOC Tree (Site Menu)
--------------------
RST uses directive ``..toctree::`` to generate the `site navigation tree
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html>`_.
Some useful parameters are ``depth`` and ``caption``.

The ``depth`` parameter determines how many heading levels to show in the TOC.
The default value of ``:maxdepth: 2`` includes heading 1 and 2 labels.

.. note:: You must add files to the TOC Tree manually. Sphinx will
    generate a warning if a file is not included in the TOC Tree.

#. First, you include the ``..toctree::`` directive in a file
   (usually in an index file for a documentation section)

   - The default ``index.rst`` file contains the root TOC file.
   - You can build on that file.

#. Then, you list the RST files that you want to include in the TOC.
#. You can include files at the same level or in subdirectories.


   .. code-block:: RST
       :caption: Default index.rst file
       :emphasize-lines: 8-12

       Welcome to Demo's documentation!
       ================================

       .. toctree::
           :maxdepth: 2
           :caption: Contents:

           getting_started
           user_guide/index
           downloads/index
           faq
           contact_us

       Indices and tables
       ==================

       * :ref:`genindex`
       * :ref:`modindex`
       * :ref:`search`

   .. hint:: Include an index file in a sub-directory that includes a
       ``toctree`` for that directory.

       In the example above, the ``user_guide.rst`` file could contain
       its own TOC that gets included here.


#. You can use *globbing*. See the `Sphinx documentation
   <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html>`_
   to learn about the topic.

   .. code-block:: RST

       .. toctree::
           :glob:

           intro*
           recipe/*
           *

Page TOC
--------

Each page can have a TOC using `.. contents
<http://docutils.sourceforge.net/docs/ref/rst/directives.html#table-of-contents>`_
directive.

The ``depth`` parameter determines how many heading levels to show in the TOC.
Omit to show all headings in the document.

.. code-block:: rst
    :caption: Sample Usage

    **********
    Page Title
    **********

    .. contents:: Table of Contents


    Section 1
    =========

References
-----------

- https://docutils.sourceforge.io/docs/ref/rst/directives.html#table-of-contents
- https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
- https://docutils.sourceforge.io/docs/ref/rst/directives.html#table-of-contents

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/workshops/rst/writing-rst-7.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.

.. _Writing in RST - Task 5:

********************
Task 5: Code Blocks
********************

.. contents:: Table of Contents

RST Code Blocks
========================

Code blocks let you display literal text with or with or without
syntax highlighting.

There are are several ways `to show code examples
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#showing-code-examples>`_
or preformatted text.

  * The ``::`` directive is a generic code of preformatted text. Sphinx tries
    to guess at the language for syntax highlighted.
  * The ``.. code-block:: language`` directive adds syntax highlighting
    and args.
  * RST supports many languages! Here is the list:

     - http://pygments.org/languages/
     - http://pygments.org/docs/lexers/

  * Code blocks can contain `additional parameters
    <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#showing-code-examples>`_,
    such as:

    - ``:linenos:``
    - ``:lineno-start:``
    - ``:emphasize-lines:``
    - ``:caption:``
    - ``:dedent:``

Here are some examples:


    .. code-block:: rst
        :caption: Generic block and syntax highlighting

        ::

            This text is in a code block. Sphinx will use generic
            text highlighting.

    .. code-block:: text
        :caption: Plain text block

        .. code-block:: text
            :caption: Plain text block

            <p>This text will have no styling. However, you can add
            parameters to the block.</p>

    .. code-block:: rst
        :caption: Basic code block

        .. code-block:: HTML

            <span class="bold>sample HTML</span>

    .. code-block:: rst
        :caption: Code block with syntax highlighting and line numbers
        :linenos:
        :emphasize-lines: 4

        .. code-block:: bash
            :caption: Code block with syntax highlighting and line numbers
            :linenos:
            :emphasize-lines: 9,10,34,39,46

            # This code block uses bash syntax highlighting
            # Also, a code-block has args for line numbers,
            # highlighting lines, and captions!


References
----------

- https://sublime-and-sphinx-guide.readthedocs.io/en/latest/code_blocks.html
- https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#showing-code-examples
- https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#what-are-directives

----


Task
============


**Task 5**: Add configuration instructions to learn about code blocks.

#. Add ``important`` and ``note`` boxes
#. Add a ``bash`` code block for the Sphinx configuration options.

   - Add a caption
   - Add line numbers and highlighting on lines 9,10,34,39,46.

----


Text to Add
============

    .. code-block:: text

        Important: Create and then change to the directory of your Sphinx
        code and RST content. The default installation directory is ``./``,
        which is the current directory.

        #. Create a project folder, then navigate to that folder

           * For example: ``mkdir -p /opt/sphinx && cd /opt/sphinx``

        #. Execute command ``sphinx-quickstart``
        #. Follow the prompts (from version 1.x)

           We've highlighted the lines that we entered a value *other than*
           default. Add the extensions that you find useful.

    .. code-block:: bash
        :caption: Output of 'sphinx-quickstart'
        :linenos:
        :emphasize-lines: 9,10,34,39,46

        > Separate source and build directories (y/n) [n]:

        Inside the root directory, two more directories will be created; "_templates"
        for custom HTML templates and "_static" for custom stylesheets and other static
        files. You can enter another prefix (such as ".") to replace the underscore.
        > Name prefix for templates and static dir [_]:

        The project name will occur in several places in the built documentation.
        > Project name: Conference Demo
        > Author name(s): BilimEdtech
        > Project release []:

        If the documents are to be written in a language other than English,
        you can select a language here by its language code. Sphinx will then
        translate text that it generates into that language.

        For a list of supported codes, see
        http://sphinx-doc.org/config.html#confval-language.
        > Project language [en]:

        The file name suffix for source files. Commonly, this is either ".txt"
        or ".rst".  Only files with this suffix are considered documents.
        > Source file suffix [.rst]:

        One document is special in that it is considered the top node of the
        "contents tree", that is, it is the root of the hierarchical structure
        of the documents. Normally, this is "index", but if your "index"
        document is a custom template, you can also set this to another filename.
        > Name of your master document (without suffix) [index]:
        Indicate which of the following Sphinx extensions should be enabled:
        > autodoc: automatically insert docstrings from modules (y/n) [n]:
        > doctest: automatically test code snippets in doctest blocks (y/n) [n]:
        > intersphinx: link between Sphinx documentation of different projects (y/n) [n]:
        > todo: write "todo" entries that can be shown or hidden on build (y/n) [n]: y
        > coverage: checks for documentation coverage (y/n) [n]:
        > imgmath: include math, rendered as PNG or SVG images (y/n) [n]:
        > mathjax: include math, rendered in the browser by MathJax (y/n) [n]:
        > ifconfig: conditional inclusion of content based on config values (y/n) [n]:
        > viewcode: include links to the source code of documented Python objects (y/n) [n]: y
        > githubpages: create .nojekyll file to publish the document on GitHub pages (y/n) [n]:

        A Makefile and a Windows command file can be generated for you so that you
        only have to run e.g. `make html` instead of invoking sphinx-build
        directly.
        > Create Makefile? (y/n) [y]:
        > Create Windows command file? (y/n) [y]: n

    .. code-block:: text

        Note:

            * The configurations for your Sphinx project are in file ``conf.py``.
            * Delete file ``conf.py`` to run ``quickstart`` again.

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/workshops/rst/writing-rst-5.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.

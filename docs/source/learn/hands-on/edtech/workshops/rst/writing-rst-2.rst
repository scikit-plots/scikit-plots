.. _Writing in RST - Task 2:

***************************
Task 2: Paragraph and Links
***************************

.. contents:: Table of Contents


Paragraphs and Text Block
=========================

Paragraphs are blocks of text with a blank line between then.
Line breaks
are ignored.

Blockquotes are intended paragraphs or other text blocks

    Here is a block quote.

    Here is the second paragraph of the block quote.

.. code-block:: rst

    Paragraphs are blocks of test with a blank line between then.
    Line breaks
    are ignored.

    Blockquotes are indented paragraphs or other text blocks

        Here is a block quote.

        Here is the second paragraph of the block quote.

.. note:: You will learn quickly that spaces and line breaks are very
    important in restructuredtext.



RST Links
---------

External Links
^^^^^^^^^^^^^^

RST provides several options for building links to external sites,
two of which are `Direct link or target definition
<https://sublime-and-sphinx-guide.readthedocs.io/en/latest/references.html>`_.
Sphinx automatically links raw URLs

Direct link (single use):

.. code-block:: rst

    `Link text <link URL>`_

    Get the latest news at `BBC <https://bbc.com/>`_.

Target definition (reuse):

.. code-block:: rst

    Get the latest news at `BBC`_.  You can get more than just
    news at `BBC`_, such as British culture and entrainment.

    .. _BBC: https://bbc.com/

Internal Links
^^^^^^^^^^^^^^

You can link to a specific heading by name *if* all heading
text is unique. We've found this unlikely in a larger project.

A way to build links to headings is to use a custom anchor that `goes
above a heading directly
<https://sublime-and-sphinx-guide.readthedocs.io/en/latest/references.html#use-a-custom-anchor>`_.

This code block defines two targets that you can link to directly using
the ``::ref:`Label``` command. By default, Sphinx displays the heading
text in the link.

.. code-block:: rst

    .. _RST Overview:

    Overview
    **********

    RST Overview content


    .. _Sphinx Overview:

    Overview
    *********

    Sphinx Overview content


Build the links using the ``:ref:`Label``` command.

.. code-block:: rst

    Content editors should read the RST :ref:`RST Overview`
    to learn about RST.

    System admins should read the Sphinx :ref:`Sphinx Overview`
    to learn how to configure the system.

For example, this page has a target defined as ``.. _Writing in RST - Task 2:``
that we can link to from any page in sphinx. Clicking
:ref:`Writing in RST - Task 2` will take you to the top of this page.

References
----------

- https://sublime-and-sphinx-guide.readthedocs.io/en/latest/references.html
- https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#internal-and-external-links
- https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst


----


Task
============

**Task 2**: Add the following text to the *overview* section to learn
about using paragraphs, inline markup, and creating hyperlinks.

* Use inline styles
* Link the URL to text *Install Sphinx guide*

----

Text to Add
============

    .. code-block:: text

        **Sphinx** is a platform that *renders RST files* to a web application
        that is suitable for documentation. This guide provides a quick
        installation guide for someone wanting to set up Sphinx on a VSP.

        The install sphinx guide <https://www.sphinx-doc.org/en/master/usage/quickstart.html>
        provides a good starting point, but we will make other changes to install
        the latest version of Sphinx from PyPI.

        The VPS setup for this workshop uses Docker image nginx\:latest
        <https://hub.docker.com/_/nginx>, which uses Debian 10.

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/workshops/rst/writing-rst-2.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.

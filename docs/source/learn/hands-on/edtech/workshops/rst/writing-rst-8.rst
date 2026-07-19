.. _Writing in RST - Task 8:

*************************
Task 8: Advanced Topics
*************************


You've learned the basics of writing in RST. There is much more to learn!
This page contains examples of how to use lesser-known or advanced features.


.. contents:: Table of Contents


Embedding HTML
==============

RST extends beyond the markdown basis to include embedding HTML.
You might want to make a link with ``target="_blank"`` attribute
or embed a video using specific HTML.

There are two ways. You can place the directive in the location where
you want the HTML to render or use `a substitution
<https://sublime-and-sphinx-guide.readthedocs.io/en/latest/reuse.html>`_.
Substitutions are a good way to reuse code!

Use the `.. raw:: <https://docutils.sourceforge.io/docs/ref/rst/directives.html#raw-data-pass-through>`_
directive to place HTML in your document directly, such as inserting a
horizontal rule with a specific width.

.. code-block:: rst

    .. raw:: html

        <hr width=50 size=10>

Or, you can create a substitution for the ``<br />`` element.
Then, you can insert the raw HTML using the ``|BR|`` substitution
anywhere in your document.

.. code-block:: rst

    .. |BR| raw:: html

        <br />

    Now, I can finally control line breaks! |BR| This text is on the next line.

Just to verify, the |BR| tag works |BR| as |BR| expected.


.. |BR| raw:: html

    <br />

Reference:

- https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#substitutions


Includes
============

RST provides the `.. include::
<https://docutils.sourceforge.io/docs/ref/rst/directives.html#including-an-external-document-fragment>`_
directive to read the contents of the referenced file.

.. code-block:: rst

    .. include:: ../shared/folder/filename.rst

References:

- https://sublime-and-sphinx-guide.readthedocs.io/en/latest/reuse.html
- https://docutils.sourceforge.io/docs/ref/rst/directives.html#including-an-external-document-fragment



Tables
============

Tables are useful but can be hard to write in RST using the default method.
This guide will show you some options.

Grid and simple tables have their place but are hard to use.
Here are two references if you are interested

    #. https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#tables
    #. https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#grid-tables

`List Tables <https://sublime-and-sphinx-guide.readthedocs.io/en/latest/tables.html#list-table-directive>`_
are a good option if you need to add content blocks in a cell.
Below is the raw markup from the tables with the colored boxes section
on :ref:`Writing in RST - Task 3`. The width parameter seems to have
no effect. Perhaps it is a problem with the RTD theme.

.. code-block:: rst

    .. list-table:: Colored Box Types
       :header-rows: 1
       :widths: 25 15 65

       * - Type/Keyword
         - Color/Note
         - Example
       * - danger, error
         - red
         - ``.. danger:: Watch out!``
       * - attention, caution, warning
         - orange
         - ``.. Caution:: Subroutine is not cross-platform.``
       * - hint, important, tip
         - blue
         - ``.. Note:: A generic blue box``
       * - admonition
         - blue
         -
           .. code-block:: rst

               .. admonition:: Custom label

                   The contents of the blue box.


CVS Tables are the most convenient if you need to display tabular data.
You can see an example from this site on page
:ref:`Step 1. Initialize OpenVPN using Docker`. As you can see from the
raw RST, you can develop your data in a spreadsheet and then paste
do the document directly.

Here is the raw RST.

.. code-block:: rst

   .. csv-table:: Suggested UDP Ports
     :header: "UPD Port Number", "Description"
     :widths: 1, 10

       22,Secure Shell (SSH)
       53,Domain Name System (DNS)
       123,Network Time Protocol (NTP)
       465,
       554,Real Time Streaming Protocol (RTSP)
       943,
       972,
       995,Post Office Protocol 3 over TLS/SSL  (POP3S)
       1935,Real Time Messaging Protocol (RTMP)
       1234,VLC media player default port for UDP/RTP stream
       10007,VoIP providers (ports 10000-20000)
       11211,Memcached
       3074,Xbox LIVE
       3748,
       5005,Real-time Transport Protocol media data (RTP)
       5730,
       8080,
       17500,Dropbox LanSync Protocol (db-lsp)
       25575,Minecraft

Reference

- https://sublime-and-sphinx-guide.readthedocs.io/en/latest/tables.html
- https://docutils.sourceforge.io/docs/ref/rst/directives.html#table
- https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#tables
- https://sublime-and-sphinx-guide.readthedocs.io/en/latest/tables.html

  - `Raw RST <https://raw.githubusercontent.com/readthedocs/sphinx_rtd_theme/a3ab477aaa23f3b7ab7d62c7abc2cc74102ab2b8/docs/demo/lists_tables.rst>`_

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/workshops/rst/writing-rst-8.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.

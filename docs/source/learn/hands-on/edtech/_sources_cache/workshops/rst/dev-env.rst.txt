********************************
Windows Development Environment
********************************

So, you want to start writing RST. This guide can help you get started.

.. contents:: Table of Contents

.. note:: These instructions assume Windows 10.

Development Overview
======================
RST tools haven't advanced as fast of Markdown tools have, but there
are some good options. A writer or developer working with RST needs
these tools:

#. A text editor that supports:

   Linting
       Detects warnings and errors; write using best practices.

   Syntax highlighting
       Improves code readability.

   IntelliSense
       Speeds up the writing process using blocks and snippets.

#. An interpreter or rendering engine

Text Editor
============
.. tip:: Use these tools:

    * `Visual Studio Code`_
    * `reStructuredText Extension for Visual Studio Code`_

`Visual Studio Code`_ (VSC) is a modern text editor with plugins that support
writing documents in RST. VSC has an abundance of extensions that are
easy to install.

The `reStructuredText Extension for Visual Studio Code`_ is highly
recommended because it adds appropriate support tools:

    * Syntax Highlighting
    * Code Snippets
    * Live Preview (not recommended, see below)
    * Section Builder
    * Linter
    * IntelliSense

Interpreter for Active Writing
===============================

.. tip:: Use Sphinx Server with LiveReload to speed up your
    development process.

Developers using RST need a way to preview their work. It is easy to make
mistakes in RST. A single spacing error (vertical or horizontal) can alter
the output drastically. Finding an easy solution to view the rendered
RST is important. Unfortunately, there are not many easy options. The
process might feel more like programming than writing documentation.

We have not had a good experience using the *live preview* tools in
Windows. For example, the `reStructuredText Extension for Visual Studio Code`_
includes a live preview tool, but can be slow slow.

Option 1 - Sphinx Server with LiveReload
-----------------------------------------
This option keeps the convenience of live preview but improves the
performance of trying to render RST in VSC. It creates an environment that:

    #. watches the active project for changes
    #. rebuilds the project automatically
    #. refreshes the browser window

.. note:: This example uses Ubuntu 18.04 in WSL.

1. Open a Linux shell.
#. Install Sphinx and extensions.

   .. code-block:: bash

      sudo apt-get update && apt upgrade
      sudo apt-get -y install python3-pip
      pip3 install Sphinx
      pip3 install sphinx-autobuild

#. Clone ``sphinx-server`` from https://github.com/dldl/sphinx-server.

   .. note:: This project needs to be in ``/opt/sphinx-server``

   .. code-block:: bash

       cd /opt
       sudo git clone https://github.com/dldl/sphinx-server.git

#. Copy file ``server.py`` to your project root.

   .. code-block:: bash

       cd /mnt/c/Users/user/source/rst/my-project
       cp /opt/sphinx-server/server.py .

#. Start the mini-server that uses ``autobuild`` and ``livereload``.

   .. code-block:: bash

       python3 server.py

   .. admonition:: python3 only

       Using ``python`` will generate this error:

       .. code-block:: bash

           python server.py
           Traceback (most recent call last):
             File "server.py", line 7, in <module>
               import http.server
           ImportError: No module named http.server

   .. tip:: Start ``python3 server.py`` using the built-in terminal in VSC!

       1. ``Terminal -> New Terminal``
       #. ``bash``
       #. ``python3 server.py``


#. Open your browser to ``http://localhost:8000/``.

Option 2 - Build Manually
-------------------------
You can build the project manually using ``make html`` from the Windows
command prompt or a Linux shell. VSC makes it easy to build using the
built-in terminal in Visual Studio Code.

Windows CMD or PowerShell
^^^^^^^^^^^^^^^^^^^^^^^^^
#. Install Python and Sphinx using `these instructions
   <https://docs.restructuredtext.net/articles/prerequisites.html>`_.
#. Open your project folder using VSC: ``File -> Open Folder``.
#. Open the terminal in Visual Studio Code and execute the ``make html``
   command.

   a. ``Terminal -> New Terminal``
   #. ``.\make.bat html``

#. Open the rendered HTML file in your browser.

Windows WSL
^^^^^^^^^^^
#. Install Python and Sphinx.

   .. code-block:: bash

       sudo apt-get -y install python3-pip
       pip3 install Sphinx

#. Open your project folder using VSC: ``File -> Open Folder``.
#. Open the terminal in Visual Studio Code, open Bash, and execute
   the ``make html``.

   a. ``Terminal -> New Terminal``
   #. ``bash``
   #. ``make html``

#. Open the rendered HTML file in your browser.

.. Section for URLs

.. _Visual Studio Code: https://code.visualstudio.com/
.. _reStructuredText Extension for Visual Studio Code: https://docs.restructuredtext.net/

:html_theme.sidebar_secondary.remove:

.. _env-index:

======================================================================
Environment builder
======================================================================

Describe a Python environment once and build the same environment on any
machine, with whichever tool the machine has: uv, ``venv`` + pip, pipenv,
micromamba, mamba or conda. The same request works from the command line
(``scikitplot env``), from Python (``scikitplot._env``), in CI and from an
AI agent::

    request  ->  plan (reads, changes nothing)  ->  approvals  ->  execute  ->  receipt

Runnable examples, each step from the command line and from Python:
:ref:`env__examples`.

.. toctree::
   :maxdepth: 2

   Getting started <getting_started.rst>
   Requests and configuration <configuration.rst>
   Targets: host, other platforms, browser, mobile <targets.rst>
   CI, containers and agents <automation.rst>
   How it works <how_it_works.rst>

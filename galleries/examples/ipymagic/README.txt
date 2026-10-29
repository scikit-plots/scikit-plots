.. _ipymagic__examples:

IPyMagic
========

.. currentmodule:: scikitplot._ipymagic

Examples for :py:mod:`~scikitplot._ipymagic`, the notebook frontend of
scikit-plots: every ``scikitplot`` command and a profiler as IPython magics,
in Jupyter, JupyterLab, VS Code, Colab or terminal IPython.

.. code-block:: ipython

    %load_ext scikitplot

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Magic
     - Does
   * - ``%scikitplot ARGS``
     - runs ``scikitplot ARGS``; returns a :class:`CommandResult`
   * - ``%%scikitplot ARGS`` + cell
     - the same, with the cell on standard input
   * - ``%scikitplot_profile STMT`` / ``%%scikitplot_profile`` + cell
     - profiles the code; returns a :class:`ProfileResult`
   * - ``%scikitplot_config key=value ...``
     - session settings (:class:`MagicSettings`)

**How to read these examples.** A gallery page runs as a plain Python script,
where magics do not exist. Each step therefore shows the magic in an
``ipython`` block, then runs the very same thing with the Python function the
magic calls (:func:`run_command`, :func:`profile_code`). In the Jupyter
notebook download, every cell magic (``%%scikitplot``,
``%%scikitplot_profile``) becomes a runnable cell next to its Python
equivalent, and the first cell loads the extension; line magics stay as text
to copy into a cell.

.. code-block:: text

    %scikitplot LINE [cell]
        ↓
    split_line       shell-like words ("my env" stays one word)
        ↓
    route            the CLI's own command registry; a typo is refused
        ↓
    in the kernel    info, doctor, show-versions ...  (logging restored)
    or a subprocess  env, mcp, cleanprompt, a cell, isolate=true
                     (streamed, timeout, interrupt, bounded capture)
        ↓
    CommandResult    exit_code, stdout, stderr, data (--format json)

Start with the commands, then the cell form (``cleanprompt``), the delegated
servers (``mcp``) and the profiler. See also :ref:`ipymagic-index`.

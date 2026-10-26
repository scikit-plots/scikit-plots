.. _ipymagic-automation:

======================================================================
CI, scripts and agents
======================================================================

Notebooks run in CI
----------------------------------------------------------------------

By default a failing command shows its error and the notebook goes on, as
with ``!command``. For a notebook that CI runs, stop at the first failure
and bound the time of long commands:

.. code-block:: ipython

   %scikitplot_config raise_on_error=true timeout=900

A failure then raises ``CommandFailedError`` (its ``.result`` holds the exit
status and the output), shown in two lines. A command stopped by the
timeout has exit status 124 and ``result.timed_out``.

==================  =========  ===============================================
Setting             Default    Meaning
==================  =========  ===============================================
``raise_on_error``  ``false``  a non-zero exit raises ``CommandFailedError``
``timeout``         ``none``   seconds a subprocess command may run
``isolate``         ``false``  run native commands in a subprocess too
``stream``          ``true``   show subprocess output as it comes
==================  =========  ===============================================

Without IPython
----------------------------------------------------------------------

The engine behind the magic is a function, for scripts, tests and agents:

.. code-block:: python

   import io

   from scikitplot._ipymagic import MagicSettings, run_command

   result = run_command("info --format json", out=io.StringIO())
   assert result.ok
   record = result.to_dict()  # argv, command, mode, exit_code, stdout, data ...
   assert record["mode"] == "in-process"

   strict = MagicSettings(raise_on_error=True, timeout=60)
   result = run_command(["env", "--format", "json", "doctor"], settings=strict,
                        out=io.StringIO())
   assert "backends" in result.data

Every run is also logged at ``DEBUG`` on the ``scikitplot._ipymagic._bridge``
logger: command, exit status, time, where it ran and the size of its output
(never the output itself).

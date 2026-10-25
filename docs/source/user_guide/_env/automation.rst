.. _env-automation:

======================================================================
CI, containers and agents
======================================================================

One file, every machine
----------------------------------------------------------------------

Commit ``scikitplot-env.toml`` and run the same command on a laptop, in CI
and in a container:

.. code-block:: console

   $ scikitplot env create --profile ci --yes

In a ``Dockerfile`` the environment is one layer, rebuilt only when the file
or a dependency file changes:

.. code-block:: dockerfile

   COPY scikitplot-env.toml requirements.txt ./
   RUN pip install scikit-plots && scikitplot env create --profile ci --yes

Offline and air-gapped machines
----------------------------------------------------------------------

``--offline`` (``network = "offline"``, preset ``airgap``) plans only steps
that need no network: uv and micromamba run with ``--offline``, pip with
``--no-index`` (give ``--find-links`` for a wheel directory), and a step
that would download is refused at planning time. To bootstrap micromamba on
a machine without network, copy the archive named in the error to
``~/.scikitplot/env/tools/cache``; its SHA-256 is checked as for a download.
A uv bootstrap is installed by pip with ``--require-hashes``: only a wheel
whose SHA-256 is pinned in the builder is accepted, from whatever index pip
is configured for.

Machine-readable output
----------------------------------------------------------------------

``--format json`` prints one JSON document on stdout: the plan, the result,
or ``{"ok": false, "error": {...}}``. ``--events FILE`` appends JSON Lines
events (``plan.started``, ``approval.granted``, ``step.started``,
``step.finished``, ``step.failed``, ``plan.finished``), each with a
``schema``, a sequence number and a time. ``-v`` and ``-vv`` log to stderr;
``-vv`` includes the tools' output.

Every error has a stable ``code`` and a ``hint``:

==============================  ==============================================
``code``                        Meaning
==============================  ==============================================
``config_invalid``              the request or a file is invalid (exit 2)
``config_authority``            a file tried to grant an approval (exit 2)
``config_schema``               a file declares another ``schema`` (exit 2)
``source_invalid``              a dependency file is invalid (exit 2)
``backend_unavailable``         no usable tool for this request (exit 3)
``target_unsupported``          the target cannot be built (exit 3)
``conversion_refused``          Conda packages for a PyPI-only backend (exit 3)
``network_forbidden``           offline, but a step needs the network (exit 3)
``plan_impossible``             another reason the plan cannot be made (exit 3)
``approval_required``           an approval was not given (exit 4)
``lock_busy``                   another run holds the environment (exit 5)
``step_failed``                 a command failed; ``step`` and ``returncode``
                                say which (exit 1)
``plan_stale``                  the environment appeared or disappeared while
                                the run waited for the lock; run again (exit 1)
``command_not_found``           ``run``: the command is not in the environment
                                nor on ``PATH`` (exit 127)
==============================  ==============================================

Agents
----------------------------------------------------------------------

An agent plans first (``scikitplot env plan --format json --profile
agent``), shows the plan or checks it, and passes approvals explicitly
(``--approve bootstrap``); it never needs a terminal, never gets a prompt,
and reads the outcome from ``--format json`` and the events. A batch of
requests runs from a JSON Lines file:

.. code-block:: json

   {"schema": "scikitplot.env.request/v1", "id": "docs", "request": {"name": "docs", "sources": ["docs/requirements.txt"]}}
   {"schema": "scikitplot.env.request/v1", "id": "lite", "request": {"name": "lite", "target": "pyodide", "packages": ["six"]}}

.. code-block:: console

   $ scikitplot env --format json batch requests.jsonl --dry-run --keep-going

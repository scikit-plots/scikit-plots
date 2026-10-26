.. _env-how-it-works:

======================================================================
How it works
======================================================================

The flow
----------------------------------------------------------------------

1. **Request.** Arguments, a preset and a file are merged into one
   immutable request; each value remembers where it came from.
2. **Plan.** The planner reads the dependency files, looks at the host
   (tools found, platform, Python) and returns a plan: each step's command,
   whether it uses the network, what it writes and whether it needs an
   approval. Nothing is changed.
3. **Approvals.** Every approval the plan needs is decided before the first
   step runs; a refusal changes nothing.
4. **Execute.** Under a lock next to the environment, the state is checked
   again (another run may have built it meanwhile: then nothing is done;
   an environment that appeared or vanished stops the run with
   ``plan_stale``). Each command runs with no terminal input and prompts
   off; ``PYTHONPATH`` and ``PYTHONHOME`` are removed from its environment.
   Conda-family tools get ``--override-channels``, so channels in a user's
   ``~/.mambarc`` or ``.condarc`` do not change the solve.
5. **Receipt.** On success the builder checks that the environment exists
   and writes ``.scikitplot-env.json`` into it: request, plan digest, state
   digest, backend and dependency-file hashes. The next ``create`` with the
   same state does nothing.

Choosing the backend
----------------------------------------------------------------------

``backend = "auto"`` follows fixed rules, and the plan names the rule it
applied (``backend_reason``):

1. a cross-build or browser target allows only micromamba and mamba;
2. a named backend is used as asked, after a check that it can do the job;
3. Conda packages or an ``environment.yml``: the first of micromamba, mamba,
   conda found on the host, else a micromamba bootstrap when allowed;
4. a single ``Pipfile`` with its ``Pipfile.lock`` and pipenv on the host:
   pipenv (``pipenv sync``, hashes checked);
5. otherwise: uv when found; ``venv`` when the requested Python is the
   running one; else a uv bootstrap when allowed; else micromamba when
   found.

A request no rule satisfies fails with the reason and a hint. With
``conversion = "allow-approximate"``, Conda packages may be installed from
PyPI by name (versions dropped, with a plan note); the default refuses.

An existing environment
----------------------------------------------------------------------

The builder writes only into an empty directory or an environment it
recognises (a ``pyvenv.cfg``, ``conda-meta`` or a receipt; a ``bin/python``
alone, as in ``/usr``, is not one); an environment without its Python is
recreated (with the ``remove`` approval). One built with another backend,
target or Python (from the request or a dependency file) is not changed in
place: the error asks for ``--recreate``. When a dependency is no longer
requested, the environment is recreated too (with the ``remove`` approval):
installers never uninstall, and an environment must not depend on its
history.

Safety
----------------------------------------------------------------------

* Tools are installed only into ``~/.scikitplot/env/tools`` and only with
  the ``bootstrap`` approval; micromamba downloads are https-only, size
  limited and checked against pinned SHA-256 values before extraction.
* Deleting checks again, at the moment of deletion, that the directory is
  an environment, and never the file system root or the home directory.
* Credentials are refused in URLs and redacted from logs, events and
  errors.

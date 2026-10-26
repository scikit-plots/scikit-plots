.. _env-configuration:

======================================================================
Requests and configuration
======================================================================

A request says *what* to build. It can come from arguments, from a file, or
both (arguments win). With neither ``--config`` nor ``-n``/``-p``, the builder
looks for ``scikitplot-env.toml`` (``.json``, ``.yaml``, ``.yml``) or a
``pyproject.toml`` with ``[tool.scikitplot.env]`` from the working directory
upwards, and stops at the project root (a ``pyproject.toml``, ``.git`` or
``.hg``): a file in a parent directory is never adopted.

.. code-block:: toml

   schema = "scikitplot.env/v1"
   name = "analysis"              # or: prefix = ".venv" (relative to this file)
   python = "3.12"
   backend = "auto"               # auto, uv, venv, pipenv, micromamba, mamba, conda
   packages = ["pandas>=2"]
   conda_packages = []
   sources = ["requirements.txt", { path = "pyproject.toml", groups = ["test"] }]
   target = "host"
   profile = "ci"

   [policy]
   interaction = "auto"           # never, auto, always
   network = "online"             # online, offline
   bootstrap = "never"            # never, if-missing
   conversion = "exact"           # exact, allow-approximate

   [verify]
   imports = ["pandas"]

The same content as JSON or YAML gives the same request and the same
digest. Unknown keys are refused, so a typo is an error, not a silently
ignored setting.

Dependency sources
----------------------------------------------------------------------

=====================  ======================================================
Source                 What is read
=====================  ======================================================
``requirements*.txt``  requirements, following ``-r``; pip and uv also get
                       the file itself, so its options (``--index-url``,
                       ``-c``) apply
``environment.yml``    Conda specs, the ``python`` pin, ``channels``, and the
                       ``pip:`` section (with ``-r`` and ``-e``)
``pyproject.toml``     ``[project] dependencies``, the extras and PEP 735
                       dependency groups named in ``groups``
``Pipfile``            ``[packages]`` (and ``[dev-packages]`` with group
                       ``dev``), ``[requires] python_version``
=====================  ======================================================

Files are read every time a plan is made and their hashes (``-r`` includes
and ``-c`` constraint files too, and ``Pipfile.lock`` for the pipenv backend)
are part of the environment's state: edit ``requirements.txt`` and the next
``create`` updates the environment. A ``pyproject.toml`` whose dependencies
are ``dynamic`` is refused (they exist only once the project is built);
list the project itself, ``packages = ["-e ."]``. ``environment.yml`` is parsed by the builder, so
the same file gives the same environment whichever backend builds it; keys
no backend can apply the same way (``variables``, pip options other than
``-r``/``-e``) are refused with a hint instead of being dropped.

Presets
----------------------------------------------------------------------

``profile`` applies a preset below the file and the arguments:

===============  ===========================================================
Preset           Sets
===============  ===========================================================
``ci``           ``interaction = "never"``
``agent``        ``interaction = "never"``, ``conversion = "exact"``
``airgap``       ``network = "offline"``, ``bootstrap = "never"``
``interactive``  ``interaction = "always"``
===============  ===========================================================

``scikitplot env validate --format json`` shows the request and, under
``origins``, which layer (default, preset, file, invocation) set each value.

Policies and approvals
----------------------------------------------------------------------

Policies say how the builder may behave; approvals say what it may change
outside the environment itself. Three things need an approval: installing a
tool (``bootstrap``), editing a shell start-up file (``shell_init``) and
deleting an environment (``remove``). They are given with ``--yes`` or
``--approve KIND`` on the command line, ``approve=`` in Python, or answered
at a terminal. A configuration file cannot approve anything: ``approve``,
``yes`` and similar keys are refused with ``code = "config_authority"``, so a
file committed by someone else cannot grant itself rights. The JupyterLite
add-on, configured by files only, reads its approvals from the
``SKPLT_ENV_APPROVE`` environment variable of whoever runs the build
(``SKPLT_ENV_APPROVE=bootstrap``).

``interaction = "auto"`` asks only at a terminal outside CI; ``never`` never
asks (a missing approval is an error); ``always`` asks at a terminal for
every approval that was not given beforehand, also in CI, and is an error
without a terminal (piped input is not an answer).

Credentials
----------------------------------------------------------------------

A URL with user info (``user:password@`` or a bare ``token@``) or an
anaconda.org ``/t/<token>/`` part is refused in every request field
(indexes, channels, ``find_links``, package specs); give credentials the way
pip, uv and micromamba read them (keyring, ``.netrc``, environment
variables). Plans, receipts, logs, events and errors are redacted, so a
token inside a dependency file does not leak either.

A ``Pipfile`` ``[[source]]`` index other than PyPI must also be given as
``index_url`` / ``extra_index_urls`` (or the ``pipenv`` backend used):
otherwise uv or pip would look its packages up on public PyPI, and the plan
is refused.

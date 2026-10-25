# scikitplot/_env/_api.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The public functions: doctor, plan, create, remove, run, shell_init, batch.

Notes
-----
**User notes.** Every function takes the request one of three ways:

* an :py:class:`~._core._request.EnvironmentRequest`;
* ``config=`` a configuration file (TOML, JSON, YAML, or ``pyproject.toml``
  with ``[tool.scikitplot.env]``) and/or request fields as keywords
  (``name="demo", packages=["numpy"]``), keywords winning;
* nothing: ``scikitplot-env.toml`` (``.json``, ``.yaml``) or a
  ``pyproject.toml`` with the table is looked up from the working directory
  upwards.

Approvals are only ever given here (``approve=``) or answered at a
terminal, never read from a file.

**Developer notes.** These functions only wire the pieces together: parse
(``_config``), plan (``_core._planner``), decide and run
(``_core._executor``). Keep logic out of this module.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ._backends import BACKENDS, get_backend
from ._bootstrap import MICROMAMBA
from ._config._schema import find_config, load_batch, load_request
from ._core._decisions import Decider
from ._core._errors import ConfigError, EnvError, ExecutionError, PlanningError
from ._core._events import EventSink
from ._core._executor import ExecutionResult, execute
from ._core._host import HostInfo, detect_host
from ._core._plan import EnvironmentPlan
from ._core._planner import plan_request, resolve_prefix
from ._core._receipt import read_receipt
from ._core._request import EnvironmentRequest
from ._targets import CONDA_SUBDIRS, TARGETS

__all__ = [
    "batch",
    "create",
    "doctor",
    "plan",
    "remove",
    "run",
    "shell_init",
    "validate",
]


def _request(
    request: EnvironmentRequest | None, config: Any, fields: Mapping[str, Any]
) -> EnvironmentRequest:
    if request is not None:
        if config is not None or fields:
            raise ConfigError(
                "pass either an EnvironmentRequest or config/fields, not both"
            )
        if not isinstance(request, EnvironmentRequest):
            raise ConfigError(
                "request must be an EnvironmentRequest",
                hint="use config= for a file or a mapping",
            )
        return request
    if config is None and fields.get("name") is None and fields.get("prefix") is None:
        config = find_config()
        if config is None:
            raise ConfigError(
                "no environment given",
                hint="pass name= or prefix=, config=, or add scikitplot-env.toml to the"
                " project",
            )
    return load_request(config, **fields)


def _sink(events: Any) -> EventSink:
    return events if isinstance(events, EventSink) else EventSink(events)


def doctor(*, host: HostInfo | None = None) -> dict[str, Any]:
    """
    Describe the host: tools found, backends usable, targets, homes.

    Parameters
    ----------
    host : HostInfo or None, default=None
        The host; detected when None.

    Returns
    -------
    dict
        ``host``, ``backends`` (``available``, ``path``, ``bootstrappable``,
        ``summary`` for each), ``targets`` and ``bootstrap_platforms``.

    Examples
    --------
    >>> from scikitplot._env import doctor
    >>> sorted(doctor()["backends"])
    ['conda', 'mamba', 'micromamba', 'pipenv', 'uv', 'venv']
    """
    host = detect_host() if host is None else host
    backends = {}
    for name in BACKENDS:
        backend = get_backend(name)
        path = (
            host.python_executable if backend.tool is None else host.tool(backend.tool)
        )
        backends[name] = {
            "available": bool(path) and host.can_subprocess,
            "path": path,
            "bootstrappable": backend.bootstrappable,
            "ecosystems": sorted(backend.ecosystems),
            "summary": backend.summary,
        }
    return {
        "host": host.to_dict(),
        "backends": backends,
        "targets": {
            **{k: t.summary for k, t in TARGETS.items()},
            **{s: f"cross-build for {s}" for s in CONDA_SUBDIRS},
            "android": "refused: an app embeds its own Python",
            "ios": "refused: an app embeds its own Python",
        },
        "bootstrap_platforms": sorted(MICROMAMBA),
    }


def validate(config: Any = None, **fields: Any) -> EnvironmentRequest:
    """
    Parse and check a request without reading dependency files or planning.

    Parameters
    ----------
    config : path-like, mapping or None, default=None
        Configuration; discovered when None and no ``name``/``prefix``.
    **fields
        Request fields.

    Returns
    -------
    EnvironmentRequest
        The validated request.

    Raises
    ------
    ConfigError
        For an invalid request.
    """
    return _request(None, config, fields)


def plan(
    request: EnvironmentRequest | None = None,
    *,
    config: Any = None,
    operation: str = "ensure",
    host: HostInfo | None = None,
    **fields: Any,
) -> EnvironmentPlan:
    """
    Plan an operation without changing anything.

    Parameters
    ----------
    request : EnvironmentRequest or None, default=None
        The request, or None to build it from ``config``/``fields``.
    config : path-like, mapping or None, default=None
        Configuration document.
    operation : {"ensure", "recreate", "remove"}, default="ensure"
        Operation.
    host : HostInfo or None, default=None
        The host; detected when None.
    **fields
        Request fields (``name``, ``packages``, ``policy`` ...).

    Returns
    -------
    EnvironmentPlan
        The plan (``plan.render()`` for people, ``plan.to_dict()`` for tools).

    Raises
    ------
    EnvError
        ``ConfigError``, ``SourceError`` or a ``PlanningError`` subclass.

    Examples
    --------
    >>> import tempfile
    >>> from scikitplot._env import plan
    >>> p = plan(prefix=tempfile.mkdtemp() + "/env", backend="venv")
    >>> p.backend, [s.id for s in p.steps]
    ('venv', ['create'])
    """
    return plan_request(
        _request(request, config, fields), operation=operation, host=host
    )


def _execute(
    plan_: EnvironmentPlan,
    *,
    approve: Any,
    interaction: str | None,
    events: Any,
    runner: Callable | None,
    lock_timeout: float,
) -> ExecutionResult:
    policy = plan_.request.get("policy", {})
    decider = Decider(interaction or policy.get("interaction", "auto"), approve=approve)
    sink = _sink(events)
    try:
        return execute(
            plan_,
            decider=decider,
            events=sink,
            runner=runner,
            lock_timeout=lock_timeout,
        )
    finally:
        if sink is not events:
            sink.close()


def create(
    request: EnvironmentRequest | None = None,
    *,
    config: Any = None,
    recreate: bool = False,
    approve: Any = (),
    interaction: str | None = None,
    events: Any = None,
    host: HostInfo | None = None,
    runner: Callable | None = None,
    lock_timeout: float = 0.0,
    **fields: Any,
) -> ExecutionResult:
    """
    Create the environment, or bring it up to date (idempotent).

    Parameters
    ----------
    request : EnvironmentRequest or None, default=None
        The request, or None to build it from ``config``/``fields``.
    config : path-like, mapping or None, default=None
        Configuration document.
    recreate : bool, default=False
        Delete and build again (needs approval ``remove``).
    approve : bool, str, sequence of str or callable, default=()
        Pre-given approvals: ``True``, kinds (``"bootstrap"``, ``"remove"``)
        or ``approve(kind, message) -> bool``.
    interaction : {"never", "auto", "always"} or None, default=None
        Overrides the request's interaction policy for the approvals.
    events : path-like, text stream, EventSink or None, default=None
        Where JSON Lines events go.
    host : HostInfo or None, default=None
        The host; detected when None.
    runner : callable or None, default=None
        Command runner (tests).
    lock_timeout : float, default=0.0
        Seconds to wait for a concurrent run on the same prefix.
    **fields
        Request fields.

    Returns
    -------
    ExecutionResult
        The outcome; ``result.changed`` is False when it was up to date.

    Raises
    ------
    EnvError
        Planning errors, ``ApprovalRequiredError``, ``LockBusyError``, ``ExecutionError``.
    """
    plan_ = plan(
        request,
        config=config,
        operation="recreate" if recreate else "ensure",
        host=host,
        **fields,
    )
    return _execute(
        plan_,
        approve=approve,
        interaction=interaction,
        events=events,
        runner=runner,
        lock_timeout=lock_timeout,
    )


def remove(
    request: EnvironmentRequest | None = None,
    *,
    config: Any = None,
    approve: Any = (),
    interaction: str | None = None,
    events: Any = None,
    host: HostInfo | None = None,
    lock_timeout: float = 0.0,
    **fields: Any,
) -> ExecutionResult:
    """
    Delete the environment (needs approval ``remove``).

    Parameters
    ----------
    request : EnvironmentRequest or None, default=None
        The request, or None to build it from ``config``/``fields``.
    config : path-like, mapping or None, default=None
        Configuration document.
    approve : bool, str, sequence of str or callable, default=()
        Pre-given approvals (``"remove"``).
    interaction : {"never", "auto", "always"} or None, default=None
        Overrides the request's interaction policy.
    events : path-like, text stream, EventSink or None, default=None
        Where JSON Lines events go.
    host : HostInfo or None, default=None
        The host; detected when None.
    lock_timeout : float, default=0.0
        Seconds to wait for a concurrent run.
    **fields
        Request fields.

    Returns
    -------
    ExecutionResult
        The outcome (no steps when there was nothing to remove).
    """
    plan_ = plan(request, config=config, operation="remove", host=host, **fields)
    return _execute(
        plan_,
        approve=approve,
        interaction=interaction,
        events=events,
        runner=None,
        lock_timeout=lock_timeout,
    )


def _run_env(
    prefix: Path, host: HostInfo, environ: Mapping[str, str]
) -> dict[str, str]:
    env = {k: v for k, v in environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    if host.is_windows:
        dirs = [prefix / "Scripts", prefix, prefix / "Library" / "bin"]
    else:
        dirs = [prefix / "bin"]
    env["PATH"] = os.pathsep.join(
        [*(str(d) for d in dirs if d.is_dir()), env.get("PATH", "")]
    )
    if (prefix / "conda-meta").is_dir():
        env["CONDA_PREFIX"] = str(prefix)
    else:
        env["VIRTUAL_ENV"] = str(prefix)
    return env


def run(
    command: Sequence[str],
    request: EnvironmentRequest | None = None,
    *,
    config: Any = None,
    host: HostInfo | None = None,
    environ: Mapping[str, str] | None = None,
    runner: Callable[..., Any] = subprocess.call,
    **fields: Any,
) -> int:
    """
    Run a command inside an existing environment.

    Parameters
    ----------
    command : sequence of str
        Command and arguments (``["python", "-m", "pytest"]``).
    request : EnvironmentRequest or None, default=None
        The request, or None to build it from ``config``/``fields``.
    config : path-like, mapping or None, default=None
        Configuration document.
    host : HostInfo or None, default=None
        The host; detected when None.
    environ : mapping or None, default=None
        Base environment; :py:data:`os.environ` when None.
    runner : callable, default=subprocess.call
        ``runner(argv, env=...)`` returning the exit status; the command
        keeps the terminal (stdin, stdout, stderr).
    **fields
        Request fields.

    Returns
    -------
    int
        Exit status of the command.

    Raises
    ------
    PlanningError
        When the environment does not exist or does not run here.

    Notes
    -----
    Conda-family environments built by micromamba or mamba run through
    ``<tool> run`` (activation scripts apply); every other environment runs
    with its ``bin`` (``Scripts``) first on ``PATH``.
    """
    if not command:
        raise ConfigError("no command given")
    req = _request(request, config, fields)
    host = detect_host() if host is None else host
    if req.target not in ("host", host.conda_subdir):
        raise PlanningError(f"a {req.target} environment does not run on this host")
    prefix = resolve_prefix(req, host)
    receipt = read_receipt(prefix)
    if (
        receipt is None
        and not (prefix / "conda-meta").is_dir()
        and not (prefix / "pyvenv.cfg").is_file()
    ):
        raise PlanningError(
            f"{prefix} is not an environment",
            hint="create it first: scikitplot env create",
        )
    env = _run_env(prefix, host, os.environ if environ is None else environ)
    argv = list(command)
    # Resolve the command against the environment's PATH here: Windows looks
    # an executable up in the parent's PATH, not in the child's ``env``.
    found = shutil.which(argv[0], path=env["PATH"])
    if found is None:
        raise ExecutionError(
            f"command {argv[0]!r} is not in {prefix} nor on PATH",
            step="run",
            returncode=127,
            code="command_not_found",
            hint="install it into the environment (add it to packages)",
        )
    argv[0] = found
    backend_name = (receipt or {}).get("backend")
    if backend_name in ("micromamba", "mamba"):
        exe = host.tool(backend_name)
        if exe:
            argv = get_backend(backend_name).run_argv(prefix, host, exe, argv)
    try:
        return int(runner(argv, env=env))
    except OSError as exc:  # not executable, wrong architecture ...
        raise ExecutionError(
            f"cannot run {argv[0]}: {exc}",
            step="run",
            returncode=126,
            code="command_not_found",
        ) from exc


def shell_init(
    shell: str,
    *,
    write: bool = False,
    rc_file: Any = None,
    approve: Any = (),
    interaction: str = "auto",
    host: HostInfo | None = None,
) -> str:
    """
    Return (and optionally install) the shell start-up block.

    Parameters
    ----------
    shell : {"bash", "zsh", "fish", "powershell"}
        Shell.
    write : bool, default=False
        Put the block into ``rc_file`` (approval ``shell_init``).
    rc_file : path-like or None, default=None
        Start-up file; the shell's default when None (PowerShell needs one).
    approve : bool, str, sequence of str or callable, default=()
        Pre-given approvals (``"shell_init"``).
    interaction : {"never", "auto", "always"}, default="auto"
        Whether a person may be asked.
    host : HostInfo or None, default=None
        The host; detected when None.

    Returns
    -------
    str
        The block.

    Raises
    ------
    ApprovalRequiredError
        When ``write`` is not approved.
    """
    from ._bootstrap._shell import rc_file as default_rc  # noqa: PLC0415
    from ._bootstrap._shell import shell_block, write_shell_block  # noqa: PLC0415

    host = detect_host() if host is None else host
    micromamba = host.tool("micromamba")
    block = shell_block(
        shell,
        str(host.tools_dir),
        micromamba=micromamba,
        root_prefix=str(Path(host.home) / "micromamba") if micromamba else None,
    )
    if write:
        target = Path(os.fspath(rc_file)) if rc_file is not None else default_rc(shell)
        Decider(interaction, approve=approve).require(
            "shell_init", f"add the scikitplot env block to {target}"
        )
        write_shell_block(target, block)
    return block


def batch(
    path: Any,
    *,
    operation: str = "ensure",
    approve: Any = (),
    interaction: str | None = None,
    events: Any = None,
    host: HostInfo | None = None,
    keep_going: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Plan or run every request of a JSON Lines batch file.

    Parameters
    ----------
    path : path-like
        Batch file (see :py:func:`~._config._schema.load_batch`).
    operation : {"ensure", "recreate", "remove"}, default="ensure"
        Operation for every request.
    approve : bool, str, sequence of str or callable, default=()
        Pre-given approvals.
    interaction : {"never", "auto", "always"} or None, default=None
        Overrides each request's interaction policy.
    events : path-like, text stream, EventSink or None, default=None
        Where JSON Lines events go (shared by all requests).
    host : HostInfo or None, default=None
        The host; detected once when None.
    keep_going : bool, default=False
        Continue after a failed request.
    dry_run : bool, default=False
        Only plan.

    Returns
    -------
    list of dict
        One record per request: ``id``, ``ok``, and ``plan``/``result`` or
        ``error``.
    """
    host = detect_host() if host is None else host
    sink = _sink(events)
    out: list[dict[str, Any]] = []
    try:
        for rid, req in load_batch(path):
            try:
                plan_ = plan_request(req, operation=operation, host=host)
                record: dict[str, Any] = {
                    "id": rid,
                    "ok": True,
                    "plan": plan_.to_dict(),
                }
                if not dry_run:
                    record["result"] = _execute(
                        plan_,
                        approve=approve,
                        interaction=interaction,
                        events=sink,
                        runner=None,
                        lock_timeout=0.0,
                    ).to_dict()
            except EnvError as exc:
                record = {"id": rid, "ok": False, "error": exc.to_dict()}
                out.append(record)
                if not keep_going:
                    break
                continue
            out.append(record)
    finally:
        if sink is not events:
            sink.close()
    return out

# scikitplot/_env/_core/_planner.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The planner: request + host -> plan. Reads files, changes nothing.

Notes
-----
**User notes.** The backend is chosen by fixed rules, and the plan says
which rule applied (``backend_reason``):

1. the target allows only some backends (cross-builds: micromamba, mamba;
   pyodide: micropip);
2. a named backend is used as asked, after a check that it can do the job;
3. ``auto`` with Conda packages or an ``environment.yml``: the first of
   micromamba, mamba, conda found on the host, else a micromamba bootstrap
   when ``bootstrap="if-missing"``;
4. ``auto`` with a single locked ``Pipfile`` and pipenv on the host: pipenv;
5. ``auto`` otherwise (PyPI only): uv when found; venv when the requested
   Python is the running one; else a uv bootstrap when allowed; else
   micromamba when found (it installs any Python).

Nothing is guessed: a request no rule satisfies fails with the reason and a
hint (install a tool, allow the bootstrap, name a backend).

**Developer notes.** Every input is an argument (host, receipt reader, file
existence), so plans are reproducible in tests for any operating system.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
from typing import Callable

from .._backends import PREFERENCE, get_backend
from .._backends._base import Backend, PlanContext, has_pypi
from .._bootstrap import MICROMAMBA, UV_REQUIREMENT, micromamba_archive
from .._sources._readers import Dependency, DependencySet, read_sources
from .._targets import Target, get_target
from ._errors import (
    BackendUnavailableError,
    ConfigError,
    ConversionRefusedError,
    NetworkForbiddenError,
    PlanningError,
)
from ._host import HostInfo, detect_host
from ._plan import EnvironmentPlan, Step, digest_of
from ._receipt import is_environment as _is_environment
from ._receipt import read_receipt
from ._request import EnvironmentRequest
from ._types import Bootstrap, Conversion, Ecosystem, Operation, choice

__all__ = ["plan_request", "resolve_prefix"]

#: Conda specs that describe the interpreter, not a package to convert.
_INTERPRETER = frozenset({"python", "pip"})


def resolve_prefix(request: EnvironmentRequest, host: HostInfo) -> Path:
    """
    Return the environment directory of ``request``.

    Parameters
    ----------
    request : EnvironmentRequest
        The request.
    host : HostInfo
        The host (its environments home).

    Returns
    -------
    pathlib.Path
        ``request.prefix``, or ``<home>/envs/<name>``.
    """
    if request.prefix is not None:
        return Path(request.prefix)
    return Path(host.home) / "envs" / str(request.name)


def _exe_name(host: HostInfo, tool: str) -> str:
    return f"{tool}.exe" if host.is_windows else tool


def _bootstrap_step(tool: str, host: HostInfo, offline: bool) -> tuple[Step, str]:
    """
    Return the bootstrap step for ``tool`` and the path it installs.

    Parameters
    ----------
    tool : {'micromamba', 'uv'}
        Tool.
    host : HostInfo
        The host (its tools directory and platform).
    offline : bool
        Offline policy (micromamba then uses only its cached archive).

    Returns
    -------
    tuple of (Step, str)
        The step and the executable it will install.
    """
    tools = host.tools_dir
    exe = str(tools / "bin" / _exe_name(host, tool))
    if tool == "micromamba":
        info = micromamba_archive(str(host.conda_subdir))
        cached = tools / "cache" / f"{host.conda_subdir}-{info['filename']}"
        return (
            Step(
                id="bootstrap-micromamba",
                title=f"install micromamba into {tools / 'bin'} (pinned, SHA-256 checked)",
                kind="action",
                action="bootstrap_micromamba",
                params=(
                    ("tools_dir", str(tools)),
                    ("subdir", str(host.conda_subdir)),
                    ("offline", offline),
                    ("url", info["url"]),
                    ("sha256", info["sha256"]),
                ),
                network=not (offline or cached.is_file()),
                writes=(str(tools),),
                approval="bootstrap",
            ),
            exe,
        )
    return (
        Step(
            id="bootstrap-uv",
            title=f"install {UV_REQUIREMENT} into {tools / 'bin'}",
            kind="action",
            action="bootstrap_uv",
            params=(
                ("tools_dir", str(tools)),
                ("requirement", UV_REQUIREMENT),
                ("python", host.python_executable),
            ),
            network=True,
            writes=(str(tools),),
            approval="bootstrap",
        ),
        exe,
    )


def _can_bootstrap(
    tool: str, host: HostInfo, request: EnvironmentRequest
) -> str | None:
    """
    Return why ``tool`` cannot be bootstrapped, or None when it can.

    Parameters
    ----------
    tool : {'micromamba', 'uv'}
        Tool.
    host : HostInfo
        The host.
    request : EnvironmentRequest
        The request (its bootstrap and network policies).

    Returns
    -------
    str or None
        The reason, or None.
    """
    if request.policy.bootstrap != Bootstrap.IF_MISSING.value:
        return "the bootstrap policy is 'never'"
    if tool == "micromamba":
        if host.conda_subdir not in MICROMAMBA:
            return f"no micromamba archive is pinned for {host.conda_subdir!r}"
        if request.policy.offline:
            info = micromamba_archive(str(host.conda_subdir))
            cached = (
                host.tools_dir / "cache" / f"{host.conda_subdir}-{info['filename']}"
            )
            if not cached.is_file():
                return f"offline and the archive is not cached at {cached}"
        return None
    if request.policy.offline:
        return "offline (uv is bootstrapped from the package index)"
    return None


def _major_minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


@dataclasses.dataclass(frozen=True)
class _Choice:
    backend: Backend
    exe: str
    reason: str
    bootstrap: Step | None = None


def _pick(names, request, host, *, reason: str, missing_hint: str) -> _Choice:
    """
    Take the first available backend of ``names``; else bootstrap one; else fail.

    Parameters
    ----------
    names : sequence of str
        Backends in preference order.
    request : EnvironmentRequest
        The request.
    host : HostInfo
        The host.
    reason : str
        Rule that led here (the plan's ``backend_reason``).
    missing_hint : str
        Hint of the error when none can be used.

    Returns
    -------
    _Choice
        The backend, its executable, the reason and a bootstrap step or None.

    Raises
    ------
    BackendUnavailableError
        With each bootstrap refusal in the message.
    """
    for name in names:
        backend = get_backend(name)
        if backend.tool is None:
            return _Choice(backend, host.python_executable, reason)
        exe = host.tool(backend.tool)
        if exe:
            return _Choice(backend, exe, f"{reason}; {name} found at {exe}")
    why = {}
    for name in names:
        backend = get_backend(name)
        if backend.bootstrappable:
            refusal = _can_bootstrap(backend.tool, host, request)
            if refusal is None:
                step, exe = _bootstrap_step(backend.tool, host, request.policy.offline)
                return _Choice(
                    backend,
                    exe,
                    f"{reason}; {name} is not installed and will be bootstrapped",
                    step,
                )
            why[name] = refusal
    detail = "; ".join(f"{n}: {w}" for n, w in why.items())
    raise BackendUnavailableError(
        f"none of {list(names)} is installed" + (f" ({detail})" if detail else ""),
        hint=missing_hint,
    )


def _choose(  # noqa: PLR0911 (one return per documented rule)
    request: EnvironmentRequest,
    host: HostInfo,
    target: Target,
    deps: DependencySet,
    python: str | None,
) -> _Choice:
    needs_conda = bool(deps.of(Ecosystem.CONDA.value) or deps.environment_files)
    allowed = target.backends
    if request.backend != "auto":
        backend = get_backend(request.backend)
        if allowed is not None and backend.name not in allowed:
            raise BackendUnavailableError(
                f"target {target.name!r} cannot be built with {backend.name!r}",
                hint=f"use one of {list(allowed)}",
            )
        if backend.tool is None:
            return _Choice(backend, host.python_executable, "named in the request")
        return _pick(
            [backend.name],
            request,
            host,
            reason="named in the request",
            missing_hint=f"install {backend.tool}, or set"
            " policy.bootstrap='if-missing' when it is bootstrappable",
        )
    if allowed is not None:
        return _pick(
            allowed,
            request,
            host,
            reason=f"target {target.name} needs a cross-building Conda backend",
            missing_hint="install micromamba, or set policy.bootstrap='if-missing'",
        )
    if needs_conda:
        return _pick(
            PREFERENCE["conda"],
            request,
            host,
            reason="the request has Conda packages",
            missing_hint="install micromamba, mamba or conda, or set"
            " policy.bootstrap='if-missing'",
        )
    if (
        len(deps.pipfiles) == 1
        and host.tool("pipenv")
        and (Path(deps.pipfiles[0]).parent / "Pipfile.lock").is_file()
        # pipenv installs into a venv of the running Python (rule 4 only then)
        and (python is None or _major_minor(python) == _major_minor(host.python))
    ):
        return _Choice(
            get_backend("pipenv"),
            str(host.tool("pipenv")),
            "a locked Pipfile and pipenv is installed",
        )
    if host.tool("uv"):
        return _Choice(
            get_backend("uv"),
            str(host.tool("uv")),
            f"PyPI only; uv found at {host.tool('uv')}",
        )
    if python is None or _major_minor(python) == _major_minor(host.python):
        return _Choice(
            get_backend("venv"),
            host.python_executable,
            "PyPI only, running Python; uv is not installed",
        )
    refusal = _can_bootstrap("uv", host, request)
    if refusal is None:
        step, exe = _bootstrap_step("uv", host, request.policy.offline)
        return _Choice(
            get_backend("uv"),
            exe,
            f"Python {python} differs from the running"
            f" {_major_minor(host.python)}; uv will be bootstrapped",
            step,
        )
    for name in ("micromamba", "mamba", "conda"):
        if host.tool(name):
            return _Choice(
                get_backend(name),
                str(host.tool(name)),
                f"Python {python} differs from the running one; {name} installs it",
            )
    raise BackendUnavailableError(
        f"Python {python} differs from the running {_major_minor(host.python)}, and no"
        f" tool that installs Pythons is available (uv bootstrap: {refusal})",
        hint="install uv or micromamba, set policy.bootstrap='if-missing', or run the"
        " builder with that Python",
    )


def _convert(
    deps: DependencySet,
    name: str,
    ecosystems: frozenset,
    request: EnvironmentRequest,
    notes: list[str],
) -> DependencySet:
    """
    Fit ``deps`` to ``ecosystems`` (of backend ``name``) under the conversion policy.

    Parameters
    ----------
    deps : DependencySet
        Dependencies.
    name : str
        Backend name, for messages.
    ecosystems : frozenset of str
        Ecosystems the backend installs from.
    request : EnvironmentRequest
        The request (its conversion policy).
    notes : list of str
        Plan notes; a conversion appends one.

    Returns
    -------
    DependencySet
        ``deps``, or the converted set.

    Raises
    ------
    ConversionRefusedError
        Under the ``exact`` policy when Conda packages meet a PyPI-only backend.
    """
    conda = [d for d in deps.of(Ecosystem.CONDA.value) if d.name not in _INTERPRETER]
    if Ecosystem.CONDA.value in ecosystems or not (conda or deps.environment_files):
        return deps
    if request.policy.conversion != Conversion.ALLOW_APPROXIMATE.value:
        names = sorted({d.name for d in conda}) or [
            Path(f).name for f in deps.environment_files
        ]
        instead = (
            "list the PyPI names in packages"
            if name == "micropip"
            else "use a Conda backend (micromamba)"
        )
        raise ConversionRefusedError(
            f"backend {name!r} installs from PyPI only, but the request has Conda"
            f" packages {names}",
            hint=f"{instead}, or set policy.conversion='allow-approximate' to install"
            " the same names from PyPI",
        )
    converted = [
        Dependency(d.name, d.name, Ecosystem.PYPI.value, d.origin) for d in conda
    ]
    notes.append(
        "approximate conversion: Conda packages installed from PyPI by name,"
        " without their version constraints: "
        + ", ".join(sorted({d.name for d in conda}))
    )
    kept = [d for d in deps.dependencies if d.ecosystem == Ecosystem.PYPI.value]
    return dataclasses.replace(
        deps, dependencies=tuple(kept + converted), environment_files=()
    )


def _python_version(request: EnvironmentRequest, deps: DependencySet) -> str | None:
    if request.python and deps.python and request.python != deps.python:
        raise ConfigError(
            f"python={request.python!r} conflicts with {deps.python!r} pinned by a"
            " dependency file",
            hint="remove one of the two pins",
        )
    return request.python or deps.python


def _channels(
    request: EnvironmentRequest, deps: DependencySet, target: Target
) -> tuple:
    from .._backends._conda import DEFAULT_CHANNELS  # noqa: PLC0415  (lazy: conda only)

    return tuple(
        request.channels or deps.channels or target.channels or DEFAULT_CHANNELS
    )


def _state(
    request: EnvironmentRequest,
    *,
    target: Target,
    backend: str,
    prefix: Path,
    python: str | None,
    inputs,
) -> str:
    # Only what changes the environment's content: how the builder may behave
    # (interaction, network, bootstrap) and the preset name do not.
    data = request.to_dict()
    data["policy"] = {"conversion": data["policy"]["conversion"]}
    data.pop("profile", None)
    return digest_of(
        {
            "request": data,
            "target": target.name,
            "backend": backend,
            "prefix": str(prefix),
            "python": python,
            "inputs": list(inputs),
        }
    )


def _lock_input(deps: DependencySet) -> tuple[tuple[str, str], ...]:
    """
    Return ``(path, sha256)`` of the ``Pipfile.lock`` the pipenv backend installs.

    Parameters
    ----------
    deps : DependencySet
        Dependencies with exactly one Pipfile (checked by the backend).

    Returns
    -------
    tuple of (str, str)
        One input entry.
    """
    lock = Path(deps.pipfiles[0]).parent / "Pipfile.lock"
    return ((str(lock), hashlib.sha256(lock.read_bytes()).hexdigest()),)


def _removed(receipt: dict, deps: DependencySet) -> list[str]:
    """
    Return the dependencies the receipt's build had and the request no longer has.

    Parameters
    ----------
    receipt : dict
        The environment's receipt.
    deps : DependencySet
        Dependencies of the request.

    Returns
    -------
    list of str
        ``"name (ecosystem)"`` entries, sorted.
    """
    before = {
        (d.get("ecosystem"), d.get("name"))
        for d in receipt.get("dependencies") or []
        if isinstance(d, dict)
    }
    now = {(d.ecosystem, d.name) for d in deps.dependencies}
    return sorted(f"{name} ({eco})" for eco, name in before - now)


def _check_indexes(
    deps: DependencySet, request: EnvironmentRequest, backend: str
) -> None:
    """
    Refuse Pipfile ``[[source]]`` indexes a non-pipenv backend would not use.

    Parameters
    ----------
    deps : DependencySet
        Dependencies (their ``indexes``).
    request : EnvironmentRequest
        The request (its ``index_url`` and ``extra_index_urls``).
    backend : str
        Chosen backend.

    Raises
    ------
    PlanningError
        When an index is neither given in the request nor used by pipenv.
    """
    given = {u.rstrip("/") for u in (request.index_url, *request.extra_index_urls) if u}
    missing = [u for u in deps.indexes if u.rstrip("/") not in given]
    if missing and backend != "pipenv":
        raise PlanningError(
            f"the Pipfile names package indexes {missing}; backend {backend!r} would"
            " resolve those packages from public PyPI instead",
            hint="add them as index_url / extra_index_urls, or use backend='pipenv'",
        )


def _verify_step(python: Path, modules) -> Step | None:
    if not modules:
        return None
    code = "import importlib, sys; [importlib.import_module(m) for m in sys.argv[1:]]"
    return Step(
        id="verify",
        title="import " + ", ".join(modules),
        argv=(str(python), "-I", "-c", code, *modules),
    )


def _micropip(deps: DependencySet, request: EnvironmentRequest) -> str:
    specs = []
    for dep in deps.of(Ecosystem.PYPI.value):
        if dep.spec.startswith("-e ") or os.path.isabs(dep.spec.split(";")[0].strip()):
            raise ConversionRefusedError(
                f"{dep.spec!r} is a local path; Pyodide cannot install it",
                hint="build a wheel and serve it by URL, then list the URL",
            )
        # per-requirement pip options (``--hash=...``) mean nothing to micropip
        specs.append(dep.spec.split(" --", 1)[0].strip())
    args = [repr(list(dict.fromkeys(specs)))]
    if request.index_url or request.extra_index_urls:
        urls = [u for u in (request.index_url, *request.extra_index_urls) if u]
        args.append(f"index_urls={urls!r}")
    return (
        "# Run inside Pyodide (JupyterLite, PyScript, a Pyodide console).\n"
        "import micropip\n"
        f"await micropip.install({', '.join(args)})\n"
    )


def _plan_remove(
    base: dict, request: EnvironmentRequest, prefix: Path, receipt_reader
) -> EnvironmentPlan:
    """
    Plan a removal: delete a recognised environment directory, no tool needed.

    Parameters
    ----------
    base : dict
        Plan fields shared by every operation.
    request : EnvironmentRequest
        The request.
    prefix : pathlib.Path
        Environment directory.
    receipt_reader : callable
        ``receipt_reader(prefix)``.

    Returns
    -------
    EnvironmentPlan
        The removal plan (no steps when there is nothing to remove).
    """
    exists = _is_environment(prefix)
    receipt = receipt_reader(prefix) if exists else None
    backend = str((receipt or {}).get("backend") or "none")
    steps: tuple[Step, ...] = ()
    notes: list[str] = []
    if exists:
        steps = (
            Step(
                id="remove",
                title=f"delete {prefix}",
                kind="action",
                action="remove_tree",
                params=(("path", str(prefix)),),
                writes=(str(prefix),),
                approval="remove",
            ),
        )
    else:
        notes.append(f"nothing to remove: {prefix} is not an environment")
    return EnvironmentPlan(
        **base,
        backend_requested=request.backend,
        backend=backend,
        backend_reason="removal deletes the directory; no tool is needed",
        python_executable=None,
        exists=exists,
        up_to_date=False,
        steps=steps,
        notes=tuple(notes),
        state_digest="",
    )


def plan_request(  # noqa: PLR0912 (the documented operation table)
    request: EnvironmentRequest,
    *,
    operation: str = Operation.ENSURE.value,
    host: HostInfo | None = None,
    receipt_reader: Callable[[Path], dict | None] = read_receipt,
) -> EnvironmentPlan:
    """
    Plan ``operation`` for ``request`` on ``host``.

    Parameters
    ----------
    request : EnvironmentRequest
        What to build.
    operation : {"ensure", "recreate", "remove"}, default="ensure"
        ``ensure`` creates or updates and does nothing when the receipt
        matches; ``recreate`` deletes first (approval ``remove``);
        ``remove`` deletes only.
    host : HostInfo or None, default=None
        The host; :py:func:`~._host.detect_host` when None.
    receipt_reader : callable, default=read_receipt
        ``receipt_reader(prefix)`` returning the receipt or None.

    Returns
    -------
    EnvironmentPlan
        The plan; nothing has been changed.

    Raises
    ------
    ConfigError, SourceError
        For an invalid request or dependency file.
    PlanningError
        (``BackendUnavailableError``, ``TargetUnsupportedError``, ``NetworkForbiddenError``,
        ``ConversionRefusedError``) when the request cannot be built here.
    """
    operation = choice(operation, Operation, field="operation").value
    host = detect_host() if host is None else host
    target = get_target(request.target, host_subdir=host.conda_subdir)
    if target.executable and not host.can_subprocess:
        raise PlanningError(
            f"this Python ({host.runtime}) cannot start processes",
            hint="plan here with target='pyodide', or build elsewhere",
        )
    prefix = resolve_prefix(request, host)
    base = {
        "request": request.to_dict(provenance=True),
        "request_digest": request.digest,
        "operation": operation,
        "target": target.name,
        "prefix": str(prefix),
        "host": host.to_dict(),
    }
    if operation == Operation.REMOVE.value:  # needs neither the sources nor a tool
        return _plan_remove(
            {**base, "dependencies": ()}, request, prefix, receipt_reader
        )
    deps = read_sources(request)
    python = _python_version(request, deps)
    notes = list(deps.notes)
    base.update(
        dependencies=tuple(d.to_dict() for d in deps.dependencies), inputs=deps.inputs
    )

    if not target.executable:  # pyodide: plan only
        converted = _convert(
            deps, "micropip", frozenset({Ecosystem.PYPI.value}), request, notes
        )
        script = _micropip(converted, request)
        return EnvironmentPlan(
            **base,
            backend_requested=request.backend,
            backend="micropip",
            backend_reason="target pyodide installs with micropip in the browser",
            python_executable=None,
            exists=False,
            up_to_date=False,
            steps=(),
            notes=tuple(notes),
            executable=False,
            artifacts=(("micropip.py", script),),
            python=python,
            state_digest=_state(
                request,
                target=target,
                backend="micropip",
                prefix=prefix,
                python=python,
                inputs=deps.inputs,
            ),
        )

    picked = _choose(request, host, target, deps, python)
    backend = picked.backend
    deps = _convert(deps, backend.name, backend.ecosystems, request, notes)
    if target.pypi == "refuse" and has_pypi(deps):
        raise ConversionRefusedError(
            f"PyPI packages cannot be installed into a cross-build for {target.name}"
            " (pip would run here, for this platform)",
            hint="use Conda packages for the cross-build, or build on that platform",
        )
    python_path = backend.python_in(prefix, host) if target.runs_here else None
    ctx = PlanContext(
        request=request,
        deps=deps,
        host=host,
        prefix=prefix,
        exe=picked.exe,
        python=python,
        channels=(
            _channels(request, deps, target)
            if Ecosystem.CONDA.value in backend.ecosystems
            else ()
        ),
        conda_subdir=target.conda_subdir,
        offline=request.policy.offline,
        relocate_prefix=target.relocate_prefix,
    )
    backend.check(ctx)
    _check_indexes(deps, request, backend.name)
    inputs = deps.inputs + (_lock_input(deps) if backend.name == "pipenv" else ())

    recognised = _is_environment(prefix)
    if not recognised and prefix.exists() and any(prefix.iterdir()):
        raise PlanningError(
            f"{prefix} exists, is not empty and is not an environment",
            hint="choose another prefix; the builder never writes into or deletes an"
            " unknown directory",
        )
    usable = recognised and (
        python_path.exists()
        if python_path is not None
        else (prefix / "conda-meta").is_dir()
    )
    state = _state(
        request,
        target=target,
        backend=backend.name,
        prefix=prefix,
        python=python,
        inputs=inputs,
    )
    receipt = receipt_reader(prefix) if recognised else None
    up_to_date = bool(
        usable
        and receipt
        and receipt.get("state_digest") == state
        and operation == Operation.ENSURE.value
    )
    rebuild = operation == Operation.RECREATE.value
    if recognised and not usable and not rebuild:
        notes.append(f"{prefix} is a broken environment (no Python); it is recreated")
        rebuild = True
    if usable and receipt and not rebuild:
        built = {
            "backend": receipt.get("backend"),
            "target": receipt.get("target"),
            "python": receipt.get("python"),
        }
        wanted = {"backend": backend.name, "target": target.name, "python": python}
        changed = [k for k in built if built[k] != wanted[k]]
        if changed:
            raise PlanningError(
                f"{prefix} was built with "
                + ", ".join(f"{k}={built[k]!r}" for k in changed)
                + "; the request asks for "
                + ", ".join(f"{k}={wanted[k]!r}" for k in changed),
                hint="use operation='recreate' (approval 'remove') to rebuild it, or ask"
                " for what it was built with",
            )
        removed = _removed(receipt, deps)
        if removed and not up_to_date:
            notes.append(
                f"no longer requested: {', '.join(removed)}; installers never uninstall,"
                " so the environment is recreated"
            )
            rebuild = True

    steps: list[Step] = []
    if picked.bootstrap is not None and not up_to_date:
        steps.append(picked.bootstrap)
    if rebuild:
        steps += (backend.remove(ctx) if recognised else []) + backend.create(ctx)
    elif not up_to_date:
        steps += backend.update(ctx) if usable else backend.create(ctx)
    exists = usable
    if not up_to_date and python_path is not None:
        verify = _verify_step(python_path, request.verify_imports)
        if verify is not None:
            steps.append(verify)
    elif request.verify_imports and python_path is None:
        notes.append(
            f"verify_imports skipped: a {target.name} environment does not run here"
        )
    if request.policy.offline and any(s.network for s in steps):
        bad = [s.id for s in steps if s.network]
        raise NetworkForbiddenError(
            f"steps {bad} need the network under the offline policy",
            hint="cache what they download, or use network='online'",
        )
    base["inputs"] = inputs
    return EnvironmentPlan(
        **base,
        backend_requested=request.backend,
        backend=backend.name,
        backend_reason=picked.reason,
        python_executable=str(python_path) if python_path else None,
        exists=exists,
        up_to_date=up_to_date,
        steps=tuple(steps),
        notes=tuple(notes),
        python=python,
        state_digest=state,
    )

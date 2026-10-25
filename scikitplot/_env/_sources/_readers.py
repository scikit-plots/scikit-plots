# scikitplot/_env/_sources/_readers.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Dependency sources: inline lists and dependency files, read into one set.

=====================  ======================================================
Source                 What is read
=====================  ======================================================
``packages``           PyPI requirement strings
``conda_packages``     Conda match specs
``requirements.txt``   PyPI requirements; pip and uv also get the file itself
                       (``-r``), so its options (``--index-url`` ...) apply
``environment.yml``    Conda specs, ``pip:`` entries (with ``-r`` and ``-e``),
                       ``channels``, the ``python`` pin
``pyproject.toml``     ``[project.dependencies]``, extras and PEP 735
                       ``[dependency-groups]`` named in ``groups``
``Pipfile``            ``[packages]`` (and ``[dev-packages]`` with group
                       ``dev``), ``[requires] python_version``
=====================  ======================================================

Notes
-----
**User notes.** The file is the source of truth: the builder does not copy
its contents into the request, it reads them every time it plans, and the
hash of every file read is part of the plan digest, so an edit of
``requirements.txt`` makes the next ``ensure`` update the environment.

One reading for every backend: ``environment.yml`` is parsed here and its
specs are given to micromamba, conda, uv or venv alike, so the same file
gives the same environment whichever tool builds it. Its ``name`` and
``prefix`` keys are ignored (the request decides where the environment
lives); keys that would change the environment and that no backend can
apply the same way (``variables``, pip options other than ``-r``/``-e``)
are refused rather than dropped.

**Developer notes.** Names are normalised as PEP 503 (PyPI) or lower case
(Conda) only to compare and display; installers always get the original
requirement strings.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Iterable

from .._config._formats import loads, read_text
from .._core._errors import ConfigError, SourceError
from .._core._request import EnvironmentRequest, SourceRef
from .._core._types import Ecosystem

__all__ = ["Dependency", "DependencySet", "read_sources"]

_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_INCLUDE = re.compile(r"^(?:-r|--requirement)(?:\s+|=)?(\S.*)$")
_PUBLIC_PYPI = frozenset({"https://pypi.org/simple", "https://pypi.python.org/simple"})
_CONSTRAINT = re.compile(r"^(?:-c|--constraint)(?:\s+|=)?(\S.*)$")
_ENV_KEYS = frozenset({"name", "prefix", "channels", "dependencies"})
_CONDA_NAME = re.compile(r"^\s*(?:[^\s:]+::)?([A-Za-z0-9_][A-Za-z0-9._-]*)")


def pypi_name(spec: str) -> str:
    """
    Return the PEP 503 normalised project name of a requirement string.

    Parameters
    ----------
    spec : str
        Requirement (``"Pandas[parquet]>=2; python_version>'3.9'"``).

    Returns
    -------
    str
        Normalised name (``"pandas"``); the whole string when it is a path or URL.

    Examples
    --------
    >>> from scikitplot._env._sources._readers import pypi_name
    >>> pypi_name("Scikit_Learn>=1.4")
    'scikit-learn'
    """
    match = _NAME.match(spec)
    if (
        not match
        or spec.strip().startswith((".", "/", "~"))
        or "://" in spec.split("@", maxsplit=1)[0]
    ):
        return spec.strip()
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def conda_name(spec: str) -> str:
    """
    Return the package name of a Conda match spec.

    Parameters
    ----------
    spec : str
        Match spec (``"conda-forge::numpy>=2"``, ``"python=3.12"``).

    Returns
    -------
    str
        Lower-case name (``"numpy"``).
    """
    match = _CONDA_NAME.match(spec)
    return match.group(1).lower() if match else spec.strip().lower()


@dataclasses.dataclass(frozen=True)
class Dependency:  # numpydoc ignore=PR02
    """
    One dependency.

    Parameters
    ----------
    name : str
        Normalised name.
    spec : str
        The requirement as written.
    ecosystem : {"pypi", "conda"}
        Where it comes from.
    origin : str
        ``"packages"``, ``"conda_packages"`` or the file it was read from.
    """

    name: str
    spec: str
    ecosystem: str
    origin: str

    def to_dict(self) -> dict[str, str]:
        """
        Return the dependency as a mapping.

        Returns
        -------
        dict
            Plain data (JSON-serialisable).
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DependencySet:  # numpydoc ignore=PR02
    """
    Everything the sources of a request ask for.

    Parameters
    ----------
    dependencies : tuple of Dependency
        Every dependency, in source order, duplicates kept.
    requirement_files : tuple of str
        ``requirements.txt`` files a pip-family installer reads itself.
    environment_files : tuple of str
        ``environment.yml`` files a Conda-family backend reads itself.
    python : str or None
        Python version pinned by a source (``environment.yml``, ``Pipfile``).
    channels : tuple of str
        Channels named by ``environment.yml`` files.
    pipfiles : tuple of str
        Pipfiles among the sources (the ``pipenv`` backend uses them).
    notes : tuple of str
        Things people should know (ignored options, inexact entries).
    inputs : tuple of (str, str)
        ``(path, sha256)`` of every file read, includes and constraint files
        too, in read order.
    requirement_origins : tuple of str
        Every file of the ``-r`` trees of ``requirement_files``: a dependency
        read from one of them reaches pip/uv through ``-r``, never inline.
    indexes : tuple of str
        Package indexes a ``Pipfile`` names in ``[[source]]``.
    """

    dependencies: tuple[Dependency, ...] = ()
    requirement_files: tuple[str, ...] = ()
    environment_files: tuple[str, ...] = ()
    python: str | None = None
    channels: tuple[str, ...] = ()
    pipfiles: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    inputs: tuple[tuple[str, str], ...] = ()
    requirement_origins: tuple[str, ...] = ()
    indexes: tuple[str, ...] = ()

    def of(self, ecosystem: str) -> tuple[Dependency, ...]:
        """
        Return the dependencies of one ecosystem.

        Parameters
        ----------
        ecosystem : {"pypi", "conda"}
            Ecosystem.

        Returns
        -------
        tuple of Dependency
            Those dependencies, in order.
        """
        return tuple(d for d in self.dependencies if d.ecosystem == ecosystem)


def _requirement_lines(
    path: Path, seen: dict[str, None], stack: tuple[str, ...] = ()
) -> Iterable[tuple[str, str]]:
    """
    Yield ``(spec, origin)`` from a requirements file, following ``-r`` and ``-c``.

    Parameters
    ----------
    path : pathlib.Path
        The file.
    seen : dict
        Resolved paths read so far (insertion-ordered); updated. A file
        included twice (a diamond) is read once.
    stack : tuple of str, default=()
        Files being read, outermost first (cycle detection).

    Yields
    ------
    tuple of (str, str)
        Requirement and the file it was read from. Constraint files (``-c``)
        are read and recorded in ``seen`` but yield nothing.

    Raises
    ------
    SourceError
        When a file includes itself.
    """
    yield from _walk(path, seen, stack, constraints=False)


def _walk(
    path: Path, seen: dict[str, None], stack: tuple[str, ...], *, constraints: bool
) -> Iterable[tuple[str, str]]:
    key = str(path.resolve())
    if key in stack:
        raise SourceError(f"{path} includes itself through -r/-c")
    if key in seen:
        return
    seen[key] = None
    text = _read(path)
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        include = _INCLUDE.match(line)
        constraint = _CONSTRAINT.match(line)
        if include or constraint:
            child = path.parent / (include or constraint).group(1).strip()
            yield from _walk(
                child, seen, (*stack, key), constraints=constraints or bool(constraint)
            )
            continue
        if line.startswith("-") or constraints:
            continue  # options (--index-url, -e ...) are read by pip/uv itself
        yield line, str(path)


def _read(path: Path) -> str:
    try:
        return read_text(path)
    except ConfigError as exc:
        raise SourceError(str(exc)) from None


def _digest(path: Path) -> tuple[str, str]:
    return str(path), hashlib.sha256(_read(path).encode("utf-8")).hexdigest()


def _requirements(source: SourceRef) -> DependencySet:
    path = Path(source.path)
    seen: dict[str, None] = {}
    deps = tuple(
        Dependency(pypi_name(s), s, Ecosystem.PYPI.value, o)
        for s, o in _requirement_lines(path, seen)
    )
    return DependencySet(
        dependencies=deps,
        requirement_files=(str(path),),
        inputs=tuple(_digest(Path(f)) for f in seen),
        requirement_origins=tuple(dict.fromkeys(d.origin for d in deps)),
    )


def _toml(path: Path) -> dict[str, Any]:
    try:
        return loads(_read(path), "toml")
    except ConfigError as exc:
        raise SourceError(f"{path}: {exc}", hint=exc.hint) from None


def _python_pin(spec: str) -> str | None:
    match = re.match(r"^\s*python\s*={1,2}\s*(3\.\d+(?:\.\d+)?)\*?\s*$", spec)
    return match.group(1) if match else None


def _environment(  # noqa: PLR0912 (one branch per entry kind)
    source: SourceRef,
) -> DependencySet:
    path = Path(source.path)
    try:
        tree = loads(_read(path), "yaml")
    except ConfigError as exc:
        raise SourceError(f"{path}: {exc}", hint=exc.hint) from None
    if not isinstance(tree, dict):
        raise SourceError(f"{path}: an environment file is a mapping")
    unknown = set(tree) - _ENV_KEYS
    if unknown:
        raise SourceError(
            f"{path}: unsupported environment-file keys {sorted(unknown)}",
            hint="supported: channels, dependencies (name and prefix are ignored);"
            " set variables in the run command or an activation script",
        )
    deps: list[Dependency] = []
    req_files: list[str] = []
    inputs: list[tuple[str, str]] = [_digest(path)]
    origins: list[str] = []
    notes: list[str] = [
        f"{path.name}: key {k!r} ignored; the request sets the location"
        for k in ("name", "prefix")
        if k in tree
    ]
    python = None
    for entry in tree.get("dependencies") or []:
        if isinstance(entry, str):
            name = conda_name(entry)
            if name == "python":
                python = _python_pin(entry) or python
            deps.append(Dependency(name, entry, Ecosystem.CONDA.value, str(path)))
        elif isinstance(entry, dict) and set(entry) == {"pip"}:
            for item in entry["pip"] or []:
                if not isinstance(item, str):
                    raise SourceError(f"{path}: pip entries must be strings")
                if item.startswith(("-r ", "--requirement ")):
                    req = (path.parent / item.split(None, 1)[1].strip()).resolve()
                    sub = _requirements(SourceRef(str(req), "requirements"))
                    deps.extend(sub.dependencies)
                    req_files.extend(sub.requirement_files)
                    inputs.extend(sub.inputs)
                    origins.extend(sub.requirement_origins)
                elif item.startswith(("-e ", "--editable ")):
                    target = item.split(None, 1)[1].strip()
                    local = path.parent / target
                    spec = str(local.resolve()) if local.exists() else target
                    deps.append(
                        Dependency(
                            pypi_name(Path(spec).name),
                            f"-e {spec}",
                            Ecosystem.PYPI.value,
                            str(path),
                        )
                    )
                elif item.startswith("-"):
                    raise SourceError(
                        f"{path}: pip option {item!r} is not supported in an"
                        " environment file",
                        hint="set index_url / extra_index_urls / find_links in the"
                        " request, or move the option into a requirements file"
                        " referenced with -r",
                    )
                else:
                    local = path.parent / item
                    spec = str(local.resolve()) if local.exists() else item
                    deps.append(
                        Dependency(
                            pypi_name(spec), spec, Ecosystem.PYPI.value, str(path)
                        )
                    )
        else:
            raise SourceError(f"{path}: unsupported dependency entry {entry!r}")
    channels = tuple(tree.get("channels") or ())
    if not all(isinstance(c, str) for c in channels):
        raise SourceError(f"{path}: channels must be strings")
    return DependencySet(
        dependencies=tuple(deps),
        requirement_files=tuple(req_files),
        environment_files=(str(path),),
        python=python,
        channels=channels,
        notes=tuple(notes),
        inputs=tuple(inputs),
        requirement_origins=tuple(origins),
    )


def _pyproject(source: SourceRef) -> DependencySet:
    path = Path(source.path)
    tree = _toml(path)
    project = tree.get("project") or {}
    dynamic = set(project.get("dynamic") or [])
    wanted_extras = [
        g for g in source.groups if g not in (tree.get("dependency-groups") or {})
    ]
    if "dependencies" in dynamic or (
        wanted_extras and "optional-dependencies" in dynamic
    ):
        raise SourceError(
            f"{path}: [project] declares its dependencies dynamic; they exist only"
            " once the project is built",
            hint="install the project itself: packages=['-e <project dir>'] (or"
            " list the requirements in a requirements file)",
        )
    deps = [
        Dependency(pypi_name(s), s, Ecosystem.PYPI.value, str(path))
        for s in project.get("dependencies") or []
    ]
    extras = project.get("optional-dependencies") or {}
    groups = tree.get("dependency-groups") or {}

    def group(name: str, stack: tuple[str, ...]) -> list[str]:
        if name in stack:
            raise SourceError(f"{path}: dependency group {name!r} includes itself")
        out: list[str] = []
        if name not in groups:
            raise SourceError(f"{path}: dependency group {name!r} does not exist")
        for item in groups[name]:
            if isinstance(item, dict) and set(item) == {"include-group"}:
                out.extend(group(item["include-group"], (*stack, name)))
            elif isinstance(item, str):
                out.append(item)
            else:
                raise SourceError(f"{path}: bad entry {item!r} in group {name!r}")
        return out

    for name in source.groups:
        if name in extras:
            specs = extras[name]
        elif name in groups:
            specs = group(name, ())
        else:
            raise SourceError(
                f"{path}: no optional-dependencies or dependency-groups entry {name!r}"
            )
        deps.extend(
            Dependency(pypi_name(s), s, Ecosystem.PYPI.value, f"{path}:{name}")
            for s in specs
        )
    notes = []
    if project.get("requires-python"):
        notes.append(f"{path.name} requires-python {project['requires-python']!r}")
    return DependencySet(
        dependencies=tuple(deps), notes=tuple(notes), inputs=(_digest(path),)
    )


def _pipfile_spec(name: str, value: Any, path: Path) -> tuple[str, str | None]:
    if isinstance(value, str):
        return (name if value.strip() in ("*", "") else f"{name}{value.strip()}"), None
    if not isinstance(value, dict):
        raise SourceError(f"{path}: bad entry for {name!r}")
    extras = value.get("extras") or []
    base = f"{name}[{','.join(extras)}]" if extras else name
    note = None
    if "git" in value:
        ref = value.get("ref")
        spec = f"{base} @ git+{value['git']}" + (f"@{ref}" if ref else "")
    elif "path" in value:
        spec = str((path.parent / value["path"]).resolve())
        if value.get("editable"):
            note = f"{path.name}: {name!r} is editable in Pipfile; installed normally"
    elif "file" in value:
        spec = f"{base} @ {value['file']}"
    else:
        version = value.get("version", "*")
        spec = base if version in ("*", "") else f"{base}{version}"
    if value.get("markers"):
        spec += f"; {value['markers']}"
    if value.get("index"):
        note = f"{path.name}: {name!r} names index {value['index']!r}; use index_url"
    return spec, note


def _pipfile(source: SourceRef) -> DependencySet:
    path = Path(source.path)
    tree = _toml(path)
    deps, notes = [], []
    tables = ["packages"] + (["dev-packages"] if "dev" in source.groups else [])
    unknown = set(source.groups) - {"dev"}
    if unknown:
        raise SourceError(
            f"{path}: Pipfile groups are only 'dev', got {sorted(unknown)}"
        )
    for table in tables:
        for name, value in (tree.get(table) or {}).items():
            spec, note = _pipfile_spec(name, value, path)
            deps.append(
                Dependency(
                    pypi_name(name), spec, Ecosystem.PYPI.value, f"{path}:{table}"
                )
            )
            if note:
                notes.append(note)
    requires = tree.get("requires") or {}
    python = requires.get("python_full_version") or requires.get("python_version")
    indexes = []
    for entry in tree.get("source") or []:
        url = entry.get("url") if isinstance(entry, dict) else None
        if not isinstance(url, str):
            raise SourceError(f"{path}: every [[source]] needs a url")
        if url.rstrip("/") not in _PUBLIC_PYPI:
            indexes.append(url)
    return DependencySet(
        dependencies=tuple(deps),
        python=python,
        pipfiles=(str(path),),
        notes=tuple(notes),
        inputs=(_digest(path),),
        indexes=tuple(indexes),
    )


_READERS = {
    "requirements": _requirements,
    "environment": _environment,
    "pyproject": _pyproject,
    "pipfile": _pipfile,
}


def read_sources(request: EnvironmentRequest) -> DependencySet:
    """
    Read every source of ``request`` into one :py:class:`DependencySet`.

    Parameters
    ----------
    request : EnvironmentRequest
        The request.

    Returns
    -------
    DependencySet
        Inline packages first, then each file in order.

    Raises
    ------
    SourceError
        For a missing or malformed file, or two sources pinning different
        Python versions.
    """
    parts = [
        DependencySet(
            dependencies=tuple(
                [
                    Dependency(pypi_name(s), s, Ecosystem.PYPI.value, "packages")
                    for s in request.packages
                ]
                + [
                    Dependency(
                        conda_name(s), s, Ecosystem.CONDA.value, "conda_packages"
                    )
                    for s in request.conda_packages
                ]
            )
        )
    ]
    for source in request.sources:
        if not os.path.isfile(source.path):
            raise SourceError(f"dependency file {source.path} does not exist")
        parts.append(_READERS[source.kind](source))
    pythons = {p.python for p in parts if p.python}
    if len(pythons) > 1:
        raise SourceError(f"sources pin different Python versions {sorted(pythons)}")
    return DependencySet(
        dependencies=tuple(d for p in parts for d in p.dependencies),
        requirement_files=tuple(f for p in parts for f in p.requirement_files),
        environment_files=tuple(f for p in parts for f in p.environment_files),
        python=next(iter(pythons), None),
        channels=tuple(dict.fromkeys(c for p in parts for c in p.channels)),
        pipfiles=tuple(f for p in parts for f in p.pipfiles),
        notes=tuple(n for p in parts for n in p.notes),
        inputs=tuple(dict.fromkeys(i for p in parts for i in p.inputs)),
        requirement_origins=tuple(
            dict.fromkeys(o for p in parts for o in p.requirement_origins)
        ),
        indexes=tuple(dict.fromkeys(i for p in parts for i in p.indexes)),
    )

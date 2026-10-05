# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Decide which tests a CI run executes and how they are split across jobs.

Notes
-----
**User notes.** Three ways to choose what runs::

    # Only what the changed files affect (the default for a pull request).
    python .github/scripts/ci_test_plan.py plan --mode auto --base <sha> --head <sha>

    # Everything, split into at most four jobs.
    python .github/scripts/ci_test_plan.py plan --mode all --max-shards 4

    # Named submodules, alone or several.
    python .github/scripts/ci_test_plan.py plan --mode custom --select "corpus _sphinx_ai_learn"

    # What can be selected.
    python .github/scripts/ci_test_plan.py units

The plan is printed as JSON. With ``--github-output`` the fields a workflow
needs (``run``, ``mode``, ``reason``, ``matrix``) are appended to that file,
and with ``--summary`` a Markdown account of the decision is appended there.

**Developer notes.** The package is a tree of *containers* and *units*. A
container is a directory whose children are independent submodules
(``scikitplot``, ``scikitplot/_externals``, ...); a unit is a child of a
container and is always tested as a whole. Everything else follows from
that, and from four rules:

1. A changed file inside a unit selects that unit, plus the ``tests``
   directory of every container above it.
2. A changed file that sits directly in a container (``__init__.py``,
   ``conftest.py``) can affect every unit below it, so it selects them all.
3. A changed file matching ``full_run_globs`` (build configuration, this
   planner) selects everything.
4. Any other changed file is not part of the library and selects nothing;
   when nothing is selected, no test job runs.

The planner fails closed: when the changed files cannot be determined, it
plans a full run and says why. In ``all`` mode it verifies that every test
file under the package root is inside a planned path before it returns.

**The plan collects what pytest collects.** A path named on the pytest
command line is collected even when ``norecursedirs`` lists it: that option
only stops pytest from *walking into* a directory. A sharded run names
paths, so the planner applies ``norecursedirs`` itself, read from the same
``pytest.ini`` pytest reads. A directory the project excludes is never a
unit, is never handed to pytest, and cannot be selected by hand; a change
inside it runs the ``tests`` directories above it. Before any plan is
returned the planner checks that no planned path is excluded.

Selected units are packed into shards by estimated duration, heaviest first
onto the lightest shard, with ties broken by name, so the same inputs always
produce the same plan. Estimates come from ``test_durations.json``; a unit
without an estimate gets ``default_unit_seconds``.

Only the standard library is used, so the plan job needs no installation.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import fnmatch
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

__all__ = [
    "MODES",
    "PYTEST_DEFAULT_NORECURSEDIRS",
    "PlanError",
    "Planner",
    "changed_files_from_git",
    "load_config",
    "load_norecursedirs",
    "main",
]

#: Modes a caller may request. ``default`` resolves through the configuration.
MODES = ("default", "auto", "all", "custom")

#: Values of the ``dependents`` setting.
DEPENDENTS = ("none", "direct")

_HERE = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _HERE.parent / "ci" / "test_plan.json"
_DEFAULT_DURATIONS = _HERE.parent / "ci" / "test_durations.json"

_REQUIRED_KEYS = {
    "schema": int,
    "package_root": str,
    "containers": list,
    "full_run_globs": list,
    "ignore_globs": list,
    "loose_test_globs": list,
    "default_mode": dict,
    "max_shards": int,
    "shard_timeout_minutes": int,
    "shard_target_minutes": (int, float),
    "default_unit_seconds": (int, float),
    "per_test_seconds": (int, float),
    "dependents": str,
}

_ZERO_SHA = "0" * 40

#: Key of the optional setting that names the pytest configuration file.
_PYTEST_INI_KEY = "pytest_ini"

#: The file pytest reads first, and the default of that setting.
_DEFAULT_PYTEST_INI = "pytest.ini"

#: What pytest uses when ``norecursedirs`` is not set (``_pytest/main.py``).
PYTEST_DEFAULT_NORECURSEDIRS = (
    "*.egg",
    ".*",
    "_darcs",
    "build",
    "CVS",
    "dist",
    "node_modules",
    "venv",
    "{arch}",
)


class PlanError(Exception):
    """Raised when the inputs do not describe a plan; the message says why."""


def load_config(  # ruff: ignore[too-many-branches]
    path,
):
    """
    Read and validate the planner configuration.

    Parameters
    ----------
    path : str or pathlib.Path
        A JSON file with the keys described in ``.github/ci/README.md``.

    Returns
    -------
    dict
        The configuration, unchanged.

    Raises
    ------
    PlanError
        If the file cannot be read, is not a JSON object, lacks a key, or a
        key has the wrong type or an impossible value.
    """
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PlanError(f"cannot read the plan configuration {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise PlanError(f"{path}: the configuration must be a JSON object")
    for key, kind in _REQUIRED_KEYS.items():
        if key not in config:
            raise PlanError(f"{path}: missing key {key!r}")
        if isinstance(config[key], bool) or not isinstance(config[key], kind):
            raise PlanError(f"{path}: {key!r} has the wrong type")
    if config["schema"] != 1:
        raise PlanError(f"{path}: unsupported schema {config['schema']!r}; expected 1")
    root = config["package_root"].strip("/")
    if not root or root not in config["containers"]:
        raise PlanError(f"{path}: 'containers' must include the package root {root!r}")
    for container in config["containers"]:
        if container != root and not container.startswith(root + "/"):
            raise PlanError(f"{path}: container {container!r} is outside {root!r}")
    if config["max_shards"] < 1:
        raise PlanError(f"{path}: 'max_shards' must be at least 1")
    _timeout_minutes = (
        1  # ruff: ignore[magic-value-comparison]
        <= config["shard_timeout_minutes"]
        <= 360  # ruff: ignore[magic-value-comparison]
    )
    if not _timeout_minutes:
        raise PlanError(f"{path}: 'shard_timeout_minutes' must be between 1 and 360")
    if config["shard_target_minutes"] <= 0:
        raise PlanError(f"{path}: 'shard_target_minutes' must be positive")
    if config["dependents"] not in DEPENDENTS:
        raise PlanError(f"{path}: 'dependents' must be one of {', '.join(DEPENDENTS)}")
    for event, mode in config["default_mode"].items():
        if mode not in ("auto", "all"):
            raise PlanError(f"{path}: default_mode[{event!r}] must be 'auto' or 'all'")
    if not isinstance(config.get(_PYTEST_INI_KEY, _DEFAULT_PYTEST_INI), str):
        raise PlanError(f"{path}: {_PYTEST_INI_KEY!r} must be a string")
    return config


def load_norecursedirs(path):
    """
    Read the ``norecursedirs`` setting pytest itself will use.

    Parameters
    ----------
    path : str or pathlib.Path or None
        A ``pytest.ini`` file. ``None`` or an empty string means the project
        has no such file.

    Returns
    -------
    tuple of str
        The patterns, in file order. :data:`PYTEST_DEFAULT_NORECURSEDIRS`
        when there is no file, no ``[pytest]`` section, or no such setting,
        because that is what pytest then applies.

    Raises
    ------
    PlanError
        If a file is named and cannot be read or parsed. A plan made without
        the project's exclusions would run tests the project never runs, so
        this is an error and not a fall-back.

    Notes
    -----
    **Developer notes.** The value is split the way pytest splits an
    ``args`` setting (:func:`shlex.split`), so several patterns may share a
    line. Only ``pytest.ini`` is understood; a project that keeps its pytest
    settings elsewhere sets ``"pytest_ini": ""`` and lists nothing here.
    """
    if not path:
        return PYTEST_DEFAULT_NORECURSEDIRS
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        text = Path(path).read_text(encoding="utf-8")
        parser.read_string(text, source=str(path))
    except (OSError, ValueError, configparser.Error) as exc:
        raise PlanError(f"cannot read the pytest configuration {path}: {exc}") from exc
    if not parser.has_option("pytest", "norecursedirs"):
        return PYTEST_DEFAULT_NORECURSEDIRS
    try:
        patterns = shlex.split(parser.get("pytest", "norecursedirs"), comments=True)
    except ValueError as exc:
        raise PlanError(f"{path}: norecursedirs cannot be split: {exc}") from exc
    return tuple(patterns)


def load_durations(path):
    """
    Read per-unit duration estimates.

    Parameters
    ----------
    path : str or pathlib.Path
        A JSON object mapping a unit path to ``{"tests": n, "work_seconds": s}``.
        A missing file is not an error: every unit then gets the default.

    Returns
    -------
    dict
        Unit path to ``(tests, work_seconds)``.

    Raises
    ------
    PlanError
        If the file exists and is not of that shape.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        units = raw["units"]
        return {
            str(unit): (int(entry["tests"]), float(entry["work_seconds"]))
            for unit, entry in units.items()
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise PlanError(f"cannot read the duration estimates {path}: {exc}") from exc


def changed_files_from_git(root, base, head):
    """
    List the files that differ between two commits.

    Parameters
    ----------
    root : str or pathlib.Path
        The repository.
    base, head : str
        Commits. The comparison is from their merge base to ``head``, which
        is what a pull request changes.

    Returns
    -------
    list of str
        POSIX paths relative to ``root``, sorted. A rename contributes both
        its old and its new path, so the unit a file left is tested too.

    Raises
    ------
    PlanError
        If either commit is missing or ``git`` fails. The caller turns this
        into a full run.
    """
    if not base or not head or set(base) == {"0"} or set(head) == {"0"}:
        raise PlanError("no base commit to compare with")
    command = [
        "git",
        "-C",
        str(root),
        "diff",
        "--name-only",
        "--no-renames",
        "-z",
        f"{base}...{head}",
    ]
    try:
        done = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PlanError(f"git could not be run: {exc}") from exc
    if done.returncode != 0:
        detail = done.stderr.decode("utf-8", "replace").strip().splitlines()
        raise PlanError(
            f"git diff {base}...{head} failed: {detail[-1] if detail else 'no output'}",
        )
    names = [
        name
        for name in done.stdout.decode("utf-8", "surrogateescape").split("\0")
        if name
    ]
    return sorted(set(names))


def _matches(path, patterns):
    """Return True if ``path`` matches any glob in ``patterns`` (``**`` crosses ``/``)."""
    for pattern in patterns:
        if fnmatch.fnmatchcase(path, pattern):
            return True
        if pattern.endswith("/**") and path.startswith(pattern[:-2]):
            return True
    return False


class Planner:
    """
    Plan a test run for one repository checkout.

    Parameters
    ----------
    root : str or pathlib.Path
        The repository checkout.
    config : dict
        From :func:`load_config`.
    durations : dict, optional
        From :func:`load_durations`.
    norecursedirs : iterable of str, optional
        From :func:`load_norecursedirs`. ``None`` means pytest's defaults.

    Notes
    -----
    **Developer notes.** The instance reads the directory tree and nothing
    else; it never imports the package, so it runs before anything is built.
    """

    def __init__(self, root, config, durations=None, norecursedirs=None):
        self.root = Path(root)
        self.config = config
        self.durations = dict(durations or {})
        self.norecursedirs = tuple(
            PYTEST_DEFAULT_NORECURSEDIRS if norecursedirs is None else norecursedirs,
        )
        self.package = config["package_root"].strip("/")
        self.containers = sorted(c.strip("/") for c in config["containers"])
        if not (self.root / self.package).is_dir():
            raise PlanError(f"{self.root}: no {self.package!r} directory here")

    # -- the tree -------------------------------------------------------

    def _matches_norecurse(self, directory):
        """Return True if pytest would not walk into ``directory`` itself."""
        name = PurePosixPath(directory).name
        for pattern in self.norecursedirs:
            # pytest's ``fnmatch_ex``: a pattern without a separator is tried
            # against the directory's name, any other against the end of its
            # path.
            if "/" in pattern:
                if fnmatch.fnmatchcase(f"/{directory}", f"*/{pattern}"):
                    return True
            elif fnmatch.fnmatchcase(name, pattern):
                return True
        return False

    def is_excluded(self, path):
        """
        Say whether a recursive ``pytest`` run would never reach ``path``.

        Parameters
        ----------
        path : str
            A POSIX path relative to the repository; a directory, or a file
            (then its directory is what counts).

        Returns
        -------
        bool
            True if ``path`` or a directory above it matches
            ``norecursedirs`` or is a ``__pycache__``.
        """
        parts = PurePosixPath(path).parts
        if (self.root / path).is_file():
            parts = parts[:-1]
        return any(
            parts[index] == "__pycache__"
            or self._matches_norecurse("/".join(parts[: index + 1]))
            for index in range(len(parts))
        )

    def _children(self, container, excluded=False):
        """
        Return the sub-directories of ``container`` pytest would walk into.

        With ``excluded=True``, return the ones it would not, instead.
        """
        directory = self.root / container
        if not directory.is_dir():
            return []
        return sorted(
            child
            for child in (
                f"{container}/{entry.name}"
                for entry in directory.iterdir()
                if entry.is_dir()
                and not entry.name.startswith(".")
                and entry.name != "__pycache__"
            )
            if self.is_excluded(child) is excluded
        )

    def excluded_units(self):
        """
        Return the directories that would be units if pytest collected them.

        Returns
        -------
        list of str
            Children of a container that ``norecursedirs`` excludes, sorted.
            A container that is itself excluded is listed once, unexpanded.
        """
        found = []
        for container in self.containers:
            if self.is_excluded(container):
                continue
            found.extend(self._children(container, excluded=True))
        return sorted(found)

    def units(self, container=None):
        """
        Return every unit at or below ``container``.

        Parameters
        ----------
        container : str, optional
            A container path; the package root by default.

        Returns
        -------
        list of str
            Unit paths, sorted. A container's own ``tests`` directory is a
            unit; a nested container is expanded, not listed.
        """
        found = []
        for child in self._children(container or self.package):
            if child in self.containers:
                found.extend(self.units(child))
            else:
                found.append(child)
        return sorted(found)

    def loose_tests(self, container=None):
        """Return test files that sit directly in a container, recursively."""
        found = []
        for current in [
            c for c in self.containers if self._is_under(c, container or self.package)
        ]:
            directory = self.root / current
            if not directory.is_dir() or self.is_excluded(current):
                continue
            found.extend(
                f"{current}/{entry.name}"
                for entry in sorted(directory.iterdir())
                if entry.is_file()
                and _matches(entry.name, self.config["loose_test_globs"])
            )
        return found

    @staticmethod
    def _is_under(path, ancestor):
        return path == ancestor or path.startswith(ancestor + "/")

    def owner(self, path):
        """
        Say which part of the tree a file belongs to.

        Parameters
        ----------
        path : str
            A POSIX path relative to the repository.

        Returns
        -------
        tuple
            ``("unit", unit)``, ``("container", container)`` for a file
            directly in a container, or ``("outside", "")``.
        """
        parts = PurePosixPath(path).parts
        if not parts or parts[0] != self.package:
            return "outside", ""
        current = self.package
        for index in range(1, len(parts)):
            is_last = index == len(parts) - 1
            candidate = f"{current}/{parts[index]}"
            if is_last and not (self.root / candidate).is_dir():
                return "container", current
            if candidate in self.containers:
                current = candidate
                continue
            return "unit", candidate
        return "container", current

    def ancestor_tests(self, unit):
        """Return the existing ``tests`` directories of the containers above ``unit``."""
        found = []
        parent = str(PurePosixPath(unit).parent)
        for container in self.containers:
            if self._is_under(parent, container):
                tests = f"{container}/tests"
                if (
                    tests != unit
                    and (self.root / tests).is_dir()
                    and not self.is_excluded(tests)
                ):
                    found.append(tests)
        return sorted(found)

    # -- selection ------------------------------------------------------

    def resolve_selection(self, text):
        """
        Turn a user's list of submodule names into unit paths.

        Parameters
        ----------
        text : str
            Names separated by spaces, commas or new lines. A name may be a
            full path (``scikitplot/corpus``), a path below the package root
            (``_externals/_sphinx_ext/_sphinx_ai_learn``), a container, or a
            bare name that identifies exactly one unit (``corpus``).

        Returns
        -------
        list of str
            Unit paths, sorted, without duplicates.

        Raises
        ------
        PlanError
            If the list is empty, or a name matches no unit or several.
        """
        names = [name for name in text.replace(",", " ").split() if name]
        if not names:
            raise PlanError("custom mode needs at least one submodule name")
        everything = self.units()
        excluded = self.excluded_units()
        chosen = []
        for name in names:
            cleaned = name.strip("/")
            candidates = [cleaned, f"{self.package}/{cleaned}"]
            # A path at or below an excluded directory is refused outright:
            # naming it would make pytest collect what the project excludes.
            barred = [c for c in candidates if self.is_excluded(c)]
            if barred:
                raise PlanError(self._excluded_message(name, barred[0]))
            hit = [c for c in candidates if c in everything]
            if hit:
                chosen.append(hit[0])
                continue
            container = [c for c in candidates if c in self.containers]
            if container:
                chosen.extend(self.units(container[0]))
                continue
            by_name = [
                unit for unit in everything if PurePosixPath(unit).name == cleaned
            ]
            if len(by_name) == 1:
                chosen.append(by_name[0])
                continue
            if by_name:
                raise PlanError(
                    f"{name!r} names several submodules: {', '.join(by_name)}",
                )
            barred = [unit for unit in excluded if PurePosixPath(unit).name == cleaned]
            if barred:
                raise PlanError(self._excluded_message(name, barred[0]))
            raise PlanError(
                f"{name!r} is not a submodule; run 'ci_test_plan.py units' to list them",
            )
        return sorted(set(chosen))

    @staticmethod
    def _excluded_message(name, path):
        return (
            f"{name!r} cannot be selected: {path} is excluded from test "
            "collection by 'norecursedirs', so no full run collects it "
            "(list such directories with 'ci_test_plan.py units --excluded')"
        )

    def select_for_changes(self, changed):
        """
        Apply the four rules of the module docstring to a list of changed files.

        Parameters
        ----------
        changed : iterable of str
            POSIX paths relative to the repository.

        Returns
        -------
        units : list of str or None
            Selected unit paths, or ``None`` when everything must run.
        reasons : list of str
            One line per decision, for the summary.
        """
        selected, reasons = set(), []
        for path in sorted(set(changed)):
            if _matches(path, self.config["full_run_globs"]):
                return None, [
                    f"`{path}` is build or test configuration: everything runs"
                ]
            if _matches(path, self.config["ignore_globs"]):
                continue
            kind, where = self.owner(path)
            if kind == "outside":
                continue
            if kind == "container":
                if where == self.package:
                    return None, [f"`{path}` sits at the package root: everything runs"]
                below = self.units(where)
                selected.update(below)
                reasons.append(
                    f"`{path}` sits in `{where}`: all {len(below)} submodules below it",
                )
            else:
                if where not in selected:
                    reasons.append(f"`{where}` changed")
                selected.add(where)
        return sorted(selected), reasons

    def with_dependents(self, units):
        """
        Add the units that import any of ``units``.

        Parameters
        ----------
        units : list of str
            Selected unit paths.

        Returns
        -------
        list of str
            ``units`` plus every unit with a module that imports one of them
            by absolute name or by relative import. One step only: importers
            of importers are not followed.
        """
        targets = set(units)
        dotted = {unit: unit.replace("/", ".") for unit in targets}
        added = set()
        for unit in self.units():
            if unit in targets:
                continue
            if self._imports_any(unit, targets, dotted):
                added.add(unit)
        return sorted(targets | added)

    def _imports_any(self, unit, targets, dotted):
        for source in sorted((self.root / unit).rglob("*.py")):
            try:
                tree = ast.parse(source.read_text(encoding="utf-8", errors="replace"))
            except (OSError, SyntaxError, ValueError):
                continue
            package = source.relative_to(self.root).parent.as_posix().split("/")
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        base = package[: len(package) - (node.level - 1)]
                        prefix = ".".join(base + ([node.module] if node.module else []))
                        names = [prefix] + [
                            f"{prefix}.{alias.name}" for alias in node.names
                        ]
                    elif node.module:
                        names = [node.module] + [
                            f"{node.module}.{alias.name}" for alias in node.names
                        ]
                for name in names:
                    for target in targets:
                        if name == dotted[target] or name.startswith(
                            dotted[target] + ".",
                        ):
                            return True
        return False

    # -- sharding -------------------------------------------------------

    def weight(self, path):
        """Return the estimated seconds a path takes, always positive."""
        if path in self.durations:
            tests, work = self.durations[path]
            return max(work + tests * float(self.config["per_test_seconds"]), 1.0)
        return max(float(self.config["default_unit_seconds"]), 1.0)

    def shards(self, paths, max_shards):
        """
        Pack ``paths`` into groups of similar duration.

        The number of groups is the estimated total divided by
        ``shard_target_minutes``, rounded up, and never more than
        ``max_shards`` or the number of paths. A small selection therefore
        runs in one job instead of paying the build once per path.

        Parameters
        ----------
        paths : list of str
            Paths pytest will be given.
        max_shards : int
            Upper bound on the number of groups.

        Returns
        -------
        list of dict
            ``{"name", "paths", "seconds"}`` per shard, in a stable order.
        """
        if max_shards < 1:
            raise PlanError("max_shards must be at least 1")
        for path in paths:
            # Paths reach the test job as one space-separated value.
            if not path or any(character.isspace() for character in path):
                raise PlanError(f"a path with white space cannot be planned: {path!r}")
        ordered = sorted(set(paths), key=lambda path: (-self.weight(path), path))
        total = sum(self.weight(path) for path in ordered)
        wanted = -(-total // (float(self.config["shard_target_minutes"]) * 60.0))
        count = max(1, min(max_shards, len(ordered), int(wanted)))
        bins = [{"paths": [], "seconds": 0.0} for _ in range(count)]
        for path in ordered:
            lightest = min(range(count), key=lambda i: (bins[i]["seconds"], i))
            bins[lightest]["paths"].append(path)
            bins[lightest]["seconds"] += self.weight(path)
        width = len(str(count))
        return [
            {
                "name": f"shard-{index + 1:0{width}d}-of-{count:0{width}d}",
                "paths": sorted(group["paths"]),
                "seconds": round(group["seconds"], 1),
            }
            for index, group in enumerate(bins)
        ]

    def collectable_test_files(self):
        """
        Return the test files a recursive ``pytest <package root>`` reaches.

        Returns
        -------
        list of str
            POSIX paths, sorted: every file matching ``loose_test_globs``
            below the package root that is not inside a directory
            ``norecursedirs`` excludes.
        """
        globs = self.config["loose_test_globs"]
        found = []
        top = self.root / self.package
        for current, directories, files in os.walk(top):
            here = Path(current).relative_to(self.root).as_posix()
            # Pruned in place, which is how pytest's own walk behaves: an
            # excluded directory hides everything below it.
            directories[:] = sorted(
                name for name in directories if not self.is_excluded(f"{here}/{name}")
            )
            found.extend(
                f"{here}/{name}"
                for name in files
                if name.endswith(".py") and _matches(name, globs)
            )
        return sorted(found)

    def _assert_complete(self, paths):
        """Raise if a test file pytest would collect is outside ``paths``."""
        missed = [
            source
            for source in self.collectable_test_files()
            if not any(self._is_under(source, path) for path in paths)
        ]
        if missed:
            shown = ", ".join(missed[:5])
            raise PlanError(
                f"a full run would not collect {len(missed)} test file(s): {shown}",
            )

    def _assert_collectable(self, paths):
        """
        Raise if a planned path is one pytest is configured never to enter.

        Naming such a path on the command line would collect it anyway, and
        the run would then include tests no recursive run of the project
        has; this is the check that keeps a sharded run equal to a whole one.
        """
        barred = sorted(path for path in paths if self.is_excluded(path))
        if barred:
            raise PlanError(
                "planned path(s) excluded by 'norecursedirs': " + ", ".join(barred),
            )

    # -- the plan -------------------------------------------------------

    def plan(  # ruff: ignore[too-many-branches, too-many-positional-arguments]
        self,
        mode,
        event="",
        changed=None,
        select="",
        max_shards=None,
        diff_error="",
    ):
        """
        Build the plan.

        Parameters
        ----------
        mode : str
            One of :data:`MODES`.
        event : str, optional
            The triggering event; used only to resolve ``default``.
        changed : list of str, optional
            Changed files, for ``auto``. ``None`` means they are unknown.
        select : str, optional
            Submodule names, for ``custom``.
        max_shards : int, optional
            Overrides the configured maximum.
        diff_error : str, optional
            Why ``changed`` is unknown, for the summary.

        Returns
        -------
        dict
            ``run`` (bool), ``mode``, ``requested_mode``, ``reason`` (list of
            str), ``units`` (list of str), ``shards`` (list of dict),
            ``timeout_minutes`` (int).

        Raises
        ------
        PlanError
            On an unknown mode or an unusable selection.
        """
        if mode not in MODES:
            raise PlanError(f"unknown mode {mode!r}; use one of {', '.join(MODES)}")
        requested = mode
        if mode == "default":
            mode = self.config["default_mode"].get(event, "all")
        limit = self.config["max_shards"] if max_shards is None else max_shards
        reasons, units = [], []
        if mode == "custom":
            units = self.resolve_selection(select)
            reasons.append(f"selected by hand: {', '.join(units)}")
        elif mode == "auto":
            if changed is None:
                mode = "all"
                reasons.append(
                    f"changed files unknown ({diff_error or 'not given'}): everything runs",
                )
            else:
                units, reasons = self.select_for_changes(changed)
                if units is None:
                    mode = "all"
                elif not units:
                    reasons.append(
                        f"none of the {len(set(changed))} changed file(s) is part of the library",
                    )
        if mode == "all":
            paths = self.units() + self.loose_tests()
            self._assert_complete(paths)
            reasons.append(f"all {len(self.units())} submodules")
            units = self.units()
        else:
            if units and self.config["dependents"] == "direct":
                widened = self.with_dependents(units)
                extra = sorted(set(widened) - set(units))
                if extra:
                    reasons.append(f"importers of the selection: {', '.join(extra)}")
                units = widened
            barred = sorted(unit for unit in units if self.is_excluded(unit))
            if barred:
                reasons.append(
                    "excluded from collection by 'norecursedirs', only the tests "
                    f"above them run: {', '.join(barred)}",
                )
            existing = [
                unit
                for unit in units
                if unit not in barred and (self.root / unit).is_dir()
            ]
            gone = sorted(set(units) - set(existing) - set(barred))
            if gone:
                reasons.append(
                    f"no longer present, only the tests above them run: {', '.join(gone)}",
                )
            paths = set(existing)
            for unit in units:
                paths.update(self.ancestor_tests(unit))
            paths = sorted(paths)
        self._assert_collectable(paths)
        shards = self.shards(paths, limit) if paths else []
        return {
            "run": bool(shards),
            "mode": mode,
            "requested_mode": requested,
            "reason": reasons,
            "units": sorted(units),
            "shards": shards,
            "timeout_minutes": self.config["shard_timeout_minutes"],
        }


def _matrix(plan):
    """Return the job matrix a workflow consumes."""
    return {
        "include": [
            {
                "name": shard["name"],
                "paths": " ".join(shard["paths"]),
                "estimate_minutes": max(1, round(shard["seconds"] / 60)),
                "timeout_minutes": plan["timeout_minutes"],
            }
            for shard in plan["shards"]
        ]
    }


def _summary(plan):
    """Return the Markdown account of ``plan``."""
    lines = ["## Test plan", ""]
    lines.append(f"- Mode: **{plan['mode']}** (requested: {plan['requested_mode']})")
    lines.append(f"- Runs tests: **{'yes' if plan['run'] else 'no'}**")
    lines.extend(f"- {reason}" for reason in plan["reason"])
    if plan["shards"]:
        lines += ["", "| Job | Estimate | Paths |", "|---|---|---|"]
        for shard in plan["shards"]:
            shown = ", ".join(f"`{path}`" for path in shard["paths"])
            lines.append(
                f"| {shard['name']} | {max(1, round(shard['seconds'] / 60))} min | {shown} |",
            )
    return "\n".join(lines) + "\n"


def _append(path, text):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0].strip(),
    )
    parser.add_argument(
        "--root",
        default=".",
        help="repository checkout (default: current directory)",
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="planner configuration",
    )
    parser.add_argument(
        "--durations",
        default=str(_DEFAULT_DURATIONS),
        help="duration estimates",
    )
    parser.add_argument(
        "--pytest-ini",
        default=None,
        help=(
            "pytest configuration to read 'norecursedirs' from, relative to "
            "--root (default: the 'pytest_ini' setting, else pytest.ini; "
            "empty: none)"
        ),
    )
    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )
    plan = commands.add_parser(
        "plan",
        help="print the plan as JSON",
    )
    plan.add_argument(
        "--mode",
        default="default",
        choices=MODES,
    )
    plan.add_argument(
        "--event",
        default="",
        help="triggering event, to resolve --mode default",
    )
    plan.add_argument(
        "--base",
        default="",
        help="commit to compare from, for auto",
    )
    plan.add_argument(
        "--head",
        default="",
        help="commit to compare to, for auto",
    )
    plan.add_argument(
        "--changed-files",
        default="",
        help="file listing changed paths, one per line, instead of git",
    )
    plan.add_argument(
        "--select",
        default="",
        help="submodule names, for custom",
    )
    plan.add_argument(
        "--max-shards",
        default="",
        help="override the configured maximum",
    )
    plan.add_argument(
        "--github-output",
        default="",
        help="append workflow outputs to this file",
    )
    plan.add_argument(
        "--summary",
        default="",
        help="append a Markdown summary to this file",
    )
    units = commands.add_parser(
        "units",
        help="list selectable submodules",
    )
    units.add_argument(
        "--excluded",
        action="store_true",
        help="list the directories 'norecursedirs' keeps out, instead",
    )
    return parser


def main(argv=None):
    """
    Run the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments; ``sys.argv[1:]`` by default.

    Returns
    -------
    int
        ``0`` on success, ``2`` when the inputs do not describe a plan.
    """
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        ini = args.pytest_ini
        if ini is None:
            ini = config.get(_PYTEST_INI_KEY, _DEFAULT_PYTEST_INI)
        planner = Planner(
            args.root,
            config,
            load_durations(args.durations),
            load_norecursedirs(Path(args.root) / ini if ini else None),
        )
        if args.command == "units":
            listed = planner.excluded_units() if args.excluded else planner.units()
            sys.stdout.write("".join(f"{unit}\n" for unit in listed))
            return 0
        max_shards = None
        if args.max_shards.strip():
            if (
                not args.max_shards.strip().isascii()
                or not args.max_shards.strip().isdigit()
            ):
                raise PlanError(
                    f"--max-shards must be a positive whole number, got {args.max_shards!r}",
                )
            max_shards = int(args.max_shards)
        mode = args.mode
        effective = (
            planner.config["default_mode"].get(args.event, "all")
            if mode == "default"
            else mode
        )
        changed, diff_error = None, ""
        if effective == "auto":
            try:
                if args.changed_files:
                    text = Path(args.changed_files).read_text(encoding="utf-8")
                    changed = [
                        line.strip() for line in text.splitlines() if line.strip()
                    ]
                else:
                    changed = changed_files_from_git(args.root, args.base, args.head)
            except (PlanError, OSError) as exc:
                diff_error = str(exc)
        plan = planner.plan(
            mode,
            event=args.event,
            changed=changed,
            select=args.select,
            max_shards=max_shards,
            diff_error=diff_error,
        )
    except PlanError as exc:
        sys.stderr.write(f"ci_test_plan: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if args.github_output:
        _append(
            args.github_output,
            "run={}\nmode={}\nmatrix={}\n".format(
                "true" if plan["run"] else "false",
                plan["mode"],
                json.dumps(_matrix(plan), separators=(",", ":")),
            ),
        )
    if args.summary:
        _append(args.summary, _summary(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

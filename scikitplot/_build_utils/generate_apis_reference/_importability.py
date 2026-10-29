# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Check that everything the API reference names can be imported, as Sphinx will.

The documentation build imports three kinds of target from
``docs/source/apis_reference.py``; one that fails stops or degrades the build:

* ``_get_submodule(parent, child)`` calls, which run when ``conf.py``
  imports the file (``strict=True``, the default, raises: the build stops
  before Sphinx reads a page);
* every ``APIS_REFERENCE`` module, imported by the page template
  (``imp0rt``) and by ``.. automodule::``;
* every ``autosummary`` and ``classes`` name, and every
  ``DEPRECATED_APIS_REFERENCE`` name, resolved by autosummary the way
  :func:`_resolve_name` does (attribute, else submodule import).

Unlike ``check``, nothing here compares the reference with the package's
public names: an undocumented public object is not a build failure.

Notes
-----
**User notes.** ``python -m scikitplot._build_utils.generate_apis_reference
imports`` prints only problems and a one-line summary. A target whose import
fails only because an optional third-party dependency is absent is reported
as ``UNAVAILABLE`` and does not fail the command; ``--strict`` makes it fail,
for an environment that installs every documentation dependency.

**Developer notes.** Classification is by the exception, not by its message,
following the cause chain (``__cause__``, then ``__context__``) to the first
``ImportError``:

* one that carries a capability ``status`` (the ``CapabilityError`` of
  ``cleanprompt``, ``corpus`` and ``mcp``; a string in any case or an Enum
  member) is *unavailable* when the status is ``absent``, else an *error*
  (``broken``: installed but failing);
* one whose ``name`` is a top-level module outside the package that is not
  installed (``importlib.util.find_spec`` is None) is *unavailable* only when
  the import ran in the package's own code (its traceback); imported from a
  third-party package, it is a broken install: an *error*. Installed but
  failing is an *error*;
* anything else, including a missing ``scikitplot.*`` module (a missing
  generated file such as ``config/__config__.py``) and a name that does not
  exist, is an *error*.

A strict ``_get_submodule`` target is always an error, whatever the cause:
``conf.py`` raises on it. A ``strict=False`` one is a warning and does not
stand for its module, which is still checked as itself.

The reference file is parsed with :mod:`ast` and never executed.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Iterable

__all__ = [
    "ImportTarget",
    "ImportabilityReport",
    "check_importability",
    "classify_import_failure",
    "import_targets",
]

#: Kinds of target, in the order a documentation build meets them.
KINDS = ("submodule", "module", "name", "deprecated")

#: The most cause-chain links followed when classifying an exception.
_MAX_CHAIN = 16


@dataclass(frozen=True)
class ImportTarget:
    """
    One thing the documentation build imports.

    Parameters
    ----------
    kind : {"submodule", "module", "name", "deprecated"}
        Where the build meets it (see the module notes).
    module : str
        Module the target belongs to (imported first).
    name : str or None
        Name resolved inside ``module``; None for a module itself.
    strict : bool
        Whether a failure stops the build (``_get_submodule(strict=False)``
        only writes a warning into the page).
    owner : str or None
        The ``APIS_REFERENCE`` entry whose description holds a submodule
        call; None for the others (their own module is the owner).
    """

    kind: str
    module: str
    name: str | None = None
    strict: bool = True
    owner: str | None = None

    @property
    def label(self) -> str:
        """The dotted target, as Sphinx names it (str)."""
        return self.module if self.name is None else f"{self.module}.{self.name}"

    @property
    def scope(self) -> str:
        """The ``APIS_REFERENCE`` entry a ``--module`` filter matches (str)."""
        return self.owner or self.module


@dataclass
class ImportabilityReport:
    """
    Outcome of :func:`check_importability`.

    Parameters
    ----------
    checked : int
        Targets checked.
    errors : list of str
        ``"<kind> <target>: <reason>"`` for failures that break the build.
    unavailable : list of str
        Same form, for targets that need an absent optional dependency.
    warnings : list of str
        Failures that do not stop the build (``strict=False`` submodules).
    """

    checked: int = 0
    errors: list = field(default_factory=list)
    unavailable: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def failed(self, *, strict: bool = False) -> bool:
        """
        Return whether the documentation build would fail.

        Parameters
        ----------
        strict : bool, default=False
            Count unavailable optional dependencies as failures.

        Returns
        -------
        bool
            True when there is an error (or, with ``strict``, an unavailable target).
        """
        return bool(self.errors or (strict and self.unavailable))

    def to_dict(self) -> dict:
        """
        Return the report as plain data.

        Returns
        -------
        dict
            ``checked``, ``errors``, ``unavailable``, ``warnings``.
        """
        return {
            "checked": self.checked,
            "errors": list(self.errors),
            "unavailable": list(self.unavailable),
            "warnings": list(self.warnings),
        }


def _literal(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None


def _call_argument(node: ast.Call, index: int, keyword: str) -> object:
    """Return the literal value of a call's argument, by position or keyword."""
    if len(node.args) > index:
        return _literal(node.args[index])
    for item in node.keywords:
        if item.arg == keyword:
            return _literal(item.value)
    return None


def _submodule_calls(entry: ast.AST, owner: str) -> list[ImportTarget]:
    """
    Return the ``_get_submodule`` calls with literal arguments in one entry.

    Parameters
    ----------
    entry : ast.AST
        The value of one ``APIS_REFERENCE`` entry.
    owner : str
        Its key.

    Returns
    -------
    list of ImportTarget
        ``kind="submodule"``, in source order. The module is
        ``f"{module_name}.{submodule_name}"``, exactly what
        ``_get_submodule`` imports (``"__init__"`` included).
    """
    found: list[ImportTarget] = []
    for node in ast.walk(entry):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_get_submodule"
        ):
            continue
        parent = _call_argument(node, 0, "module_name")
        child = _call_argument(node, 1, "submodule_name")
        if not (isinstance(parent, str) and isinstance(child, str)):
            continue
        strict = next(
            (_literal(k.value) for k in node.keywords if k.arg == "strict"), True
        )
        strict = strict if isinstance(strict, bool) else True
        found.append(
            ImportTarget("submodule", f"{parent}.{child}", None, strict, owner)
        )
    return found


def _assignment(tree: ast.Module, name: str) -> ast.AST | None:
    """Return the value assigned to the top-level ``name``, or None."""
    for node in tree.body:
        targets = (
            [node.target]
            if isinstance(node, ast.AnnAssign)
            else (node.targets if isinstance(node, ast.Assign) else [])
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return node.value
    return None


def _deprecated_names(tree: ast.Module) -> list[str]:
    node = _assignment(tree, "DEPRECATED_APIS_REFERENCE")
    value = _literal(node) if node is not None else None
    if not isinstance(value, dict):
        return []
    return [
        name
        for names in value.values()
        if isinstance(names, list)
        for name in names
        if isinstance(name, str)
    ]


def import_targets(model, *, package_root: str = "scikitplot") -> list[ImportTarget]:
    """
    Return every target the documentation build imports, without importing.

    Parameters
    ----------
    model : ReferenceModel
        The parsed reference (:func:`load_reference`).
    package_root : str, default="scikitplot"
        Package that ``DEPRECATED_APIS_REFERENCE`` names are relative to.

    Returns
    -------
    list of ImportTarget
        Unique targets: submodule calls, then modules with their names, then
        deprecated names.
    """
    tree = ast.parse(model.source, filename=str(model.path))
    targets: list[ImportTarget] = []
    reference = _assignment(tree, "APIS_REFERENCE")
    if isinstance(reference, ast.Dict):
        for key, value in zip(reference.keys, reference.values):
            owner = _literal(key) if key is not None else None
            if isinstance(owner, str):
                targets.extend(_submodule_calls(value, owner))
    for module_name in sorted(model.modules):
        module_ref = model.modules[module_name]
        targets.append(ImportTarget("module", module_name))
        names = list(module_ref.documented)
        for section in module_ref.sections:
            names.extend(section.classes)
        targets.extend(
            ImportTarget("name", module_name, name) for name in dict.fromkeys(names)
        )
    targets.extend(
        ImportTarget("deprecated", package_root, name)
        for name in _deprecated_names(tree)
    )
    return list(dict.fromkeys(targets))


def _chain(exc: BaseException) -> Iterable[BaseException]:
    seen = 0
    current: BaseException | None = exc
    while current is not None and seen < _MAX_CHAIN:
        yield current
        seen += 1
        current = current.__cause__ or current.__context__


def _importer(exc: BaseException) -> str | None:
    """Return the module whose code ran the failing import (importlib skipped)."""
    name = None
    tb = exc.__traceback__
    while tb is not None:
        module = tb.tb_frame.f_globals.get("__name__", "")
        if isinstance(module, str) and not module.startswith(
            ("importlib", "_frozen_importlib")
        ):
            name = module
        tb = tb.tb_next
    return name


def classify_import_failure(exc: BaseException, *, package_root: str) -> str:
    """
    Return ``"unavailable"`` or ``"error"`` for an import failure.

    Parameters
    ----------
    exc : BaseException
        What importing or resolving the target raised.
    package_root : str
        The package being documented; its own missing modules are errors.

    Returns
    -------
    str
        ``"unavailable"`` when the failure is an optional third-party
        dependency that the documented package itself imports and that is not
        installed; otherwise ``"error"`` (see the module notes).
    """
    for item in _chain(exc):
        if not isinstance(item, ImportError):
            continue
        status = getattr(item, "status", None)
        status = getattr(status, "value", status)  # an Enum member or a string
        if isinstance(status, str):  # a capability report decides, absent or not
            return "unavailable" if status.lower() == "absent" else "error"
        name = getattr(item, "name", None)
        if not isinstance(name, str) or not name:
            continue
        top = name.split(".")[0]
        if top == package_root:
            return "error"
        try:
            spec = importlib.util.find_spec(top)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            return "error"  # installed but failing
        importer = _importer(item)
        own = importer is not None and (
            importer == package_root or importer.startswith(package_root + ".")
        )
        return "unavailable" if own else "error"  # else: a broken third-party install
    return "error"


def _reason(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:300]}"


def _file(
    report: ImportabilityReport, target: ImportTarget, exc: BaseException, root: str
) -> None:
    """Add the failure of ``target`` to the report's errors, unavailable or warnings."""
    cause = exc.__cause__ if exc.__cause__ is not None else exc
    line = f"{target.kind} {target.label}: {_reason(cause)}"
    if target.kind == "submodule":  # conf.py raises on a strict one: the build stops
        (report.errors if target.strict else report.warnings).append(line)
    elif classify_import_failure(exc, package_root=root) == "unavailable":
        report.unavailable.append(line)
    else:
        report.errors.append(line)


def check_importability(
    model,
    *,
    package_root: str = "scikitplot",
    modules: set | None = None,
    resolve=None,
) -> ImportabilityReport:
    """
    Import every target of the reference and classify each failure.

    Parameters
    ----------
    model : ReferenceModel
        The parsed reference (:func:`load_reference`).
    package_root : str, default="scikitplot"
        The documented package.
    modules : set of str or None, default=None
        Limit to these ``APIS_REFERENCE`` entries: their modules, names and
        the submodule calls of their descriptions (deprecated names belong
        to the package root).
    resolve : callable or None, default=None
        ``resolve(module_object, module_name, name)``; the package's
        ``_resolve_name`` when None.

    Returns
    -------
    ImportabilityReport
        Counts and one line per failure, in target order.
    """
    if resolve is None:
        from . import _resolve_name as resolve  # noqa: PLC0415 - avoid a cycle

    report = ImportabilityReport()
    imported: dict = {}
    failed_modules: dict = {}
    for target in import_targets(model, package_root=package_root):
        if modules is not None and target.scope not in modules:
            continue
        if target.module in failed_modules:
            continue  # reported once, at the module
        report.checked += 1
        try:
            module = imported.get(target.module)
            if module is None:
                try:
                    module = importlib.import_module(target.module)
                except (Exception, SystemExit) as exc:  # noqa: BLE001 - reported
                    if target.strict:  # a warning does not stand for the module
                        failed_modules[target.module] = exc
                    raise
                imported[target.module] = module
            if target.name is not None:
                resolve(module, target.module, target.name)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - reported
            _file(report, target, exc, package_root)
    return report

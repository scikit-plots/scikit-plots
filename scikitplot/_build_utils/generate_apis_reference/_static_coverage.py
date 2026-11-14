# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Compare each module's ``__all__`` with the API reference, from the source alone.

``check`` and ``plan`` import the package, so they need a built install; a
public name added without its API-reference entry (Rule 172) was therefore
seen only on a machine with a build. This module reads ``__all__`` from the
source with :mod:`ast`, never importing it, and compares it with the
``autosummary`` lists of ``docs/source/apis_reference.py``:

* **missing**: in ``__all__`` but in no section of its module (and not in
  the module's ``ignore`` or a section's ``exclude``);
* **stale**: documented, but no longer in ``__all__``;
A documented entry that names a submodule (``"_seaborn"``), or a name
through one (``"_probscale.probplot"``), is not stale when that resolves in
the source (:func:`documented_target_exists`), as autosummary resolves it.

A module's lazily exported names (imported under ``if TYPE_CHECKING:`` in
a module with a module-level ``__getattr__``, :func:`static_lazy`) count
as public: documented, they are not stale; undocumented, they are missing.

* **private**: a ``_name`` (not a dunder) in ``__all__``: exported by
  ``import *`` although its name says it is internal. Import it by name
  from its own module instead.

Existing gaps are recorded in a baseline file next to the template; the
check fails only on a *new* gap, or on a baseline entry that is no longer a
gap (so a fixed gap is removed from the baseline in the same change, and
the baseline only shrinks). Both directions keep the rule exact: nothing
new slips in, nothing fixed is left listed.

Notes
-----
**User notes.** ``python -m scikitplot._build_utils.generate_apis_reference
coverage`` prints new gaps and fixed ones, and exits 1 if there are any.
``coverage --write`` rewrites the baseline from the current tree: run it
only to accept the tree as it is (after fixing, it removes fixed entries).

**Developer notes.** ``__all__`` is resolved from module-level statements:
list, tuple and string literals; ``+``, ``+=``, ``.extend`` and
``.append``; ``list(...)``, ``tuple(...)``, ``sorted(...)`` and
``dict.fromkeys(...)``; names assigned a resolvable value earlier in the
module; ``__all__`` itself; and ``<module>.__all__`` for a module imported
by name (followed recursively, relative imports included). Anything else,
a module without ``__all__``, or one with no Python source (compiled), is
reported as *unresolved* and not compared: the importing ``check`` covers
it on a build. Deterministic: sorted output, a baseline written as sorted
JSON with a trailing newline.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

#: Baseline of known gaps, beside the template it describes.
BASELINE_NAME = "coverage_baseline.json"


class UnresolvedError(Exception):
    """``__all__`` of a module cannot be read from its source alone."""


def module_file(package_root: Path, dotted: str) -> Path | None:
    """
    Return the source file of module *dotted*, or ``None``.

    Parameters
    ----------
    package_root : Path
        Directory that holds the top-level package (the repository root).
    dotted : str
        Module name, for example ``"scikitplot.corpus"``.

    Returns
    -------
    Path or None
        ``<dotted>/__init__.py`` or ``<dotted>.py``; ``None`` when neither
        exists (a compiled module, or no such module).
    """
    base = package_root.joinpath(*dotted.split("."))
    for candidate in (base / "__init__.py", base.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    return None


def _import_aliases(tree: ast.Module, package: str) -> dict[str, str]:
    """Return ``{local name: dotted module or object}`` of a module's imports."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                kept = parts[: len(parts) - (node.level - 1)]
                base = ".".join(kept + ([base] if base else []))
            for alias in node.names:
                if alias.name != "*":
                    target = f"{base}.{alias.name}" if base else alias.name
                    aliases[alias.asname or alias.name] = target
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    aliases[top] = top
    return aliases


def static_all(  # noqa: PLR0912
    package_root: Path, dotted: str, _seen: frozenset = frozenset()
) -> list[str]:
    """
    Return the ``__all__`` of module *dotted*, read from its source.

    Parameters
    ----------
    package_root : Path
        Directory that holds the top-level package.
    dotted : str
        Module name.

    Returns
    -------
    list of str
        The names, in order, each once.

    Raises
    ------
    UnresolvedError
        If the module has no Python source, no ``__all__``, or builds it in
        a way this reader does not follow; the message says which.
    """
    if dotted in _seen:
        raise UnresolvedError(
            f"{dotted}: __all__ refers to itself through {sorted(_seen)}"
        )
    seen = _seen | {dotted}
    path = module_file(package_root, dotted)
    if path is None:
        raise UnresolvedError(f"{dotted}: no Python source (compiled or absent)")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = dotted if path.name == "__init__.py" else dotted.rpartition(".")[0]
    aliases = _import_aliases(tree, package)
    variables: dict[str, list[str]] = {}
    state: dict[str, list[str] | None] = {"all": None}

    def value(expr: ast.AST) -> list[str]:  # noqa: PLR0911, PLR0912
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            out: list[str] = []
            for element in expr.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    out.append(element.value)
                elif isinstance(element, ast.Starred):
                    out.extend(value(element.value))
                else:
                    raise UnresolvedError(f"{dotted}: an __all__ item is not a string")
            return out
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return value(expr.left) + value(expr.right)
        if isinstance(expr, ast.Attribute) and expr.attr == "__all__":
            if isinstance(expr.value, ast.Name) and expr.value.id in aliases:
                return static_all(package_root, aliases[expr.value.id], seen)
            raise UnresolvedError(f"{dotted}: __all__ of an unknown name")
        if isinstance(expr, ast.Call) and len(expr.args) == 1 and not expr.keywords:
            func = expr.func
            if isinstance(func, ast.Name) and func.id in ("list", "tuple", "sorted"):
                inner = value(expr.args[0])
                return sorted(inner) if func.id == "sorted" else inner
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "fromkeys"
                and isinstance(func.value, ast.Name)
                and func.value.id == "dict"
            ):
                return value(expr.args[0])
        if isinstance(expr, ast.Name):
            if expr.id == "__all__" and state["all"] is not None:
                return list(state["all"])
            if expr.id in variables:
                return list(variables[expr.id])
        raise UnresolvedError(
            f"{dotted}: __all__ is built in a way not read statically"
        )

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if "__all__" in names:
                state["all"] = value(node.value)
            for name in names:
                if name != "__all__":
                    try:
                        variables[name] = value(node.value)
                    except UnresolvedError:
                        variables.pop(name, None)
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            state["all"] = (state["all"] or []) + value(node.value)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "__all__"
            ):
                if func.attr == "extend" and len(node.value.args) == 1:
                    state["all"] = (state["all"] or []) + value(node.value.args[0])
                elif func.attr == "append" and len(node.value.args) == 1:
                    item = node.value.args[0]
                    if not (
                        isinstance(item, ast.Constant) and isinstance(item.value, str)
                    ):
                        raise UnresolvedError(
                            f"{dotted}: __all__.append of a non-string"
                        )
                    state["all"] = (state["all"] or []) + [item.value]
                else:
                    raise UnresolvedError(f"{dotted}: __all__.{func.attr}(...)")
    if state["all"] is None:
        raise UnresolvedError(f"{dotted}: no __all__")
    return list(dict.fromkeys(state["all"]))


def static_lazy(package_root: Path, dotted: str) -> list[str]:
    """
    Return the names a module exports lazily, read from its source.

    Parameters
    ----------
    package_root : Path
        Directory that holds the top-level package.
    dotted : str
        Module name.

    Returns
    -------
    list of str
        The names the package itself provides (relative imports, or
        imports from its own top-level package) in a top-level
        ``if TYPE_CHECKING:`` block of a module that defines a module-level
        ``__getattr__``: the convention
        for a public name resolved on first access (an optional tier kept
        out of ``__all__`` so that ``import *`` does not import its
        dependency). Empty for any other module.
    """
    path = module_file(package_root, dotted)
    if path is None:
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    has_getattr = any(
        isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
        for node in tree.body
    )
    if not has_getattr:
        return []
    top = dotted.split(".", maxsplit=1)[0]
    names: list[str] = []
    for node in tree.body:
        test = node.test if isinstance(node, ast.If) else None
        guarded = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if not guarded:
            continue
        for inner in ast.walk(node):
            # Only the package's own names: a typing helper imported for
            # annotations (``from typing import Optional``) is not an export.
            if isinstance(inner, ast.ImportFrom) and (
                inner.level or (inner.module or "").split(".")[0] == top
            ):
                names.extend(a.asname or a.name for a in inner.names)
            elif isinstance(inner, ast.Import):
                names.extend(
                    a.asname or a.name.split(".")[-1]
                    for a in inner.names
                    if a.name.split(".")[0] == top
                )
    return list(dict.fromkeys(n for n in names if n != "*"))


def _defines(  # noqa: PLR0911, PLR0912
    package_root: Path, dotted: str, name: str, _seen: frozenset = frozenset()
) -> bool:
    """
    Whether module *dotted*'s source binds *name* at its top level.

    A definition, an assignment, an import, or ``from .x import *`` of a
    module whose ``__all__`` (or, without one, whose public top-level
    names) includes it; star imports are followed, each module once.
    """
    path = module_file(package_root, dotted)
    if path is None or dotted in _seen:
        return False
    seen = _seen | {dotted}
    try:
        if name in static_all(package_root, dotted):
            return True
    except UnresolvedError:
        pass
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*" and isinstance(node, ast.ImportFrom):
                    source = _absolute(dotted, path, node)
                    star_takes = not name.startswith("_") or _star_lists(
                        package_root, source, name
                    )
                    if star_takes and _defines(package_root, source, name, seen):
                        return True
                    continue
                bound = alias.asname or alias.name.split(".")[0]
                if bound == name:
                    return True
    return False


def _absolute(dotted: str, path: Path, node: ast.ImportFrom) -> str:
    """Return the module an ``ImportFrom`` in *dotted* (at *path*) names."""
    if not node.level:
        return node.module or ""
    package = dotted if path.name == "__init__.py" else dotted.rpartition(".")[0]
    parts = package.split(".")
    kept = parts[: len(parts) - (node.level - 1)]
    return ".".join(kept + ([node.module] if node.module else []))


def _star_lists(package_root: Path, dotted: str, name: str) -> bool:
    """Whether ``import *`` of *dotted* takes the private *name* (it is in ``__all__``)."""
    try:
        return name in static_all(package_root, dotted)
    except UnresolvedError:
        return False


def documented_target_exists(package_root: Path, module: str, entry: str) -> bool:
    """
    Whether a dotted reference entry ``"sub.Name"`` names something real.

    Parameters
    ----------
    package_root : Path
        Directory that holds the top-level package.
    module : str
        The reference module the entry is listed under.
    entry : str
        The entry, relative to *module*: ``"sub"``, ``"sub.inner"`` or
        ``"sub.Name"``.

    Returns
    -------
    bool
        ``True`` when ``<module>.<entry>`` is a module with source, or
        when the part before the last dot is one and its source binds the
        last part (as autosummary resolves the entry).
    """
    if module_file(package_root, f"{module}.{entry}") is not None:
        return True
    head, _, tail = entry.rpartition(".")
    return bool(head) and _defines(package_root, f"{module}.{head}", tail)


@dataclass
class CoverageReport:
    """Gaps between ``__all__`` and the reference, per module."""

    missing: dict[str, list[str]] = field(default_factory=dict)
    stale: dict[str, list[str]] = field(default_factory=dict)
    private: dict[str, list[str]] = field(default_factory=dict)
    unresolved: dict[str, str] = field(default_factory=dict)

    def gaps(self) -> dict[str, list[str]]:
        """Return the gaps as ``{"missing:<module>": names, "stale:<module>": names}``."""
        out: dict[str, list[str]] = {}
        for kind, table in (
            ("missing", self.missing),
            ("stale", self.stale),
            ("private", self.private),
        ):
            for module, names in table.items():
                if names:
                    out[f"{kind}:{module}"] = sorted(names)
        return dict(sorted(out.items()))


def deprecated_entries(source: str) -> list[str]:
    """
    Return the entries of ``DEPRECATED_APIS_REFERENCE`` in ``source``.

    Parameters
    ----------
    source : str
        Text of the API-reference file.

    Returns
    -------
    list of str
        Every entry, in file order; empty when the table is absent.

    Raises
    ------
    UnresolvedError
        If the table is not a literal mapping of version to list of strings.

    Notes
    -----
    Entries are relative to the top-level package, as the "Recently
    deprecated" autosummary resolves them (``"api.metrics.plot_roc_curve"``
    is ``scikitplot.api.metrics.plot_roc_curve``).
    """
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "DEPRECATED_APIS_REFERENCE"
            for t in targets
        ):
            continue
        try:
            table = ast.literal_eval(value)
        except ValueError as exc:
            raise UnresolvedError(
                "DEPRECATED_APIS_REFERENCE is not a literal mapping"
            ) from exc
        if not isinstance(table, dict) or not all(
            isinstance(names, list) and all(isinstance(n, str) for n in names)
            for names in table.values()
        ):
            raise UnresolvedError(
                "DEPRECATED_APIS_REFERENCE must map each version to a list of strings"
            )
        return [n for names in table.values() for n in names]
    return []


def coverage(
    package_root: Path,
    modules: Mapping[str, object],
    *,
    deprecated: Iterable[str] = (),
) -> CoverageReport:
    """
    Compare each reference module's statically read ``__all__`` with its sections.

    Parameters
    ----------
    package_root : Path
        Directory that holds the top-level package.
    modules : Mapping
        ``ReferenceModel.modules``: module name to an object with
        ``documented``, ``ignore`` and ``sections`` (each with ``exclude``).
    deprecated : iterable of str, default=()
        :func:`deprecated_entries`. An entry under a module documents its
        last part for that module, as a section entry would.

    Returns
    -------
    CoverageReport
    """
    report = CoverageReport()
    deprecated = list(deprecated)
    for name in sorted(modules):
        entry = modules[name]
        try:
            public = static_all(package_root, name)
        except UnresolvedError as exc:
            report.unresolved[name] = str(exc)
            continue
        lazy = static_lazy(package_root, name)
        public = public + [n for n in lazy if n not in public]
        # "sub.Name" documents Name through its submodule (as autosummary
        # resolves it); a name counts as documented either way.
        documented = set(entry.documented)
        documented |= {n.rsplit(".", 1)[-1] for n in entry.documented if "." in n}
        top = name.split(".", 1)[0]
        documented |= {
            n.rsplit(".", 1)[-1]
            for n in deprecated
            if f"{top}.{n}".startswith(f"{name}.")
        }
        skipped = set(entry.ignore)
        for section in entry.sections:
            skipped |= set(section.exclude)
        private = [n for n in public if n.startswith("_") and not n.startswith("__")]
        report.private[name] = private
        report.missing[name] = [
            n
            for n in public
            if n not in documented and n not in skipped and n not in private
        ]
        listed = set(public)
        # A documented name is stale when it is not public and, for a
        # dotted or submodule entry, does not resolve in the source either.
        report.stale[name] = sorted(
            n
            for n in entry.documented
            if n not in listed
            and n.rsplit(".", 1)[-1] not in listed
            and not documented_target_exists(package_root, name, n)
        )
    return report


def load_baseline(path: Path) -> dict[str, list[str]]:
    """Return the baseline gaps; an absent file is no gaps."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: sorted(values) for key, values in data.get("gaps", {}).items()}


def baseline_text(gaps: Mapping[str, Iterable[str]]) -> str:
    """Return the baseline file text for *gaps*: sorted JSON, one trailing newline."""
    payload = {
        "about": (
            "Known gaps between each module's __all__ and the API reference."
            " Generated by: python -m scikitplot._build_utils."
            "generate_apis_reference coverage --write. It only shrinks: fix a"
            " gap, then rewrite this file (Rule 172)."
        ),
        "gaps": {key: sorted(values) for key, values in sorted(gaps.items()) if values},
    }
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def compare(
    current: Mapping[str, Iterable[str]], baseline: Mapping[str, Iterable[str]]
) -> tuple[list[str], list[str]]:
    """
    Return ``(new, fixed)``: gaps not in the baseline, and baseline gaps now gone.

    Each item reads ``"<kind>:<module>: <name>"``, kind ``missing``,
    ``stale`` or ``private``.
    """
    now = {f"{key}: {n}" for key, names in current.items() for n in names}
    before = {f"{key}: {n}" for key, names in baseline.items() for n in names}
    return sorted(now - before), sorted(before - now)


def _run_as_script(argv: list[str] | None = None) -> int:
    """
    Run ``coverage`` from this file's path, without importing ``scikitplot``.

    ``python -m scikitplot._build_utils.generate_apis_reference`` imports
    the ``scikitplot`` package first, which needs a build; the pre-commit
    hook runs ``python -I -B <this file>`` instead, which loads only this
    folder as a package.
    """
    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415

    folder = Path(__file__).resolve().parent
    name = "_scikitplot_generate_apis_reference"
    spec = importlib.util.spec_from_file_location(
        name, folder / "__init__.py", submodule_search_locations=[str(folder)]
    )
    if spec is None or spec.loader is None:  # pragma: no cover - always found
        raise ImportError(f"cannot load {folder}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    args = sys.argv[1:] if argv is None else argv
    return package.main(["--repo-root", ".", "coverage", *args])


if __name__ == "__main__":
    raise SystemExit(_run_as_script())

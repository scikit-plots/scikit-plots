#!/usr/bin/env python3
"""Static contract checker for ``scikitplot.cleanprompt``.

Parses the runtime tree and the maintenance plane and reports whether each
still satisfies its stated contract. It never imports the runtime, so it works
on a tree whose dependencies are absent and cannot be fooled by an import-time
side effect.

Notes
-----
**Developer notes.** This is a *structural* checker. It proves that the shape of
the code still matches what the documents claim — that optional imports are
still deferred, that no sibling submodule crept in, that every source module
still has an owning test module. It cannot prove behaviour; that is what the
focused suite and the probes in ``evidence/`` are for, and
``maintenance_status`` is deliberately reported separately from
``runtime_status`` so the two are never conflated.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

SUBSYSTEM = "scikitplot.cleanprompt"

#: The one module allowed a deferred import of scikitplot.corpus.
CORPUS_BRIDGE = "_corpus.py"
RUNTIME = ("scikitplot", "cleanprompt")
MAINTENANCE = ("maintenances", "cleanprompt")
SKILL = ("skills", "cleanprompt", "SKILL.md")

#: Every tree committed for this subsystem. None may hold a whole
#: credential-shaped value (CP-REST-001).
AT_REST_TREES = (
    RUNTIME,
    MAINTENANCE,
    ("skills", "cleanprompt"),
    ("galleries", "examples", "cleanprompt"),
)

#: Distributions that must never be imported at module scope.
OPTIONAL_DISTRIBUTIONS = {
    "spacy",
    "nltk",
    "flask",
    "cryptography",
    "numpy",
    "pandas",
    "pydantic",
}

#: Names the public facade must export.
REQUIRED_SYMBOLS = {
    "Redactor",
    "restore",
    "resolve_spans",
    "Vault",
    "RedactionPolicy",
    "TagStyle",
    "Limits",
    "OverlapStrategy",
    "Span",
    "Entry",
    "RedactionResult",
    "RestorationResult",
    "Stats",
    "Detector",
    "DetectorRegistry",
    "RegexDetector",
    "LiteralDetector",
    "default_registry",
    "default_patterns",
    "get_pattern",
    "PATTERNS",
    "capabilities",
    "probe",
    "require",
    "CapabilityStatus",
    # engine selection and the vocabulary that makes engines interchangeable
    "CANONICAL_LABELS",
    "ENGINES",
    "canonical_label",
    "resolve_engine",
    "build_detectors",
    # language and model selection
    "LANGUAGES",
    "resolve_model",
    # the encode/decode surface an LLM caller uses
    "encode",
    "decode",
    "session",
    "Session",
    "Handle",
    # logging, which must be reachable without touching an optional tier
    "get_logger",
    "configure_logging",
    "redacting",
}

#: Names that must resolve lazily and must NOT appear in ``__all__``.
LAZY_SYMBOLS = {
    "NerDetector",
    "spacy_detector",
    "create_app",
    "SessionStore",
    "DEFAULT_ENTITY_LABELS",
    "DEFAULT_MODEL",
    "NltkDetector",
    "nltk_detector",
    "corpora_status",
}

#: Keys that would turn a metadata document into an execution surface.
FORBIDDEN_META_KEYS = {
    "command",
    "commands",
    "cmd",
    "shell",
    "exec",
    "executable",
    "script",
    "scripts",
    "argv",
}

REQUIRED_MAINTENANCE_FILES = (
    "MAINTAINING.md",
    "MAINTENANCE.json",
    "REVIEW.json",
    "_maintenance/DESIGN.md",
    "_maintenance/STATE.json",
    "_maintenance/EVIDENCE.json",
    "_maintenance/FRESH_CHAT_HANDOFF.md",
    "_maintenance/VERIFICATION.md",
    "_maintenance/SUBMODULE_STRUCTURE.md",
    "_maintenance/FAMILY.md",
    "_maintenance/tests/test_contract.py",
)


class ContractError(RuntimeError):
    """The checker could not run at all."""


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def discover_repo(start: Path) -> Path:
    """Locate the wide checkout root.

    Parameters
    ----------
    start : Path
        Where to begin looking.

    Returns
    -------
    Path
        The directory containing ``scikitplot``, ``maintenances`` and ``skills``.

    Raises
    ------
    ContractError
        If no such directory is found.
    """
    for origin in (start.resolve(), Path(__file__).resolve()):
        for candidate in (origin, *origin.parents):
            if all(
                (candidate / name).is_dir()
                for name in ("scikitplot", "maintenances", "skills")
            ):
                return candidate
    raise ContractError("could not locate the wide repository root")


#: Directories tools create inside a package. None is source, and hashing one
#: made the fingerprint depend on whether a linter had run (round 15).
CACHE_DIRECTORIES = frozenset(
    {"__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache", ".hypothesis"}
)


def runtime_files(root: Path) -> List[Path]:
    """Return every file in the runtime package, sorted and cache-free."""
    package = root.joinpath(*RUNTIME)
    if not package.is_dir():
        return []
    return sorted(
        path
        for path in package.rglob("*")
        if path.is_file() and not CACHE_DIRECTORIES.intersection(path.parts)
    )


def runtime_sources(root: Path) -> List[Path]:
    """Return the runtime ``.py`` files, excluding the test package."""
    package = root.joinpath(*RUNTIME)
    return [
        path
        for path in runtime_files(root)
        if path.suffix == ".py" and "tests" not in path.relative_to(package).parts
    ]


def tree_fingerprint(root: Path):
    """Return a stable digest of the runtime tree, or ``None`` if absent."""
    package = root.joinpath(*RUNTIME)
    if not package.is_dir():
        return None
    digest = hashlib.sha256()
    for path in runtime_files(root):
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def parse(path: Path) -> ast.Module:
    """Parse one source file."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def read_json(path: Path) -> Any:
    """Read one JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def absolute_imports(tree: ast.Module):
    """Yield ``(node, module_name)`` for every absolute import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node, node.module or ""


def symbols(tree: ast.Module) -> set:
    """Return every function and class name defined in a module."""
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def nodes_inside_functions(tree: ast.Module):
    """Return the ids of every node nested inside a function definition."""
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                inside.add(id(inner))
    return inside


def _literal_strings(node: ast.AST):
    """Return the string constants of a list or tuple literal, else ``None``."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return [
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
    return None


def string_list(tree: ast.Module, name: str):
    """Return the string elements of a module-level list or tuple assignment."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            literal = _literal_strings(node.value)
            if literal is not None:
                return literal
    return None


def resolve_all(package: Path, tree: ast.Module):
    """Resolve a module's ``__all__``, following ``+=`` aggregation.

    Parameters
    ----------
    package : Path
        Directory holding the sibling modules an aggregated ``__all__`` may
        extend from.
    tree : ast.Module
        The parsed facade.

    Returns
    -------
    list of str or None
        Every exported name, or ``None`` when ``__all__`` is absent or built in
        a way this resolver cannot follow.

    Notes
    -----
    **Developer notes.** The facade builds ``__all__`` by extending it with each
    private module's own ``__all__``, which keeps one source of truth per module
    and is worth supporting. But a checker that cannot read the contract it
    guards is worthless, and this exact contract has already regressed once: a
    reformatting pass appended the six optional-tier names here, and the static
    view reported every base name missing rather than the one real problem.

    So the resolver simulates the aggregation: a literal assignment seeds the
    list, ``__all__ += <literal>`` extends it, and ``__all__ += _mod.__all__``
    extends it with that sibling module's own literal ``__all__``. Anything it
    cannot follow returns ``None``, which callers must treat as "unknown", never
    as "empty" — reporting a contract as violated because the checker could not
    read it is the failure mode this replaces.
    """
    names: List[str] = []
    seen_assignment = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            literal = _literal_strings(node.value)
            if literal is None:
                return None
            names = list(literal)
            seen_assignment = True
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
            and isinstance(node.op, ast.Add)
        ):
            literal = _literal_strings(node.value)
            if literal is not None:
                names.extend(literal)
                continue
            # __all__ += _module.__all__
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "__all__"
                and isinstance(value.value, ast.Name)
            ):
                sibling = package / "{0}.py".format(value.value.id)
                if not sibling.is_file():
                    return None
                exported = string_list(parse(sibling), "__all__")
                if exported is None:
                    return None
                names.extend(exported)
                continue
            return None
    return names if seen_assignment else None



def _function_node(tree: ast.Module, name: str):
    """Return the top-level function ``name`` in a module, or ``None``."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None

def _references_name(tree: ast.Module, name: str) -> bool:
    """Return whether ``name`` is actually referenced in code.

    Parameters
    ----------
    tree : ast.Module
        The parsed module.
    name : str
        The identifier to look for.

    Returns
    -------
    bool
        ``True`` when the name appears as a bare name, an attribute, or an
        imported alias — never merely inside a string or a comment.

    Notes
    -----
    **Developer notes.** Written after this checker reported a false positive
    against the very module whose docstring explains why ``find_spec`` is
    unsuitable. A checker that cannot tell an explanation from a call trains
    its readers to ignore it.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == name for alias in node.names):
                return True
    return False


def _assigns_name(tree: ast.Module, name: str) -> bool:
    """Return whether ``name`` is assigned at module level.

    Notes
    -----
    **Developer notes.** Distinct from :func:`_references_name`: a rename
    leaves every use site referring to the old name, so a reference check
    reports a table that no longer exists as present.
    """
    for node in tree.body:
        targets: list = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            return True
    return False


def reject_command_surface(obj: Any, where: str = "metadata") -> None:
    """Refuse a metadata document that names an executable field."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in FORBIDDEN_META_KEYS:
                raise ContractError(
                    "{0} contains unsupported executable field {1!r}".format(where, key)
                )
            reject_command_surface(value, where)
    elif isinstance(obj, list):
        for value in obj:
            reject_command_surface(value, where)


# ---------------------------------------------------------------------------
# runtime contract
# ---------------------------------------------------------------------------


def runtime_findings(root: Path) -> List[str]:
    """Return every runtime contract violation."""
    package = root.joinpath(*RUNTIME)
    findings: List[str] = []
    if not package.is_dir():
        return ["scikitplot/cleanprompt package is missing"]

    facade = package / "__init__.py"
    if not facade.is_file():
        return ["scikitplot/cleanprompt/__init__.py is missing"]

    for name in (
        "_engine.py",
        "_policy.py",
        "_patterns.py",
        "_detectors.py",
        "_types.py",
        "_vault.py",
        "_capabilities.py",
        "_exceptions.py",
        "_cli.py",
        "_spec.py",
        "_frontends.py",
        "_diagnostics.py",
        "__main__.py",
    ):
        if not (package / name).is_file():
            findings.append("CP-STRUCT: missing runtime module {0}".format(name))

    sources = runtime_sources(root)

    # CP-TIER-001: optional dependencies must never be imported at module scope.
    for path in sources:
        tree = parse(path)
        deferred = nodes_inside_functions(tree)
        for node, module in absolute_imports(tree):
            head = module.split(".")[0]
            if head in OPTIONAL_DISTRIBUTIONS and id(node) not in deferred:
                findings.append(
                    "CP-TIER-001: {0} imports {1} at module scope".format(
                        path.name, module
                    )
                )

    # CP-INDEP-001: no sibling submodule, no developer plane.
    #
    # One exception, stated exactly rather than worked around: _corpus.py may
    # import scikitplot.corpus, and only inside a function, so that the two
    # submodules can cooperate without either loading the other at import
    # time. Any other module, any other sibling, or a module-scope import in
    # _corpus.py is still a finding.
    for path in sources:
        tree = parse(path)
        deferred = nodes_inside_functions(tree)
        for node, module in absolute_imports(tree):
            head = module.split(".")[0]
            if head == "scikitplot":
                permitted = (
                    path.name == CORPUS_BRIDGE
                    and (module == "scikitplot.corpus"
                         or module.startswith("scikitplot.corpus."))
                    and id(node) in deferred
                )
                if not permitted:
                    findings.append(
                        "CP-INDEP-001: {0} imports sibling {1}".format(
                            path.name, module
                        )
                    )
            if head in ("maintenances", "skills"):
                findings.append(
                    "CP-PLANE-001: {0} imports the developer plane {1}".format(
                        path.name, module
                    )
                )

    # CP-FACADE-001/002/003: the lazy facade contract.
    facade_tree = parse(facade)
    exported = resolve_all(package, facade_tree)
    if exported is None:
        findings.append(
            "CP-FACADE-004: the facade's __all__ cannot be resolved statically, "
            "so the star-import surface is unverifiable; keep it to literal "
            "lists and '__all__ += _module.__all__' aggregation"
        )
    else:
        missing = sorted(REQUIRED_SYMBOLS - set(exported))
        if missing:
            findings.append(
                "CP-FACADE-001: __all__ omits required symbols: "
                + ", ".join(missing)
            )
        leaked = sorted(LAZY_SYMBOLS & set(exported))
        if leaked:
            findings.append(
                "CP-FACADE-002: __all__ lists optional-tier names, which makes "
                "star import resolve them: " + ", ".join(leaked)
            )
    facade_functions = {
        node.name
        for node in facade_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for required in ("__getattr__", "__dir__"):
        if required not in facade_functions:
            findings.append(
                "CP-FACADE-003: the facade defines no {0}".format(required)
            )

    # CP-CAPS-001: the seven-state vocabulary must be complete.
    capabilities = package / "_capabilities.py"
    if capabilities.is_file():
        source = capabilities.read_text(encoding="utf-8")
        for member in (
            "AVAILABLE",
            "ABSENT",
            "BROKEN",
            "INCOMPATIBLE",
            "MISCONFIGURED",
            "UNREACHABLE",
            "UNKNOWN",
        ):
            if '{0} = "{0}"'.format(member) not in source:
                findings.append(
                    "CP-CAPS-001: CapabilityStatus is missing {0}".format(member)
                )
        # Parsed, not searched: the module's own docstring explains why
        # find_spec is unsuitable, and a substring search cannot tell an
        # explanation from a call. Only a real reference counts.
        if _references_name(parse(capabilities), "find_spec"):
            findings.append(
                "CP-CAPS-002: capability probing uses find_spec, which cannot "
                "distinguish a usable install from a shadowing stub and yields "
                "no version"
            )

    # CP-FRONT-001: the frontends must render from one neutral declaration.
    cli = package / "_cli.py"
    if cli.is_file():
        tree = parse(cli)
        # Checked as a module-level ASSIGNMENT, not as a reference. A rename
        # leaves every use site intact, so "is it mentioned" stays true while
        # the table itself is gone — which is exactly what the mutation test
        # caught this rule failing to notice.
        if not _assigns_name(tree, "COMMANDS"):
            findings.append(
                "CP-FRONT-001: _cli.py declares no COMMANDS table; both "
                "frontends must render from one neutral declaration"
            )
        # Dispatch is a two-hop property: _cli.py hands the parse to
        # `load_runner`, and `load_runner` returns one of the two runners.
        # The first version checked that _cli.py *mentioned* run_argparse and
        # run_click, which was satisfied only by an import _cli.py never used.
        # A linter removed that dead import, the rule failed, and nothing about
        # dispatch had changed (CP-047). The rule now follows the real path.
        if not _references_name(tree, "load_runner"):
            findings.append(
                "CP-FRONT-001: _cli.py does not dispatch through load_runner; "
                "a frontend that owns behaviour can diverge"
            )
        frontends = package / "_frontends.py"
        if frontends.is_file():
            runner = _function_node(parse(frontends), "load_runner")
            for name in ("run_argparse", "run_click"):
                if runner is None or not _references_name(runner, name):
                    findings.append(
                        "CP-FRONT-001: load_runner never returns {0}; a "
                        "frontend that owns behaviour can diverge".format(name)
                    )

    # CP-FRONT-002: click is optional and must never load at import time.
    for path in sources:
        tree = parse(path)
        deferred = nodes_inside_functions(tree)
        for node, module in absolute_imports(tree):
            if module.split(".")[0] == "click" and id(node) not in deferred:
                findings.append(
                    "CP-FRONT-002: {0} imports click at module scope; the "
                    "base tier must work on the standard library alone".format(
                        path.name
                    )
                )

    # CP-SAFE-001: no dynamic evaluation anywhere in the runtime.
    for path in sources:
        tree = parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "compile"):
                    findings.append(
                        "CP-SAFE-001: {0} calls {1}()".format(path.name, node.func.id)
                    )

    # CP-SAFE-002: no bare except, no silent pass.
    for path in sources:
        for node in ast.walk(parse(path)):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    findings.append("CP-SAFE-002: {0} has a bare except".format(path.name))
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    findings.append(
                        "CP-SAFE-002: {0} swallows an exception with pass".format(
                            path.name
                        )
                    )

    # CP-TEST-001: module-mirrored test ownership.
    tests_dir = package / "tests"
    if not tests_dir.is_dir():
        findings.append("CP-TEST-001: the focused test package is missing")
    else:
        test_names = {path.name for path in tests_dir.glob("test_*.py")}
        for path in sources:
            if path.name == "__main__.py":
                continue
            expected = "test_{0}".format(path.name)
            if expected not in test_names:
                findings.append(
                    "CP-TEST-001: {0} has no owning test module {1}".format(
                        path.name, expected
                    )
                )
        if "test_regressions.py" not in test_names:
            findings.append("CP-TEST-002: the regression module is missing")

    # CP-DOC-001: every runtime module documents itself.
    for path in sources:
        if not ast.get_docstring(parse(path)):
            findings.append("CP-DOC-001: {0} has no module docstring".format(path.name))

    # CP-DOC-001b: the diagnosis surface must exist and be wired in.
    diagnostics = package / "_diagnostics.py"
    if diagnostics.is_file():
        tree = parse(diagnostics)
        for required in ("diagnose", "suggest_terms", "describe_outcome"):
            if required not in symbols(tree):
                findings.append(
                    "CP-DIAG-001: _diagnostics.py does not define {0}()".format(
                        required
                    )
                )
        app = package / "_app.py"
        if app.is_file() and not _references_name(parse(app), "diagnose"):
            findings.append(
                "CP-DIAG-002: the web tier does not consult the diagnosis, so "
                "a visitor cannot see which detectors are switched off"
            )

    # CP-WEB-001: the web tier must not build an app at import time.
    app = package / "_app.py"
    if app.is_file():
        source = app.read_text(encoding="utf-8")
        if "\napp = Flask(" in source:
            findings.append("CP-WEB-001: _app.py builds a Flask app at module scope")
        if "def create_app(" not in source:
            findings.append("CP-WEB-001: _app.py has no create_app factory")
        if "compare_digest" not in source:
            findings.append(
                "CP-WEB-002: the web tier does not compare its CSRF token in "
                "constant time"
            )

    return list(dict.fromkeys(findings))


# ---------------------------------------------------------------------------
# maintenance contract
# ---------------------------------------------------------------------------


def at_rest_errors(root: Path) -> List[str]:
    """
    Return every whole credential-shaped value stored in a cleanprompt tree.

    Notes
    -----
    **Developer notes.** Runtime invariant ``I14`` covers the package; this is
    the same rule for everything else that is committed with it - this plane,
    the skill and the gallery - because a secret scanner reads a push, not a
    package. The patterns are read from the compiled catalogue with the
    standard library, so the checker still never imports the runtime. A
    pattern's validator cannot be applied from here, so a pattern that names
    one is matched without it: that can only report more, never less.

    A finding names the file, the line and the kind, and never the value.
    """
    package = root.joinpath(*RUNTIME)
    compiled = package / "_config" / "_compiled.json"
    catalog = package / "_catalog.py"
    if not compiled.is_file() or not catalog.is_file():
        return []
    names = string_list(parse(catalog), "AT_REST_PACKS")
    if not names:
        return ["CP-REST-002: _catalog.py declares no AT_REST_PACKS, so nothing "
                "holds committed files to the credential patterns"]
    try:
        packs = read_json(compiled).get("packs", {})
    except ValueError as exc:
        return ["CP-REST-002: _compiled.json is not valid JSON: {0}".format(exc)]
    patterns = []
    for name in names:
        if name not in packs:
            return ["CP-REST-002: credential pack {0!r} is not in the compiled "
                    "catalogue".format(name)]
        for item in packs[name].get("patterns", []):
            flags = 0
            for flag in item.get("flags") or []:
                flags |= getattr(re, flag)
            patterns.append((item["kind"], re.compile(item["pattern"], flags)))
    if not patterns:
        return ["CP-REST-002: the credential packs define no patterns"]
    found = set()
    for tree in AT_REST_TREES:
        base = root.joinpath(*tree)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or CACHE_DIRECTORIES.intersection(path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for kind, compiled_pattern in patterns:
                for match in compiled_pattern.finditer(text):
                    if match.group():
                        found.add((
                            path.relative_to(root).as_posix(),
                            text.count("\n", 0, match.start()) + 1,
                            kind,
                        ))
    return [
        "CP-REST-001: {0}:{1} holds a whole {2}-shaped value; build it from "
        "fragments".format(*item)
        for item in sorted(found)
    ]


def maintenance_errors(root: Path) -> List[str]:
    """Return every maintenance-plane violation."""
    errors: List[str] = []
    base = root.joinpath(*MAINTENANCE)
    errors.extend(at_rest_errors(root))

    for relative in REQUIRED_MAINTENANCE_FILES:
        if not (base / relative).is_file():
            errors.append("missing maintenance file: {0}".format(relative))

    skill = root.joinpath(*SKILL)
    if not skill.is_file():
        errors.append("missing skills/cleanprompt/SKILL.md")
    else:
        text = skill.read_text(encoding="utf-8")
        if len(text.splitlines()) < 80:
            errors.append("the cleanprompt skill is too shallow")
        for marker in ("CP-006", "CP-015", "lazy", "verification", "_engine.py"):
            if marker.lower() not in text.lower():
                errors.append("skill is missing marker {0}".format(marker))

    for name in ("MAINTENANCE.json", "REVIEW.json"):
        path = base / name
        if path.is_file():
            try:
                reject_command_surface(read_json(path), name)
            except ContractError as exc:
                errors.append(str(exc))
            except ValueError as exc:
                errors.append("{0} is not valid JSON: {1}".format(name, exc))

    review = base / "REVIEW.json"
    if review.is_file():
        try:
            document = read_json(review)
        except ValueError as exc:
            errors.append("REVIEW.json is not valid JSON: {0}".format(exc))
        else:
            for finding in document.get("findings", []):
                if finding.get("status") == "closed" and not finding.get("regression"):
                    errors.append(
                        "REVIEW.json: finding {0} is closed with no named "
                        "regression".format(finding.get("id"))
                    )

    evidence = base / "_maintenance" / "EVIDENCE.json"
    if evidence.is_file():
        try:
            document = read_json(evidence)
        except ValueError as exc:
            errors.append("EVIDENCE.json is not valid JSON: {0}".format(exc))
        else:
            recorded = document.get("runtime_tree_fingerprint")
            actual = tree_fingerprint(root)
            if recorded is not None and recorded != actual:
                errors.append(
                    "EVIDENCE runtime_tree_fingerprint does not match the "
                    "scikitplot/cleanprompt tree"
                )
            for lane in document.get("lanes", []):
                if lane.get("status") == "UNAVAILABLE" and not lane.get("reason"):
                    errors.append(
                        "EVIDENCE lane {0} is UNAVAILABLE with no reason".format(
                            lane.get("id")
                        )
                    )

    return errors


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def payload(root: Path) -> Dict[str, Any]:
    """Return the full report."""
    findings = runtime_findings(root)
    errors = maintenance_errors(root)
    package = root.joinpath(*RUNTIME)
    return {
        "subsystem": SUBSYSTEM,
        "maintenance_status": "PASS" if not errors else "FAIL",
        "runtime_status": "PASS" if not findings else "FAIL",
        "release_status": "BLOCKED" if (errors or findings) else "UNVERIFIED",
        "maintenance_errors": errors,
        "runtime_findings": findings,
        "inventory": {
            "runtime_files": len(runtime_files(root)),
            "runtime_modules": len(runtime_sources(root)),
            "focused_test_modules": len(list((package / "tests").glob("test_*.py")))
            if (package / "tests").is_dir()
            else 0,
        },
        "runtime_tree_fingerprint": tree_fingerprint(root),
    }


def main(argv=None) -> int:
    """Run the checker."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, help="wide checkout root")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--update",
        action="store_true",
        help="refresh the evidence fingerprint; refused while runtime is FAIL",
    )
    args = parser.parse_args(argv)
    root = args.repo.resolve() if args.repo else discover_repo(Path.cwd())
    report = payload(root)

    if args.update:
        if report["runtime_status"] != "PASS":
            raise SystemExit("refusing --update while the runtime contract is FAIL")
        path = root.joinpath(*MAINTENANCE) / "_maintenance" / "EVIDENCE.json"
        document = read_json(path)
        document["runtime_tree_fingerprint"] = report["runtime_tree_fingerprint"]
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(
            "maintenance={0} runtime={1} release={2}\n".format(
                report["maintenance_status"],
                report["runtime_status"],
                report["release_status"],
            )
        )
        for item in report["maintenance_errors"] + report["runtime_findings"]:
            sys.stdout.write("  - {0}\n".format(item))
    return 0 if report["maintenance_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

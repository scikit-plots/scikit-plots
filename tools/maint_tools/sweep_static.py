#!/usr/bin/env python3
"""
Sweep first-party scikit-plots code for known risk patterns, with file:line evidence.

Why this exists
---------------
A review of one submodule finds what its reviewer looks for; a sweep finds
the same class of problem in every submodule at once, and repeats exactly.
This tool walks ``scikitplot/`` (vendored code and tests left out), parses
each file once, and reports every match of a fixed set of checks
(``CATEGORIES``): code a Python floor cannot run, files opened without an
encoding, unsafe deserialisation and shell use, network calls without a
time limit, archive extraction without a filter, silent exception handling,
``print`` and ``assert`` in library code, mutable default arguments.

Usage
-----
::

    python -B tools/maint_tools/sweep_static.py                  # summary
    python -B tools/maint_tools/sweep_static.py --list pickle-load
    python -B tools/maint_tools/sweep_static.py --json /tmp/sweep.json
    python -B tools/maint_tools/sweep_static.py --floor 3.9 --fail-on floor-isinstance-union

``--root DIR`` sweeps another checkout (default: this one). ``--floor X.Y``
is the Python version the code must run on (default 3.8, the package's
``requires-python``). ``--list CATEGORY`` prints every finding of one
category; ``--fail-on CATEGORY`` (repeatable) exits 1 when that category
has a finding that no ``# noqa`` acknowledges, so a fixed class of problem
can be kept fixed. ``--exclude GLOB`` (repeatable) leaves files out.

Notes
-----
**Acknowledged findings.** A finding whose line carries ``# noqa`` with the
matching ruff code (``S301`` for pickle, ``T201`` for print, ...; bare
``# noqa`` for any) is counted as acknowledged, not new: someone decided
about it. The summary shows both counts.

**What a check cannot know.** Each check matches a syntactic pattern. A
``pickle.load`` of a file the package wrote itself, or a ``print`` in a
command-line entry point, may be right; the sweep lists candidates for a
reviewer, it does not judge them. ``floor-syntax`` uses ``ast.parse(...,
feature_version=...)``, which Python documents as best effort: it refuses
``match``, ``except*`` and type parameters but accepts parenthesised
context managers (3.10), so run the code on the floor too (Rule 137).

**Developer notes.** Standard library only; the package is never imported
(lesson CI-12). Python 3.8 and newer (Rule 137). Output is sorted, so two
sweeps of the same tree are identical.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

#: Version of the JSON this tool writes.
FORMAT = 1

#: Vendored third-party code: findings there belong upstream.
VENDORED = ("scikitplot/externals/", "scikitplot/cexternals/")

#: Category -> (lowest Python the construct needs or None, ruff codes that
#: acknowledge it, what it is).
CATEGORIES: Dict[str, Tuple[Any, Tuple[str, ...], str]] = {
    "floor-syntax": (None, (), "a file the floor's parser refuses"),
    "floor-annotation": (None, (), "an annotation evaluated at definition time using list[...] (3.9) or X | Y (3.10), without `from __future__ import annotations`"),
    "floor-isinstance-union": ((3, 10), (), "isinstance/issubclass with an X | Y type (3.10)"),
    "floor-zip-strict": ((3, 10), (), "zip(..., strict=...) (3.10)"),
    "floor-str-removeprefix": ((3, 9), (), "str.removeprefix/removesuffix (3.9)"),
    "open-without-encoding": (None, ("PLW1514",), "open() in text mode without encoding= (the locale's encoding; cp1252 on Windows)"),
    "path-text-without-encoding": (None, ("PLW1514",), "Path.read_text/write_text without encoding="),
    "pickle-load": (None, ("S301",), "pickle/cloudpickle/dill load: runs code from the file"),
    "joblib-load": (None, ("S301",), "joblib.load: unpickles, runs code from the file"),
    "numpy-allow-pickle": (None, ("S301",), "numpy.load(allow_pickle=...) not False"),
    "torch-load-without-weights-only": (None, ("S614",), "torch.load without weights_only="),
    "yaml-unsafe-load": (None, ("S506",), "yaml.load without Loader="),
    "eval-exec": (None, ("S102", "S307"), "eval/exec"),
    "subprocess-shell": (None, ("S602", "S604"), "a subprocess call with shell= not False"),
    "os-system": (None, ("S605", "S607"), "os.system/os.popen"),
    "tempfile-mktemp": (None, ("S306",), "tempfile.mktemp (a race)"),
    "extractall-without-filter": (None, ("S202",), "extractall without filter= (path traversal on old Pythons)"),
    "requests-without-timeout": (None, ("S113",), "requests call without timeout= (can wait forever)"),
    "urlopen-without-timeout": (None, (), "urlopen without timeout (can wait forever)"),
    "weak-hash": (None, ("S324",), "md5/sha1 without usedforsecurity=False"),
    "stdlib-xml-parse": (None, ("S314", "S318", "S320"), "xml.etree/minidom parsing (untrusted XML needs a hardened parser)"),
    "silent-broad-except": (None, ("S110", "BLE001", "SIM105"), "except Exception/bare except whose body is only pass/continue"),
    "bare-except": (None, ("E722",), "except: with no type (also catches KeyboardInterrupt)"),
    "print-in-library": (None, ("T201",), "print() inside a function: bypasses logging"),
    "assert-in-library": (None, ("S101",), "assert outside tests: removed by python -O"),
    "mutable-default": (None, ("B006",), "a list/dict/set default argument (shared between calls)"),
    "regex-unbounded-scan": (None, (), "a regex whose scan from an unclosed opener runs to the end (<[^>]*>, (.*?)</): quadratic on text full of openers"),
}

_NOQA = re.compile(r"#\s*noqa(?::\s*(?P<codes>[A-Za-z0-9, ]+))?", re.IGNORECASE)
_HTTP = ("get", "post", "put", "delete", "head", "patch", "request")
_GENERIC = ("list", "dict", "tuple", "set", "frozenset", "type")


def area_of(rel: str) -> str:
    """
    Return the owning area of a repository-relative path.

    Parameters
    ----------
    rel : str
        Path such as ``scikitplot/corpus/_base.py``.

    Returns
    -------
    str
        The first folder under ``scikitplot/`` (``corpus``); a Sphinx
        extension as ``_sphinx_ext/<name>``; a top-level module as
        ``(package)``.
    """
    parts = rel.split("/")
    if len(parts) > 3 and parts[1] == "_externals" and parts[2] == "_sphinx_ext":
        return "_sphinx_ext" if parts[3].endswith(".py") else f"_sphinx_ext/{parts[3]}"
    return "(package)" if parts[1].endswith(".py") else parts[1]


#: Calls that return a module named by their first argument: their result
#: stands for that module, so ``safe_import("pickle").load`` is ``pickle.load``.
_MODULE_RETURNING = frozenset({"safe_import", "import_module", "importlib.import_module", "__import__"})


def _dotted(node: ast.AST) -> str:
    """Return ``a.b.c`` for a call's function expression, ``""`` otherwise."""
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
        return ".".join(reversed(names))
    if (
        isinstance(node, ast.Call)
        and _dotted(node.func) in _MODULE_RETURNING
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        names.append(node.args[0].value)
        return ".".join(reversed(names))
    return ".".join(reversed(names)) if names else ""


def _keyword(node: ast.Call, name: str) -> ast.AST | None:
    for item in node.keywords:
        if item.arg == name:
            return item.value
    return None


def _encoding_given(node: ast.Call, position: int) -> bool:
    """
    Return whether a call names its encoding or may receive it.

    ``encoding=`` given, the encoding passed by position (``position`` is its
    index), or a ``**mapping`` that can supply it: a splat such as
    ``**UTF8`` (``{"encoding": "utf-8"}``) cannot be decided statically, so
    the check reports only calls whose encoding is provably missing.
    """
    if _keyword(node, "encoding") is not None or len(node.args) > position:
        return True
    return any(item.arg is None for item in node.keywords)


def _is_false(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _annotation_needs(annotation: ast.AST) -> Tuple[int, int] | None:
    """Return the Python an evaluated annotation needs: (3, 10) for ``X | Y``, (3, 9) for ``list[...]``."""
    needed = None
    for node in ast.walk(annotation):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return (3, 10)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in _GENERIC:
            needed = (3, 9)
    return needed


#: ``re`` functions that search every position of the text.
_RE_SEARCHING = frozenset({"compile", "sub", "subn", "findall", "finditer", "search", "split"})
#: An opener, then a repeated class that excludes its closer but not the
#: opener itself: a scan from each unclosed opener runs to the text's end.
_UNBOUNDED_SCAN = re.compile(r"(<|\\{)[^\[]{0,40}?\[\^(?P<cls>[^\]]*)\][*+]")
_GROUP_SYNTAX = re.compile(r"\(\?(?:P<\w+>|<[=!]|P=\w+\))")
_LAZY_TO_CLOSING_TAG = re.compile(r"\.\*\??\)?</")


def _unbounded_scan(pattern: str) -> bool:
    """
    Return whether *pattern* scans from each unclosed opener to the end.

    ``<[^>]*>`` (or ``{[^}]*}``) tried at every ``<`` of a text full of
    them scans to the end each time: quadratic. ``<[^<>]*>`` stops at the
    next opener and is linear. ``(.*?)</tag>`` under ``re.DOTALL`` does the
    same for every opening tag. A pattern anchored at the start (``^``,
    ``\\A``) is tried at one position and is not flagged. A heuristic for a
    sweep, not a proof.

    Examples
    --------
    >>> _unbounded_scan(r"<[^>]+>"), _unbounded_scan(r"<[^<>]+>")
    (True, False)
    >>> _unbounded_scan(r"<object\\b[^<>]*>(.*?)</object>"), _unbounded_scan(r"^<a[^>]*>$")
    (True, False)
    >>> _unbounded_scan(r"(?P<pre>://[^/]+)"), _unbounded_scan(r"(?<=x)[^y]+")
    (False, False)
    """
    if pattern.startswith(("^", "\\A")):
        return False
    # "<" in a group name or a lookbehind is syntax, not an opener.
    pattern = _GROUP_SYNTAX.sub("(", pattern)
    for match in _UNBOUNDED_SCAN.finditer(pattern):
        opener = "<" if match.group(1) == "<" else "{"
        if opener not in match.group("cls"):
            return True
    return bool(_LAZY_TO_CLOSING_TAG.search(pattern))


class _Checker(ast.NodeVisitor):
    """Collect findings of one parsed file."""

    def __init__(self, emit: Any, future_annotations: bool, xml_aliases: frozenset) -> None:
        self.emit = emit
        self.future = future_annotations
        self.xml_aliases = xml_aliases  # names bound to xml.etree / xml.dom.minidom
        self.depth = 0  # function nesting

    def visit_Call(self, node: ast.Call) -> None:  # noqa: C901, PLR0912
        name = _dotted(node.func)
        last = name.rsplit(".", 1)[-1]
        # Decided by the call's shape, not its dotted name: ``Path(p).open``
        # has the dotted name ``open``, like the builtin.
        builtin = (isinstance(node.func, ast.Name) and node.func.id == "open") or name in ("io.open", "codecs.open")
        method = (
            not builtin
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "open"
            and (
                name.split(".")[0] in ("Path", "path", "pathlib")
                or (isinstance(node.func.value, ast.Call) and _dotted(node.func.value.func) in ("Path", "pathlib.Path"))
            )
        )
        if builtin or method:
            # open(file, mode, buffering, encoding) and io/codecs.open take the
            # file first; Path.open(mode, buffering, encoding) does not.
            first = 1 if builtin else 0
            mode = node.args[first] if len(node.args) > first else _keyword(node, "mode")
            text = mode.value if isinstance(mode, ast.Constant) else ("r" if mode is None else None)
            if isinstance(text, str) and "b" not in text and not _encoding_given(node, first + 2):
                self.emit("open-without-encoding", node)
        if last in ("read_text", "write_text") and isinstance(node.func, ast.Attribute):
            # read_text(encoding, errors); write_text(data, encoding, errors)
            if not _encoding_given(node, 0 if last == "read_text" else 1):
                self.emit("path-text-without-encoding", node)
        if name in ("pickle.load", "pickle.loads", "cloudpickle.load", "cloudpickle.loads", "dill.load", "dill.loads"):
            self.emit("pickle-load", node)
        if name == "joblib.load":
            self.emit("joblib-load", node)
        if name in ("np.load", "numpy.load") and _keyword(node, "allow_pickle") is not None and not _is_false(_keyword(node, "allow_pickle")):
            self.emit("numpy-allow-pickle", node)
        if name == "torch.load" and _keyword(node, "weights_only") is None:
            self.emit("torch-load-without-weights-only", node)
        if name in ("yaml.load", "yaml.load_all") and _keyword(node, "Loader") is None and len(node.args) < 2:
            self.emit("yaml-unsafe-load", node)
        if name in ("eval", "exec", "builtins.eval", "builtins.exec"):
            self.emit("eval-exec", node)
        if (name.startswith("subprocess.") or last in ("Popen", "check_output", "check_call")) and _keyword(node, "shell") is not None and not _is_false(_keyword(node, "shell")):
            self.emit("subprocess-shell", node)
        if name in ("os.system", "os.popen"):
            self.emit("os-system", node)
        if name in ("tempfile.mktemp", "mktemp"):
            self.emit("tempfile-mktemp", node)
        if last == "extractall" and isinstance(node.func, ast.Attribute) and _keyword(node, "filter") is None:
            self.emit("extractall-without-filter", node)
        if name.startswith("requests.") and last in _HTTP and _keyword(node, "timeout") is None:
            self.emit("requests-without-timeout", node)
        if last == "urlopen" and _keyword(node, "timeout") is None and len(node.args) < 3:
            self.emit("urlopen-without-timeout", node)
        if name in ("hashlib.md5", "hashlib.sha1") and _keyword(node, "usedforsecurity") is None:
            self.emit("weak-hash", node)
        if last in ("parse", "fromstring", "parseString", "iterparse", "XML") and (
            name.startswith(("xml.etree.", "xml.dom.minidom.")) or name.split(".")[0] in self.xml_aliases
        ):
            self.emit("stdlib-xml-parse", node)
        if (
            name.split(".")[0] in ("re", "regex")
            and last in _RE_SEARCHING
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and _unbounded_scan(node.args[0].value)
        ):
            self.emit("regex-unbounded-scan", node)
        if name == "print" and self.depth:
            self.emit("print-in-library", node)
        if last in ("removeprefix", "removesuffix") and isinstance(node.func, ast.Attribute):
            self.emit("floor-str-removeprefix", node)
        if name == "zip" and _keyword(node, "strict") is not None:
            self.emit("floor-zip-strict", node)
        if name in ("isinstance", "issubclass") and len(node.args) == 2 and isinstance(node.args[1], ast.BinOp) and isinstance(node.args[1].op, ast.BitOr):
            self.emit("floor-isinstance-union", node)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.emit("assert-in-library", node)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        broad = node.type is None or (isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"))
        if broad and len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Continue)):
            self.emit("silent-broad-except", node)
        if node.type is None:
            self.emit("bare-except", node)
        self.generic_visit(node)

    def _function(self, node: Any) -> None:
        defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
        for default in defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)) or (
                isinstance(default, ast.Call) and _dotted(default.func) in ("list", "dict", "set") and not default.args
            ):
                self.emit("mutable-default", node, node.name)
                break
        if not self.future:
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            annotations = [a.annotation for a in arguments if a.annotation is not None]
            annotations += [a.annotation for a in (node.args.vararg, node.args.kwarg) if a is not None and a.annotation is not None]
            if node.returns is not None:
                annotations.append(node.returns)
            needs = [n for n in map(_annotation_needs, annotations) if n is not None]
            if needs:
                self.emit("floor-annotation", node, node.name, max(needs))
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function


def _xml_aliases(tree: ast.AST) -> frozenset:
    """Return the names a module binds to ``xml.etree.*`` or ``xml.dom.minidom``."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("xml.etree", "xml.dom.minidom")) and alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("xml.etree", "xml.dom")):
            names.update(alias.asname or alias.name for alias in node.names)
    return frozenset(names)


def _excluded(rel: str, exclude: Sequence[str]) -> bool:
    parts = rel.split("/")
    if rel.startswith(VENDORED) or "tests" in parts[:-1]:
        return True
    if parts[-1].startswith("test_") or parts[-1] == "conftest.py":
        return True
    return any(fnmatch.fnmatchcase(rel, pattern) for pattern in exclude)


def _acknowledged(line: str, category: str) -> bool:
    match = _NOQA.search(line)
    if not match:
        return False
    codes = match.group("codes")
    if not codes:
        return True
    given = {c.strip().upper() for c in codes.split(",") if c.strip()}
    return bool(given & set(CATEGORIES[category][1]))


def sweep(root: Path, *, floor: Tuple[int, int] = (3, 8), exclude: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """
    Return every finding under ``root/scikitplot``, sorted by file and line.

    Parameters
    ----------
    root : pathlib.Path
        Repository checkout (holds ``scikitplot/``).
    floor : tuple of (int, int), optional
        Lowest Python the code must run on; floor checks for constructs
        that need a newer one. Default: ``(3, 8)``.
    exclude : sequence of str, optional
        ``fnmatch`` patterns of repository-relative paths to leave out.

    Returns
    -------
    list of dict
        ``category``, ``area``, ``file``, ``line``, ``code`` (the source
        line, stripped, at most 160 characters), ``detail`` and
        ``acknowledged``.

    Raises
    ------
    ValueError
        If ``root/scikitplot`` is not a folder.
    """
    package = root / "scikitplot"
    if not package.is_dir():
        raise ValueError(f"sweep_static: {package} is not a folder.")
    findings: List[Dict[str, Any]] = []
    for path in sorted(package.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        if _excluded(rel, exclude):
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()

        def emit(
            category: str,
            node: Any,
            detail: str = "",
            needed: Tuple[int, int] | None = None,
            rel: str = rel,
            lines: List[str] = lines,
        ) -> None:
            needed = needed or CATEGORIES[category][0]
            if needed is not None and tuple(floor) >= needed:
                return
            number = getattr(node, "lineno", 0)
            text = lines[number - 1] if 0 < number <= len(lines) else ""
            findings.append(
                {
                    "category": category,
                    "area": area_of(rel),
                    "file": rel,
                    "line": number,
                    "code": text.strip()[:160],
                    "detail": detail,
                    "acknowledged": _acknowledged(text, category),
                }
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                tree = ast.parse(source)
            except SyntaxError as exc:
                findings.append(
                    {"category": "floor-syntax", "area": area_of(rel), "file": rel, "line": exc.lineno or 0,
                     "code": (exc.text or "").strip()[:160], "detail": f"does not parse here: {exc.msg}", "acknowledged": False}
                )
                continue
            try:
                ast.parse(source, feature_version=tuple(floor))
            except SyntaxError as exc:
                findings.append(
                    {"category": "floor-syntax", "area": area_of(rel), "file": rel, "line": exc.lineno or 0,
                     "code": (exc.text or "").strip()[:160], "detail": exc.msg, "acknowledged": False}
                )
        future = any(
            isinstance(n, ast.ImportFrom) and n.module == "__future__" and any(a.name == "annotations" for a in n.names)
            for n in tree.body
        )
        _Checker(emit, future, _xml_aliases(tree)).visit(tree)
    findings.sort(key=lambda f: (f["file"], f["line"], f["category"]))
    return findings


def summary(findings: Sequence[Dict[str, Any]]) -> List[str]:
    """
    Return one line per category: new and acknowledged counts, top areas.

    Parameters
    ----------
    findings : sequence of dict
        From :func:`sweep`.

    Returns
    -------
    list of str
        Sorted by category name.
    """
    new: Dict[str, Counter] = defaultdict(Counter)
    known: Counter = Counter()
    for f in findings:
        if f["acknowledged"]:
            known[f["category"]] += 1
        else:
            new[f["category"]][f["area"]] += 1
    lines = []
    for category in sorted(set(new) | set(known)):
        areas = new[category]
        top = ", ".join(f"{a} {n}" for a, n in sorted(areas.items(), key=lambda kv: (-kv[1], kv[0]))[:6])
        lines.append(f"{category:32} new {sum(areas.values()):4}  acknowledged {known[category]:4}  {top}")
    return lines


def _floor(text: str) -> Tuple[int, int]:
    match = re.fullmatch(r"3\.(\d+)", text)
    if not match:
        raise argparse.ArgumentTypeError(f"floor must look like 3.8; got {text!r}")
    return 3, int(match.group(1))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--floor", type=_floor, default=(3, 8), metavar="X.Y")
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    parser.add_argument("--json", type=Path, default=None, metavar="OUT")
    parser.add_argument("--list", choices=sorted(CATEGORIES), default=None, metavar="CATEGORY")
    parser.add_argument("--fail-on", action="append", default=[], choices=sorted(CATEGORIES), metavar="CATEGORY")
    parser.add_argument("--categories", action="store_true", help="describe every category and exit")
    args = parser.parse_args(argv)
    if args.categories:
        for name in sorted(CATEGORIES):
            print(f"{name:32} {CATEGORIES[name][2]}")  # noqa: T201
        return 0
    try:
        findings = sweep(args.root, floor=args.floor, exclude=args.exclude)
    except ValueError as exc:
        parser.error(str(exc))
    if args.list:
        for f in findings:
            if f["category"] == args.list:
                mark = " (acknowledged)" if f["acknowledged"] else ""
                print(f"{f['file']}:{f['line']}: {f['code']}{mark}")  # noqa: T201
    else:
        for line in summary(findings):
            print(line)  # noqa: T201
    if args.json is not None:
        data = {"format": FORMAT, "floor": ".".join(map(str, args.floor)), "findings": findings}
        args.json.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    failing = [f for f in findings if f["category"] in args.fail_on and not f["acknowledged"]]
    if failing:
        print(f"{len(failing)} unacknowledged finding(s) in {', '.join(sorted(set(args.fail_on)))}.", file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

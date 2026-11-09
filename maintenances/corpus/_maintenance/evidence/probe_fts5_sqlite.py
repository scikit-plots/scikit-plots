"""
Probe the FTS5 quoting helpers against the SQLite this interpreter bundles (CX-05).

Standard library only, so it runs on every interpreter, including one where
``scikitplot.corpus`` itself does not import: the helpers are read from
``scikitplot/corpus/_storage/_storage.py`` of a checkout and executed alone.

Run from the repository root:
``python -B maintenances/corpus/_maintenance/evidence/probe_fts5_sqlite.py [ROOT]``.
``ROOT`` defaults to this checkout. Writes one line per character class and
exits non-zero if a quoted query raises, a kept character is changed, or a
query differs from the same query with a space in place of the character.
"""

import __future__

import ast
import pathlib
import platform
import re
import sqlite3
import sys

ROOT = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[4]
)
NAMES = ("_FTS5_UNREPRESENTABLE", "_FTS5_TOKEN", "_fts5_phrase", "_fts5_any")

#: Characters FTS5 cannot hold in a string, and the boundaries around them.
SPACED = ["\x00", "\ud800", "\udbff", "\udc00", "\udfff"]
#: Characters that must reach FTS5 unchanged, the boundaries included.
KEPT = (
    [chr(c) for c in range(0x01, 0x20)]
    + ["\x7f", "\x80", "\x85", "\x9f", "\u00a0", "\u200d", "\u2028", "\u2029"]
    + ["\ud7ff", "\ue000", "\ufdd0", "\ufeff", "\ufffe", "\uffff"]
    + ["\U0001f600", "\U00020000", "\U0010ffff"]
)
TEMPLATES = ["{c}", "a{c}b", "roc{c}auc", '"{c}"', "{c}{c}null", "AND{c}OR"]
DOCS = ["roc auc curve", "a b", "null value", "AND OR NOT", "plain text"]


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")


def _load(root: pathlib.Path) -> dict:
    """Execute only the helpers of ``_storage.py``, with its ``__future__`` flags."""
    path = root / "scikitplot" / "corpus" / "_storage" / "_storage.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            keep.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) in NAMES for t in targets):
                keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    flags = __future__.annotations.compiler_flag
    namespace = {"re": re}
    code = compile(module, str(path), "exec", flags=flags, dont_inherit=True)
    # The checkout's own helpers, read from its source: the point of the probe.
    exec(code, namespace)  # noqa: S102
    return namespace


def _run(con, expression):
    """Return the matching rowids, or the error a query raised."""
    try:
        rows = con.execute(
            "SELECT rowid FROM t WHERE t MATCH ? ORDER BY rowid", (expression,)
        ).fetchall()
    except Exception as exc:  # noqa: BLE001  # the probe records every failure
        return f"{type(exc).__name__}: {exc}"
    return [r[0] for r in rows]


def main() -> int:
    """Check every class on this interpreter's SQLite and print the verdict."""
    ns = _load(ROOT)
    phrase, any_ = ns["_fts5_phrase"], ns["_fts5_any"]
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE VIRTUAL TABLE t USING fts5(text)")
    except sqlite3.OperationalError as exc:
        _say(
            f"python {platform.python_version()} sqlite {sqlite3.sqlite_version}: no FTS5 ({exc})"
        )
        return 0
    con.executemany("INSERT INTO t(text) VALUES (?)", [(d,) for d in DOCS])
    _say(
        f"python {platform.python_version()} gil={'on' if gil else 'off'} "
        f"sqlite {sqlite3.sqlite_version} root={ROOT}"
    )
    bad = 0
    raw_rejected = []
    for char in SPACED + KEPT:
        spaced_char = char in SPACED
        raw = '"a' + char.replace('"', '""') + 'b"'
        if isinstance(_run(con, raw), str):
            raw_rejected.append(f"U+{ord(char):04X}")
        for template in TEMPLATES:
            query = template.format(c=char)
            spaced = template.format(c=" ")
            for build in (phrase, any_):
                expression = build(query)
                got = _run(con, expression) if expression is not None else []
                if isinstance(got, str):
                    bad += 1
                    _say(f"  RAISE {build.__name__}({query!r}): {got}")
                    continue
                if spaced_char:
                    other = build(spaced)
                    want = _run(con, other) if other is not None else []
                    if got != want or expression != other:
                        bad += 1
                        _say(f"  DIFF {build.__name__}({query!r}) != spaced query")
                elif build is phrase and char not in expression:
                    bad += 1
                    _say(
                        f"  CHANGED {build.__name__}({query!r}) dropped U+{ord(char):04X}"
                    )
    _say(f"  raw phrase rejected by SQLite: {', '.join(raw_rejected) or 'none'}")
    _say(f"  helpers: {len(SPACED + KEPT) * len(TEMPLATES) * 2} queries, FAILED: {bad}")
    con.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

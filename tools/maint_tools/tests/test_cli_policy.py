# SPDX-License-Identifier: BSD-3-Clause
"""
Every Python helper in tools/maint_tools/ follows the README's CLI policy.

The policy (``tools/maint_tools/README.md``, "CLI policy for Python helpers")
says importing a helper must not parse arguments, read stdin, use the network
or run its work. These tests check it for every ``*.py`` file in the
directory, so a new helper is covered without being listed:

* the module body holds only a docstring, imports, definitions, constant
  assignments, ``try`` blocks of those, and one ``if __name__ == "__main__":``
  guard; no top-level call statement, loop or ``with``;
* nothing outside a function reads ``sys.argv``/``sys.stdin`` or calls
  ``parse_args``;
* it defines ``main`` and calls it only under the guard;
* ``--help`` exits 0, run with ``python -I`` from a directory outside the
  checkout, and prints the same text twice;
* it names no ``sklearn`` path: the inherited scikit-learn helpers were removed
  (PR 907), so such a name would be a stale copy.

Notes
-----
**Developer notes.** The module-body rule is checked on the AST, so it is
deterministic and needs none of a helper's optional dependencies. A
module-level assignment may call a pure constructor (``re.compile``,
``Path(...)``); what it may not do is name ``sys.argv``, ``sys.stdin`` or
``parse_args``, which is the import-time work the policy forbids.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
HELPERS = sorted(HERE.glob("*.py"))
_ALLOWED = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
)
_SKLEARN = re.compile(r"\bsklearn\b")


def _is_main_guard(node: ast.stmt) -> bool:
    test = getattr(node, "test", None)
    return (
        isinstance(node, ast.If)
        and isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _body_problems(body: list, where: str) -> list:
    problems = []
    for node in body:
        if _is_docstring(node) or _is_main_guard(node) or isinstance(node, _ALLOWED):
            continue
        if isinstance(node, ast.Try):
            for part in (node.body, node.orelse, node.finalbody):
                problems += _body_problems(part, where)
            for handler in node.handlers:
                problems += _body_problems(handler.body, where)
            continue
        problems.append(f"{where}:{node.lineno}: top-level {type(node).__name__}")
    return problems


def _import_time_reads(tree: ast.Module, where: str) -> list:
    """Names of argv/stdin/parse_args used outside every function body."""
    problems = []

    def visit(node, inside):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            inside = True
        if not inside and _is_main_guard(node):
            return
        if not inside:
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
                and node.attr in {"argv", "stdin"}
            ):
                problems.append(f"{where}:{node.lineno}: sys.{node.attr} at import")
            if isinstance(node, ast.Attribute) and node.attr == "parse_args":
                problems.append(f"{where}:{node.lineno}: parse_args at import")
        for child in ast.iter_child_nodes(node):
            visit(child, inside)

    visit(tree, False)
    return problems


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class CliPolicyTests(unittest.TestCase):
    def test_there_are_helpers_to_check(self):
        self.assertGreaterEqual(len(HELPERS), 8)

    def test_importing_a_helper_runs_no_work(self):
        problems = []
        for path in HELPERS:
            tree = _tree(path)
            problems += _body_problems(tree.body, path.name)
            problems += _import_time_reads(tree, path.name)
        self.assertEqual(problems, [])

    def test_every_helper_has_main_called_only_under_the_guard(self):
        for path in HELPERS:
            tree = _tree(path)
            names = {
                n.name
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            guards = [n for n in tree.body if _is_main_guard(n)]
            with self.subTest(helper=path.name):
                self.assertIn("main", names)
                self.assertEqual(len(guards), 1)
                calls = {
                    n.func.id
                    for n in ast.walk(guards[0])
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                }
                self.assertIn("main", calls)

    def test_help_is_safe_and_deterministic_from_another_directory(self):
        with tempfile.TemporaryDirectory() as cwd:
            for path in HELPERS:
                runs = [
                    subprocess.run(  # noqa: S603 - this interpreter, a repo file, --help
                        [sys.executable, "-I", "-B", str(path), "--help"],
                        cwd=cwd,
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                    for _ in range(2)
                ]
                with self.subTest(helper=path.name):
                    self.assertEqual(runs[0].returncode, 0, runs[0].stderr)
                    self.assertTrue(runs[0].stdout.strip())
                    self.assertEqual(runs[0].stdout, runs[1].stdout)

    def test_no_helper_names_a_sklearn_path(self):
        offenders = [
            f"{path.name}:{number}"
            for path in HELPERS
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            )
            if _SKLEARN.search(line)
        ]
        self.assertEqual(offenders, [])

    def test_the_removed_scikit_learn_helpers_stay_removed(self):
        for name in (
            "bump-dependencies-versions.py",
            "check_xfailed_checks.py",
            "sort_whats_new.py",
            "update_tracking_issue.py",
            "whats_missing.sh",
        ):
            with self.subTest(helper=name):
                self.assertFalse((HERE / name).exists())


class PolicyCheckerTests(unittest.TestCase):
    """The checks above catch each import-time pattern the removed helpers had."""

    def _problems(self, source: str) -> list:
        tree = ast.parse(source)
        return _body_problems(tree.body, "x.py") + _import_time_reads(tree, "x.py")

    def test_the_legacy_patterns_are_caught(self):
        cases = {
            "args = parser.parse_args()\n": "parse_args",
            "text = ''.join(sys.stdin)\n": "sys.stdin",
            "for e in items():\n    pass\n": "For",
            "print('x')\n": "Expr",
            "with open('f') as h:\n    pass\n": "With",
            "x = sys.argv[1] if len(sys.argv) > 1 else 'today'\n": "sys.argv",
        }
        for source, label in cases.items():
            with self.subTest(source=source):
                problems = self._problems(source)
                self.assertTrue(problems)
                self.assertIn(label, " ".join(problems))

    def test_the_policy_shape_passes(self):
        source = (
            '"""Doc."""\nimport re\n_X = re.compile("a")\n'
            "try:\n    import yaml\nexcept ImportError:\n    yaml = None\n"
            "def main(argv=None):\n    return len(sys.argv)\n"
            'if __name__ == "__main__":\n    raise SystemExit(main())\n'
        )
        self.assertEqual(self._problems(source), [])


if __name__ == "__main__":
    unittest.main()

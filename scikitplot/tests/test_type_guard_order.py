"""
No ``len()`` runs on a value before the check that proves it has a length.

Notes
-----
**User notes.** A configuration value of the wrong type must be refused with
the project's own configuration error, which names the field. When ``len()``
runs first, an integer or ``None`` raises a bare ``TypeError`` from the
interpreter instead, and the message no longer says which field was wrong.

**Developer notes.** The regression this guards against came from a lint
fix that hoisted ``len(item) > 1024`` above ``isinstance(item, str)`` inside
one condition. Both forms of that reordering are detected, for the same
expression (compared by its AST, so ``value`` and ``self.value`` differ):

* **One condition.** In ``len(x) > N or not isinstance(x, str)`` the
  ``len`` operand runs first, so the type check never protects it.
* **Sibling statements.** An ``if`` whose test calls ``len(x)`` followed,
  in the same block, by an ``if`` whose test checks ``isinstance(x, ...)``:
  the later type check shows that ``x`` may not have a length.

A ``len(x)`` counts as guarded, and is not reported, when an earlier operand
of the same ``and``/``or`` chain, the test of an enclosing ``if`` or
conditional expression, or an earlier sibling ``if`` already calls
``isinstance(x, ...)``. The check is purely syntactic and never imports the
scanned code, so it runs in a second with nothing installed. A file Python
cannot parse cannot be imported either, so it is skipped.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap
import warnings

PACKAGE = pathlib.Path(__file__).resolve().parents[1]

#: Directories that hold no package source.
_SKIP_DIRS = frozenset({"__pycache__", "build", "node_modules"})


def _call_targets(node, name):
    """
    Return the first-argument dumps of every ``name(...)`` call under *node*.

    Parameters
    ----------
    node : ast.AST
        Expression or statement to search.
    name : str
        Bare function name, such as ``"len"`` or ``"isinstance"``.

    Returns
    -------
    frozenset of str
        ``ast.dump`` of each call's first positional argument.
    """
    return frozenset(
        ast.dump(sub.args[0])
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == name
        and sub.args
    )


def _unguarded_len_targets(expr, guarded):
    """
    Return targets of ``len()`` calls in *expr* that no type check precedes.

    Parameters
    ----------
    expr : ast.AST
        Expression to search.
    guarded : frozenset of str
        Targets already proven by an enclosing or earlier ``isinstance``.

    Returns
    -------
    set of str
        ``ast.dump`` of each unguarded ``len()`` argument.
    """
    if isinstance(expr, ast.BoolOp):
        found = set()
        seen = set(guarded)
        for operand in expr.values:
            found |= _unguarded_len_targets(operand, frozenset(seen))
            seen |= _call_targets(operand, "isinstance")
        return found
    if isinstance(expr, ast.IfExp):
        inner = guarded | _call_targets(expr.test, "isinstance")
        return (
            _unguarded_len_targets(expr.test, guarded)
            | _unguarded_len_targets(expr.body, inner)
            | _unguarded_len_targets(expr.orelse, inner)
        )
    found = set()
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "len"
        and expr.args
    ):
        target = ast.dump(expr.args[0])
        if target not in guarded:
            found.add(target)
    for child in ast.iter_child_nodes(expr):
        found |= _unguarded_len_targets(child, guarded)
    return found


def _condition_violations(expr, guarded):
    """
    Return line numbers where one condition runs ``len(x)`` before its check.

    Parameters
    ----------
    expr : ast.AST
        Expression to search.
    guarded : frozenset of str
        Targets already proven by an enclosing ``isinstance``.

    Returns
    -------
    list of int
        Line of each ``and``/``or`` chain with the reordered form.
    """
    lines = []
    for sub in ast.walk(expr):
        if not isinstance(sub, ast.BoolOp):
            continue
        seen = set(guarded)
        for index, operand in enumerate(sub.values):
            early = _unguarded_len_targets(operand, frozenset(seen))
            later = set()
            for following in sub.values[index + 1 :]:
                later |= _call_targets(following, "isinstance")
            if early & later:
                lines.append(sub.lineno)
                break
            seen |= _call_targets(operand, "isinstance")
    return lines


def _block_violations(body, guarded, lines):
    """
    Collect violations in one statement list and the blocks nested in it.

    Parameters
    ----------
    body : list of ast.stmt
        Statements of one block, in source order.
    guarded : frozenset of str
        Targets proven by the enclosing ``if`` tests.
    lines : list of int
        Receives the line of every violation found.
    """
    seen = set(guarded)
    for index, stmt in enumerate(body):
        if isinstance(stmt, ast.If):
            lines.extend(_condition_violations(stmt.test, frozenset(seen)))
            early = _unguarded_len_targets(stmt.test, frozenset(seen))
            for following in body[index + 1 :]:
                if isinstance(following, ast.If) and early & _call_targets(
                    following.test, "isinstance"
                ):
                    lines.append(stmt.lineno)
                    break
            inner = frozenset(seen | _call_targets(stmt.test, "isinstance"))
            _block_violations(stmt.body, inner, lines)
            _block_violations(stmt.orelse, inner, lines)
            seen |= _call_targets(stmt.test, "isinstance")
            continue
        for field in ("body", "orelse", "finalbody", "handlers"):
            nested = getattr(stmt, field, None)
            if isinstance(nested, list) and nested:
                if isinstance(nested[0], ast.ExceptHandler):
                    for handler in nested:
                        _block_violations(handler.body, frozenset(seen), lines)
                else:
                    _block_violations(nested, frozenset(seen), lines)
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.expr):
                lines.extend(_condition_violations(child, frozenset(seen)))


def _violations(source):
    """
    Return the sorted lines in *source* that run ``len()`` before its check.

    Parameters
    ----------
    source : str
        Python source code.

    Returns
    -------
    list of int
        Sorted, de-duplicated line numbers; empty when the source is clean or
        cannot be parsed.
    """
    try:
        with warnings.catch_warnings():
            # An invalid escape sequence in a scanned file is that file's
            # concern; it must not fail this check under ``-W error``.
            warnings.simplefilter("ignore", (SyntaxWarning, DeprecationWarning))
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = []
    _block_violations(tree.body, frozenset(), lines)
    return sorted(set(lines))


def _source_files(root):
    """Yield every ``.py`` file under *root* in a stable order."""
    for path in sorted(root.rglob("*.py")):
        if not _SKIP_DIRS.intersection(path.relative_to(root).parts):
            yield path


def _check(source):
    return _violations(textwrap.dedent(source))


def test_the_check_finds_the_reordered_condition():
    assert _check("""
        def f(item):
            if len(item) > 1024 or not isinstance(item, (str, int)):
                raise ValueError(item)
        """) == [3]


def test_the_check_finds_a_length_test_before_a_sibling_type_test():
    assert _check("""
        def f(item):
            if len(item) > 1024:
                raise ValueError(item)
            if not isinstance(item, str):
                raise ValueError(item)
        """) == [3]


def test_the_check_finds_the_form_outside_an_if():
    assert _check("""
        def f(item):
            return len(item) > 1024 and isinstance(item, str)
        """) == [3]


def test_the_check_accepts_every_guarded_form():
    assert _check("""
            def f(item, other):
                if isinstance(item, str) and len(item) > 1024:
                    raise ValueError(item)
                if not isinstance(item, str) or len(item) > 1024:
                    raise ValueError(item)
                n = len(item) if isinstance(item, str) else 0
                if isinstance(item, str):
                    if len(item) > 1024:
                        raise ValueError(item)
                if isinstance(other, list) and len(other) == 0:
                    return n
                if isinstance(other, list):
                    return n
            """) == []


def test_the_check_accepts_a_type_check_proved_by_an_earlier_sibling():
    assert _check("""
            def f(item):
                if not isinstance(item, str):
                    raise TypeError(item)
                if len(item) > 1024:
                    raise ValueError(item)
                if isinstance(item, str):
                    return item
            """) == []


def test_the_check_compares_the_same_expression_only():
    assert _check("""
            def f(self, item):
                if len(self.item) > 1024 or not isinstance(item, str):
                    raise ValueError(item)
            """) == []


def test_the_check_skips_unparseable_source():
    assert _check("def f(:\n") == []


def test_no_package_source_runs_len_before_its_type_check():
    offenders = []
    for path in _source_files(PACKAGE):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.relative_to(PACKAGE.parent)
        offenders.extend(f"{relative}:{line}" for line in _violations(source))
    assert not offenders, (
        "len() runs before the isinstance() check that proves the value has "
        "a length; check the type first (engineering rule 68):\n" + "\n".join(offenders)
    )

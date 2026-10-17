"""
Architecture guard: each submodule stands on its own (slice S-39).

A feature submodule must be usable without the others. ``scikitplot.corpus``
should import and work whether or not ``scikitplot.annoy`` is installable, and
``scikitplot.mcp`` should import whether or not a corpus can be built. That is
true today, and this module is what keeps it true: a module-level import of a
sibling submodule is easy to add and impossible to notice afterwards.

The rule is a tier, not a ban:

No submodule imports another at module level. A contract that several of them
need is carried as a byte-identical copy inside each, not shared from a common
package, so ``scikitplot.corpus`` can be read, tested and reasoned about without
``scikitplot.rank_bm25`` or anything else being present.

Duplication has an obvious failure mode: the copies drift, one concept ends up
answered two different ways, and that is precisely the defect the shared
contract was written to remove. So the copies are compared by digest here. A
change to one must be a change to all, or this fails.

An optional integration with a sibling is still allowed, imported inside the
function that needs it, so a missing sibling is an unavailable capability rather
than an ``ImportError``.

See Also
--------
scikitplot._utils._canonical
scikitplot._utils._indexing
"""

import ast
import difflib
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

TIER1 = frozenset({"corpus", "rank_bm25", "annoy", "_cli", "mcp", "agent"})
SUBMODULES = TIER1 | frozenset({"_utils", "config", "cexternals", "_lib"})

#: Contracts that several submodules need and none may import from another.
#: Each carries a byte-identical copy; the copies are compared below, because
#: duplication is only safe while something enforces that it stays duplication
#: rather than becoming divergence.
DUPLICATED_CONTRACTS = {
    "_validation.py": ("corpus", "rank_bm25", "mcp"),
    # R2-1: one rank-fusion rule for every hybrid retriever.
    "_fusion.py": ("corpus", "mcp"),
}

#: Module-level dependencies that are part of a submodule's definition rather
#: than coupling: a backend binding and the configuration it is built against.
ALLOWED_MODULE_LEVEL = {
    ("annoy", "cexternals"),
    ("_cli", "config"),
}


_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _is_type_checking(node):
    """Whether ``node`` is ``if TYPE_CHECKING:`` (or ``typing.TYPE_CHECKING``)."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _import_targets(node, depth):
    """Return the ``scikitplot`` submodules an import statement names."""
    targets = []
    if isinstance(node, ast.Import):
        targets = [
            alias.name.split(".")[1]
            for alias in node.names
            if alias.name.startswith("scikitplot.")
        ]
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        names = [alias.name for alias in node.names]
        if node.level and node.level > depth:
            # Leaves the submodule: ``from ..corpus import x``, ``from .. import corpus``.
            targets = [module.split(".")[0]] if module else names
        elif not node.level and module == "scikitplot":
            targets = names  # ``from scikitplot import corpus``
        elif not node.level and module.startswith("scikitplot."):
            targets = [module.split(".")[1]]
    return targets


def _cross_submodule_imports(path, owner):
    """Yield ``(target, is_module_level, lineno)`` for imports of other submodules.

    Module level means "runs when the module is imported": anything outside a
    function body (class bodies and module-level ``if``/``try`` blocks
    included), except the body of ``if TYPE_CHECKING:``. ``import x.y``,
    ``from x.y import z``, ``from scikitplot import y`` and relative imports
    that leave the submodule all count.
    """
    depth = len(path.relative_to(ROOT).parts) - 1
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:  # pragma: no cover - vendored or templated sources
        return

    def walk(nodes, at_module_level):
        for child in nodes:
            for target in _import_targets(child, depth):
                if target in SUBMODULES and target != owner:
                    yield target, at_module_level, child.lineno
            if _is_type_checking(child):
                yield from walk(child.body, False)
                yield from walk(child.orelse, at_module_level)
                continue
            yield from walk(
                ast.iter_child_nodes(child),
                at_module_level and not isinstance(child, _FUNCTION_NODES),
            )

    yield from walk(tree.body, True)


def _sources(submodule):
    """Yield the production sources of ``submodule``."""
    for path in (ROOT / submodule).rglob("*.py"):
        parts = path.relative_to(ROOT).parts
        if "tests" in parts or "__pycache__" in parts:
            continue
        yield path


@pytest.mark.parametrize("submodule", sorted(TIER1))
def test_a_submodule_does_not_import_a_sibling_at_module_level(submodule):
    """A sibling's absence must be an unavailable capability, not an ImportError."""
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno} imports {target}"
        for path in _sources(submodule)
        for target, module_level, lineno in _cross_submodule_imports(path, submodule)
        if module_level and (submodule, target) not in ALLOWED_MODULE_LEVEL
    ]
    assert offenders == [], (
        f"{submodule} imports a sibling submodule at module level, so it can no "
        f"longer be used on its own: {offenders}. Move the import inside the "
        "function that needs it and report absence as an unavailable capability."
    )


@pytest.mark.parametrize(("filename", "owners"), sorted(DUPLICATED_CONTRACTS.items()))
def test_a_duplicated_contract_has_not_drifted(filename, owners):
    """Every copy of a shared contract is byte-identical to every other."""
    texts, digests = {}, {}
    for owner in owners:
        path = ROOT / owner / filename
        assert path.is_file(), f"{owner} is missing its copy of {filename}"
        data = path.read_bytes()
        texts[owner] = data.decode("utf-8", errors="replace")
        digests[owner] = hashlib.sha256(data).hexdigest()
    # Name the copy that differs and show how: a digest alone says nothing
    # about whether the drift is a formatter's reflow or a changed rule.
    reference = owners[0]
    diffs = [
        "".join(
            difflib.unified_diff(
                texts[reference].splitlines(keepends=True),
                texts[owner].splitlines(keepends=True),
                fromfile=f"{reference}/{filename}",
                tofile=f"{owner}/{filename}",
            )
        )
        for owner in owners[1:]
        if digests[owner] != digests[reference]
    ]
    assert len(set(digests.values())) == 1, (
        f"copies of {filename} have drifted apart: {digests}. One concept "
        "answered two ways is the defect this contract removes; change every "
        "copy together.\n" + "\n".join(diffs)
    )


@pytest.mark.parametrize(("filename", "owners"), sorted(DUPLICATED_CONTRACTS.items()))
def test_every_copy_of_a_duplicated_contract_is_tested(filename, owners):
    """A copy nobody exercises is a copy that can rot unnoticed."""
    stem = Path(filename).stem
    missing = [
        owner for owner in owners
        if not (ROOT / owner / "tests" / f"test_{stem}.py").is_file()
    ]
    assert missing == [], f"no test for {filename} in: {missing}"


@pytest.mark.parametrize("submodule", sorted(TIER1))
def test_every_submodule_is_declared(submodule):
    """A new submodule joins this rule deliberately, rather than by default."""
    assert (ROOT / submodule).is_dir()


def test_the_scanner_tells_function_imports_from_module_imports(tmp_path, monkeypatch):
    """Only code that runs at import time counts as module level (R2-5).

    A function's own statements used to count as module level, and the body
    of a module-level ``try`` nested one block deeper did not.
    """
    source = tmp_path / "corpus" / "probe.py"
    source.parent.mkdir()
    source.write_text(
        "from scikitplot.annoy import A\n"
        "try:\n"
        "    if True:\n"
        "        from scikitplot.annoy import B\n"
        "except ImportError:\n"
        "    pass\n"
        "class C:\n"
        "    from scikitplot.annoy import D\n"
        "def f():\n"
        "    from scikitplot.annoy import E\n"
        "    if True:\n"
        "        from scikitplot.annoy import F\n"
        "g = lambda: __import__('x')\n"
        "import scikitplot.annoy\n"
        "import scikitplot.annoy as G\n"
        "from scikitplot import annoy\n"
        "if TYPE_CHECKING:\n"
        "    from scikitplot.annoy import H\n"
        "else:\n"
        "    from scikitplot.annoy import I\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "ROOT", tmp_path)  # the scanner's base
    found = sorted(
        (lineno, module_level)
        for _, module_level, lineno in _cross_submodule_imports(source, "corpus")
    )
    assert found == [
        (1, True),
        (4, True),
        (8, True),
        (10, False),
        (12, False),
        (14, True),
        (15, True),
        (16, True),
        (18, False),
        (20, True),
    ]

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
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

TIER1 = frozenset({"corpus", "rank_bm25", "annoy", "_cli", "mcp"})
SUBMODULES = TIER1 | frozenset({"_utils", "config", "cexternals", "_lib"})

#: Contracts that several submodules need and none may import from another.
#: Each carries a byte-identical copy; the copies are compared below, because
#: duplication is only safe while something enforces that it stays duplication
#: rather than becoming divergence.
DUPLICATED_CONTRACTS = {
    "_validation.py": ("corpus", "rank_bm25", "mcp"),
}

#: Module-level dependencies that are part of a submodule's definition rather
#: than coupling: a backend binding and the configuration it is built against.
ALLOWED_MODULE_LEVEL = {
    ("annoy", "cexternals"),
    ("_cli", "config"),
}


def _cross_submodule_imports(path, owner):
    """Yield ``(target, is_module_level, lineno)`` for imports of other submodules."""
    depth = len(path.relative_to(ROOT).parts) - 1
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:  # pragma: no cover - vendored or templated sources
        return

    def walk(node, at_module_level):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ImportFrom):
                module = child.module or ""
                if child.level and child.level > depth:
                    target = module.split(".")[0] if module else None
                elif child.level:
                    target = None
                elif module.startswith("scikitplot."):
                    target = module.split(".")[1]
                else:
                    target = None
                if target in SUBMODULES and target != owner:
                    yield target, at_module_level, child.lineno
            yield from walk(child, at_module_level and isinstance(node, ast.Module))

    yield from walk(tree, True)


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
    digests = {}
    for owner in owners:
        path = ROOT / owner / filename
        assert path.is_file(), f"{owner} is missing its copy of {filename}"
        digests[owner] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(set(digests.values())) == 1, (
        f"copies of {filename} have drifted apart: {digests}. One concept "
        "answered two ways is the defect this contract removes; change every "
        "copy together."
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

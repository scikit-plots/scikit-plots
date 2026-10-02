"""
Tests for the public facade, :mod:`scikitplot.cleanprompt`.

Notes
-----
**Developer notes.** This module is the gate on the import contract. The
optional tiers are absent or present depending on the machine, so every
assertion here is written to hold either way: presence is asserted through the
capability report, never assumed.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

from .. import _capabilities as caps

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
ROOT = PACKAGE.parents[1]

#: Third-party distributions the base tier must never pull in.
FORBIDDEN_AT_IMPORT = ("spacy", "flask", "cryptography", "numpy", "pandas", "pydantic")


def _runtime_sources():
    """Every runtime source file, excluding the test package."""
    return sorted(
        path
        for path in PACKAGE.rglob("*.py")
        if "tests" not in path.relative_to(PACKAGE).parts
    )


#: Registers an empty stand-in for the parent package before anything imports it.
#:
#: ``import scikitplot.cleanprompt`` first runs ``scikitplot/__init__.py``, and
#: that file imports NumPy unconditionally. What the parent loads is not a
#: property of this package, so the isolation claims are measured with the
#: parent replaced by an empty module whose ``__path__`` is the real package
#: directory: every submodule still resolves from the real files, and nothing
#: outside ``cleanprompt`` runs.
_ISOLATED_PARENT = (
    "import sys, types\n"
    "_parent = types.ModuleType('scikitplot')\n"
    "_parent.__path__ = [{0!r}]\n"
    "sys.modules['scikitplot'] = _parent\n"
).format(str(PACKAGE.parent))

#: Imports the real parent package from the checkout the tests run in.
_REAL_PARENT = "import sys;sys.path.insert(0, {0!r})\n".format(str(ROOT))


def _in_subprocess(body, *, prelude=_ISOLATED_PARENT):
    """
    Run ``body`` in a fresh interpreter and return its stdout.

    Parameters
    ----------
    body : str
        Source to execute after ``prelude``.
    prelude : str, default=_ISOLATED_PARENT
        Source executed first. ``_ISOLATED_PARENT`` measures this package
        alone; ``_REAL_PARENT`` measures it under the real ``scikitplot``.

    Returns
    -------
    str
        Captured standard output.

    Raises
    ------
    AssertionError
        If the interpreter exits with a non-zero status.
    """
    completed = subprocess.run(
        [sys.executable, "-c", prelude + body], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


class TestImportIsolation:
    """The central claim: importing costs nothing and pulls in nothing."""

    def test_plain_import_loads_no_third_party_package(self):
        output = _in_subprocess(
            "import scikitplot.cleanprompt\n"
            "print([n for n in {0!r} if n in sys.modules])".format(FORBIDDEN_AT_IMPORT)
        )
        assert output.strip() == "[]"

    def test_star_import_is_base_safe(self):
        """``__all__`` is the star-import surface; it must resolve with no extras."""
        output = _in_subprocess(
            "exec('from scikitplot.cleanprompt import *')\n"
            "print([n for n in {0!r} if n in sys.modules])".format(FORBIDDEN_AT_IMPORT)
        )
        assert output.strip() == "[]"

    def test_dir_does_not_resolve_optional_names(self):
        output = _in_subprocess(
            "import scikitplot.cleanprompt as cp\n"
            "names = dir(cp)\n"
            "print('spacy_detector' in names, [n for n in {0!r} if n in sys.modules])".format(
                FORBIDDEN_AT_IMPORT
            )
        )
        assert output.strip() == "True []"

    def test_hasattr_on_a_missing_name_imports_nothing(self):
        output = _in_subprocess(
            "import scikitplot.cleanprompt as cp\n"
            "hasattr(cp, 'definitely_not_here')\n"
            "print([n for n in {0!r} if n in sys.modules])".format(FORBIDDEN_AT_IMPORT)
        )
        assert output.strip() == "[]"

    def test_module_entry_point_help_is_base_safe(self):
        completed = subprocess.run(
            [sys.executable, "-m", "scikitplot.cleanprompt", "--help"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert completed.returncode == 0
        assert "redact" in completed.stdout

    def test_capabilities_subcommand_is_base_safe(self):
        completed = subprocess.run(
            [sys.executable, "-m", "scikitplot.cleanprompt", "capabilities"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert completed.returncode == 0
        assert "ner" in completed.stdout

    def test_import_under_a_blocker_still_succeeds(self):
        """The strongest form: the optional packages are made unimportable."""
        body = (
            "import builtins\n"
            "real = builtins.__import__\n"
            "blocked = {0!r}\n"
            "def guard(name, *a, **k):\n"
            "    if name.split('.')[0] in blocked:\n"
            "        raise ImportError('blocked: ' + name)\n"
            "    return real(name, *a, **k)\n"
            "builtins.__import__ = guard\n"
            "import scikitplot.cleanprompt as cp\n"
            "exec('from scikitplot.cleanprompt import *')\n"
            "r = cp.Redactor().redact('mail a@b.co')\n"
            "print(r.text, cp.restore(r.text, r.vault).text == 'mail a@b.co')\n"
        ).format(FORBIDDEN_AT_IMPORT)
        assert _in_subprocess(body).strip() == "mail [EMAIL-1] True"


class TestLazyResolution:
    """PEP 562 behaviour of the facade."""

    def test_optional_names_are_absent_from_all(self):
        from .. import __all__
        from .._capabilities import TIERS  # noqa: F401 - readability

        from .. import _LAZY

        assert set(__all__).isdisjoint(set(_LAZY))

    def test_optional_names_are_listed_by_dir(self):
        import scikitplot.cleanprompt as package

        from .. import _LAZY

        assert set(_LAZY) <= set(dir(package))

    def test_dir_is_sorted_and_unique(self):
        import scikitplot.cleanprompt as package

        names = dir(package)
        assert names == sorted(set(names))

    def test_unknown_attribute_raises_attribute_error(self):
        import scikitplot.cleanprompt as package

        with pytest.raises(AttributeError, match="has no attribute"):
            package.definitely_not_here

    def test_optional_name_reflects_its_tier(self):
        """Available resolves; unavailable raises with an install command."""
        import scikitplot.cleanprompt as package
        from .. import CapabilityError

        for name, (_module, tier) in package._LAZY.items():
            if caps.probe(tier).available:
                assert getattr(package, name) is not None
            else:
                with pytest.raises(CapabilityError) as caught:
                    getattr(package, name)
                assert caught.value.tier == tier
                assert "pip install" in str(caught.value)

    def test_resolution_is_cached(self):
        """A second access must not re-enter ``__getattr__``."""
        import scikitplot.cleanprompt as package

        if not caps.probe("web").available:
            pytest.skip("web tier unavailable")
        first = package.create_app
        assert "create_app" in vars(package)
        assert package.create_app is first


class TestPublicSurface:
    """What ``__all__`` promises."""

    def test_every_exported_name_resolves(self):
        import scikitplot.cleanprompt as package

        missing = [name for name in package.__all__ if not hasattr(package, name)]
        assert missing == []

    def test_all_is_sorted_within_its_sections(self):
        import scikitplot.cleanprompt as package

        assert len(package.__all__) == len(set(package.__all__))

    def test_version_is_present_and_plausible(self):
        import scikitplot.cleanprompt as package

        assert package.__version__.count(".") == 2

    def test_core_names_are_exported(self):
        import scikitplot.cleanprompt as package

        for name in ("Redactor", "restore", "Vault", "RedactionPolicy", "capabilities"):
            assert name in package.__all__

    def test_no_private_name_is_exported(self):
        import scikitplot.cleanprompt as package

        assert [
            n for n in package.__all__ if n.startswith("_") and n != "__version__"
        ] == []


class TestArchitecture:
    """Structural rules, enforced by parsing rather than by convention."""

    def test_no_file_in_the_package_holds_a_whole_credential(self):
        """
        Invariant ``I14``, for the whole package rather than ``_config`` alone.

        Everything under the package is committed and published, tests and
        documentation included, so it is all read by secret scanners. A
        fixture that needs a key-shaped value builds it from pieces at run
        time; a literal one is found here, by the package's own patterns,
        before a push is refused for it.
        """
        from .._catalog import (  # noqa: PLC0415
            AT_REST_PACKS,
            at_rest_findings,
            builtin_catalog,
        )

        packs = [builtin_catalog().packs[name] for name in AT_REST_PACKS]
        texts = {}
        for path in sorted(PACKAGE.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                texts[path.relative_to(PACKAGE).as_posix()] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue  # not text, so not something a pattern can read
        assert len(texts) > 50, "the walk found too few files to mean anything"
        assert at_rest_findings(texts, packs) == []

    def test_no_sibling_scikitplot_submodule_is_imported(self):
        """
        Each submodule must be usable on its own.

        The one exception is the corpus bridge, ``_corpus.py``, which may
        import ``scikitplot.corpus`` inside a function and nowhere else — the
        same rule the contract check ``CP-INDEP-001`` enforces.
        """
        offenders = []
        for path in _runtime_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            deferred = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    deferred.update(
                        id(inner) for inner in ast.walk(node) if inner is not node
                    )
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                for name in names:
                    if name.split(".")[0] != "scikitplot":
                        continue
                    bridge = (
                        path.name == "_corpus.py"
                        and (
                            name == "scikitplot.corpus"
                            or name.startswith("scikitplot.corpus.")
                        )
                        and id(node) in deferred
                    )
                    if not bridge:
                        offenders.append("{0}: {1}".format(path.name, name))
        assert offenders == []

    def test_no_runtime_module_imports_the_maintenance_plane(self):
        offenders = []
        for path in _runtime_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                if any(n.startswith(("maintenances", "skills")) for n in names):
                    offenders.append(path.name)
        assert offenders == []

    def test_optional_dependencies_are_imported_inside_functions_only(self):
        """A module-scope optional import would break the tier contract."""
        offenders = []
        for path in _runtime_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            scoped = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for inner in ast.walk(node):
                        scoped.add(id(inner))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                for name in names:
                    if (
                        name.split(".")[0] in FORBIDDEN_AT_IMPORT
                        and id(node) not in scoped
                    ):
                        offenders.append("{0}: {1}".format(path.name, name))
        assert offenders == []

    def test_every_source_module_has_a_test_module(self):
        """``foo.py`` is owned by ``tests/test_foo.py``."""
        tests = {path.name for path in (PACKAGE / "tests").glob("test_*.py")}
        missing = []
        for path in _runtime_sources():
            if path.name == "__main__.py":
                continue  # exercised through test__cli.py; see its module note
            expected = "test_{0}".format(path.name)
            if expected not in tests:
                missing.append(expected)
        assert missing == []

    def test_every_test_module_owns_a_source_module(self):
        allowed = {"test_regressions.py"}
        sources = {path.name for path in _runtime_sources()}
        orphans = [
            path.name
            for path in (PACKAGE / "tests").glob("test_*.py")
            if path.name not in allowed and path.name[len("test_") :] not in sources
        ]
        assert orphans == []

    def test_every_public_callable_is_documented(self):
        import scikitplot.cleanprompt as package

        undocumented = []
        for name in package.__all__:
            if name.startswith("__"):
                continue
            obj = getattr(package, name)
            if callable(obj) and not (obj.__doc__ or "").strip():
                undocumented.append(name)
        assert undocumented == []

    def test_docstrings_follow_numpydoc_section_order(self):
        """Parameters, Returns, Raises, See Also, Notes, References, Examples."""
        order = [
            "Parameters",
            "Returns",
            "Yields",
            "Raises",
            "See Also",
            "Notes",
            "References",
            "Examples",
        ]
        canonical = {name: index for index, name in enumerate(order)}
        canonical["Yields"] = canonical["Returns"]
        problems = []
        for path in _runtime_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module),
                ):
                    continue
                doc = ast.get_docstring(node) or ""
                seen = [
                    canonical[name]
                    for name in order
                    if "\n{0}\n{1}".format(name, "-" * len(name)) in doc
                ]
                if seen != sorted(seen):
                    problems.append(
                        "{0}:{1}".format(path.name, getattr(node, "name", "<module>"))
                    )
        assert problems == []

    def test_no_todo_or_fixme_markers(self):
        offenders = []
        for path in _runtime_sources():
            text = path.read_text(encoding="utf-8")
            for marker in ("TODO", "FIXME", "XXX", "HACK"):
                if marker in text:
                    offenders.append("{0}: {1}".format(path.name, marker))
        assert offenders == []

    def test_no_bare_except_or_silent_pass(self):
        offenders = []
        for path in _runtime_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    if node.type is None:
                        offenders.append("{0}: bare except".format(path.name))
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                        offenders.append("{0}: silent pass".format(path.name))
        assert offenders == []

    def test_no_print_in_runtime_code(self):
        """Diagnostics go to an injected stream, never to stdout implicitly."""
        offenders = []
        for path in _runtime_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    offenders.append(path.name)
        assert offenders == []

    def test_every_module_declares_all(self):
        for path in _runtime_sources():
            if path.name in ("__init__.py", "__main__.py"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            names = {
                target.id
                for node in tree.body
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            assert "__all__" in names, "{0} declares no __all__".format(path.name)

    def test_future_annotations_everywhere(self):
        """Keeps annotations lazy, so the floor Python version stays supported."""
        for path in _runtime_sources():
            text = path.read_text(encoding="utf-8")
            assert "from __future__ import annotations" in text, path.name


class TestDoctests:
    """Every documented example runs."""

    def test_doctests_pass(self):
        import doctest

        from .. import (
            _api,
            _capabilities,
            _detectors,
            _engine,
            _engines,
            _languages,
            _logging,
            _patterns,
            _policy,
            _render,
            _types,
            _vault,
        )

        # Every base-tier module. An optional-tier module is excluded because
        # running its examples would import the dependency this facade exists
        # to keep out of the base install.
        failures = 0
        for module in (
            _api,
            _capabilities,
            _detectors,
            _engine,
            _engines,
            _languages,
            _logging,
            _patterns,
            _policy,
            _render,
            _types,
            _vault,
        ):
            result = doctest.testmod(module, verbose=False, report=False)
            failures += result.failed
        assert failures == 0


_LOADED_BY_THE_PACKAGE = (
    "before = set(sys.modules)\n"
    "import scikitplot.cleanprompt, scikitplot.cleanprompt._runtime\n"
    "print(sorted({m.split('.')[0] for m in set(sys.modules) - before}))\n"
)


def _foreign(loaded):
    """Top-level modules in ``loaded`` that are neither stdlib nor scikitplot."""
    return [
        name
        for name in loaded
        if name not in sys.stdlib_module_names
        and name != "scikitplot"
        and not name.startswith("_")
    ]


@pytest.mark.skipif(sys.version_info < (3, 10), reason="sys.stdlib_module_names is 3.10+")
def test_importing_the_package_loads_only_the_standard_library():
    """
    CP-054: the base tier is standard library only, measured, not listed.

    A block-list of known optional packages missed ``typing_extensions``, which
    two modules imported at module scope; this asserts the positive instead —
    every top-level module a fresh import loads is stdlib or scikitplot. The
    parent package is replaced by an empty stand-in, so the measurement covers
    this package and nothing else.
    """
    loaded = ast.literal_eval(_in_subprocess(_LOADED_BY_THE_PACKAGE).strip())
    assert "scikitplot" in loaded
    assert _foreign(loaded) == []


@pytest.mark.skipif(sys.version_info < (3, 10), reason="sys.stdlib_module_names is 3.10+")
def test_the_package_adds_nothing_foreign_to_what_the_parent_loads():
    """
    CP-085: under the real parent, this package adds no third-party module.

    ``scikitplot/__init__.py`` loads NumPy, so "nothing third-party is loaded"
    cannot hold for ``import scikitplot.cleanprompt`` and is not this package's
    to promise. What it does promise is that it adds nothing: the parent is
    imported first and only the modules loaded after it are examined.
    """
    body = "import scikitplot\n" + _LOADED_BY_THE_PACKAGE
    loaded = ast.literal_eval(_in_subprocess(body, prelude=_REAL_PARENT).strip())
    assert "scikitplot" in loaded
    assert _foreign(loaded) == []


def test_the_isolated_parent_runs_no_parent_code():
    """The stand-in is what was imported: the real ``__init__`` never ran."""
    output = _in_subprocess(
        "import scikitplot, scikitplot.cleanprompt\n"
        "print(scikitplot.__file__ if hasattr(scikitplot, '__file__') else None)"
    )
    assert output.strip() == "None"

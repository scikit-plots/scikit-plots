"""
Verify ``apis_reference.py`` against importable Scikit-Plots modules.

Unlike the historical check, this test never imports/executes
``docs/source/apis_reference.py``.  The documentation configuration contains
``_get_submodule(...)`` calls that intentionally inspect optional/compiled
submodules while Sphinx renders the API pages; executing those calls merely to
read ``APIS_REFERENCE`` makes a source-tree test fail before it can validate any
symbol.

The canonical blueprint/parser/validator lives in
``scikitplot._build_utils.generate_apis_reference`` and its adjacent ``apis_reference.py.in`` template. The first static check
also proves that the checked-in generated file exactly matches a clean full
rebuild. Runtime name resolution happens only after the target module imports.
Optional modules unavailable in the current environment remain explicitly
skipped rather than being mistaken for stale documentation.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_HELPER = _REPO_ROOT / "scikitplot" / "_build_utils" / "generate_apis_reference" / "__init__.py"
_SPEC = importlib.util.spec_from_file_location("_api_reference_sync_helper", _HELPER)
assert _SPEC is not None and _SPEC.loader is not None
_api_ref = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _api_ref
_SPEC.loader.exec_module(_api_ref)

_MODEL = _api_ref.load_reference(Path(__file__).with_name("apis_reference.py"))
_POLICY = _api_ref.Policy()

try:
    importlib.metadata.distribution("scikit-plots")
except importlib.metadata.PackageNotFoundError:
    _INSTALLED_DISTRIBUTION = False
else:
    _INSTALLED_DISTRIBUTION = True


def test_api_reference_matches_canonical_template() -> None:
    """The generated file must be reproducible without importing Scikit-Plots."""

    path = Path(__file__).with_name("apis_reference.py")
    assert path.read_text(encoding="utf-8") == _api_ref.build_reference_source()


def test_api_reference_structure_is_valid() -> None:
    """Static duplicate/class-list mistakes fail without importing Scikit-Plots."""

    report = _api_ref.inspect_reference(
        _MODEL,
        _POLICY,
        modules=set(),
        require_distribution=False,
    )
    assert not report.structural_errors, "\n".join(report.structural_errors)


@pytest.mark.parametrize("module_path", sorted(_MODEL.modules))
def test_api_reference_names_resolve(module_path: str) -> None:
    """Every documented name must resolve when its owning module is importable."""

    if not _INSTALLED_DISTRIBUTION:
        pytest.skip("scikit-plots distribution is not installed; runtime API verification requires an installed build")

    report = _api_ref.inspect_reference(
        _MODEL,
        _POLICY,
        modules={module_path},
        require_distribution=True,
    )
    result = report.modules[module_path]
    if result.import_error:
        pytest.skip(f"{module_path} not importable in this environment: {result.import_error}")

    problems = []
    if result.stale:
        problems.append("stale autosummary names: " + ", ".join(result.stale))
    if result.stale_classes:
        problems.append("stale classes names: " + ", ".join(result.stale_classes))
    assert not problems, f"{module_path}: " + "; ".join(problems)

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "tools" / "maint_tools" / "generate_affiliated_index.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("generate_affiliated_index", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_affiliated_index_is_current():
    helper = _load_helper()
    current, _expected = helper.check(ROOT)
    assert current


def test_every_partial_distribution_is_rendered_once():
    helper = _load_helper()
    distributions, _registry, _packages = helper._metadata(ROOT)
    source = helper.render(ROOT)
    registry_block = source.split("Partial-distribution registry", 1)[1].split(
        "Install from a checkout", 1
    )[0]
    for dist in distributions.DISTRIBUTIONS:
        assert registry_block.count(f"``{dist.name}``") == 1


def test_registry_and_lib_directories_agree():
    helper = _load_helper()
    distributions, registry, _packages = helper._metadata(ROOT)
    expected = {registry.directory_of(dist.name) for dist in distributions.DISTRIBUTIONS}
    actual = {
        path.name
        for path in (ROOT / "libs").iterdir()
        if path.is_dir() and (path / "pyproject.toml").is_file()
    }
    assert actual == expected


def test_render_is_import_safe_and_deterministic():
    helper = _load_helper()
    first = helper.render(ROOT)
    second = helper.render(ROOT)
    assert first == second
    assert "import scikitplot" not in first
    assert "scikit-plots-skinny" in first
    assert "libs/_tools" in first

"""
Tests for ``_sphinx_collection.contract`` and the package's lazy exports.

The contract constants are consumed by the Python directives, the generated
JavaScript and the post-build integrity check, so they must stay mutually
consistent and importable without Sphinx or docutils.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOST_ROOT = ROOT.parents[2]
EXTERNALS = HOST_ROOT / "scikitplot" / "_externals"


def _module(name: str = ""):
    externals = str(EXTERNALS)
    if externals not in sys.path:
        sys.path.insert(0, externals)
    return importlib.import_module("_sphinx_ext._sphinx_collection" + name)


@pytest.fixture(scope="module")
def contract():
    return _module(".contract")


@pytest.fixture(scope="module")
def package():
    return _module()


EXPECTED = {
    "COLLECTION_UI_CONTRACT": "controls-status-results-v4",
    "CONTRACT_CLASS": "sk-collection-controls-status-results-v4",
    "STATUS_CLASS": "sk-collection-status",
    "STATUS_SOURCE_ATTRIBUTE": "data-sk-collection-status-source",
    "STATUS_SOURCE_DOCUMENT": "document",
    "STATUS_PLACEMENT_ATTRIBUTE": "data-sk-collection-status-placement",
    "STATUS_PLACEMENT_SIBLING": "sibling",
}


@pytest.mark.parametrize(("name", "value"), sorted(EXPECTED.items()), ids=sorted(EXPECTED))
def test_contract_constant(contract, name, value):
    assert getattr(contract, name) == value


def test_contract_all_lists_exactly_the_constants(contract):
    assert sorted(contract.__all__) == sorted(EXPECTED)
    assert len(set(contract.__all__)) == len(contract.__all__)


def test_contract_class_embeds_the_contract_version(contract):
    assert contract.CONTRACT_CLASS == "sk-collection-" + contract.COLLECTION_UI_CONTRACT


@pytest.mark.parametrize("name", sorted(EXPECTED), ids=sorted(EXPECTED))
def test_contract_values_are_safe_html_tokens(contract, name):
    # They are interpolated unescaped into class and attribute positions.
    assert re.fullmatch(r"[a-z][a-z0-9-]*", getattr(contract, name))


def test_contract_matches_the_browser_asset(contract):
    assets = _module(".assets")
    for value in (
        contract.COLLECTION_UI_CONTRACT,
        contract.STATUS_CLASS,
        contract.STATUS_SOURCE_ATTRIBUTE,
        contract.STATUS_PLACEMENT_ATTRIBUTE,
    ):
        assert value in assets.ASSET_JS, value


def test_contract_imports_without_sphinx_or_docutils():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import _sphinx_ext._sphinx_collection as pkg;"
        "from _sphinx_ext._sphinx_collection import contract;"
        "bad = sorted(m for m in sys.modules"
        " if m.split('.')[0] in ('sphinx', 'docutils', 'yaml', 'sphinx_design'));"
        "print(contract.COLLECTION_UI_CONTRACT); print(bad)"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(EXTERNALS)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split("\n")[:2] == ["controls-status-results-v4", "[]"]


# -- package lazy exports -----------------------------------------------------

LAZY_SOURCES = {
    "COLLECTION_UI_CONTRACT": ".contract",
    "CONTRACT_CLASS": ".contract",
    "STATUS_CLASS": ".contract",
    "CONTAINER_CLASS": ".assets",
    "SEARCHABLE_CLASS": ".assets",
    "SEARCH_VARIANTS": ".assets",
    "collection_asset_revision": ".setup",
    "collection_assets_outdated": ".setup",
    "ensure_assets": ".setup",
    "register_collection_asset_revision": ".setup",
    "remember_collection_asset_revision": ".setup",
    "verify_collection_assets": ".setup",
    "SECTION_STYLES": ".sections",
    "render_sections": ".sections",
    "sections_allowed": ".sections",
    "FilterError": ".select",
    "Selection": ".select",
    "apply_selection": ".select",
    "group_records": ".select",
    "has_field": ".select",
    "parse_filter": ".select",
}


def test_package_all_is_the_documented_export_set(package):
    assert sorted(package.__all__) == sorted(LAZY_SOURCES)
    assert len(set(package.__all__)) == len(package.__all__)


@pytest.mark.parametrize(
    ("name", "source"), sorted(LAZY_SOURCES.items()), ids=sorted(LAZY_SOURCES)
)
def test_lazy_export_resolves_to_the_owning_module(package, name, source):
    value = getattr(package, name)
    assert value is getattr(_module(source), name)
    # A second lookup returns the identical object.
    assert getattr(package, name) is value


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("does_not_exist", id="unknown-name"),
        pytest.param("parse_sort", id="not-re-exported"),
        pytest.param("__wrapped__", id="dunder-probe"),
        pytest.param("", id="empty-name"),
    ],
)
def test_unknown_package_attribute_raises_attribute_error(package, name):
    with pytest.raises(AttributeError):
        getattr(package, name)
    assert hasattr(package, name) is False


def test_from_import_of_lazy_names_works(package):
    namespace: dict = {}
    exec(  # noqa: S102 - fixed literal source, exercising the import protocol
        "from _sphinx_ext._sphinx_collection import Selection, STATUS_CLASS",
        namespace,
    )
    assert namespace["Selection"] is _module(".select").Selection
    assert namespace["STATUS_CLASS"] == "sk-collection-status"


def test_asset_class_constants(package):
    assert package.CONTAINER_CLASS == "sk-collection"
    assert package.SEARCHABLE_CLASS == "sk-collection-searchable"
    assert package.SEARCH_VARIANTS == ("pill-overflow", "classic")

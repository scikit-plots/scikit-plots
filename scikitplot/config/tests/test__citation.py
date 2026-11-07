# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :func:`scikitplot.config.citation`.

The module under test is generated at build time from ``_citation.py.in``.
These tests render the template the way ``config/meson.build`` does
(``@CITATION@`` from ``CITATION.bib``, ``@CITATION_FIELDS@`` from
``tools/citation/citation_ref.py fields``), load it inside a stand-in package
whose ``version`` module is written by the real ``gitversion.py`` template,
and compare ``citation("build")`` with the version entry the citation tool
renders for the same version, commit and date. No compiled code is needed.

Notes
-----
**Developer notes.** The comparison with the tool is the contract: the
installed package and ``CITATION.bib`` write the same entry for one build.
Every version kind, a release with and without a known DOI, and a build
without git are covered.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "_citation.py.in"
MESON = HERE.parent / "meson.build"
REPO = HERE.parents[2]
TOOL = REPO / "tools" / "citation" / "citation_ref.py"
GITVERSION = REPO / "scikitplot" / "_build_utils" / "gitversion.py"
COMMIT = "b656b70f3c2d9e8a1b4c5d6e7f8091a2b3c4d5e6"
_ENTRY = re.compile(r"^@software\{.*?^\}\n", re.MULTILINE | re.DOTALL)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    if not TOOL.is_file() or not GITVERSION.is_file():  # pragma: no cover
        pytest.skip("not a source checkout")
    return _load("_skplt_test_citation_ref", TOOL)


@pytest.fixture(scope="module")
def gitversion():
    if not GITVERSION.is_file():  # pragma: no cover - not a source checkout
        pytest.skip("not a source checkout")
    return _load("_skplt_test_gitversion", GITVERSION)


@pytest.fixture
def make_citation(tmp_path, monkeypatch, tool, gitversion):
    """Return ``build(version, commit, date)`` -> the rendered module."""

    def build(version, commit=None, date=None, *, version_source=None):
        source = TEMPLATE.read_text(encoding="utf-8")
        source = source.replace(
            "@CITATION_FIELDS@", json.dumps(tool.build_fields(), sort_keys=True)
        )
        source = source.replace(
            "@CITATION@", (REPO / "CITATION.bib").read_text(encoding="utf-8").strip()
        )
        package = f"_skplt_cite_{len(list(tmp_path.iterdir()))}"
        root = tmp_path / package
        (root / "config").mkdir(parents=True)
        path = root / "config" / "_citation.py"
        path.write_text(source, encoding="utf-8")
        if version_source is None:
            info = gitversion.GitVersionInfo(
                version, commit or "", f"{date}T00:00:00Z" if date else ""
            )
            version_source = gitversion.generate_version_template(info)
        (root / "version.py").write_text(version_source, encoding="utf-8")
        for name, where in ((package, root), (f"{package}.config", root / "config")):
            module = types.ModuleType(name)
            module.__path__ = [str(where)]
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setitem(
            sys.modules,
            f"{package}.version",
            _load(f"{package}.version", root / "version.py"),
        )
        sys.modules[package].version = sys.modules[f"{package}.version"]
        spec = importlib.util.spec_from_file_location(
            f"{package}.config._citation", path
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    return build


def _tool_entry(tool, version, commit, date):
    bib = tool.render(tool.Provenance(version, commit, date))["bib"]
    entries = _ENTRY.findall(bib)
    assert len(entries) == 2  # noqa: PLR2004 - all versions, then this version
    return entries[1]


@pytest.mark.parametrize(
    ("version", "commit", "date"),
    [
        ("0.5.dev0", COMMIT, "2026-10-28"),  # development
        ("0.5.0rc1", COMMIT, "2026-11-01"),  # pre-release
        ("0.4.0.post11", COMMIT, "2026-02-17"),  # post-release
        ("0.4.0", COMMIT, "2025-01-15"),  # release with a known DOI
        ("0.6.0", COMMIT, "2027-03-02"),  # release whose DOI is not known yet
        ("0.5.dev0", None, None),  # development, built without git
        ("0.4.0", None, None),  # release, built without git
    ],
)
def test_the_build_entry_is_the_one_the_tool_writes(
    make_citation, tool, version, commit, date
):
    module = make_citation(version, commit, date)
    assert module.citation("build") == _tool_entry(tool, version, commit, date)


def test_the_build_entry_names_the_installed_version_and_tree(make_citation):
    entry = make_citation("0.5.dev0", COMMIT, "2026-10-28").citation("build")
    assert "version  = { 0.5.dev0 }," in entry
    assert (
        f"url      = {{ https://github.com/scikit-plots/scikit-plots/tree/{COMMIT} }},"
        in entry
    )
    assert "build 0.5.dev0+git.20261028.b656b70" in entry
    assert "year     = { 2026 }," in entry


def test_a_build_without_git_has_no_date_and_cites_the_repository(make_citation):
    entry = make_citation("0.5.dev0").citation("build")
    assert "year" not in entry
    assert "month" not in entry
    assert "url      = { https://github.com/scikit-plots/scikit-plots }," in entry
    assert "generated without git: no commit recorded" in entry


def test_only_a_known_release_gets_its_own_doi(make_citation):
    assert "doi      = { 10.5281/zenodo.13367095 }," in make_citation(
        "0.4.0", COMMIT, "2025-01-15"
    ).citation("build")
    assert "  doi      =" not in make_citation("0.6.0", COMMIT, "2027-03-02").citation(
        "build"
    )


def test_all_is_the_default_and_is_bibtex_unchanged(make_citation):
    module = make_citation("0.5.dev0", COMMIT, "2026-10-28")
    assert module.citation() is module.__bibtex__
    assert module.citation("all") is module.__citation__
    assert module.__bibtex__.startswith("@software{The-scikit-plots-developers_2024,")
    assert "citation" in module.__all__


@pytest.mark.parametrize(
    ("which", "error"), [(None, TypeError), (b"all", TypeError), ("latest", ValueError)]
)
def test_an_unknown_which_is_refused(make_citation, which, error):
    with pytest.raises(error, match="which must"):
        make_citation("0.5.dev0").citation(which)


@pytest.mark.parametrize(
    ("lines", "match"),
    [
        ('git_revision = "' + COMMIT + '"\n__version_iso_8601__ = ""\n', "without"),
        ('git_revision = ""\n__version_iso_8601__ = "2026.10.28"\n', "without"),
        ('git_revision = "XYZ"\n__version_iso_8601__ = "2026.10.28"\n', "commit"),
        (
            'git_revision = "' + COMMIT + '"\n__version_iso_8601__ = "28.10.2026"\n',
            "date",
        ),
    ],
)
def test_an_inconsistent_version_module_is_refused(make_citation, lines, match):
    source = 'short_version = "0.5.dev0"\nfull_version = "0.5.dev0"\n' + lines
    module = make_citation("0.5.dev0", version_source=source)
    with pytest.raises(RuntimeError, match=match):
        module.citation("build")


def test_a_version_that_is_not_pep440_is_refused(make_citation):
    source = (
        'short_version = "nightly"\nfull_version = "nightly"\n'
        'git_revision = ""\n__version_iso_8601__ = ""\n'
    )
    with pytest.raises(RuntimeError, match="PEP 440"):
        make_citation("0.5.dev0", version_source=source).citation("build")


def test_the_pep440_grammar_is_the_tools(make_citation, tool):
    module = make_citation("0.5.dev0")
    assert module._PEP440.pattern == tool._PEP440.pattern
    assert module._PEP440.flags == tool._PEP440.flags


def test_the_fields_survive_meson_substitution(tool):
    line = json.dumps(tool.build_fields(), sort_keys=True)
    assert "@" not in line  # Meson would read @NAME@ as a variable
    assert '"""' not in line
    assert "\\" not in line
    assert "\n" not in line
    assert set(json.loads(line)) == {*tool.BUILD_FIELD_KEYS, "version_dois"}


def test_the_fields_command_prints_build_fields(tool, capsys):
    assert tool.main(["fields"]) == 0
    assert json.loads(capsys.readouterr().out) == tool.build_fields()


def test_meson_substitutes_both_placeholders():
    text = MESON.read_text(encoding="utf-8")
    assert "conf_data_bib.set('CITATION_FIELDS', _citation_fields)" in text
    assert "'citation_ref.py', 'fields']" in text
    template = TEMPLATE.read_text(encoding="utf-8")
    assert set(re.findall(r"@([A-Za-z0-9_]+)@", template)) == {
        "CITATION",
        "CITATION_FIELDS",
    }

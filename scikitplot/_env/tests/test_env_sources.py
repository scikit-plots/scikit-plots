# scikitplot/_env/tests/test_env_sources.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency sources: requirements, environment.yml, pyproject, Pipfile."""

from __future__ import annotations

from pathlib import Path

import pytest

from .._core._errors import SourceError
from .._core._request import EnvironmentRequest, SourceRef
from .._sources._readers import conda_name, pypi_name, read_sources
from ._helpers import write

FIXTURES = Path(__file__).parent


def _read(*sources, **fields):
    return read_sources(EnvironmentRequest(name="t", sources=tuple(sources), **fields))


def test_names():
    assert pypi_name("Foo_Bar.baz[x]>=1") == "foo-bar-baz"
    assert conda_name("conda-forge::NumPy>=2") == "numpy"


class TestRequirements:
    def test_includes_and_options(self, tmp_path):
        write(tmp_path / "base.txt", "six==1.17.0  # pinned\n--index-url https://x/simple\n")
        main = write(tmp_path / "requirements.txt", "-r base.txt\n\n# c\ntomli-w\n")
        deps = _read(SourceRef(str(main), "requirements"))
        assert [d.spec for d in deps.dependencies] == ["six==1.17.0", "tomli-w"]
        assert deps.dependencies[0].origin == str(tmp_path / "base.txt")
        assert deps.requirement_files == (str(main),)
        assert [p for p, _ in deps.inputs] == [str(main.resolve()), str(
            (tmp_path / "base.txt").resolve())]

    def test_include_cycle(self, tmp_path):
        write(tmp_path / "a.txt", "-r b.txt\n")
        write(tmp_path / "b.txt", "-r a.txt\n")
        with pytest.raises(SourceError, match="includes itself"):
            _read(SourceRef(str(tmp_path / "a.txt"), "requirements"))

    def test_missing_file(self, tmp_path):
        with pytest.raises(SourceError, match="does not exist"):
            _read(SourceRef(str(tmp_path / "nope.txt"), "requirements"))

    def test_edit_changes_input_digest(self, tmp_path):
        path = write(tmp_path / "r.txt", "six\n")
        before = _read(SourceRef(str(path), "requirements")).inputs
        write(path, "six\ntomli-w\n")
        assert _read(SourceRef(str(path), "requirements")).inputs != before


class TestEnvironmentFile:
    def test_original_fixture(self):
        pytest.importorskip("yaml")
        deps = _read(SourceRef(str(FIXTURES / "environment-1.yml"), "environment"))
        assert [d.spec for d in deps.of("conda")] == [
            "numpy", "matplotlib", "pillow", "ipywidgets"]
        assert [d.spec for d in deps.of("pypi")] == ["ipycanvas", "py2vega"]
        assert any("'name' ignored" in n for n in deps.notes)

    def test_local_path_pip_entry(self):
        pytest.importorskip("yaml")
        path = FIXTURES / "test_package" / "environment-3.yml"
        deps = _read(SourceRef(str(path), "environment"))
        assert deps.of("pypi")[0].spec == str((FIXTURES / "test_package").resolve())

    def test_python_pin_channels_editable_include(self, tmp_path):
        pytest.importorskip("yaml")
        write(tmp_path / "req.txt", "six\n")
        write(tmp_path / "pkg" / "pyproject.toml", "[project]\nname='p'\n")
        env = write(tmp_path / "environment.yml", (
            "channels: [conda-forge]\n"
            "dependencies:\n  - python=3.11\n  - numpy\n"
            "  - pip:\n    - -r req.txt\n    - -e ./pkg\n"))
        deps = _read(SourceRef(str(env), "environment"))
        assert deps.python == "3.11"
        assert deps.channels == ("conda-forge",)
        assert deps.requirement_files == (str((tmp_path / "req.txt").resolve()),)
        assert f"-e {(tmp_path / 'pkg').resolve()}" in [d.spec for d in deps.of("pypi")]
        assert len(deps.inputs) == 2

    @pytest.mark.parametrize(("text", "match"), [
        ("dependencies:\n  - pip:\n    - --index-url https://x\n", "pip option"),
        ("variables:\n  A: 1\ndependencies: [numpy]\n", "unsupported"),
        ("dependencies:\n  - {conda: x}\n", "unsupported dependency"),
        ("- numpy\n", "mapping"),
    ])
    def test_refused_rather_than_dropped(self, tmp_path, text, match):
        pytest.importorskip("yaml")
        env = write(tmp_path / "environment.yml", text)
        with pytest.raises(SourceError, match=match):
            _read(SourceRef(str(env), "environment"))


class TestPyproject:
    TEXT = """
[project]
name = "p"
dependencies = ["requests>=2"]
requires-python = ">=3.10"
[project.optional-dependencies]
plot = ["matplotlib"]
[dependency-groups]
test = ["pytest", {include-group = "lint"}]
lint = ["ruff"]
"""

    def test_groups_and_extras(self, tmp_path):
        path = write(tmp_path / "pyproject.toml", self.TEXT)
        deps = _read(SourceRef(str(path), "pyproject", groups=("plot", "test")))
        assert [d.spec for d in deps.dependencies] == [
            "requests>=2", "matplotlib", "pytest", "ruff"]
        assert any("requires-python" in n for n in deps.notes)

    def test_unknown_group(self, tmp_path):
        path = write(tmp_path / "pyproject.toml", self.TEXT)
        with pytest.raises(SourceError, match="docs"):
            _read(SourceRef(str(path), "pyproject", groups=("docs",)))


class TestPipfile:
    TEXT = """
[packages]
requests = "*"
six = ">=1.16"
tool = {git = "https://example.org/tool.git", ref = "v1"}
[dev-packages]
pytest = {version = ">=8", markers = "python_version >= '3.9'"}
[requires]
python_version = "3.12"
"""

    def test_packages_and_dev(self, tmp_path):
        path = write(tmp_path / "Pipfile", self.TEXT)
        deps = _read(SourceRef(str(path), "pipfile", groups=("dev",)))
        assert [d.spec for d in deps.dependencies] == [
            "requests", "six>=1.16", "tool @ git+https://example.org/tool.git@v1",
            "pytest>=8; python_version >= '3.9'"]
        assert deps.python == "3.12"
        assert deps.pipfiles == (str(path),)

    def test_bad_group(self, tmp_path):
        path = write(tmp_path / "Pipfile", self.TEXT)
        with pytest.raises(SourceError, match="only 'dev'"):
            _read(SourceRef(str(path), "pipfile", groups=("test",)))

    def test_conflicting_python_pins(self, tmp_path):
        pytest.importorskip("yaml")
        pipfile = write(tmp_path / "Pipfile", self.TEXT)
        env = write(tmp_path / "environment.yml", "dependencies:\n  - python=3.11\n")
        with pytest.raises(SourceError, match="different Python"):
            _read(SourceRef(str(pipfile), "pipfile"), SourceRef(str(env), "environment"))


def test_inline_packages_first():
    deps = read_sources(EnvironmentRequest(name="t", packages=["a"], conda_packages=["b"]))
    assert [(d.spec, d.ecosystem, d.origin) for d in deps.dependencies] == [
        ("a", "pypi", "packages"), ("b", "conda", "conda_packages")]


class TestReviewFindings:
    def test_diamond_includes_and_constraints(self, tmp_path):
        write(tmp_path / "common.txt", "six\n")
        write(tmp_path / "a.txt", "-r common.txt\n")
        write(tmp_path / "b.txt", "-r common.txt\n")
        write(tmp_path / "c.txt", "six<2\n")
        main = write(tmp_path / "requirements.txt", "-r a.txt\n-r b.txt\n-c c.txt\n")
        deps = _read(SourceRef(str(main), "requirements"))
        assert [d.spec for d in deps.dependencies] == ["six"]  # read once
        assert str((tmp_path / "c.txt").resolve()) in [p for p, _ in deps.inputs]
        before = deps.inputs
        write(tmp_path / "c.txt", "six<1.17\n")
        assert _read(SourceRef(str(main), "requirements")).inputs != before

    def test_missing_include_group_and_dynamic(self, tmp_path):
        path = write(tmp_path / "pyproject.toml", (
            '[project]\nname = "p"\n[dependency-groups]\n'
            'test = [{include-group = "lint"}]\n'))
        with pytest.raises(SourceError, match="'lint' does not exist"):
            _read(SourceRef(str(path), "pyproject", groups=("test",)))
        write(path, '[project]\nname = "p"\ndynamic = ["dependencies"]\n')
        with pytest.raises(SourceError, match="dynamic"):
            _read(SourceRef(str(path), "pyproject"))

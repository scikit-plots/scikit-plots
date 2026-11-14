# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ``generate_apis_reference/_static_coverage.py`` (Rule 172)."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "generate_apis_reference"
_SPEC = importlib.util.spec_from_file_location(
    "_generate_apis_reference_cov",
    PACKAGE / "__init__.py",
    submodule_search_locations=[str(PACKAGE)],
)
assert _SPEC is not None
assert _SPEC.loader is not None
api_ref = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = api_ref
_SPEC.loader.exec_module(api_ref)
sc = importlib.import_module(f"{_SPEC.name}._static_coverage")


def _module(root: Path, dotted: str, source: str) -> None:
    parts = dotted.split(".")
    path = root.joinpath(*parts) / "__init__.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


class TestStaticAll:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("__all__ = ['a', 'b']", ["a", "b"]),
            ("__all__ = ('a',)\n__all__ += ['b']", ["a", "b"]),
            (
                "__all__ = ['a']\n__all__.extend(['b'])\n__all__.append('c')",
                ["a", "b", "c"],
            ),
            ("_X = ['b', 'a']\n__all__ = sorted(_X)", ["a", "b"]),
            (
                "__all__ = ['a', 'a', 'b']\n__all__ = list(dict.fromkeys(__all__))",
                ["a", "b"],
            ),
            ("__all__ = [*['a'], 'b'] + ['c']", ["a", "b", "c"]),
            ("__all__: list = ['a']", ["a"]),
        ],
    )
    def test_literal_forms(self, tmp_path, source, expected):
        _module(tmp_path, "pkg", source)
        assert sc.static_all(tmp_path, "pkg") == expected

    def test_a_submodules_all_is_followed(self, tmp_path):
        _module(
            tmp_path,
            "pkg",
            "from . import _a\nfrom .sub import _b\n__all__ = []\n__all__ += _a.__all__\n__all__ += _b.__all__\n",
        )
        (tmp_path / "pkg" / "_a.py").write_text("__all__ = ['A']\n", encoding="utf-8")
        _module(tmp_path, "pkg.sub", "")
        (tmp_path / "pkg" / "sub" / "_b.py").write_text(
            "__all__ = ['B']\n", encoding="utf-8"
        )
        assert sc.static_all(tmp_path, "pkg") == ["A", "B"]

    @pytest.mark.parametrize(
        ("source", "reason"),
        [
            ("x = 1", "no __all__"),
            ("__all__ = [n for n in dir()]", "not read statically"),
            ("__all__ = [1]", "not a string"),
            ("__all__ = unknown.__all__", "unknown name"),
            ("__all__ = []\n__all__.insert(0, 'a')", "__all__.insert"),
        ],
    )
    def test_unreadable_forms_are_named(self, tmp_path, source, reason):
        _module(tmp_path, "pkg", source)
        with pytest.raises(sc.UnresolvedError, match=reason):
            sc.static_all(tmp_path, "pkg")

    def test_a_module_without_source(self, tmp_path):
        with pytest.raises(sc.UnresolvedError, match="no Python source"):
            sc.static_all(tmp_path, "pkg.compiled")

    def test_a_cycle_ends(self, tmp_path):
        _module(tmp_path, "pkg", "from . import a\n__all__ = a.__all__\n")
        (tmp_path / "pkg" / "a.py").write_text(
            "from pkg import a as me\n__all__ = me.__all__\n", encoding="utf-8"
        )
        with pytest.raises(sc.UnresolvedError, match="refers to itself"):
            sc.static_all(tmp_path, "pkg")


REFERENCE = """
APIS_REFERENCE = {
    "pkg": {
        "sections": [
            {"title": "One", "autosummary": ["a", "gone"]},
            {"title": "Two", "autosummary": ["b"], "exclude": ["hidden"]},
        ],
    },
    "nopkg": {"sections": [{"title": "X", "autosummary": ["z"]}]},
}
"""


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.scikitplot.maintenance.api_reference]\n"
        "reference_file = 'docs/source/apis_reference.py'\n",
        encoding="utf-8",
    )
    ref = tmp_path / "docs" / "source" / "apis_reference.py"
    ref.parent.mkdir(parents=True)
    ref.write_text(REFERENCE, encoding="utf-8")
    _module(
        tmp_path,
        "pkg",
        "__all__ = ['a', 'b', 'c', 'hidden', '_inner', '__version__']\n",
    )
    return tmp_path


class TestCoverage:
    def test_missing_stale_excluded_and_unresolved(self, repo):
        model = api_ref.load_reference(repo / "docs" / "source" / "apis_reference.py")
        report = sc.coverage(repo, model.modules)
        assert report.gaps() == {
            "missing:pkg": ["__version__", "c"],
            "private:pkg": ["_inner"],
            "stale:pkg": ["gone"],
        }
        assert list(report.unresolved) == ["nopkg"]

    def test_compare_reports_new_and_fixed(self):
        new, fixed = sc.compare(
            {"missing:pkg": ["c", "d"]}, {"missing:pkg": ["c", "e"]}
        )
        assert (new, fixed) == (["missing:pkg: d"], ["missing:pkg: e"])

    def test_baseline_text_is_deterministic(self):
        one = sc.baseline_text({"stale:b": ["y", "x"], "missing:a": ["k"], "empty": []})
        assert one == sc.baseline_text({"missing:a": ["k"], "stale:b": ["x", "y"]})
        assert one.endswith("}\n")
        assert list(json.loads(one)["gaps"]) == ["missing:a", "stale:b"]


class TestCommand:
    def _run(self, repo, monkeypatch, tmp_path, *args):
        monkeypatch.setattr(
            api_ref, "default_template_path", lambda: tmp_path / "t" / "x.in"
        )
        (tmp_path / "t").mkdir(exist_ok=True)
        out = io.StringIO()
        code = api_ref.main(["--repo-root", str(repo), "coverage", *args], stdout=out)
        return code, out.getvalue()

    def test_a_new_gap_fails_until_the_baseline_records_it(
        self, repo, monkeypatch, tmp_path
    ):
        code, text = self._run(repo, monkeypatch, tmp_path)
        assert code == 1
        assert "NEW      missing:pkg: c" in text
        assert "NEW      stale:pkg: gone" in text
        assert self._run(repo, monkeypatch, tmp_path, "--write")[0] == 0
        baseline = (tmp_path / "t" / sc.BASELINE_NAME).read_text(encoding="utf-8")
        assert self._run(repo, monkeypatch, tmp_path)[0] == 0
        assert self._run(repo, monkeypatch, tmp_path, "--write")[0] == 0
        assert (tmp_path / "t" / sc.BASELINE_NAME).read_text(
            encoding="utf-8"
        ) == baseline

    def test_a_fixed_gap_fails_until_the_baseline_drops_it(
        self, repo, monkeypatch, tmp_path
    ):
        self._run(repo, monkeypatch, tmp_path, "--write")
        _module(repo, "pkg", "__all__ = ['a', 'b', 'hidden', 'gone', '__version__']\n")
        code, text = self._run(repo, monkeypatch, tmp_path)
        assert code == 1
        assert "FIXED    missing:pkg: c" in text
        assert "FIXED    stale:pkg: gone" in text
        assert "FIXED    private:pkg: _inner" in text

    def test_json(self, repo, monkeypatch, tmp_path):
        code, text = self._run(repo, monkeypatch, tmp_path, "--json")
        data = json.loads(text)
        assert code == 1
        assert data["new"] == [
            "missing:pkg: __version__",
            "missing:pkg: c",
            "private:pkg: _inner",
            "stale:pkg: gone",
        ]
        assert list(data["unresolved"]) == ["nopkg"]


def test_the_repository_has_no_new_gap():
    """The real tree against its committed baseline (Rule 172)."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "docs" / "source" / "apis_reference.py").is_file():
        pytest.skip("not a source checkout (installed package)")
    out = io.StringIO()
    code = api_ref.main(["--repo-root", str(root), "coverage"], stdout=out)
    assert code == 0, out.getvalue()


def test_the_script_entry_point_runs_without_the_package(tmp_path):
    """The pre-commit hook's command: no ``scikitplot`` import, exit 0."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "docs" / "source" / "apis_reference.py").is_file():
        pytest.skip("not a source checkout (installed package)")
    script = PACKAGE / "_static_coverage.py"
    probe = (
        "import runpy, sys; sys.argv = ['x']; "
        f"code = 0\ntry:\n    runpy.run_path({str(script)!r}, run_name='__main__')\n"
        "except SystemExit as e:\n    code = e.code\n"
        "print(code, 'scikitplot' in sys.modules)"
    )
    done = subprocess.run(  # noqa: S603 - fixed command
        [sys.executable, "-I", "-B", "-c", probe],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.stdout.strip().splitlines()[-1] == "0 False", done.stdout + done.stderr


class TestLazyExports:
    def test_type_checking_imports_of_a_lazy_module(self, tmp_path):
        _module(
            tmp_path,
            "pkg",
            """
            from typing import TYPE_CHECKING
            import typing
            __all__ = ["a"]
            if TYPE_CHECKING:
                from typing import Optional
                import numpy
                from ._opt import Heavy, helper as alias
            if typing.TYPE_CHECKING:
                import pkg.other
            def __getattr__(name):
                raise AttributeError(name)
            """,
        )
        assert sc.static_lazy(tmp_path, "pkg") == ["Heavy", "alias", "other"]

    def test_without_getattr_nothing_is_lazy(self, tmp_path):
        _module(
            tmp_path,
            "pkg",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from x import Y\n",
        )
        assert sc.static_lazy(tmp_path, "pkg") == []

    def test_lazy_names_are_neither_stale_nor_left_undocumented(self, repo):
        _module(
            repo,
            "pkg",
            """
            from typing import TYPE_CHECKING
            __all__ = ['a', 'b', 'hidden']
            if TYPE_CHECKING:
                from ._opt import gone, extra
            def __getattr__(name):
                raise AttributeError(name)
            """,
        )
        model = api_ref.load_reference(repo / "docs" / "source" / "apis_reference.py")
        assert sc.coverage(repo, model.modules).gaps() == {"missing:pkg": ["extra"]}


class TestDottedEntries:
    def test_submodule_and_name_through_a_submodule(self, tmp_path):
        _module(tmp_path, "pkg", "__all__ = []\n")
        _module(
            tmp_path,
            "pkg._sub",
            "__all__ = ['Exported']\nclass Defined: pass\nVALUE = 1\n",
        )
        (tmp_path / "pkg" / "_mod.py").write_text(
            "from x import thing as alias\n", encoding="utf-8"
        )
        exists = sc.documented_target_exists
        assert exists(tmp_path, "pkg", "_sub")
        assert exists(tmp_path, "pkg", "_mod")
        assert exists(tmp_path, "pkg", "_sub.Exported")
        assert exists(tmp_path, "pkg", "_sub.Defined")
        assert exists(tmp_path, "pkg", "_sub.VALUE")
        assert exists(tmp_path, "pkg", "_mod.alias")
        assert not exists(tmp_path, "pkg", "_sub.Gone")
        assert not exists(tmp_path, "pkg", "_missing")
        assert not exists(tmp_path, "pkg", "_missing.Name")

    def test_names_through_a_star_import(self, tmp_path):
        _module(tmp_path, "pkg", "__all__ = []\n")
        _module(tmp_path, "pkg._sub", "from ._impl import *\nfrom ._loop import *\n")
        (tmp_path / "pkg" / "_sub" / "_impl.py").write_text(
            "__all__ = ['listed', '_private_listed']\n"
            "def listed(): pass\ndef _private_listed(): pass\n"
            "def _private_unlisted(): pass\n",
            encoding="utf-8",
        )
        (tmp_path / "pkg" / "_sub" / "_loop.py").write_text(
            "from ._loop import *\n", encoding="utf-8"
        )
        exists = sc.documented_target_exists
        assert exists(tmp_path, "pkg", "_sub.listed")
        assert exists(tmp_path, "pkg", "_sub._private_listed")
        assert not exists(tmp_path, "pkg", "_sub._private_unlisted")
        assert not exists(tmp_path, "pkg", "_sub.absent")

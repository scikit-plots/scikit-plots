"""
``scikitplot._api_listing`` and ``python -m scikitplot api``.

A synthetic package is mounted below ``scikitplot`` (on ``scikitplot.__path__``)
so every case is controlled: a class, an enum, a protocol, an exception, a
long signature, a pipe in a summary, a lazy submodule, a name that does not
resolve, and a module whose optional dependency is missing.
"""

from __future__ import annotations

import io
import json
import sys
import textwrap

import pytest

import scikitplot
from scikitplot import _api_listing as api

PKG = "scikitplot._apitestpkg"

_INIT = '''
"""A synthetic package. Second sentence, not shown.

More text.
"""
import enum
import typing

__all__ = [
    "Color", "Shape", "Proto", "Oops", "long_function", "piped", "lazy",
    "missing", "CONSTANT",
]


class Color(enum.Enum):
    """Colors."""

    RED = 1


class Shape:
    """A shape with a :class:`~scikitplot.corpus.ZipReader` role.

    Parameters
    ----------
    size : int
    """

    def __init__(self, size: int = 1, *, name: str | None = None) -> None:
        self.size = size


class Proto(typing.Protocol):
    """A protocol."""

    def read(self) -> str: ...


class Oops(ValueError):
    """A failure."""


def long_function(first_parameter_name: int = 1, second_parameter_name: str = "x",
                  third_parameter_name: float = 2.5, fourth: tuple = (1, 2)) -> dict:
    """Do something long."""
    return {}


def piped(a: int | None = None) -> str:
    """Return a | b."""
    return ""


CONSTANT = 3


def __getattr__(name):
    if name == "missing":
        raise AttributeError("missing is not built here")
    raise AttributeError(name)
'''

_LAZY = '''
"""A lazy submodule: importing it is recorded."""
import sys
sys.modules.setdefault("_apitestpkg_lazy_imported", True)
__all__ = []
'''

_NEEDS_DEP = '''
"""Needs an optional dependency."""
import not_installed_optional_dependency_xyz  # noqa: F401
__all__ = []
'''


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    """Mount the synthetic package as ``scikitplot._apitestpkg``."""
    root = tmp_path / "_apitestpkg"
    root.mkdir()
    (root / "__init__.py").write_text(textwrap.dedent(_INIT), encoding="utf-8")
    (root / "lazy.py").write_text(textwrap.dedent(_LAZY), encoding="utf-8")
    (root / "noall.py").write_text('"""No __all__."""\nx = 1\n', encoding="utf-8")
    (root / "needs_dep.py").write_text(textwrap.dedent(_NEEDS_DEP), encoding="utf-8")
    monkeypatch.setattr(scikitplot, "__path__", [*scikitplot.__path__, str(tmp_path)])
    for name in [m for m in sys.modules if m.startswith(PKG)] + ["_apitestpkg_lazy_imported"]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield PKG
    for name in [m for m in sys.modules if m.startswith(PKG)] + ["_apitestpkg_lazy_imported"]:
        sys.modules.pop(name, None)


def _rows(listing):
    return {r["name"]: r for r in listing["rows"]}


class TestListing:
    def test_rows_follow_all_and_describe_each_kind(self, pkg):
        listing = api.module_listing(pkg)
        assert [r["name"] for r in listing["rows"]] == [
            "Color", "Shape", "Proto", "Oops", "long_function", "piped", "lazy",
            "missing", "CONSTANT",
        ]
        rows = _rows(listing)
        assert {n: r["kind"] for n, r in rows.items()} == {
            "Color": "enum", "Shape": "class", "Proto": "protocol",
            "Oops": "exception", "long_function": "function", "piped": "function",
            "lazy": "module", "missing": "unavailable", "CONSTANT": "int",
        }
        assert rows["Color"]["signature"] == "Color(value)"
        assert rows["Proto"]["signature"] == "Proto"
        assert rows["Shape"]["signature"] == (
            "Shape(size: int = 1, *, name: str | None = None) -> None"
        )
        assert rows["Shape"]["summary"] == "A shape with a `ZipReader` role."
        assert rows["Oops"]["signature"] == "Oops(...)"
        assert listing["doc"] == "A synthetic package. Second sentence, not shown."
        assert listing["version"] == scikitplot.__version__

    def test_a_long_signature_is_cut_in_a_listing_and_whole_in_detail(self, pkg):
        cut = _rows(api.module_listing(pkg))["long_function"]["signature"]
        # max_length - 4 characters, then " ...)": one over, as the skills show it.
        assert cut.endswith(" ...)")
        assert len(cut) == len("long_function") + api.SIGNATURE_MAX + 1
        whole = api.name_detail(pkg, "long_function")["signature"]
        assert whole.endswith("fourth: tuple = (1, 2)) -> dict")

    def test_a_submodule_is_listed_without_being_imported(self, pkg):
        row = _rows(api.module_listing(pkg))["lazy"]
        assert row == {"name": "lazy", "kind": "module", "signature": f"{pkg}.lazy", "summary": ""}
        assert "_apitestpkg_lazy_imported" not in sys.modules
        assert f"{pkg}.lazy" not in sys.modules

    def test_an_unresolved_name_is_reported_not_fatal(self, pkg):
        row = _rows(api.module_listing(pkg))["missing"]
        assert row["summary"] == "AttributeError: missing is not built here"

    def test_detail_gives_the_whole_docstring(self, pkg):
        detail = api.name_detail(pkg, "Shape")
        assert detail["kind"] == "class"
        assert "Parameters\n----------\nsize : int" in detail["doc"]

    def test_an_unknown_name_suggests_close_ones(self, pkg):
        with pytest.raises(api.ApiListingError) as caught:
            api.name_detail(pkg, "Shpe")
        assert caught.value.reason == "api"
        assert "Did you mean: Shape?" in str(caught.value)

    def test_detail_resolves_only_the_requested_name(self, pkg):
        # "missing" does not resolve and "lazy" is not imported: neither hides Shape.
        assert api.name_detail(pkg, "Shape")["name"] == "Shape"
        assert f"{pkg}.lazy" not in sys.modules
        with pytest.raises(api.ApiListingError, match="missing does not resolve here"):
            api.name_detail(pkg, "missing")

    @pytest.mark.parametrize(
        "module", ["os", "scikitplotx", "scikitplot..x", "scikitplot.", "", "scikitplot.a-b", None]
    )
    def test_only_scikitplot_modules_are_imported(self, module):
        with pytest.raises(api.ApiListingError) as caught:
            api.module_listing(module)
        assert caught.value.reason == "name"

    def test_an_import_failure_keeps_its_cause(self, pkg):
        with pytest.raises(api.ApiListingError) as caught:
            api.module_listing(f"{pkg}.needs_dep")
        assert caught.value.reason == "import"
        assert isinstance(caught.value.__cause__, ModuleNotFoundError)

    def test_a_module_without_all_is_refused(self, pkg):
        for call in (lambda: api.module_listing(f"{pkg}.noall"), lambda: api.name_detail(f"{pkg}.noall", "x")):
            with pytest.raises(api.ApiListingError, match="declares no __all__"):
                call()

    def test_deterministic(self, pkg):
        assert api.render_text(api.module_listing(pkg)) == api.render_text(api.module_listing(pkg))


class TestRender:
    def test_markdown_escapes_pipes(self, pkg):
        out = api.render_markdown(api.module_listing(pkg))
        row = next(line for line in out.splitlines() if line.startswith("| `piped"))
        assert row == "| `piped(a: int \\| None = None) -> str` | function | Return a \\| b. |"

    def test_text_aligns_on_the_longest_kind(self, pkg):
        out = api.render_text(api.module_listing(pkg))
        assert "unavailable missing" in out
        assert "class       Shape(" in out


def _cli(argv):
    from scikitplot._cli.app import main

    out, err = io.StringIO(), io.StringIO()
    old = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        code = main(argv)
    finally:
        sys.stdout, sys.stderr = old
    return code, out.getvalue(), err.getvalue()


class TestCommand:
    def test_text_json_markdown(self, pkg):
        code, out, _ = _cli(["api", pkg])
        assert code == 0 and "enum        Color(value)" in out
        code, out, _ = _cli(["api", pkg, "--format", "json"])
        assert code == 0 and json.loads(out)["rows"][0]["name"] == "Color"
        code, out, _ = _cli(["api", pkg, "--format", "markdown"])
        assert code == 0 and "| Name | Kind | Summary |" in out
        code, out, _ = _cli(["api", pkg, "--name", "Shape", "--format", "markdown"])
        assert code == 0 and out.startswith(f"# `{pkg}.Shape` (class, scikit-plots")

    @pytest.mark.parametrize(
        ("argv", "code", "text"),
        [
            (["api", "os"], 2, "is not scikitplot or a module below it"),
            (["api", "scikitplot._apitestpkg_nope"], 1, "may predate this module"),
            (["api", "scikitplot._apitestpkg.needs_dep"], 69, "not_installed_optional_dependency_xyz"),
            (["api", "scikitplot._apitestpkg", "--name", "Shpe"], 1, "Did you mean: Shape?"),
        ],
    )
    def test_exit_codes(self, pkg, argv, code, text):
        got, out, err = _cli(argv)
        assert got == code
        assert text in out + err

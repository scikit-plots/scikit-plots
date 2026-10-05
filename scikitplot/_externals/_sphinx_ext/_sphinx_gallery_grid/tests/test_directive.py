"""
Tests for :mod:`.directive`: the theme-independent ``gallery-grid`` directive.

Three layers are exercised, cheapest first:

* the pure helpers (quote stripping, indentation, fence sizing, YAML shape
  validation, source-tree confinement, markup-language detection);
* card and grid source generation on a directive instance wired to small
  fake Sphinx objects;
* a handful of tiny real Sphinx builds (reStructuredText, bundled
  ``alabaster`` theme) whose produced HTML and warning stream are asserted.
"""

from __future__ import annotations

import html as html_module
import itertools
import json
import logging
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import sphinx
from sphinx.errors import ConfigError, ExtensionError
from sphinx.testing.util import SphinxTestApp

from .. import directive as directive_module
from ..directive import (
    _CONFDIR_ATTRIBUTE,
    _record_confdir,
    MARKDOWN,
    MIN_CARD_FENCE,
    MIN_GRID_FENCE,
    RESTRUCTUREDTEXT,
    RST_INDENT,
    GalleryGridDirective,
    _card_option_names,
    _coerce_items,
    _confined_path,
    _fence_for,
    _indent,
    _max_run,
    _source_format,
    _strip_quotes,
)

EXTENSION = __package__.rsplit(".", 1)[0]
HOSTILE = "<script>alert(1)</script>"

#: ``_confined_path`` reads ``env.app``, which Sphinx 9 deprecates. It is
#: reported by one dedicated test below; elsewhere it must not mask the
#: behaviour under test.
ignore_env_app_deprecation = pytest.mark.filterwarnings(
    "ignore:.*BuildEnvironment.app.*"
)


# -- pure helpers -------------------------------------------------------------


class TestStripQuotes:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ('"1 2 3 4"', "1 2 3 4"),
            ("'1 2 3 4'", "1 2 3 4"),
            ("plain", "plain"),
            ('""', ""),
            ("''", ""),
            ('"', '"'),
            ("", ""),
            ("\"mixed'", "\"mixed'"),
            ('"only-leading', '"only-leading'),
            ('only-trailing"', 'only-trailing"'),
            ('""double""', '"double"'),
            ('a "quoted" word', 'a "quoted" word'),
            ("`tick`", "`tick`"),
        ],
        ids=["double", "single", "plain", "empty-double", "empty-single",
             "lone-quote", "empty", "mismatched", "leading-only", "trailing-only",
             "one-layer-only", "inner-quotes", "backticks"],
    )
    def test_one_matching_layer_is_removed(self, value, expected):
        assert _strip_quotes(value) == expected

    @pytest.mark.parametrize("value", [None, 5, 1.5, ["'a'"], ("x",), True])
    def test_a_non_string_is_returned_unchanged(self, value):
        assert _strip_quotes(value) is value


class TestIndent:
    def test_the_default_prefix_is_three_spaces(self):
        assert RST_INDENT == "   "
        assert _indent("a") == "   a"

    def test_blank_lines_gain_no_trailing_whitespace(self):
        assert _indent("a\n\nb\n   \nc") == "   a\n\n   b\n   \n   c"

    def test_existing_indentation_is_preserved(self):
        assert _indent("a\n  b") == "   a\n     b"

    def test_a_custom_prefix(self):
        assert _indent("a\nb", "> ") == "> a\n> b"

    def test_the_line_count_is_preserved(self):
        text = "one\n\ntwo\nthree"
        assert len(_indent(text).splitlines()) == len(text.splitlines())

    def test_empty_text(self):
        assert _indent("") == ""

    def test_indenting_twice_nests(self):
        assert _indent(_indent("a")) == "      a"


class TestMaxRun:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("", 0),
            ("abc", 0),
            ("~", 1),
            ("a~b~~c~~~d", 3),
            ("~~~~\n~~", 4),
            ("~~\n~~", 2),
            ("~ ~ ~", 1),
            ("```", 0),
        ],
        ids=["empty", "none", "one", "longest", "multi-line", "newline-breaks-run",
             "spaced", "other-char"],
    )
    def test_longest_run_of_tildes(self, text, expected):
        assert _max_run(text, "~") == expected

    def test_the_character_is_a_parameter(self):
        assert _max_run("a```b`", "`") == 3


class TestFenceFor:
    def test_minimum_lengths_keep_plain_content_unchanged(self):
        assert (MIN_CARD_FENCE, MIN_GRID_FENCE) == (4, 5)
        assert _fence_for("plain", MIN_CARD_FENCE) == "~~~~"
        assert _fence_for("plain", MIN_GRID_FENCE) == "~~~~~"

    @pytest.mark.parametrize("run", [1, 3, 4, 5, 9])
    def test_the_fence_is_longer_than_any_run_it_wraps(self, run):
        fence = _fence_for("x " + "~" * run + " y", MIN_CARD_FENCE)
        assert set(fence) == {"~"}
        assert len(fence) == max(MIN_CARD_FENCE, run + 1)
        assert len(fence) > run

    def test_backticks_in_the_content_do_not_matter(self):
        assert _fence_for("``````` code", MIN_CARD_FENCE) == "~~~~"

    def test_another_fence_character(self):
        assert _fence_for("a ::::: b", 3, ":") == "::::::"


class TestCoerceItems:
    def test_an_empty_body_is_an_empty_list(self):
        assert _coerce_items(None, "x") == []

    def test_an_empty_list_is_kept(self):
        assert _coerce_items([], "x") == []

    def test_a_list_of_mappings_is_returned_as_is(self):
        payload = [{"title": "a"}, {}]
        assert _coerce_items(payload, "x") is payload

    @pytest.mark.parametrize(
        "payload, name",
        [({"title": "a"}, "dict"), ("text", "str"), (5, "int"), (1.5, "float"),
         (True, "bool"), (("a",), "tuple")],
        ids=["mapping", "string", "int", "float", "bool", "tuple"],
    )
    def test_a_non_list_is_rejected_with_origin_and_type(self, payload, name):
        with pytest.raises(ValueError) as caught:
            _coerce_items(payload, "data.yaml")
        message = str(caught.value)
        assert message.startswith(f"data.yaml: expected a YAML list of items, got {name}.")
        assert "- title: My card" in message

    @pytest.mark.parametrize(
        "item, name",
        [("text", "str"), (5, "int"), (None, "NoneType"), (["nested"], "list")],
        ids=["string", "int", "null", "list"],
    )
    def test_a_non_mapping_item_is_rejected_with_its_index(self, item, name):
        with pytest.raises(ValueError) as caught:
            _coerce_items([{"title": "ok"}, {"title": "ok"}, item], "data.yaml")
        assert str(caught.value).startswith(f"data.yaml: item 2 is a {name}, ")

    def test_the_item_limit_is_inclusive(self):
        limit = directive_module.MAX_COLLECTION_ITEMS
        assert len(_coerce_items([{}] * limit, "x")) == limit

    def test_a_gallery_over_the_item_limit_is_rejected(self):
        limit = directive_module.MAX_COLLECTION_ITEMS
        with pytest.raises(ValueError) as caught:
            _coerce_items([{}] * (limit + 1), "data.yaml")
        message = str(caught.value)
        assert message.startswith(f"data.yaml: gallery contains {limit + 1:,} items")
        assert f"the limit is {limit:,}" in message


class TestCardOptionNames:
    def test_the_names_track_sphinx_design_plus_name(self):
        from sphinx_design.grids import GridItemCardDirective

        assert _card_option_names() == (
            frozenset(GridItemCardDirective.option_spec) | {"name"}
        )

    @pytest.mark.parametrize(
        "name", ["link", "link-alt", "img-top", "img-alt", "class-card", "shadow", "name"]
    )
    def test_core_card_options_are_known(self, name):
        assert name in _card_option_names()
        assert name in directive_module._FALLBACK_CARD_OPTIONS

    @pytest.mark.parametrize(
        "name", ["title", "header", "image", "content", "category", "tags", "description"]
    )
    def test_item_data_keys_are_not_card_options(self, name):
        assert name not in _card_option_names()

    def test_the_result_is_stable(self):
        assert _card_option_names() is _card_option_names()


# -- source-tree confinement --------------------------------------------------


def path_directive(srcdir, document="index.rst", confdir=None):
    """Return the minimal directive-shaped object ``_confined_path`` reads."""
    env = SimpleNamespace(srcdir=str(srcdir))
    if confdir is not None:
        # What _record_confdir stores before any document is read.
        setattr(env, _CONFDIR_ATTRIBUTE, confdir)
    return SimpleNamespace(
        env=env, get_source_info=lambda: (str(Path(srcdir) / document), 3)
    )


class TestConfinedPath:
    def test_a_sibling_file_resolves_inside_the_source_tree(self, tmp_path):
        assert _confined_path(path_directive(tmp_path), "data.yaml") == (
            tmp_path.resolve() / "data.yaml"
        )

    def test_the_reference_is_relative_to_the_current_document(self, tmp_path):
        directive = path_directive(tmp_path, "guide/page.rst")
        assert _confined_path(directive, "data.yaml") == (
            tmp_path.resolve() / "guide" / "data.yaml"
        )
        assert _confined_path(directive, "../shared/data.yaml") == (
            tmp_path.resolve() / "shared" / "data.yaml"
        )

    def test_a_missing_file_inside_the_tree_is_still_confined_not_rejected(self, tmp_path):
        assert not (tmp_path / "nope.yaml").exists()
        assert _confined_path(path_directive(tmp_path), "nope.yaml").name == "nope.yaml"

    def test_the_source_root_itself_is_inside(self, tmp_path):
        assert _confined_path(path_directive(tmp_path), ".") == tmp_path.resolve()

    @pytest.mark.parametrize(
        "reference",
        ["../outside.yaml", "../../etc/passwd", "/etc/passwd", "a/b/../../../x.yaml",
         "..", "./../x"],
        ids=["parent", "deep-parent", "absolute", "normalised", "bare-parent", "dot-parent"],
    )
    def test_a_path_outside_the_source_tree_is_refused(self, tmp_path, reference):
        source = tmp_path / "docs"
        source.mkdir()
        with pytest.raises(ValueError) as caught:
            _confined_path(path_directive(source), reference)
        message = str(caught.value)
        assert repr(reference) in message
        assert "outside the documentation source directory" in message
        assert str(source.resolve()) in message

    def test_a_sibling_directory_sharing_the_name_prefix_is_outside(self, tmp_path):
        source = tmp_path / "docs"
        source.mkdir()
        (tmp_path / "docs-private").mkdir()
        with pytest.raises(ValueError, match="outside"):
            _confined_path(path_directive(source), "../docs-private/secret.yaml")

    def test_a_symlink_leaving_the_tree_is_refused(self, tmp_path):
        source = tmp_path / "docs"
        source.mkdir()
        secret = tmp_path / "secret.yaml"
        secret.write_text("- title: secret\n", encoding="utf-8")
        try:
            (source / "link.yaml").symlink_to(secret)
        except OSError:  # pragma: no cover - platform without symlinks
            pytest.skip("symlinks are not available on this platform")
        with pytest.raises(ValueError, match="outside"):
            _confined_path(path_directive(source), "link.yaml")

    def test_a_symlink_staying_inside_the_tree_is_followed(self, tmp_path):
        (tmp_path / "real.yaml").write_text("[]", encoding="utf-8")
        try:
            (tmp_path / "link.yaml").symlink_to(tmp_path / "real.yaml")
        except OSError:  # pragma: no cover - platform without symlinks
            pytest.skip("symlinks are not available on this platform")
        assert _confined_path(path_directive(tmp_path), "link.yaml") == (
            tmp_path.resolve() / "real.yaml"
        )

    def test_the_configuration_directory_is_a_second_allowed_root(self, tmp_path):
        source = tmp_path / "project" / "source"
        conf = tmp_path / "project" / "config"
        source.mkdir(parents=True)
        conf.mkdir()
        directive = path_directive(source, confdir=str(conf))
        assert _confined_path(directive, "../config/data.yaml") == (
            conf.resolve() / "data.yaml"
        )
        with pytest.raises(ValueError, match="outside"):
            _confined_path(directive, "../other/data.yaml")


class TestSourceFormat:
    @staticmethod
    def env(source_suffix, path="page.rst"):
        return SimpleNamespace(
            docname="page",
            doc2path=lambda name: f"/src/{path}",
            config=SimpleNamespace(source_suffix=source_suffix),
        )

    @pytest.mark.parametrize(
        "source_suffix, path, expected",
        [
            ({".rst": "restructuredtext"}, "page.rst", RESTRUCTUREDTEXT),
            ({".rst": "restructuredtext", ".md": "markdown"}, "page.md", MARKDOWN),
            ({".txt": "restructuredtext"}, "page.txt", RESTRUCTUREDTEXT),
            ({".rst": "markdown"}, "page.rst", MARKDOWN),
            ({".md": "markdown"}, "page.rst", MARKDOWN),
            ({".md": "myst"}, "page.md", MARKDOWN),
            ({}, "page.rst", MARKDOWN),
            (".rst", "page.rst", RESTRUCTUREDTEXT),
            ([".rst", ".txt"], "page.txt", RESTRUCTUREDTEXT),
            ((".rst",), "page.rst", RESTRUCTUREDTEXT),
            (None, "page.rst", RESTRUCTUREDTEXT),
            (None, "page.md", MARKDOWN),
            (None, "page", MARKDOWN),
        ],
        ids=["mapping-rst", "mapping-md", "mapping-txt-as-rst", "mapping-rst-remapped",
             "mapping-suffix-missing", "mapping-other-parser", "mapping-empty",
             "legacy-string", "legacy-list", "legacy-tuple", "unavailable-rst",
             "unavailable-md", "unavailable-no-suffix"],
    )
    def test_the_language_follows_source_suffix(self, source_suffix, path, expected):
        assert _source_format(self.env(source_suffix, path)) == expected

    def test_the_two_languages_use_sphinx_parser_names(self):
        assert (MARKDOWN, RESTRUCTUREDTEXT) == ("markdown", "restructuredtext")


# -- source generation on a fake directive ------------------------------------


def make_directive(options=None, source_suffix=None):
    """Build a directive wired to the few fake attributes generation needs."""
    directive = object.__new__(GalleryGridDirective)
    directive.options = dict(options or {})
    directive._browser_records = {}
    counter = itertools.count()
    env = SimpleNamespace(
        new_serialno=lambda category: next(counter),
        docname="index",
        doc2path=lambda name: "/src/index.rst",
        config=SimpleNamespace(
            source_suffix=source_suffix or {".rst": "restructuredtext"}
        ),
    )
    directive.state = SimpleNamespace(
        document=SimpleNamespace(settings=SimpleNamespace(env=env))
    )
    return directive


class TestOptionsBlock:
    def test_rst_options_are_indented_field_lines(self):
        block = make_directive()._build_options_block(
            {"link": "https://example.org", "shadow": "none"}, rst=True
        )
        assert block == "   :link: https://example.org\n   :shadow: none"

    def test_myst_options_end_with_a_hard_break(self):
        block = make_directive()._build_options_block({"link": "x"}, rst=False)
        assert block == ":link: x  \n"

    @pytest.mark.parametrize("rst", [True, False], ids=["rst", "myst"])
    def test_a_none_value_is_a_valueless_flag(self, rst):
        block = make_directive()._build_options_block({"reverse": None}, rst=rst)
        assert block.strip() == ":reverse:"
        assert "None" not in block

    def test_insertion_order_is_kept(self):
        block = make_directive()._build_options_block({"b": 1, "a": 2}, rst=True)
        assert block.splitlines() == ["   :b: 1", "   :a: 2"]

    def test_no_options(self):
        assert make_directive()._build_options_block({}, rst=True) == ""


class TestBuildCard:
    def test_an_rst_card_with_every_body_part(self):
        card = make_directive()._build_card(
            {"title": "Title", "header": "Head", "image": "a.png", "content": "Body",
             "link": "https://example.org"},
            rst=True,
        )
        assert card == (
            "\n.. grid-item-card:: Title\n"
            "   :link: https://example.org\n"
            "   :class-card: sk-collection-item-0\n"
            "\n"
            "   Head\n\n   ^^^\n\n"
            "   .. image:: a.png\n\n"
            "   Body\n"
        )

    def test_a_myst_card_with_every_body_part(self):
        card = make_directive()._build_card(
            {"title": "Title", "header": "Head", "image": "a.png", "content": "Body"},
            rst=False,
        )
        assert card == (
            "\n~~~~{grid-item-card} Title\n"
            ":class-card: sk-collection-item-0  \n"
            "\n\n"
            "Head  \n^^^  \n"
            "![image](a.png)  \n"
            "Body  \n"
            "\n~~~~\n"
        )

    @pytest.mark.parametrize("rst", [True, False], ids=["rst", "myst"])
    def test_an_empty_item_still_renders_a_card(self, rst):
        card = make_directive()._build_card({}, rst=rst)
        assert "grid-item-card" in card
        assert ":class-card: sk-collection-item-0" in card

    def test_multi_line_content_is_indented_as_one_block_in_rst(self):
        card = make_directive()._build_card(
            {"title": "T", "content": ".. youtube:: abc\n   :width: 100%\n\nMore"}, rst=True
        )
        assert (
            "   .. youtube:: abc\n      :width: 100%\n\n   More\n" in card
        )

    def test_the_item_mapping_is_not_mutated(self):
        item = {"title": "T", "header": "H", "content": "C", "category": "x",
                "link": "https://example.org", "_sk_collection_browser_title": "B",
                "_sk_collection_metadata_only": ["category"]}
        snapshot = {key: (list(v) if isinstance(v, list) else v) for key, v in item.items()}
        directive = make_directive()
        first = directive._build_card(item, rst=True)
        assert item == snapshot
        # ... so the same item can be rendered again (a list-valued group-by).
        second = directive._build_card(item, rst=True)
        assert first.replace("item-0", "item-1") == second

    def test_each_card_gets_a_unique_serial_token(self):
        directive = make_directive()
        cards = [directive._build_card({"title": str(n)}, rst=True) for n in range(3)]
        tokens = [re.search(r"sk-collection-item-\d+", card).group() for card in cards]
        assert tokens == [f"sk-collection-item-{n}" for n in range(3)]
        assert list(directive._browser_records) == tokens

    def test_data_keys_are_not_forwarded_as_card_options(self):
        card = make_directive()._build_card(
            {"title": "T", "category": "ml", "tags": ["a"], "stars": 5, "bogus-key": 1,
             "link": "https://example.org"},
            rst=True,
        )
        options = re.findall(r"^\s+:([\w-]+):", card, re.M)
        assert options == ["link", "class-card"]

    def test_metadata_only_keys_are_withheld_even_when_they_name_an_option(self):
        card = make_directive()._build_card(
            {"title": "T", "link": "https://example.org", "shadow": "lg",
             "_sk_collection_metadata_only": ["shadow"]},
            rst=True,
        )
        assert ":link: https://example.org" in card
        assert ":shadow:" not in card
        assert "_sk_collection" not in card

    @pytest.mark.parametrize("marker", ["shadow", 5, None, {"shadow": 1}])
    def test_a_malformed_metadata_only_marker_is_ignored(self, marker):
        card = make_directive()._build_card(
            {"title": "T", "shadow": "lg", "_sk_collection_metadata_only": marker},
            rst=True,
        )
        assert ":shadow: lg" in card

    def test_the_directive_class_card_applies_to_every_card_unquoted(self):
        directive = make_directive({"class-card": '"shared"'})
        card = directive._build_card({"title": "T"}, rst=True)
        assert ":class-card: shared sk-collection-item-0" in card

    def test_an_item_class_card_is_kept_beside_the_token(self):
        card = make_directive()._build_card({"title": "T", "class-card": "mine"}, rst=True)
        assert ":class-card: mine sk-collection-item-0" in card

    def test_namespaced_card_options_apply_and_override_the_alias(self):
        directive = make_directive(
            {"class-card": "alias", "card-class-card": "namespaced", "card-shadow": "none",
             "card-link-type": "url"}
        )
        card = directive._build_card({"title": "T", "shadow": "lg"}, rst=True)
        assert ":class-card: namespaced sk-collection-item-0" in card
        assert ":shadow: none" in card and ":shadow: lg" not in card
        assert ":link-type: url" in card

    def test_grid_options_are_not_applied_to_cards(self):
        card = make_directive({"grid-gutter": "1"})._build_card({"title": "T"}, rst=True)
        assert "gutter" not in card

    @pytest.mark.parametrize(
        "title, content, length",
        [("plain", "plain", 4), ("a ~~~ b", "x", 4), ("a ~~~~ b", "x", 5),
         ("t", "~~~~~~{youtube} id\n~~~~~~", 7), ("`code` title", "```{x}\n```", 4)],
        ids=["plain", "short-run", "equal-run", "nested-tilde-fence", "backticks"],
    )
    def test_the_myst_fence_outgrows_every_tilde_run_inside(self, title, content, length):
        card = make_directive()._build_card({"title": title, "content": content}, rst=False)
        lines = card.strip("\n").splitlines()
        fence = "~" * length
        assert lines[0] == f"{fence}{{grid-item-card}} {title}"
        assert lines[-1] == fence
        assert all(not line.startswith(fence) for line in lines[1:-1])

    def test_the_browser_record_holds_only_requested_fields(self):
        directive = make_directive(
            {"filter-fields": ("category",), "sort-fields": ("stars",),
             "search-fields": ("blurb",)}
        )
        directive._build_card(
            {"title": "T", "link-alt": "Alt", "category": "ml", "stars": 5,
             "blurb": "words", "secret": "never"},
            rst=True,
        )
        assert directive._browser_records == {
            "sk-collection-item-0": {
                "title": "T",
                "fields": {"category": "ml", "stars": 5},
                "search": "T Alt ml words",
            }
        }

    def test_the_browser_title_overrides_the_escaped_source_title(self):
        directive = make_directive()
        card = directive._build_card(
            {"title": r"C\+\+", "_sk_collection_browser_title": "C++"}, rst=True
        )
        assert r".. grid-item-card:: C\+\+" in card
        assert directive._browser_records["sk-collection-item-0"]["title"] == "C++"
        assert directive._browser_records["sk-collection-item-0"]["fields"] == {
            "title": "C++"
        }

    def test_a_search_base_replaces_the_default_search_corpus(self):
        directive = make_directive()
        directive._build_card(
            {"title": "T", "link-alt": "Open T on YouTube",
             "_sk_collection_search_base": ["Only this"]},
            rst=True,
        )
        assert directive._browser_records["sk-collection-item-0"]["search"] == "Only this"


class TestRenderGrid:
    def test_an_rst_grid_nests_indented_cards(self):
        source = make_directive()._render_grid([{"title": "A"}, {"title": "B"}], rst=True)
        assert [line for line in source.splitlines() if line.strip()] == [
            ".. grid:: 1 2 3 4",
            "   :gutter: 2",
            "   :class-container: gallery-directive",
            "   .. grid-item-card:: A",
            "      :class-card: sk-collection-item-0",
            "   .. grid-item-card:: B",
            "      :class-card: sk-collection-item-1",
        ]
        # Options are separated from the nested cards by a blank line, and no
        # blank line carries trailing whitespace.
        assert "gallery-directive\n\n" in source
        assert all(line == line.rstrip() for line in source.splitlines())

    def test_a_myst_grid_wraps_cards_in_a_longer_fence(self):
        source = make_directive()._render_grid([{"title": "A"}], rst=False)
        lines = [line for line in source.splitlines() if line.strip()]
        assert lines[0] == "~~~~~{grid} 1 2 3 4"
        assert lines[1] == ":gutter: 2"
        assert lines[2].rstrip() == ":class-container: gallery-directive"
        assert lines[3] == "~~~~{grid-item-card} A"
        assert lines[-2:] == ["~~~~", "~~~~~"]

    def test_the_myst_grid_fence_outgrows_a_grown_card_fence(self):
        source = make_directive()._render_grid(
            [{"title": "A", "content": "~~~~~~~~ deep"}], rst=False
        )
        lines = [line for line in source.splitlines() if line.strip()]
        assert lines[0].startswith("~" * 10 + "{grid}")
        assert lines[-1] == "~" * 10
        assert lines[-2] == "~" * 9

    @pytest.mark.parametrize("rst", [True, False], ids=["rst", "myst"])
    def test_columns_and_container_class_are_unquoted(self, rst):
        directive = make_directive(
            {"grid-columns": '"1 2 2 3"', "class-container": "'wide'"}
        )
        source = directive._render_grid([{"title": "A"}], rst=rst)
        assert "grid:: 1 2 2 3" in source or "{grid} 1 2 2 3" in source
        assert ":class-container: gallery-directive wide" in source
        assert '"' not in source and "'" not in source

    def test_namespaced_grid_options_override_defaults_and_extend_classes(self):
        directive = make_directive(
            {"class-container": "alias", "grid-class-container": "extra",
             "grid-gutter": "1", "grid-reverse": None, "grid-margin": "0"}
        )
        source = directive._render_grid([{"title": "A"}], rst=True)
        head = source.split("grid-item-card")[0]
        assert "   :gutter: 1\n" in head and ":gutter: 2" not in head
        assert "   :class-container: gallery-directive alias extra\n" in head
        assert "   :reverse:\n" in head
        assert "   :margin: 0\n" in head

    def test_card_options_are_not_applied_to_the_grid(self):
        source = make_directive({"card-shadow": "none"})._render_grid([], rst=True)
        assert "shadow" not in source

    def test_identical_input_renders_identical_source(self):
        items = [{"title": "A", "link": "https://example.org"}, {"title": "B"}]
        assert make_directive()._render_grid(items, rst=True) == (
            make_directive()._render_grid(items, rst=True)
        )

    def test_get_source_format_delegates_to_source_suffix(self):
        assert make_directive()._get_source_format() == RESTRUCTUREDTEXT
        assert make_directive(source_suffix={".rst": "markdown"})._get_source_format() == (
            MARKDOWN
        )


class TestParse:
    def test_generated_source_is_parsed_line_by_line_and_all_children_returned(self):
        from docutils import nodes

        received = []

        def nested_parse(content, offset, container):
            received.append((list(content), content.source(0), offset))
            container += nodes.rubric(text="heading")
            container += nodes.paragraph(text="grid")

        directive = make_directive()
        directive.state.nested_parse = nested_parse
        children = directive._parse(".. grid:: 1\n\n   body\n")
        assert received == [([".. grid:: 1", "", "   body"], "<gallery-grid>", 0)]
        assert [type(child).__name__ for child in children] == ["rubric", "paragraph"]

    def test_parsing_nothing_returns_no_nodes(self):
        directive = make_directive()
        directive.state.nested_parse = lambda content, offset, container: None
        assert directive._parse("") == []


# -- option spec and setup ----------------------------------------------------


class TestOptionSpec:
    def test_the_directive_shape(self):
        assert GalleryGridDirective.name == "gallery-grid"
        assert GalleryGridDirective.has_content is True
        assert GalleryGridDirective.required_arguments == 0
        assert GalleryGridDirective.optional_arguments == 1
        assert GalleryGridDirective.final_argument_whitespace is True

    @pytest.mark.parametrize(
        "name",
        ["grid-columns", "class-container", "class-card", "filter", "sort", "group-by",
         "limit", "offset", "show-count", "section-style", "searchable", "interactive",
         "search-variant", "search_variant", "filter-fields", "sort-fields",
         "search-fields", "search-label", "collection-id", "grid-gutter", "card-link"],
    )
    def test_every_documented_option_is_registered(self, name):
        assert callable(GalleryGridDirective.option_spec[name])

    @pytest.mark.parametrize("name", ["limit", "offset"])
    @pytest.mark.parametrize("raw", ["-1", "x", "1.5", ""])
    def test_pagination_options_reject_non_naturals(self, name, raw):
        with pytest.raises(ValueError):
            GalleryGridDirective.option_spec[name](raw)

    @pytest.mark.parametrize(
        "raw, expected",
        [(None, "auto"), ("", "auto"), (" Rubric ", "rubric"), ("SECTION", "section")],
    )
    def test_section_style_defaults_to_auto(self, raw, expected):
        assert GalleryGridDirective.option_spec["section-style"](raw) == expected

    def test_section_style_rejects_an_unknown_value(self):
        with pytest.raises(ValueError):
            GalleryGridDirective.option_spec["section-style"]("heading")

    @pytest.mark.parametrize("name", ["searchable", "interactive", "search-variant"])
    def test_search_options_accept_a_flag_or_a_variant(self, name):
        converter = GalleryGridDirective.option_spec[name]
        assert converter(None) is None
        assert converter(" Classic ") == "classic"
        assert converter("pill-overflow") == "pill-overflow"
        with pytest.raises(ValueError, match="pill-overflow"):
            converter("fancy")

    @pytest.mark.parametrize(
        "raw", ["9lives", "", "a b", "x" * 65, "<b>", "a/b"],
        ids=["digit-first", "empty", "space", "too-long", "html", "slash"],
    )
    def test_collection_id_rejects_unsafe_names(self, raw):
        with pytest.raises(ValueError, match="collection-id"):
            GalleryGridDirective.option_spec["collection-id"](raw)

    def test_collection_id_accepts_the_longest_safe_name(self):
        name = "a" + "b" * 63
        assert GalleryGridDirective.option_spec["collection-id"](f" {name} ") == name

    @pytest.mark.parametrize(
        "raw", ["javascript:alert(1)", "data:text/html,x", "vbscript:x", "JaVaScRiPt:x"]
    )
    @pytest.mark.parametrize("name", ["card-link", "card-img-top", "card-img-background"])
    def test_url_options_reject_active_schemes(self, name, raw):
        with pytest.raises(ValueError, match="HTTP"):
            GalleryGridDirective.option_spec[name](raw)

    @pytest.mark.parametrize(
        "raw", ["https://example.org/x", "http://example.org", "mailto:a@example.org",
                "relative/page", "../up.png"]
    )
    def test_url_options_accept_web_mail_and_relative_targets(self, raw):
        assert GalleryGridDirective.option_spec["card-link"](raw) == raw

    @pytest.mark.parametrize("name", ["card-link-alt", "grid-class-container", "card-img-alt"])
    def test_presentation_options_must_be_one_line(self, name):
        with pytest.raises(ValueError, match="single line"):
            GalleryGridDirective.option_spec[name]("a\n:link: javascript:alert(1)")


class FakeApp:
    """Record what an extension registers, without a Sphinx application."""

    def __init__(self, extensions=()):
        self.config = SimpleNamespace(extensions=list(extensions))
        self.extensions = {}
        self.loaded = []
        self.config_values = {}
        self.directives = {}
        self.events = []

    def setup_extension(self, name):
        self.loaded.append(name)

    def add_config_value(self, name, default, rebuild, types=()):
        self.config_values[name] = (default, rebuild)

    def add_directive(self, name, cls):
        self.directives[name] = cls

    def connect(self, event, callback):
        self.events.append(event)


class TestSetup:
    def test_setup_registers_the_directive_config_and_hooks(self):
        app = FakeApp()
        metadata = directive_module.setup(app)
        assert metadata == {"parallel_read_safe": True, "parallel_write_safe": True}
        assert app.directives == {"gallery-grid": GalleryGridDirective}
        assert app.loaded == ["sphinx_design"]
        assert app.config_values["collection_search_variant"] == ("pill-overflow", "env")
        assert app.events == [
            "config-inited",
            "env-before-read-docs",
            "builder-inited",
            "env-get-outdated",
            "env-updated",
            "build-finished",
        ]
        assert app._scikitplot_sphinx_extension_root == EXTENSION.rsplit(".", 1)[0]

    def test_asset_changes_are_an_html_rebuild_dependency(self):
        app = FakeApp()
        directive_module.setup(app)
        revision, rebuild = app.config_values["sk_collection_asset_revision"]
        assert rebuild == "html"
        assert re.fullmatch(r"[0-9a-f]{64}", revision)

    def test_a_mixed_namespace_is_refused_before_anything_is_registered(self):
        root = EXTENSION.rsplit(".", 1)[0]
        other = "elsewhere._sphinx_ext" if root != "elsewhere._sphinx_ext" else "_sphinx_ext"
        app = FakeApp(extensions=[other + "._sphinx_collection"])
        with pytest.raises(ExtensionError, match="Mixed"):
            directive_module.setup(app)
        assert app.directives == {} and app.loaded == [] and app.events == []

    def test_the_retired_theme_bucket_is_refused(self):
        app = FakeApp(extensions=["_sphinx_ext._pydata_sphinx_theme"])
        with pytest.raises(ExtensionError, match="_sphinx_gallery_grid"):
            directive_module.setup(app)

    @pytest.mark.parametrize("variant", ["pill-overflow", "classic"])
    def test_a_known_search_variant_passes_validation(self, variant):
        config = SimpleNamespace(collection_search_variant=variant)
        assert directive_module._validate_collection_search_variant(None, config) is None

    @pytest.mark.parametrize("variant", ["", "fancy", "Classic", None, 5])
    def test_an_unknown_search_variant_is_a_config_error(self, variant):
        config = SimpleNamespace(collection_search_variant=variant)
        with pytest.raises(ConfigError, match="collection_search_variant must be"):
            directive_module._validate_collection_search_variant(None, config)


class TestPackage:
    def test_public_names_resolve_lazily_to_the_directive_module(self):
        from .. import GalleryGridDirective as exported_directive
        from .. import setup as exported_setup

        assert exported_directive is GalleryGridDirective
        assert exported_setup is directive_module.setup

    def test_the_package_declares_its_public_names(self):
        import importlib

        package = importlib.import_module(EXTENSION)
        assert sorted(package.__all__) == ["GalleryGridDirective", "setup"]

    def test_an_unknown_attribute_is_an_attribute_error(self):
        import importlib

        package = importlib.import_module(EXTENSION)
        with pytest.raises(AttributeError, match="nope"):
            package.nope


# -- real builds --------------------------------------------------------------


@pytest.fixture
def build(tmp_path):
    """
    Build a throwaway one-page reStructuredText project and return its output.

    The Sphinx logger configuration is restored afterwards so no global
    logging state leaks into other tests.
    """
    sphinx_logger = logging.getLogger("sphinx")
    saved = (list(sphinx_logger.handlers), sphinx_logger.level, sphinx_logger.propagate)
    counter = iter(range(1000))

    def run(body, files=None, conf=""):
        root = tmp_path / f"project{next(counter)}"
        source = root / "src"
        source.mkdir(parents=True)
        (source / "conf.py").write_text(
            f"extensions = [{EXTENSION!r}]\nhtml_theme = 'alabaster'\n{conf}",
            encoding="utf-8",
        )
        (source / "index.rst").write_text("Page\n====\n\n" + body, encoding="utf-8")
        for name, text in (files or {}).items():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(text, bytes):
                target.write_bytes(text)
            else:
                target.write_text(text, encoding="utf-8")
        app = SphinxTestApp("html", srcdir=source, builddir=root / "out", freshenv=True)
        try:
            app.build()
            outdir = Path(app.outdir)
            page = (outdir / "index.html").read_text(encoding="utf-8")
            log_text = re.sub(r"\x1b\[[0-9;]*m", "", app.warning.getvalue())
            dependencies = {
                str(path) for path in app.env.dependencies.get("index", set())
            }
        finally:
            app.cleanup()
        start = page.index('<div class="body"')
        end = page.index('<div class="sphinxsidebar"')
        body_html = page[start:end]
        carrier = r'<span hidden class="sk-collection-data">(.*?)</span>'
        payloads = [
            json.loads(html_module.unescape(match))
            for match in re.findall(carrier, body_html, re.S)
        ]
        return SimpleNamespace(
            html=re.sub(carrier, "", body_html, flags=re.S),
            page=page,
            warnings=log_text,
            payloads=payloads,
            source=source,
            outdir=outdir,
            dependencies=dependencies,
        )

    yield run
    sphinx_logger.handlers[:] = saved[0]
    sphinx_logger.setLevel(saved[1])
    sphinx_logger.propagate = saved[2]


def card_titles(page_html):
    """Return the visible card titles, in document order."""
    return [
        html_module.unescape(title.strip())
        for title in re.findall(
            r'<div class="sd-card-title[^"]*">\s*(.*?)</div>', page_html, re.S
        )
    ]


THREE = (
    "   - title: One\n     category: a\n     stars: 30\n"
    "   - title: Two\n     category: b\n     stars: 200\n"
    "   - title: Three\n     category: a\n"
)


class TestBuildCards:
    def test_cards_render_title_header_image_content_and_link(self, build):
        result = build(
            ".. gallery-grid::\n"
            '   :grid-columns: "1 2 2 3"\n'
            '   :class-card: "mycard"\n'
            "   :class-container: 'wide'\n\n"
            "   - title: One\n"
            "     header: Head text\n"
            "     image: https://example.org/a.png\n"
            "     content: Body **bold**\n"
            "     category: data-only\n"
            "     link: https://example.org/one\n"
            "     link-alt: Read one\n"
            "   - title: Two\n"
        )
        assert result.warnings == ""
        assert card_titles(result.html) == ["One", "Two"]
        assert 'class="sk-collection docutils container"' in result.html
        assert "sd-row-cols-1 sd-row-cols-xs-1 sd-row-cols-sm-2 sd-row-cols-md-2 " \
            "sd-row-cols-lg-3" in result.html
        assert "gallery-directive wide docutils" in result.html
        assert result.html.count("mycard sk-collection-item-") == 2
        assert '<div class="sd-card-header docutils">' in result.html
        assert "Head text" in result.html
        assert 'src="https://example.org/a.png"' in result.html
        assert "Body <strong>bold</strong>" in result.html
        assert 'href="https://example.org/one"' in result.html
        assert "<span>Read one</span>" in result.html
        # A data field must neither be rendered nor make the card disappear.
        assert "data-only" not in result.html

    def test_the_shared_assets_are_written_and_referenced(self, build):
        result = build(".. gallery-grid::\n\n   - title: One\n")
        assert result.warnings == ""
        for name in ("sk-collection.css", "sk-collection.js"):
            assert (result.outdir / "_static" / name).is_file()
            assert f"_static/{name}" in result.page

    def test_a_plain_gallery_carries_metadata_but_no_controls(self, build):
        result = build(".. gallery-grid::\n\n   - title: One\n     link-alt: Alt text\n")
        assert "sk-collection-searchable" not in result.html
        assert "sk-collection-status" not in result.html
        assert "sk-collection-label" not in result.html
        (payload,) = result.payloads
        assert payload == {
            "version": 1,
            "interactive": False,
            "searchVariant": "pill-overflow",
            "facets": [],
            "sorts": ["title"],
            "records": {
                "sk-collection-item-0": {
                    "title": "One",
                    "fields": {"title": "One"},
                    "search": "One Alt text",
                }
            },
            "collectionId": "",
        }

    def test_the_site_wide_search_variant_is_the_default(self, build):
        result = build(
            ".. gallery-grid::\n   :searchable:\n\n   - title: One\n",
            conf="collection_search_variant = 'classic'\n",
        )
        assert result.warnings == ""
        assert result.payloads[0]["searchVariant"] == "classic"
        assert result.payloads[0]["interactive"] is False

    def test_interactive_controls_emit_label_metadata_status_then_cards(self, build):
        result = build(
            ".. gallery-grid::\n"
            "   :interactive: classic\n"
            "   :filter-fields: category, meta.level\n"
            "   :sort-fields: title,stars\n"
            "   :search-fields: blurb\n"
            "   :search-label: Find <b>\n"
            "   :collection-id: g1\n\n"
            f'   - title: "One </span>{HOSTILE}"\n'
            "     category: a\n"
            "     stars: 3\n"
            "     blurb: hidden words\n"
            "     secret: never exported\n"
            "     meta:\n"
            "       level: deep\n"
        )
        assert result.warnings == ""
        assert (
            'class="sk-collection sk-collection-searchable '
            'sk-collection-controls-status-results-v4 docutils container"' in result.html
        )
        assert '<p class="sk-collection-label">Find &lt;b&gt;</p>' in result.html
        assert (
            '<p hidden class="sk-collection-status" role="status" aria-live="polite" '
            'aria-atomic="true" data-sk-collection-status-source="document">'
            "1 of 1 cards</p>" in result.html
        )
        assert result.html.index("sk-collection-status") < result.html.index("sd-card ")
        assert "<script" not in result.html
        (payload,) = result.payloads
        assert payload["interactive"] is True
        assert payload["searchVariant"] == "classic"
        assert payload["facets"] == ["category", "meta.level"]
        assert payload["sorts"] == ["title", "stars"]
        assert payload["collectionId"] == "g1"
        (record,) = payload["records"].values()
        assert record["title"] == f"One </span>{HOSTILE}"
        assert record["fields"] == {
            "category": "a",
            "meta.level": "deep",
            "title": f"One </span>{HOSTILE}",
            "stars": 3,
        }
        assert "hidden words" in record["search"]
        assert "never exported" not in json.dumps(payload)

    def test_the_default_search_label(self, build):
        result = build(".. gallery-grid::\n   :searchable:\n\n   - title: One\n")
        assert '<p class="sk-collection-label">Filter this gallery</p>' in result.html


class TestBuildSelection:
    def test_filter_sort_and_pagination_with_an_honest_count(self, build):
        result = build(
            ".. gallery-grid::\n   :sort: -title\n   :offset: 1\n   :limit: 1\n"
            "   :show-count:\n\n" + THREE
        )
        assert result.warnings == ""
        assert card_titles(result.html) == ["Three"]
        assert "<p>Showing 1 of 3 items.</p>" in result.html

    def test_no_count_line_when_everything_is_shown(self, build):
        result = build(".. gallery-grid::\n   :show-count:\n\n" + THREE)
        assert "Showing" not in result.html

    def test_a_filter_keeps_matching_items_in_source_order(self, build):
        result = build(".. gallery-grid::\n   :filter: category=a\n\n" + THREE)
        assert result.warnings == ""
        assert card_titles(result.html) == ["One", "Three"]

    def test_a_numeric_sort_puts_missing_values_last(self, build):
        result = build(".. gallery-grid::\n   :sort: -stars\n\n" + THREE)
        assert card_titles(result.html) == ["Two", "One", "Three"]

    def test_grouping_makes_real_sections_at_the_top_level(self, build):
        result = build(".. gallery-grid::\n   :group-by: category\n\n" + THREE)
        assert result.warnings == ""
        assert re.findall(r'<section id="(\w+)">\s*<h2>(\w+)<', result.html) == [
            ("a", "a"),
            ("b", "b"),
        ]
        assert card_titles(result.html) == ["One", "Three", "Two"]

    def test_group_labels_are_plain_text(self, build):
        result = build(
            ".. gallery-grid::\n   :group-by: category\n\n"
            f'   - title: One\n     category: "{HOSTILE} *x*"\n'
        )
        assert "<script" not in result.html
        assert "&lt;script&gt;alert(1)&lt;/script&gt; *x*" in result.html

    def test_rubric_style_uses_rubrics_even_at_the_top_level(self, build):
        result = build(
            ".. gallery-grid::\n   :group-by: category\n   :section-style: rubric\n\n" + THREE
        )
        assert result.warnings == ""
        assert re.findall(r'<p class="rubric">(\w+)</p>', result.html) == ["a", "b"]
        assert "<h2>" not in result.html

    def test_grouping_inside_another_element_falls_back_to_rubrics_quietly(self, build):
        result = build(
            ".. note::\n\n   .. gallery-grid::\n      :group-by: category\n\n"
            "      - title: One\n        category: a\n"
        )
        assert result.warnings == ""
        assert '<p class="rubric">a</p>' in result.html
        assert card_titles(result.html) == ["One"]

    def test_an_explicit_section_request_that_cannot_be_met_warns(self, build):
        result = build(
            ".. note::\n\n   .. gallery-grid::\n      :group-by: category\n"
            "      :section-style: section\n\n"
            "      - title: One\n        category: a\n"
        )
        assert "index.rst:6: WARNING: ':section-style: section' was requested" in (
            result.warnings
        )
        assert '<p class="rubric">a</p>' in result.html

    def test_an_item_with_a_list_valued_group_field_appears_in_each_group(self, build):
        result = build(
            ".. gallery-grid::\n   :group-by: tags\n\n"
            "   - title: Both\n     tags: [x, y]\n"
            "   - title: OnlyY\n     tags: [y]\n"
        )
        assert result.warnings == ""
        assert card_titles(result.html) == ["Both", "Both", "OnlyY"]
        # Each rendered card keeps a unique token even when its item repeats.
        tokens = re.findall(r"sk-collection-item-\d+", result.html)
        assert len(tokens) == len(set(tokens)) == 3


class TestBuildErrors:
    @pytest.mark.parametrize(
        "body, fragment",
        [
            (".. gallery-grid::\n\n   foo: bar\n", "expected a YAML list of items, got dict"),
            (".. gallery-grid::\n\n   - a\n   - b\n", "item 0 is a str"),
            (".. gallery-grid::\n\n   just text\n", "expected a YAML list of items, got str"),
            (".. gallery-grid::\n\n   - title: [unclosed\n", "could not parse YAML"),
            (".. gallery-grid::\n\n   - !!python/object/apply:os.system [id]\n",
             "could not parse YAML"),
            (".. gallery-grid::\n   :filter: category ?? a\n\n   - title: One\n",
             "cannot parse filter term"),
            (".. gallery-grid::\n   :interactive:\n   :filter-fields: nope\n\n   - title: One\n",
             ":filter-fields: field 'nope' does not exist in any gallery record"),
            (".. gallery-grid::\n   :sort-fields: nope\n\n   - title: One\n",
             ":sort-fields: field 'nope' does not exist"),
            (".. gallery-grid::\n   :search-fields: a.b\n\n   - title: One\n     a: 1\n",
             ":search-fields: field 'a.b' does not exist"),
        ],
        ids=["mapping", "scalar-items", "plain-text", "bad-yaml", "python-tag",
             "bad-filter", "filter-field-typo", "sort-field-typo", "search-path-typo"],
    )
    def test_author_errors_are_one_located_error_and_no_cards(self, build, body, fragment):
        result = build(body)
        assert "index.rst:4: ERROR: gallery-grid: " in result.warnings
        assert fragment in result.warnings
        assert result.warnings.count("ERROR") == 1
        assert "sd-card" not in result.html

    @pytest.mark.parametrize(
        "option",
        [":limit: -1", ":offset: x", ":section-style: heading", ":collection-id: 9bad",
         ":filter-fields: a b", ":searchable: fancy", ":card-link: javascript:alert(1)",
         ":no-such-option: 1"],
        ids=["limit", "offset", "section-style", "collection-id", "filter-fields",
             "search-variant", "card-link-scheme", "unknown-option"],
    )
    def test_an_invalid_option_is_a_docutils_error(self, build, option):
        result = build(f".. gallery-grid::\n   {option}\n\n   - title: One\n")
        assert 'index.rst:4: ERROR: Error in "gallery-grid" directive' in result.warnings
        assert "sd-card" not in result.html

    def test_conflicting_search_variants_fail_closed(self, build):
        result = build(
            ".. gallery-grid::\n   :interactive: classic\n"
            "   :search-variant: pill-overflow\n\n   - title: One\n"
        )
        assert "index.rst:4: ERROR: conflicting search variants were supplied" in (
            result.warnings
        )
        assert "sd-card" not in result.html

    def test_an_invalid_site_wide_search_variant_stops_the_build(self, build):
        with pytest.raises(ConfigError, match="collection_search_variant must be"):
            build(
                ".. gallery-grid::\n\n   - title: One\n",
                conf="collection_search_variant = 'fancy'\n",
            )

    def test_a_field_missing_from_some_records_is_not_an_error(self, build):
        result = build(
            ".. gallery-grid::\n   :interactive:\n   :filter-fields: category\n\n"
            "   - title: One\n     category: a\n   - title: Two\n"
        )
        assert result.warnings == ""
        assert card_titles(result.html) == ["One", "Two"]

    @pytest.mark.parametrize(
        "body",
        [".. gallery-grid::\n", ".. gallery-grid::\n\n   []\n",
         ".. gallery-grid::\n   :limit: 0\n\n   - title: One\n"],
        ids=["no-content", "empty-list", "limit-zero"],
    )
    def test_an_empty_gallery_is_not_an_error(self, build, body):
        # _coerce_items: "An empty body yields an empty list, which the caller
        # renders as an empty grid rather than an error."
        result = build(body)
        assert '<p class="sk-collection-empty">No items matched.</p>' in result.html
        assert "ERROR" not in result.warnings

    def test_an_empty_gallery_still_tells_the_reader(self, build):
        result = build(".. gallery-grid::\n")
        assert '<p class="sk-collection-empty">No items matched.</p>' in result.html
        assert "sd-card" not in result.html

    def test_a_filter_matching_nothing_is_exactly_one_warning(self, build):
        result = build(".. gallery-grid::\n   :filter: category=zzz\n\n" + THREE)
        assert "ERROR" not in result.warnings
        assert result.warnings.count("WARNING") == 1

    def test_a_filter_matching_nothing_warns_with_the_source_count(self, build):
        result = build(".. gallery-grid::\n   :filter: category=zzz\n\n" + THREE)
        assert (
            "index.rst:4: WARNING: gallery-grid: no items matched the filter "
            "(3 in source)." in result.warnings
        )
        assert '<p class="sk-collection-empty">No items matched.</p>' in result.html
        assert "sd-card" not in result.html


@ignore_env_app_deprecation
class TestBuildFiles:
    def test_a_data_file_is_read_and_registered_as_a_dependency(self, build):
        result = build(
            ".. gallery-grid:: data/items.yaml\n",
            {"data/items.yaml": "- title: From file ç\n"},
        )
        assert result.warnings == ""
        assert card_titles(result.html) == ["From file ç"]
        assert any(
            path.replace("\\", "/").endswith("data/items.yaml")
            for path in result.dependencies
        )

    def test_an_empty_data_file_is_an_empty_gallery_not_a_crash(self, build):
        result = build(".. gallery-grid:: items.yaml\n", {"items.yaml": ""})
        assert '<p class="sk-collection-empty">No items matched.</p>' in result.html

    def test_a_missing_data_file_is_a_located_warning_and_no_dependency(self, build):
        result = build(".. gallery-grid:: nope.yaml\n")
        assert "index.rst:4: WARNING: gallery-grid: no grid data found at " in (
            result.warnings
        )
        assert "ERROR" not in result.warnings
        assert "No grid data found at " in result.html
        assert not any("nope.yaml" in path for path in result.dependencies)

    @pytest.mark.parametrize(
        "argument",
        ["../../../etc/passwd", "/etc/passwd", "../outside.yaml"],
        ids=["traversal", "absolute", "parent"],
    )
    def test_a_path_outside_the_source_tree_is_a_located_error(self, build, argument):
        result = build(f".. gallery-grid:: {argument}\n")
        assert "index.rst:4: ERROR: gallery-grid: " in result.warnings
        assert "outside the documentation source directory" in result.warnings
        assert "root:" not in result.html
        assert "sd-card" not in result.html
        assert result.dependencies == set()

    def test_a_non_utf8_data_file_is_a_located_error_not_a_crash(self, build):
        result = build(
            ".. gallery-grid:: items.yaml\n",
            {"items.yaml": "- title: caf\xe9\n".encode("latin-1")},
        )
        assert "index.rst:4: ERROR: gallery-grid: could not read" in result.warnings
        assert "Data files must be UTF-8 encoded" in result.warnings
        assert "sd-card" not in result.html

    def test_file_errors_name_the_file(self, build):
        result = build(".. gallery-grid:: items.yaml\n", {"items.yaml": "foo: bar\n"})
        assert "items.yaml: expected a YAML list of items, got dict" in result.warnings


def test_reading_a_data_file_uses_no_deprecated_sphinx_api(build):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build(".. gallery-grid:: items.yaml\n", {"items.yaml": "- title: One\n"})
    deprecations = [
        str(item.message)
        for item in caught
        if issubclass(item.category, (DeprecationWarning, PendingDeprecationWarning))
        and "BuildEnvironment.app" in str(item.message)
    ]
    assert deprecations == []


def test_the_configuration_directory_is_recorded_before_documents_are_read():
    """The environment, not the deprecated ``env.app``, carries ``confdir``."""
    env = SimpleNamespace()
    _record_confdir(SimpleNamespace(confdir=Path("/project/config")), env, ["index"])
    assert getattr(env, _CONFDIR_ATTRIBUTE) == str(Path("/project/config"))


def test_the_directive_source_reads_no_deprecated_environment_attribute():
    source = Path(directive_module.__file__).read_text(encoding="utf-8")
    assert "env.app" not in source.replace("BuildEnvironment.app", "")



class TestItemDataIsNotMarkup:
    """
    Item values cannot leave the line they are written on or carry a script link.

    Notes
    -----
    **Developer notes.** A card is rendered by generating directive source
    from the item. Before these checks a line break in ``title`` or ``link``
    ended the generated construct and the rest was parsed as markup, so a
    data file could inject a ``raw`` directive; and ``link: javascript:...``
    was written into ``href`` unchanged.
    """

    @pytest.mark.parametrize(
        "link",
        ["javascript:alert(1)", "JavaScript:alert(1)", "data:text/html,x", "vbscript:x", "file:///etc/passwd"],
        ids=["javascript", "mixed-case", "data", "vbscript", "file"],
    )
    def test_a_link_with_an_executable_or_local_scheme_is_refused(self, build, link):
        result = build(f'.. gallery-grid::\n\n   - title: One\n     link: "{link}"\n')
        assert "ERROR" in result.warnings and "scheme" in result.warnings
        assert 'href="' + link.split(":")[0].lower() not in result.html.lower()

    @pytest.mark.parametrize(
        "link",
        ["https://example.com/a", "http://example.com", "mailto:a@example.com", "page.html#part", "../other/"],
        ids=["https", "http", "mailto", "relative", "parent"],
    )
    def test_an_ordinary_link_is_rendered(self, build, link):
        result = build(f'.. gallery-grid::\n\n   - title: One\n     link: "{link}"\n')
        assert result.warnings == ""
        # Sphinx obfuscates a mail address, so only its scheme is compared.
        expected = "mailto:" if link.startswith("mailto:") else link + '"'
        assert f'href="{expected}' in html_module.unescape(result.html)

    @pytest.mark.parametrize("key", ["title", "link", "img-alt"])
    def test_a_line_break_in_a_value_cannot_start_new_markup(self, build, key):
        payload = r"x\n\n.. raw:: html\n\n   <script>alert(1)</script>"
        value = payload if key != "link" else "https://example.com/a" + payload[1:]
        extra = "" if key == "title" else f'     {key}: "{value}"\n'
        title = value if key == "title" else "One"
        result = build(f'.. gallery-grid::\n\n   - title: "{title}"\n{extra}')
        assert "<script>alert(1)</script>" not in result.html

    def test_a_folded_title_is_rendered_on_one_line(self, build):
        result = build(".. gallery-grid::\n\n   - title: >\n       Two\n       words\n")
        assert result.warnings == ""
        assert card_titles(result.html) == ["Two words"]


class TestSingleLineAndLinkHelpers:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("a\nb", "a b"), ("  a \t b\r\n", "a b"), ("plain", "plain"), ("", ""), (3, 3), (None, None)],
    )
    def test_single_line(self, value, expected):
        assert directive_module._single_line(value) == expected

    @pytest.mark.parametrize("value", ["https://e.x", "HTTP://e.x", "mailto:a@e.x", "a/b.html", "#top", "", 5, None])
    def test_an_accepted_link_is_returned_unchanged(self, value):
        assert directive_module._checked_link(value, "T") == value

    @pytest.mark.parametrize("value", ["javascript:x", "data:x", "ftp://e.x/a", "x-custom:y"])
    def test_a_refused_link_names_the_item_and_the_scheme(self, value):
        with pytest.raises(ValueError, match=r"item 'T': link uses the '[^']+' scheme"):
            directive_module._checked_link(value, "T")

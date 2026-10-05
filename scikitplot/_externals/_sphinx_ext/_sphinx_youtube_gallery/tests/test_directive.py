"""
Tests for :mod:`.directive`: the ``youtube-gallery`` Sphinx directive.

Three layers are exercised, cheapest first:

* the pure option converters and text helpers;
* the directive's own methods (catalog loading, query building, mode
  resolution, card and source generation) on a directive instance wired to
  small fake Sphinx objects, with catalog files in ``tmp_path``;
* a handful of tiny real Sphinx builds (reStructuredText, bundled
  ``alabaster`` theme) whose produced HTML and warning stream are asserted.

Nothing here touches the network: the directive only ever reads a catalog.
"""

from __future__ import annotations

import datetime as dt
import html as html_module
import json
import logging
import re
import warnings
from types import SimpleNamespace

import pytest
import sphinx
import yaml
from docutils import nodes
from sphinx.errors import ExtensionError
from sphinx.testing.util import SphinxTestApp

from .. import directive as directive_module
from ..directive import (
    DEFAULT_MAX_EMBEDS,
    MODES,
    THUMBNAIL_URL,
    VIEWS,
    YouTubeGalleryDirective,
    _comma_list,
    _format_duration,
    _literal_metadata,
    _mode_choice,
    _view_choice,
)
from ..model import (
    CatalogError,
    ChannelRecord,
    VideoRecord,
    normalize_channel_record,
    normalize_record,
)
from ..query import Query

UTC = dt.timezone.utc
VID = "dQw4w9WgXcQ"
A, B, C = (letter * 11 for letter in "abc")
UC_A = "UC" + "a" * 22
PL_A = "PL" + "a" * 16
EXTENSION = __package__.rsplit(".", 1)[0]
HOSTILE = "<script>alert(1)</script>"

#: ``_confined_path`` reads ``env.app``, which Sphinx 9 deprecates. It is
#: reported by one dedicated test below; elsewhere it must not mask the
#: behaviour under test.
ignore_env_app_deprecation = pytest.mark.filterwarnings(
    "ignore:.*BuildEnvironment.app.*"
)


# -- fakes --------------------------------------------------------------------


class RecordingLogger:
    """Stand-in for the module's Sphinx logger that records warnings."""

    def __init__(self):
        self.warnings = []

    def warning(self, message, *args, **kwargs):
        self.warnings.append((message % args if args else message, kwargs))


@pytest.fixture
def log(monkeypatch):
    """Replace the directive module's logger with a recorder."""
    recorder = RecordingLogger()
    monkeypatch.setattr(directive_module, "logger", recorder)
    return recorder


def make_directive(
    options=None, content=(), arguments=(), config=None, srcdir="/src", docname="index"
):
    """
    Build a directive instance wired to minimal fake Sphinx objects.

    Only the attributes the methods under test read are provided, so a test
    that strays into real Sphinx machinery fails loudly.
    """
    directive = object.__new__(YouTubeGalleryDirective)
    directive.options = dict(options or {})
    directive.content = list(content)
    directive.arguments = list(arguments)
    directive.lineno = 7
    settings = {"source_suffix": {".rst": "restructuredtext"}}
    settings.update(config or {})
    env = SimpleNamespace(
        config=SimpleNamespace(**settings),
        srcdir=str(srcdir),
        docname=docname,
        dependencies=[],
        app=SimpleNamespace(confdir=str(srcdir)),
    )
    env.note_dependency = env.dependencies.append
    env.doc2path = lambda name: f"{srcdir}/{name}.rst"
    directive.state = SimpleNamespace(
        document=SimpleNamespace(settings=SimpleNamespace(env=env))
    )
    directive.state_machine = SimpleNamespace(
        get_source_and_line=lambda lineno=None: (f"{srcdir}/{docname}.rst", lineno)
    )
    return directive


def video(**keys):
    """Return a normalized video record (id defaults to a fixed one)."""
    keys.setdefault("id", VID)
    return normalize_record(keys)


# -- option converters and text helpers ---------------------------------------


class TestLiteralMetadata:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("plain words", "plain words"),
            ("", ""),
            ("a*b", r"a\*b"),
            ("`code`", r"\`code\`"),
            (":ref:`x`", r"\:ref\:\`x\`"),
            ("|sub|", r"\|sub\|"),
            ("back\\slash", "back\\\\slash"),
            ("<b>", r"\<b\>"),
            ("[x](y)", r"\[x\]\(y\)"),
            ("_under_", r"\_under\_"),
            ("C++", r"C\+\+"),
        ],
        ids=["plain", "empty", "emphasis", "literal", "role", "substitution",
             "backslash", "html", "md-link", "underscore", "plus"],
    )
    def test_ascii_punctuation_is_backslash_escaped(self, text, expected):
        assert _literal_metadata(text) == expected

    def test_every_ascii_punctuation_character_is_escaped(self):
        import string

        escaped = _literal_metadata(string.punctuation)
        assert escaped == "".join("\\" + char for char in string.punctuation)

    @pytest.mark.parametrize(
        "text", ["a\nb", "a\r\nb", "a\n\n.. raw:: html", " a \t b ", "a\x0bb", "a\N{LINE SEPARATOR}b"],
        ids=["newline", "crlf", "directive", "padding", "vtab", "line-separator"],
    )
    def test_the_result_is_always_one_line(self, text):
        result = _literal_metadata(text)
        assert result == result.strip()
        assert len(result.splitlines()) == 1
        assert "  " not in result

    def test_whitespace_runs_collapse_to_one_space(self):
        assert _literal_metadata("  a \n\t b  ") == "a b"

    def test_non_ascii_text_is_left_alone(self):
        assert _literal_metadata("Çok güzel — \U0001f600") == (
            "Çok güzel — \U0001f600"
        )

    def test_a_directive_or_role_cannot_survive_as_markup(self):
        escaped = _literal_metadata(".. raw:: html\n\n   <script>")
        assert escaped.startswith(r"\.\. raw\:\: html")
        assert "\n" not in escaped


class TestChoices:
    @pytest.mark.parametrize("mode", MODES)
    def test_every_mode_is_accepted(self, mode):
        assert _mode_choice(mode) == mode

    @pytest.mark.parametrize("raw", ["  Embed ", "LIST", "Thumbnail\n"])
    def test_mode_is_trimmed_and_lowercased(self, raw):
        assert _mode_choice(raw) == raw.strip().lower()

    @pytest.mark.parametrize("raw", ["", "grid", "embed list", "<b>"])
    def test_an_unknown_mode_is_rejected(self, raw):
        with pytest.raises(ValueError, match="auto"):
            _mode_choice(raw)

    @pytest.mark.parametrize("view", VIEWS)
    def test_every_view_is_accepted(self, view):
        assert _view_choice(f" {view.upper()} ") == view

    @pytest.mark.parametrize("raw", ["", "channel", "playlists"])
    def test_an_unknown_view_is_rejected(self, raw):
        with pytest.raises(ValueError, match="channels"):
            _view_choice(raw)

    def test_the_documented_vocabularies(self):
        assert MODES == ("auto", "embed", "thumbnail", "list")
        assert VIEWS == ("auto", "videos", "channels")


class TestCommaList:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("pca, clustering", ["pca", "clustering"]),
            ("one", ["one"]),
            ("", []),
            (None, []),
            (" , ,", []),
            ("a,,b, ", ["a", "b"]),
            ("a b, c", ["a b", "c"]),
            ("b,a,b", ["b", "a", "b"]),
            ("ü, ç", ["ü", "ç"]),
        ],
        ids=["two", "one", "empty", "none", "only-commas", "blank-entries",
             "inner-space", "order-and-duplicates", "unicode"],
    )
    def test_splitting(self, raw, expected):
        assert _comma_list(raw) == expected


class TestFormatDuration:
    @pytest.mark.parametrize(
        "seconds, expected",
        [
            (None, ""),
            (0, "0:00"),
            (5, "0:05"),
            (59, "0:59"),
            (60, "1:00"),
            (150, "2:30"),
            (3599, "59:59"),
            (3600, "1:00:00"),
            (3750, "1:02:30"),
            (86400, "24:00:00"),
            (360000, "100:00:00"),
        ],
    )
    def test_clock_rendering(self, seconds, expected):
        assert _format_duration(seconds) == expected


class TestOptionSpec:
    def test_the_directive_shape(self):
        assert YouTubeGalleryDirective.name == "youtube-gallery"
        assert YouTubeGalleryDirective.has_content is True
        assert YouTubeGalleryDirective.required_arguments == 0
        assert YouTubeGalleryDirective.optional_arguments == 1
        assert YouTubeGalleryDirective.final_argument_whitespace is True

    @pytest.mark.parametrize(
        "name",
        ["catalog", "channel", "playlist", "tags", "match", "match-regex", "since",
         "until", "sort", "group-by", "limit", "offset", "mode", "view", "columns",
         "grid-columns", "show-duration", "show-count", "class-card",
         "class-container", "section-style", "searchable", "interactive",
         "search-variant", "filter-fields", "sort-fields", "search-fields",
         "search-label", "collection-id", "video-width", "video-privacy-mode",
         "video-url-parameters", "grid-gutter", "card-shadow"],
    )
    def test_every_documented_option_is_registered(self, name):
        assert callable(YouTubeGalleryDirective.option_spec[name])

    @pytest.mark.parametrize("name", ["limit", "offset"])
    @pytest.mark.parametrize("raw", ["-1", "x", "1.5", ""])
    def test_pagination_options_reject_non_naturals(self, name, raw):
        with pytest.raises(ValueError):
            YouTubeGalleryDirective.option_spec[name](raw)

    def test_pagination_options_convert_to_int(self):
        assert YouTubeGalleryDirective.option_spec["limit"]("0") == 0
        assert YouTubeGalleryDirective.option_spec["offset"](" 12 ") == 12

    @pytest.mark.parametrize(
        "raw, expected",
        [(None, "auto"), ("", "auto"), (" Rubric ", "rubric"), ("section", "section")],
    )
    def test_section_style_defaults_to_auto(self, raw, expected):
        assert YouTubeGalleryDirective.option_spec["section-style"](raw) == expected

    def test_section_style_rejects_an_unknown_value(self):
        with pytest.raises(ValueError):
            YouTubeGalleryDirective.option_spec["section-style"]("heading")

    @pytest.mark.parametrize("name", ["searchable", "interactive", "search-variant"])
    def test_search_options_accept_a_flag_or_a_variant(self, name):
        converter = YouTubeGalleryDirective.option_spec[name]
        assert converter(None) is None
        assert converter(" Classic ") == "classic"
        with pytest.raises(ValueError, match="pill-overflow"):
            converter("fancy")

    @pytest.mark.parametrize(
        "raw", ["9lives", "", "a b", "x" * 65, "<b>", "a/b"],
        ids=["digit-first", "empty", "space", "too-long", "html", "slash"],
    )
    def test_collection_id_rejects_unsafe_names(self, raw):
        with pytest.raises(ValueError, match="collection-id"):
            YouTubeGalleryDirective.option_spec["collection-id"](raw)

    @pytest.mark.parametrize(
        "raw", ["", " , ", "a b", "1a", "a..b", "<b>"],
        ids=["empty", "only-commas", "space", "digit-first", "double-dot", "html"],
    )
    @pytest.mark.parametrize("name", ["filter-fields", "sort-fields", "search-fields"])
    def test_field_lists_reject_unsafe_names(self, name, raw):
        with pytest.raises(ValueError, match="field names"):
            YouTubeGalleryDirective.option_spec[name](raw)

    def test_field_lists_are_ordered_and_deduplicated(self):
        converter = YouTubeGalleryDirective.option_spec["filter-fields"]
        assert converter("tags, channel,tags , a.b") == ("tags", "channel", "a.b")

    @pytest.mark.parametrize(
        "name, raw",
        [("video-width", "-3"), ("video-width", "0"), ("video-height", "1em"),
         ("video-aspect", "16x9"), ("video-align", "top"),
         ("video-privacy-mode", "maybe"), ("video-title", "a\nb")],
        ids=["negative-width", "zero-width", "unit", "aspect", "align", "privacy",
             "multi-line-title"],
    )
    def test_player_options_reject_invalid_values(self, name, raw):
        with pytest.raises(ValueError):
            YouTubeGalleryDirective.option_spec[name](raw)

    @pytest.mark.parametrize(
        "raw", ["javascript:alert(1)", "data:text/html,x", "vbscript:x"]
    )
    def test_card_link_rejects_active_schemes(self, raw):
        with pytest.raises(ValueError, match="HTTP"):
            YouTubeGalleryDirective.option_spec["card-link"](raw)


# -- catalog loading ----------------------------------------------------------


class TestParseYaml:
    def test_valid_yaml_is_parsed(self):
        assert YouTubeGalleryDirective._parse_yaml("- a\n- b: 1\n", "x") == [
            "a",
            {"b": 1},
        ]

    def test_empty_text_parses_to_none(self):
        assert YouTubeGalleryDirective._parse_yaml("", "x") is None

    @pytest.mark.parametrize(
        "text",
        ["- id: [unclosed", "a: b: c", "\tkey: value", "- !!python/object:os.system x"],
        ids=["flow", "nested-colon", "tab", "python-tag"],
    )
    def test_invalid_or_unsafe_yaml_is_a_located_catalog_error(self, text):
        with pytest.raises(CatalogError, match=r"^page\.rst: could not parse YAML"):
            YouTubeGalleryDirective._parse_yaml(text, "page.rst")

    def test_an_alias_bomb_is_refused(self):
        text = "a: &a [x]\nb: [" + ", ".join(["*a"] * 101) + "]\n"
        with pytest.raises(CatalogError, match="aliases"):
            YouTubeGalleryDirective._parse_yaml(text, "page.rst")

    def test_excessive_nesting_is_refused(self):
        with pytest.raises(CatalogError, match="nesting"):
            YouTubeGalleryDirective._parse_yaml("[" * 40 + "]" * 40, "page.rst")


@ignore_env_app_deprecation
class TestLoadRecords:
    def test_inline_content_is_a_video_catalog(self):
        directive = make_directive(content=[f"- {A}", f"- id: {B}", "  title: Bee"])
        kind, records = directive._load_records()
        assert kind == "video"
        assert [(record.id, record.title) for record in records] == [(A, A), (B, "Bee")]

    def test_inline_channels_are_a_channel_catalog(self):
        directive = make_directive(content=["channels:", '  - "@Foo"'])
        kind, records = directive._load_records()
        assert kind == "channel"
        assert [record.id for record in records] == ["@foo"]

    def test_inline_errors_name_the_directive_content(self):
        directive = make_directive(content=["foo: bar"])
        with pytest.raises(CatalogError, match=r"^youtube-gallery directive content: "):
            directive._load_records()

    def test_no_source_at_all_explains_every_way_to_give_one(self):
        directive = make_directive(config={"youtube_catalog_path": ""})
        with pytest.raises(CatalogError) as caught:
            directive._load_records()
        message = str(caught.value)
        for hint in ("directive argument", ":catalog:", "directive content",
                     "youtube_catalog_path"):
            assert hint in message

    def test_a_missing_config_value_behaves_like_an_empty_one(self):
        with pytest.raises(CatalogError, match="no catalog given"):
            make_directive()._load_records()

    def test_the_argument_names_a_file_relative_to_the_document(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "c.yaml").write_text(f"- {A}\n", encoding="utf-8")
        directive = make_directive(arguments=[" data/c.yaml "], srcdir=tmp_path)
        kind, records = directive._load_records()
        assert (kind, [record.id for record in records]) == ("video", [A])

    def test_the_catalog_file_is_registered_as_a_build_dependency(self, tmp_path):
        target = tmp_path / "c.yaml"
        target.write_text(f"- {A}\n", encoding="utf-8")
        directive = make_directive(arguments=["c.yaml"], srcdir=tmp_path)
        directive._load_records()
        assert directive.env.dependencies == [str(target.resolve())]

    def test_inline_content_registers_no_dependency(self):
        directive = make_directive(content=[f"- {A}"])
        directive._load_records()
        assert directive.env.dependencies == []

    def test_the_catalog_option_is_equivalent_to_the_argument(self, tmp_path):
        (tmp_path / "c.yaml").write_text(f"videos:\n  - {B}\n", encoding="utf-8")
        directive = make_directive(options={"catalog": "c.yaml"}, srcdir=tmp_path)
        assert [record.id for record in directive._load_records()[1]] == [B]

    def test_the_configured_path_is_the_last_resort(self, tmp_path):
        (tmp_path / "c.yaml").write_text(f"- {C}\n", encoding="utf-8")
        directive = make_directive(
            config={"youtube_catalog_path": "c.yaml"}, srcdir=tmp_path
        )
        assert [record.id for record in directive._load_records()[1]] == [C]

    def test_exactly_one_source_is_used_in_precedence_order(self, tmp_path):
        for name, video_id in (("arg.yaml", A), ("opt.yaml", B), ("conf.yaml", C)):
            (tmp_path / name).write_text(f"- {video_id}\n", encoding="utf-8")
        config = {"youtube_catalog_path": "conf.yaml"}

        def ids(**keys):
            directive = make_directive(config=config, srcdir=tmp_path, **keys)
            return [record.id for record in directive._load_records()[1]]

        everything = {"arguments": ["arg.yaml"], "options": {"catalog": "opt.yaml"}}
        assert ids(content=[f"- {VID}"], **everything) == [VID]
        assert ids(**everything) == [A]
        assert ids(options={"catalog": "opt.yaml"}) == [B]
        assert ids() == [C]

    def test_a_missing_file_is_a_catalog_error(self, tmp_path):
        directive = make_directive(arguments=["nope.yaml"], srcdir=tmp_path)
        with pytest.raises(CatalogError, match="catalog file not found"):
            directive._load_records()
        assert directive.env.dependencies == []

    @pytest.mark.parametrize(
        "reference",
        ["../outside.yaml", "../../etc/passwd", "/etc/passwd", "a/../../outside.yaml"],
        ids=["parent", "deep-parent", "absolute", "normalised-parent"],
    )
    def test_a_path_outside_the_source_tree_is_refused(self, tmp_path, reference):
        source = tmp_path / "docs"
        source.mkdir()
        (tmp_path / "outside.yaml").write_text(f"- {A}\n", encoding="utf-8")
        directive = make_directive(arguments=[reference], srcdir=source)
        with pytest.raises(CatalogError, match="outside the documentation source"):
            directive._load_records()
        assert directive.env.dependencies == []

    def test_a_symlink_escaping_the_source_tree_is_refused(self, tmp_path):
        source = tmp_path / "docs"
        source.mkdir()
        secret = tmp_path / "secret.yaml"
        secret.write_text(f"- {A}\n", encoding="utf-8")
        try:
            (source / "link.yaml").symlink_to(secret)
        except OSError:  # pragma: no cover - platform without symlinks
            pytest.skip("symlinks are not available on this platform")
        directive = make_directive(arguments=["link.yaml"], srcdir=source)
        with pytest.raises(CatalogError, match="outside the documentation source"):
            directive._load_records()

    def test_a_non_utf8_file_is_a_catalog_error_not_a_crash(self, tmp_path):
        (tmp_path / "c.yaml").write_bytes(f"- id: {A}\n  title: caf\xe9\n".encode("latin-1"))
        directive = make_directive(arguments=["c.yaml"], srcdir=tmp_path)
        with pytest.raises(CatalogError, match="UTF-8"):
            directive._load_records()

    def test_a_directory_is_a_catalog_error_not_a_crash(self, tmp_path):
        (tmp_path / "folder").mkdir()
        directive = make_directive(arguments=["folder"], srcdir=tmp_path)
        with pytest.raises(CatalogError, match="could not read"):
            directive._load_records()

    def test_file_errors_name_the_file(self, tmp_path):
        target = tmp_path / "c.yaml"
        target.write_text("- id: 5\n", encoding="utf-8")
        directive = make_directive(arguments=["c.yaml"], srcdir=tmp_path)
        with pytest.raises(CatalogError) as caught:
            directive._load_records()
        assert str(caught.value).startswith(f"{target.resolve()}: record 0:")

    def test_an_empty_file_is_an_empty_video_catalog(self, tmp_path):
        (tmp_path / "c.yaml").write_text("", encoding="utf-8")
        directive = make_directive(arguments=["c.yaml"], srcdir=tmp_path)
        assert directive._load_records() == ("video", [])

    def test_unicode_content_round_trips(self, tmp_path):
        (tmp_path / "c.yaml").write_text(
            f"- id: {A}\n  title: Çok güzel \U0001f600\n", encoding="utf-8"
        )
        directive = make_directive(arguments=["c.yaml"], srcdir=tmp_path)
        assert directive._load_records()[1][0].title == "Çok güzel \U0001f600"


# -- query building -----------------------------------------------------------


class TestBuildQuery:
    def test_no_options_is_the_default_query(self):
        assert make_directive()._build_query() == Query()

    def test_every_option_is_translated(self):
        directive = make_directive(
            options={
                "channel": " StatQuest ",
                "playlist": " Intro ",
                "tags": ["pca", "stats"],
                "match": " text ",
                "since": " 2023-01-01 ",
                "until": "2024-01-01T00:00:00Z",
                "sort": " -published ",
                "group-by": " year ",
                "limit": 5,
                "offset": 2,
            }
        )
        assert directive._build_query() == Query(
            channel="StatQuest",
            playlist="Intro",
            tags=("pca", "stats"),
            match="text",
            since=dt.datetime(2023, 1, 1, tzinfo=UTC),
            until=dt.datetime(2024, 1, 1, tzinfo=UTC),
            sort_by="-published",
            group_by="year",
            limit=5,
            offset=2,
        )

    def test_match_regex_is_translated(self):
        directive = make_directive(options={"match-regex": r" ^lecture \d{1,2} "})
        assert directive._build_query().match_regex == r"^lecture \d{1,2}"

    @pytest.mark.parametrize("name", ["sort", "group-by"])
    def test_a_blank_sort_or_group_means_none(self, name):
        query = make_directive(options={name: "   "})._build_query()
        assert (query.sort_by, query.group_by) == ("none", "none")

    @pytest.mark.parametrize("name", ["since", "until"])
    def test_a_blank_bound_is_no_bound(self, name):
        query = make_directive(options={name: ""})._build_query()
        assert (query.since, query.until) == (None, None)

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("https://www.youtube.com/@StatQuest", "StatQuest"),
            ("https://www.youtube.com/@StatQuest/videos", "StatQuest"),
            (f"https://www.youtube.com/channel/{UC_A}", UC_A),
            (f"https://www.youtube.com/channel/{UC_A}/playlists", UC_A),
            ("https://www.youtube.com/c/Name", "Name"),
            ("https://www.youtube.com/user/Name", "Name"),
            ("@StatQuest", "@StatQuest"),
            ("Plain Display Name", "Plain Display Name"),
            (UC_A, UC_A),
        ],
        ids=["handle-url", "handle-tab", "id-url", "id-tab", "custom", "legacy",
             "bare-handle", "display-name", "bare-id"],
    )
    def test_a_channel_url_is_reduced_to_its_identifier(self, raw, expected):
        assert make_directive(options={"channel": raw})._build_query().channel == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (f"https://www.youtube.com/playlist?list={PL_A}", PL_A),
            (f"https://www.youtube.com/watch?v={A}&list={PL_A}", PL_A),
            (PL_A, PL_A),
            ("Intro to ML", "Intro to ML"),
        ],
        ids=["playlist-url", "watch-with-list", "bare-id", "display-name"],
    )
    def test_a_playlist_url_is_reduced_to_its_identifier(self, raw, expected):
        query = make_directive(options={"playlist": raw})._build_query()
        assert query.playlist == expected

    @pytest.mark.parametrize(
        "name, raw, fragment",
        [
            ("channel", "https://evil.example/@Foo", "not a YouTube URL"),
            ("playlist", "https://evil.example/playlist?list=PLx", "not a YouTube URL"),
            ("channel", f"https://youtu.be/{A}", "carries no channel identifier"),
            ("playlist", f"https://youtu.be/{A}", "carries no playlist identifier"),
            ("playlist", "https://www.youtube.com/@Foo", "carries no playlist"),
            ("since", "yesterday", "ISO-8601"),
            ("until", "2024-13-01", "ISO-8601"),
        ],
        ids=["channel-foreign", "playlist-foreign", "channel-is-video",
             "playlist-is-video", "playlist-is-channel", "since-word", "until-month"],
    )
    def test_a_bad_option_value_is_reported_with_the_option_name(
        self, name, raw, fragment
    ):
        with pytest.raises(CatalogError) as caught:
            make_directive(options={name: raw})._build_query()
        message = str(caught.value)
        assert message.startswith(f"option ':{name}:' -- ")
        assert fragment in message

    @pytest.mark.parametrize(
        "options, fragment",
        [
            ({"sort": "no way"}, "invalid sort key"),
            ({"group-by": "no way"}, "invalid group key"),
            ({"match": "a", "match-regex": "b"}, "mutually exclusive"),
            ({"match-regex": "(a+)+"}, "not supported"),
            ({"since": "2024-01-01", "until": "2023-01-01"}, "must be earlier than"),
        ],
        ids=["sort", "group", "both-matchers", "unsafe-regex", "empty-interval"],
    )
    def test_conflicting_or_invalid_selection_options(self, options, fragment):
        with pytest.raises(CatalogError, match=fragment):
            make_directive(options=options)._build_query()


# -- mode resolution ----------------------------------------------------------


class TestResolveMode:
    def test_the_default_embed_budget(self):
        assert DEFAULT_MAX_EMBEDS == 24

    @pytest.mark.parametrize(
        "count, expected",
        [(0, "embed"), (1, "embed"), (DEFAULT_MAX_EMBEDS, "embed"),
         (DEFAULT_MAX_EMBEDS + 1, "thumbnail"), (5000, "thumbnail")],
        ids=["zero", "one", "at-budget", "over-budget", "huge"],
    )
    def test_auto_degrades_to_thumbnails_over_the_budget(self, log, count, expected):
        assert make_directive()._resolve_mode(count) == expected
        assert log.warnings == []

    def test_auto_is_the_default_and_never_returned(self, log):
        directive = make_directive(options={"mode": "auto"})
        assert directive._resolve_mode(3) == "embed"

    def test_the_budget_is_configurable(self, log):
        directive = make_directive(config={"youtube_catalog_max_embeds": 2})
        assert directive._resolve_mode(2) == "embed"
        assert directive._resolve_mode(3) == "thumbnail"

    @pytest.mark.parametrize("mode", ["thumbnail", "list"])
    def test_an_explicit_light_mode_is_honoured_silently(self, log, mode):
        directive = make_directive(options={"mode": mode})
        assert directive._resolve_mode(10_000) == mode
        assert log.warnings == []

    def test_explicit_embed_within_budget_is_silent(self, log):
        directive = make_directive(options={"mode": "embed"})
        assert directive._resolve_mode(DEFAULT_MAX_EMBEDS) == "embed"
        assert log.warnings == []

    def test_explicit_embed_over_budget_is_honoured_with_a_located_warning(self, log):
        directive = make_directive(
            options={"mode": "embed"}, config={"youtube_catalog_max_embeds": 3}
        )
        assert directive._resolve_mode(9) == "embed"
        ((message, keys),) = log.warnings
        assert "9 inline players" in message
        assert "budget of 3" in message
        assert "youtube_catalog_max_embeds" in message
        assert keys == {"location": "/src/index.rst:7"}


# -- cards --------------------------------------------------------------------


class TestCard:
    def test_a_thumbnail_card_links_out_with_accessible_text(self):
        item = make_directive()._card(video(title="PCA  explained"), "thumbnail", True)
        assert item["title"] == "PCA explained"
        assert item["link"] == f"https://www.youtube.com/watch?v={VID}"
        assert item["img-top"] == f"https://i.ytimg.com/vi/{VID}/hqdefault.jpg"
        assert item["img-alt"] == "Video thumbnail: PCA explained"
        assert item["link-alt"] == "Watch PCA explained on YouTube"
        assert item["kind"] == "video"
        assert "content" not in item

    def test_the_thumbnail_template_uses_the_always_available_rendition(self):
        assert THUMBNAIL_URL.format(id=VID) == (
            f"https://i.ytimg.com/vi/{VID}/hqdefault.jpg"
        )

    def test_metadata_is_carried_for_the_browser_controls(self):
        record = video(
            title="T",
            description="D",
            channel="Ch",
            channel_id=UC_A,
            handle="hh",
            playlist="P",
            playlist_id=PL_A,
            tags=["a", "b"],
            published="2024-03-01T10:00:00Z",
            duration=90,
            position=4,
        )
        item = make_directive()._card(record, "thumbnail", True)
        assert {key: item[key] for key in (
            "description", "channel", "channel_id", "handle", "playlist",
            "playlist_id", "tags", "published", "year", "duration", "position",
            "video_count",
        )} == {
            "description": "D",
            "channel": "Ch",
            "channel_id": UC_A,
            "handle": "hh",
            "playlist": "P",
            "playlist_id": PL_A,
            "tags": ["a", "b"],
            "published": "2024-03-01T10:00:00+00:00",
            "year": "2024",
            "duration": 90,
            "position": 4,
            "video_count": None,
        }

    def test_the_card_does_not_alias_the_record_tags(self):
        record = video(tags=["a"])
        item = make_directive()._card(record, "thumbnail", True)
        item["tags"].append("mutated")
        assert record.tags == ["a"]

    def test_an_undated_video_has_no_published_value(self):
        item = make_directive()._card(video(), "thumbnail", True)
        assert item["published"] is None
        assert item["year"] == "unknown"

    def test_the_description_is_metadata_only(self):
        item = make_directive()._card(video(description="long text"), "thumbnail", True)
        assert "content" not in item and "header" not in item

    def test_the_default_search_corpus_is_the_title_only(self):
        item = make_directive()._card(video(title="T", description="D"), "thumbnail", True)
        assert item["_sk_collection_search_base"] == ["T"]

    def test_show_duration_appends_the_clock_to_the_visible_title_only(self):
        directive = make_directive(options={"show-duration": None})
        item = directive._card(video(title="T", duration=3750), "thumbnail", True)
        assert item["title"] == r"T \(1\:02\:30\)"
        assert item["_sk_collection_browser_title"] == "T (1:02:30)"
        assert item["link-alt"] == "Watch T on YouTube"
        assert item["img-alt"] == "Video thumbnail: T"
        assert item["_sk_collection_search_base"] == ["T"]

    def test_show_duration_is_silent_for_an_unknown_duration(self):
        directive = make_directive(options={"show-duration": None})
        assert directive._card(video(title="T"), "thumbnail", True)["title"] == "T"

    def test_a_zero_duration_is_known_and_shown(self):
        directive = make_directive(options={"show-duration": None})
        item = directive._card(video(title="T", duration=0), "thumbnail", True)
        assert item["_sk_collection_browser_title"] == "T (0:00)"

    def test_without_show_duration_the_title_is_unchanged(self):
        item = make_directive()._card(video(title="T", duration=90), "thumbnail", True)
        assert item["title"] == "T"

    def test_custom_fields_are_flattened_and_declared_metadata_only(self):
        record = video(fields={"category": "ml", "audience": {"level": "x"}})
        item = make_directive()._card(record, "thumbnail", True)
        assert item["category"] == "ml"
        assert item["audience"] == {"level": "x"}
        assert item["_sk_collection_metadata_only"] == ["category", "audience"]

    def test_no_metadata_only_marker_without_custom_fields(self):
        item = make_directive()._card(video(), "thumbnail", True)
        assert "_sk_collection_metadata_only" not in item

    @pytest.mark.parametrize("rst", [True, False], ids=["rst", "myst"])
    def test_hostile_title_markup_is_escaped_in_the_card_argument(self, rst):
        title = f"{HOSTILE} `x` **b** :ref:`y`\n.. raw:: html"
        item = make_directive()._card(video(title=title), "thumbnail", rst)
        assert "\n" not in item["title"]
        # Every punctuation character that could start markup is escaped.
        assert re.search(r"(?<!\\)[`*<>:|_.]", item["title"]) is None
        for key in ("link-alt", "img-alt", "_sk_collection_browser_title"):
            assert "\n" not in item[key]

    def test_the_link_is_never_taken_from_catalog_text(self):
        record = video(title="javascript:alert(1)", channel="javascript:alert(1)")
        item = make_directive()._card(record, "thumbnail", True)
        assert item["link"] == f"https://www.youtube.com/watch?v={VID}"
        assert item["img-top"].startswith("https://i.ytimg.com/vi/")

    def test_an_rst_embed_card_nests_a_youtube_directive(self):
        item = make_directive()._card(video(title="A  `b`\nc"), "embed", True)
        assert item["content"] == f".. youtube:: {VID}\n   :title: A `b` c\n"
        assert "link" not in item and "img-top" not in item

    def test_a_myst_embed_card_nests_a_fenced_youtube_directive(self):
        item = make_directive()._card(video(title="T"), "embed", False)
        assert item["content"] == f"~~~{{youtube}} {VID}\n:title: T\n~~~\n"

    @pytest.mark.parametrize(
        "title, fence",
        [("plain", "~~~"), ("a ~~ b", "~~~"), ("a ~~~ b", "~~~~"), ("~~~~~~", "~~~~~~~")],
        ids=["none", "short-run", "equal-run", "long-run"],
    )
    def test_the_myst_fence_outgrows_any_tilde_run_in_the_title(self, title, fence):
        content = make_directive()._card(video(title=title), "embed", False)["content"]
        lines = content.splitlines()
        assert lines[0] == f"{fence}{{youtube}} {VID}"
        assert lines[-1] == fence
        assert all(not line.startswith(fence) for line in lines[1:-1])

    def test_player_options_are_forwarded_to_the_embed(self):
        directive = make_directive(
            options={
                "video-width": "100%",
                "video-height": "360",
                "video-aspect": "4:3",
                "video-align": "center",
                "video-title": "Custom",
                "video-privacy-mode": "",
                "video-url-parameters": "?start=5",
            }
        )
        content = directive._card(video(title="T"), "embed", True)["content"]
        assert content.splitlines() == [
            f".. youtube:: {VID}",
            "   :title: Custom",
            "   :width: 100%",
            "   :height: 360",
            "   :aspect: 4:3",
            "   :align: center",
            "   :privacy_mode:",
            "   :url_parameters: ?start=5",
        ]

    @pytest.mark.parametrize("value", ["false", "off", "no", "0"])
    def test_a_false_privacy_mode_omits_the_flag(self, value):
        directive = make_directive(options={"video-privacy-mode": value})
        assert "privacy_mode" not in directive._card(video(), "embed", True)["content"]

    @pytest.mark.parametrize("value", ["true", "on", "yes", "1", ""])
    def test_a_true_privacy_mode_is_a_valueless_flag(self, value):
        directive = make_directive(options={"video-privacy-mode": value})
        content = directive._card(video(), "embed", True)["content"]
        assert "   :privacy_mode:\n" in content

    @pytest.mark.parametrize("mode", ["embed", "thumbnail"])
    def test_a_channel_card_is_a_title_only_link_in_every_card_mode(self, mode):
        record = normalize_channel_record(
            {"id": "@Foo", "title": "Foo  Channel", "description": "d", "tags": ["t"]}
        )
        directive = make_directive(options={"show-duration": None})
        item = directive._card(record, mode, True)
        assert item["kind"] == "channel"
        assert item["title"] == "Foo Channel"
        assert item["link"] == "https://www.youtube.com/@Foo"
        assert item["link-alt"] == "Open Foo Channel on YouTube"
        assert item["description"] == "d" and item["tags"] == ["t"]
        for absent in ("content", "img-top", "img-alt"):
            assert absent not in item

    def test_a_derived_channel_card_carries_its_video_count(self):
        record = ChannelRecord(id="@foo", title="@Foo", handle="Foo",
                               url="https://www.youtube.com/@Foo", video_count=3)
        assert make_directive()._card(record, "thumbnail", True)["video_count"] == 3


# -- generated gallery-grid source --------------------------------------------


def split_rst_source(source):
    """Split generated RST into (first line, option lines, parsed YAML body)."""
    head, _, body = source.partition("\n\n")
    first, *option_lines = head.splitlines()
    dedented = "\n".join(line[3:] if line.strip() else "" for line in body.splitlines())
    return first, [line.strip() for line in option_lines], yaml.safe_load(dedented)


class TestRenderGrid:
    @pytest.fixture
    def records(self):
        return [video(id=A, title="One"), video(id=B, title="Two: `x`")]

    def test_rst_source_is_a_gallery_grid_with_a_yaml_body(self, records):
        source = make_directive()._render_grid(records, "thumbnail", True, Query())
        first, options, items = split_rst_source(source)
        assert first == ".. gallery-grid::"
        assert options == [":grid-columns: 1 2 2 3"]
        assert [item["link"] for item in items] == [
            f"https://www.youtube.com/watch?v={A}",
            f"https://www.youtube.com/watch?v={B}",
        ]
        assert items[1]["title"] == r"Two\: \`x\`"
        assert source.endswith("\n")

    def test_myst_source_uses_a_colon_fence(self, records):
        source = make_directive()._render_grid(records, "thumbnail", False, Query())
        lines = source.splitlines()
        assert lines[0] == ":::::{gallery-grid}"
        assert lines[1] == ":grid-columns: 1 2 2 3"
        assert lines[-1] == ":::::"
        items = yaml.safe_load("\n".join(lines[3:-1]))
        assert [item["kind"] for item in items] == ["video", "video"]

    def test_an_empty_selection_still_produces_a_valid_yaml_list(self):
        source = make_directive()._render_grid([], "thumbnail", True, Query())
        assert split_rst_source(source)[2] == []

    def test_unicode_survives_the_yaml_round_trip(self):
        record = video(title="Çok güzel \U0001f600", description="ü\nç")
        source = make_directive()._render_grid([record], "thumbnail", True, Query())
        (item,) = split_rst_source(source)[2]
        assert item["title"] == "Çok güzel \U0001f600"
        assert item["description"] == "ü\nç"

    @pytest.mark.parametrize(
        "text",
        [HOSTILE, "a: b\n- c", "line\n\n.. raw:: html\n\n   <script>", "'\"\\", "# x",
         "{a: 1}", "*alias", "\ttab"],
        ids=["html", "yaml-syntax", "rst-directive", "quotes", "comment", "flow-map",
             "alias", "tab"],
    )
    def test_hostile_metadata_round_trips_as_inert_yaml_strings(self, text):
        record = normalize_record({"id": A, "description": "x" + text, "channel": "x" + text,
                                   "fields": {"category": "x" + text}})
        source = make_directive()._render_grid([record], "thumbnail", True, Query())
        first, options, items = split_rst_source(source)
        assert first == ".. gallery-grid::"
        assert len(options) == 1 and len(items) == 1
        assert items[0]["description"] == record.description
        assert items[0]["channel"] == record.channel
        assert items[0]["category"] == record.fields["category"]
        # Nothing the catalog supplied sits at the directive's own indentation.
        body = source.partition("\n\n")[2]
        assert all(line.startswith("   ") for line in body.splitlines() if line.strip())

    @pytest.mark.parametrize(
        "text",
        ["a\x85- id: x", "a\N{LINE SEPARATOR}   :limit: 0", "a\x0c.. raw:: html",
         "a\r:x: y"],
        ids=["next-line", "line-separator", "form-feed", "carriage-return"],
    )
    def test_exotic_line_breaks_cannot_inject_items_or_options(self, text):
        record = normalize_record({"id": A, "description": text, "channel": text})
        source = make_directive()._render_grid([record], "thumbnail", True, Query())
        first, options, items = split_rst_source(source)
        assert first == ".. gallery-grid::"
        assert options == [":grid-columns: 1 2 2 3"]
        assert len(items) == 1
        assert items[0]["link"] == f"https://www.youtube.com/watch?v={A}"
        assert items[0]["kind"] == "video"

    @pytest.mark.parametrize(
        "options, expected",
        [
            ({"columns": "1 2"}, "1 2"),
            ({"grid-columns": "2 2 3 4"}, "2 2 3 4"),
            ({"columns": "1", "grid-columns": "4"}, "4"),
        ],
        ids=["alias", "canonical", "canonical-wins"],
    )
    def test_column_options(self, options, expected):
        source = make_directive(options=options)._render_grid([], "thumbnail", True, Query())
        assert f":grid-columns: {expected}" in split_rst_source(source)[1]

    def test_selection_options_are_delegated_not_reapplied(self):
        query = Query(group_by="playlist", limit=3, offset=1, sort_by="title",
                      channel="x", match="y")
        source = make_directive()._render_grid([], "thumbnail", True, query)
        assert split_rst_source(source)[1] == [
            ":grid-columns: 1 2 2 3",
            ":group-by: playlist",
            ":limit: 3",
            ":offset: 1",
        ]

    def test_limit_zero_is_forwarded(self):
        source = make_directive()._render_grid([], "thumbnail", True, Query(limit=0))
        assert ":limit: 0" in split_rst_source(source)[1]

    def test_flags_are_emitted_valueless_never_as_none(self):
        directive = make_directive(
            options={"show-count": None, "searchable": None, "interactive": None,
                     "section-style": "rubric"},
            config={"collection_search_variant": "pill-overflow"},
        )
        options = split_rst_source(
            directive._render_grid([], "thumbnail", True, Query())
        )[1]
        assert ":show-count:" in options
        assert ":searchable:" in options
        assert ":interactive:" in options
        assert ":section-style: rubric" in options
        assert ":search-variant: pill-overflow" in options
        assert not any("None" in line for line in options)

    def test_reader_control_options_are_forwarded(self):
        directive = make_directive(
            options={
                "interactive": None,
                "filter-fields": ("channel", "tags"),
                "sort-fields": ("title", "published"),
                "search-fields": ("description",),
                "search-label": "Find a video",
                "collection-id": "learn",
                "class-card": "my-card",
                "class-container": "my-grid",
                "grid-gutter": "1",
                "card-shadow": "none",
            },
            config={"collection_search_variant": "classic"},
        )
        options = split_rst_source(
            directive._render_grid([], "thumbnail", True, Query())
        )[1]
        for line in (
            ":filter-fields: channel,tags",
            ":sort-fields: title,published",
            ":search-fields: description",
            ":search-label: Find a video",
            ":collection-id: learn",
            ":class-card: my-card",
            ":class-container: my-grid",
            ":grid-gutter: 1",
            ":card-shadow: none",
            ":search-variant: classic",
        ):
            assert line in options

    def test_no_search_variant_without_reader_controls(self):
        directive = make_directive(config={"collection_search_variant": "classic"})
        options = split_rst_source(
            directive._render_grid([], "thumbnail", True, Query())
        )[1]
        assert not any(line.startswith(":search-variant:") for line in options)

    @pytest.mark.parametrize(
        "options, expected",
        [
            ({"interactive": "classic"}, "classic"),
            ({"searchable": "classic"}, "classic"),
            ({"interactive": None, "search-variant": "classic"}, "classic"),
            ({"interactive": None, "search_variant": "classic"}, "classic"),
            ({"interactive": "classic", "search-variant": "classic"}, "classic"),
            ({"interactive": None}, "pill-overflow"),
        ],
        ids=["interactive-value", "searchable-value", "dedicated", "underscore-alias",
             "repeated-same", "site-default"],
    )
    def test_the_search_variant_is_resolved_once(self, options, expected):
        directive = make_directive(
            options=options, config={"collection_search_variant": "pill-overflow"}
        )
        lines = split_rst_source(directive._render_grid([], "thumbnail", True, Query()))[1]
        assert f":search-variant: {expected}" in lines
        # The activating flag itself stays valueless.
        for key in ("interactive", "searchable"):
            if key in options:
                assert f":{key}:" in lines

    def test_identical_input_renders_identical_source(self, records):
        first = make_directive()._render_grid(records, "embed", True, Query())
        second = make_directive()._render_grid(records, "embed", True, Query())
        assert first == second


class TestRenderList:
    def test_rst_links(self):
        records = [video(id=A, title="One"), video(id=B, title="Two")]
        assert make_directive()._render_list(records, True) == (
            f"* `One <https://www.youtube.com/watch?v={A}>`__\n"
            f"* `Two <https://www.youtube.com/watch?v={B}>`__\n"
        )

    def test_markdown_links(self):
        records = [video(id=A, title="One")]
        assert make_directive()._render_list(records, False) == (
            f"- [One](https://www.youtube.com/watch?v={A})\n"
        )

    @pytest.mark.parametrize("rst", [True, False], ids=["rst", "myst"])
    def test_label_delimiters_in_a_title_are_escaped(self, rst):
        record = video(title="a `b` <c> [d](e) *f*\ng")
        (line,) = make_directive()._render_list([record], rst).splitlines()
        assert r"a \`b\` \<c\> \[d\]\(e\) \*f\* g" in line
        assert line.count(f"https://www.youtube.com/watch?v={VID}") == 1

    @pytest.mark.parametrize("rst", [True, False], ids=["rst", "myst"])
    def test_show_duration_adds_a_suffix_only_when_known(self, rst):
        directive = make_directive(options={"show-duration": None})
        timed, untimed = directive._render_list(
            [video(id=A, title="One", duration=150), video(id=B, title="Two")], rst
        ).splitlines()
        assert timed.endswith(" — 2:30")
        assert "—" not in untimed

    def test_channels_are_listed_by_their_channel_url(self):
        record = normalize_channel_record("@Foo")
        assert make_directive()._render_list([record], True) == (
            "* `\\@Foo <https://www.youtube.com/@Foo>`__\n"
        )


class TestIsRst:
    @pytest.mark.parametrize(
        "source_suffix, expected",
        [
            ({".rst": "restructuredtext"}, True),
            ({".rst": "markdown"}, False),
            ({".md": "markdown"}, False),
            (".rst", True),
            ([".rst", ".txt"], True),
        ],
        ids=["mapping-rst", "mapping-remapped", "mapping-missing", "string", "list"],
    )
    def test_the_page_language_follows_source_suffix(self, source_suffix, expected):
        directive = make_directive(config={"source_suffix": source_suffix})
        assert directive._is_rst() is expected


# -- delegated root marking ---------------------------------------------------


def collection_root(*extra_classes, statuses=0):
    """Return a container shaped like gallery-grid's collection root."""
    from ..._sphinx_collection._browser import status_node

    root = nodes.container(classes=["sk-collection", *extra_classes])
    for _ in range(statuses):
        root += status_node(1)
    return root


class TestMarkGalleryRoot:
    def test_the_plain_root_gains_the_compatibility_class(self):
        root = collection_root()
        YouTubeGalleryDirective._mark_gallery_root([nodes.paragraph(), root])
        assert root["classes"] == ["sk-collection", "youtube-gallery"]

    def test_marking_is_idempotent(self):
        root = collection_root()
        YouTubeGalleryDirective._mark_gallery_root([root])
        YouTubeGalleryDirective._mark_gallery_root([root])
        assert root["classes"].count("youtube-gallery") == 1

    def test_only_the_first_collection_root_is_marked(self):
        first, second = collection_root(), collection_root()
        YouTubeGalleryDirective._mark_gallery_root([first, second])
        assert "youtube-gallery" in first["classes"]
        assert "youtube-gallery" not in second["classes"]

    def test_non_collection_and_non_element_nodes_are_ignored(self):
        other = nodes.container(classes=["something"])
        YouTubeGalleryDirective._mark_gallery_root([nodes.Text("x"), other])
        assert other["classes"] == ["something"]

    def test_nothing_to_mark_is_not_an_error(self):
        assert YouTubeGalleryDirective._mark_gallery_root([]) is None

    def test_a_searchable_root_with_contract_and_one_status_is_accepted(self):
        root = collection_root(
            "sk-collection-searchable",
            "sk-collection-controls-status-results-v4",
            statuses=1,
        )
        YouTubeGalleryDirective._mark_gallery_root([root])
        assert "youtube-gallery" in root["classes"]

    def test_a_searchable_root_without_the_contract_class_is_refused(self):
        root = collection_root("sk-collection-searchable", statuses=1)
        with pytest.raises(ExtensionError, match="controls -> status -> results"):
            YouTubeGalleryDirective._mark_gallery_root([root])

    @pytest.mark.parametrize("count", [0, 2])
    def test_a_searchable_root_needs_exactly_one_status_sibling(self, count):
        root = collection_root(
            "sk-collection-searchable",
            "sk-collection-controls-status-results-v4",
            statuses=count,
        )
        with pytest.raises(ExtensionError, match=f"got {count}"):
            YouTubeGalleryDirective._mark_gallery_root([root])


# -- extension setup ----------------------------------------------------------


class FakeApp:
    """Record what an extension registers, without a Sphinx application."""

    def __init__(self, extensions=()):
        self.config = SimpleNamespace(extensions=list(extensions))
        self.extensions = {}
        self.loaded = []
        self.config_values = {}
        self.directives = {}

    def setup_extension(self, name):
        self.loaded.append(name)

    def add_config_value(self, name, default, rebuild, types=()):
        self.config_values[name] = (default, rebuild, list(types))

    def add_directive(self, name, cls):
        self.directives[name] = cls


class TestSetup:
    def test_setup_registers_the_directive_config_and_dependencies(self):
        app = FakeApp()
        metadata = directive_module.setup(app)
        assert metadata == {"parallel_read_safe": True, "parallel_write_safe": True}
        assert app.directives == {"youtube-gallery": YouTubeGalleryDirective}
        assert app.config_values == {
            "youtube_catalog_path": ("", "env", [str]),
            "youtube_catalog_max_embeds": (DEFAULT_MAX_EMBEDS, "env", [int]),
        }
        root = EXTENSION.rsplit(".", 1)[0]
        assert app.loaded == [
            f"{root}._sphinx_gallery_grid",
            f"{root}._sphinxcontrib_youtube",
        ]

    def test_the_package_setup_delegates_to_the_directive_module(self):
        from .. import setup as package_setup

        app = FakeApp()
        assert package_setup(app) == {
            "parallel_read_safe": True,
            "parallel_write_safe": True,
        }
        assert "youtube-gallery" in app.directives

    def test_dependencies_live_in_the_same_namespace_as_this_package(self):
        root = EXTENSION.rsplit(".", 1)[0]
        assert all(
            name.startswith(root + ".") for name in directive_module.REQUIRED_EXTENSIONS
        )

    def test_a_mixed_namespace_is_refused_before_anything_is_registered(self):
        root = EXTENSION.rsplit(".", 1)[0]
        other = "elsewhere._sphinx_ext" if root != "elsewhere._sphinx_ext" else "_sphinx_ext"
        app = FakeApp(extensions=[other + "._sphinx_collection"])
        with pytest.raises(ExtensionError, match="Mixed"):
            directive_module.setup(app)
        assert app.directives == {} and app.loaded == []

    def test_the_renamed_legacy_extension_is_refused(self):
        app = FakeApp(extensions=["_sphinx_ext.youtube_catalog"])
        with pytest.raises(ExtensionError, match="renamed"):
            directive_module.setup(app)


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
            page = (app.outdir / "index.html").read_text(encoding="utf-8")
            log_text = re.sub(r"\x1b\[[0-9;]*m", "", app.warning.getvalue())
        finally:
            app.cleanup()
        start = page.index('<div class="body"')
        end = page.index('<div class="sphinxsidebar"')
        body_html = page[start:end]
        payloads = [
            json.loads(html_module.unescape(match))
            for match in re.findall(
                r'<span hidden class="sk-collection-data">(.*?)</span>', body_html, re.S
            )
        ]
        visible = re.sub(
            r'<span hidden class="sk-collection-data">.*?</span>', "", body_html, flags=re.S
        )
        return SimpleNamespace(
            html=visible, warnings=log_text, payloads=payloads, source=source
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


class TestBuildVideos:
    def test_thumbnail_cards_link_out_and_escape_hostile_text(self, build):
        result = build(
            ".. youtube-gallery::\n"
            "   :mode: thumbnail\n"
            "   :show-duration:\n\n"
            f"   - id: {VID}\n"
            f'     title: "{HOSTILE} `x` **b** :ref:`y` |sub|"\n'
            '     description: "<img src=x onerror=alert(1)>"\n'
            "     duration: 90\n"
            f"   - https://youtu.be/{A}?t=5\n"
        )
        assert result.warnings == ""
        assert card_titles(result.html) == [
            f"{HOSTILE} `x` **b** :ref:`y` |sub| (1:30)",
            A,
        ]
        assert "<script" not in result.html
        assert "onerror" not in result.html
        assert f'href="https://www.youtube.com/watch?v={VID}"' in result.html
        assert f'href="https://www.youtube.com/watch?v={A}"' in result.html
        assert f'src="https://i.ytimg.com/vi/{VID}/hqdefault.jpg"' in result.html
        assert (
            'alt="Video thumbnail: &lt;script&gt;alert(1)&lt;/script&gt; `x` **b** '
            ':ref:`y` |sub|"' in result.html
        )
        assert "<iframe" not in result.html
        assert 'class="sk-collection youtube-gallery docutils container"' in result.html

    def test_embed_cards_hold_a_player_titled_from_the_record(self, build):
        result = build(
            ".. youtube-gallery::\n"
            "   :mode: embed\n"
            "   :video-privacy-mode:\n"
            "   :video-url-parameters: start=5&rel=0\n\n"
            f"   - id: {VID}\n"
            '     title: "T \\"q\\" <i>"\n'
        )
        assert result.warnings == ""
        assert card_titles(result.html) == ['T "q" <i>']
        (frame,) = re.findall(r"<iframe[^>]*>", result.html)
        assert 'title="T &quot;q&quot; &lt;i&gt;"' in frame
        assert f"https://www.youtube-nocookie.com/embed/{VID}?start=5&amp;rel=0" in frame
        assert "<i>" not in result.html
        assert "hqdefault" not in result.html

    def test_auto_mode_degrades_to_thumbnails_over_the_configured_budget(self, build):
        result = build(
            ".. youtube-gallery::\n",
            {"data/c.yaml": f"videos:\n  - {A}\n  - {B}\n"},
            "youtube_catalog_path = 'data/c.yaml'\nyoutube_catalog_max_embeds = 1\n",
        )
        assert result.warnings == ""
        assert card_titles(result.html) == [A, B]
        assert "<iframe" not in result.html
        assert result.html.count("hqdefault.jpg") == 2

    def test_explicit_embed_over_budget_warns_with_a_location(self, build):
        result = build(
            f".. youtube-gallery::\n   :mode: embed\n\n   - {A}\n   - {B}\n",
            conf="youtube_catalog_max_embeds = 1\n",
        )
        assert result.html.count("<iframe") == 2
        assert "index.rst:4: WARNING: youtube-gallery: rendering 2 inline players" in (
            result.warnings
        )

    def test_list_mode_is_a_plain_bullet_list_of_escaped_links(self, build):
        result = build(
            ".. youtube-gallery::\n"
            "   :mode: list\n"
            "   :show-duration:\n\n"
            f"   - id: {VID}\n"
            '     title: "<b>bold</b> `a <http://evil.example>`__ end"\n'
            "     duration: 61\n"
        )
        assert result.warnings == ""
        assert (
            f'<li><p><a class="reference external" '
            f'href="https://www.youtube.com/watch?v={VID}">&lt;b&gt;bold&lt;/b&gt; '
            f"`a &lt;http://evil.example&gt;`__ end</a> — 1:01</p></li>"
        ) in result.html
        assert 'href="http://evil.example"' not in result.html
        assert "sk-collection" not in result.html
        assert result.payloads == []

    def test_selection_sorting_grouping_and_pagination(self, build):
        result = build(
            ".. youtube-gallery::\n"
            "   :group-by: playlist\n"
            "   :mode: thumbnail\n"
            "   :sort: title\n"
            "   :limit: 2\n"
            "   :show-count:\n\n"
            f"   - id: {A}\n     title: B\n     playlist: P1\n"
            f'   - id: {B}\n     title: A\n     playlist: "<i>P2</i>"\n'
            f"   - id: {C}\n     title: C\n"
        )
        assert result.warnings == ""
        assert card_titles(result.html) == ["A", "B"]
        assert re.findall(r"<h2>(.*?)<a", result.html) == ["&lt;i&gt;P2&lt;/i&gt;", "P1"]
        assert "<i>P2</i>" not in result.html
        assert "<p>Showing 2 of 3 items.</p>" in result.html

    def test_list_mode_sorts_paginates_groups_and_counts(self, build):
        result = build(
            ".. youtube-gallery::\n"
            "   :mode: list\n"
            "   :sort: title\n"
            "   :offset: 1\n"
            "   :limit: 2\n"
            "   :group-by: playlist\n"
            "   :show-count:\n\n"
            f"   - id: {A}\n     title: B\n     playlist: P1\n"
            f"   - id: {B}\n     title: A\n     playlist: P2\n"
            f"   - id: {C}\n     title: C\n"
            f"   - id: {VID}\n     title: D\n     playlist: P1\n"
        )
        assert result.warnings == ""
        assert re.findall(r"<h2>(.*?)<a", result.html) == ["P1", "Ungrouped"]
        assert re.findall(r'watch\?v=(\w+)">(\w)</a>', result.html) == [(A, "B"), (C, "C")]
        assert "<p>Showing 2 of 4 items.</p>" in result.html

    def test_list_mode_shows_no_count_when_nothing_is_cut(self, build):
        result = build(
            f".. youtube-gallery::\n   :mode: list\n   :show-count:\n\n   - {A}\n"
        )
        assert result.warnings == ""
        assert "Showing" not in result.html

    def test_filters_select_before_rendering(self, build):
        result = build(
            ".. youtube-gallery::\n"
            "   :mode: thumbnail\n"
            "   :channel: https://www.youtube.com/@Foo/videos\n"
            "   :tags: x\n"
            "   :since: 2023-01-01\n\n"
            f"   - id: {A}\n     handle: foo\n     tags: [x, y]\n     published: 2024-01-01\n"
            f"   - id: {B}\n     handle: foo\n     tags: [x]\n     published: 2020-01-01\n"
            f"   - id: {C}\n     handle: bar\n     tags: [x]\n     published: 2024-01-01\n"
        )
        assert result.warnings == ""
        assert card_titles(result.html) == [A]

    def test_interactive_controls_delegate_one_shared_collection_root(self, build):
        result = build(
            ".. youtube-gallery::\n"
            "   :interactive: classic\n"
            "   :filter-fields: channel,category\n"
            "   :search-fields: description\n"
            "   :search-label: Find <b>\n"
            "   :collection-id: learn\n"
            "   :mode: thumbnail\n\n"
            f"   - id: {A}\n"
            "     title: Tee\n"
            "     channel: Chan\n"
            '     description: "Desc </span><script>x</script>"\n'
            "     fields:\n"
            "       category: cat\n"
        )
        assert result.warnings == ""
        assert result.html.count("sk-collection-searchable") == 1
        assert (
            'class="sk-collection sk-collection-searchable '
            "sk-collection-controls-status-results-v4 youtube-gallery docutils "
            'container"' in result.html
        )
        assert '<p class="sk-collection-label">Find &lt;b&gt;</p>' in result.html
        assert result.html.count('class="sk-collection-status"') == 1
        assert "1 of 1 cards" in result.html
        assert "<script" not in result.html
        (payload,) = result.payloads
        assert payload["interactive"] is True
        assert payload["searchVariant"] == "classic"
        assert payload["facets"] == ["channel", "category"]
        assert payload["collectionId"] == "learn"
        (record,) = payload["records"].values()
        assert record["title"] == "Tee"
        assert record["fields"] == {"channel": "Chan", "category": "cat", "title": "Tee"}
        assert "Desc </span><script>x</script>" in record["search"]
        assert "on YouTube" not in record["search"]
        # The custom field is data: it never becomes a rejected card option.
        assert card_titles(result.html) == ["Tee"]

    def test_the_metadata_carrier_cannot_be_broken_out_of(self, build):
        result = build(
            ".. youtube-gallery::\n   :searchable:\n   :mode: thumbnail\n\n"
            f'   - id: {A}\n     title: "x</span><script>alert(1)</script>"\n'
        )
        assert result.warnings == ""
        assert "<script" not in result.html
        (payload,) = result.payloads
        assert [record["title"] for record in payload["records"].values()] == [
            "x</span><script>alert(1)</script>"
        ]


class TestBuildChannels:
    def test_a_channel_catalog_renders_title_only_link_cards(self, build):
        result = build(
            ".. youtube-gallery::\n\n"
            "   channels:\n"
            '     - "@Foo"\n'
            '     - id: "@Bar"\n'
            "       title: Bar T\n"
            "       description: never shown\n"
        )
        assert result.warnings == ""
        assert card_titles(result.html) == ["@Foo", "Bar T"]
        assert 'href="https://www.youtube.com/&#64;Foo"' in result.html
        assert 'href="https://www.youtube.com/&#64;Bar"' in result.html
        assert "<span>Open Bar T on YouTube</span>" in result.html
        assert "never shown" not in result.html
        assert "<img" not in result.html and "<iframe" not in result.html

    def test_a_video_catalog_projects_to_sorted_channels(self, build):
        result = build(
            ".. youtube-gallery::\n"
            "   :view: channels\n"
            "   :sort: -title\n\n"
            f"   - id: {A}\n     title: zzz video\n     handle: Alpha\n"
            f"   - id: {B}\n     title: aaa video\n     channel_id: {UC_A}\n"
            "     channel: Zed <b>\n"
            f"   - id: {C}\n     channel: plain name\n"
            f"   - id: {VID}\n     handle: alpha\n"
        )
        assert result.warnings == ""
        # Sorted by *channel* title, not by the contributing video titles.
        assert card_titles(result.html) == ["Zed <b>", "@Alpha"]
        assert f'href="https://www.youtube.com/channel/{UC_A}"' in result.html
        assert "<b>" not in result.html
        assert "watch?v=" not in result.html

    def test_predicates_choose_the_videos_feeding_the_projection(self, build):
        result = build(
            ".. youtube-gallery::\n"
            "   :view: channels\n"
            "   :tags: keep\n\n"
            f"   - id: {A}\n     handle: Alpha\n     tags: keep\n"
            f"   - id: {B}\n     handle: Beta\n     tags: drop\n"
        )
        assert result.warnings == ""
        assert card_titles(result.html) == ["@Alpha"]

    def test_view_channels_on_a_channel_catalog_is_the_same_as_auto(self, build):
        body = '\n   channels:\n     - "@Foo"\n     - "@Bar"\n'
        auto = build(".. youtube-gallery::\n" + body)
        explicit = build(".. youtube-gallery::\n   :view: channels\n" + body)
        assert explicit.warnings == auto.warnings == ""
        assert card_titles(explicit.html) == card_titles(auto.html) == ["@Foo", "@Bar"]

    def test_channel_list_mode(self, build):
        result = build(
            '.. youtube-gallery::\n   :mode: list\n\n   channels:\n     - "@Foo"\n'
        )
        assert result.warnings == ""
        assert (
            '<li><p><a class="reference external" '
            'href="https://www.youtube.com/&#64;Foo">&#64;Foo</a></p></li>'
        ) in result.html


class TestBuildErrors:
    @pytest.mark.parametrize(
        "body, fragment",
        [
            (".. youtube-gallery::\n", "no catalog given"),
            (".. youtube-gallery::\n\n   - id: [unclosed\n", "could not parse YAML"),
            (".. youtube-gallery::\n\n   foo: bar\n", "must contain a 'videos' key"),
            (".. youtube-gallery::\n\n   - id: nope\n", "record 0:"),
            (f".. youtube-gallery::\n\n   - id: {A}\n     titles: x\n",
             "record 0: unknown key(s) ['titles']"),
            (f".. youtube-gallery::\n\n   videos: [{A}]\n   channels: []\n",
             "use either 'videos' or 'channels'"),
            (f".. youtube-gallery::\n   :columns: 1 2\n   :grid-columns: 2 2\n\n   - {A}\n",
             ":columns: and :grid-columns: disagree"),
            ('.. youtube-gallery::\n   :view: videos\n\n   channels:\n     - "@Foo"\n',
             ":view: videos cannot be built from a channels catalog"),
            (f".. youtube-gallery::\n   :view: channels\n\n   - id: {A}\n     channel: plain\n",
             "none carry a stable channel_id"),
            (f".. youtube-gallery::\n   :mode: list\n   :searchable:\n\n   - {A}\n",
             "cannot use reader-side controls (:searchable:)"),
            (f".. youtube-gallery::\n   :sort: nope\n\n   - {A}\n",
             "option ':sort:' field 'nope' is not present"),
            (f".. youtube-gallery::\n   :group-by: nope\n\n   - {A}\n",
             "option ':group-by:' field 'nope' is not present"),
            (f".. youtube-gallery::\n   :since: nope\n\n   - {A}\n",
             "option ':since:' -- 'nope' is not a valid ISO-8601"),
            (f".. youtube-gallery::\n   :channel: https://evil.example/@Foo\n\n   - {A}\n",
             "option ':channel:' -- 'https://evil.example/@Foo' is not a YouTube URL"),
            (f".. youtube-gallery::\n   :match: a\n   :match-regex: b\n\n   - {A}\n",
             "mutually exclusive"),
            (f".. youtube-gallery::\n   :match-regex: (a+)+\n\n   - {A}\n",
             "match-regex construct '(' is not supported"),
        ],
        ids=["no-source", "bad-yaml", "mapping-shape", "bad-id", "unknown-key",
             "mixed-kinds", "columns-conflict", "videos-from-channels",
             "channels-without-identity", "list-with-controls", "sort-typo",
             "group-typo", "bad-date", "foreign-channel-url", "both-matchers",
             "unsafe-regex"],
    )
    def test_a_recognised_failure_is_one_located_error_and_no_cards(
        self, build, body, fragment
    ):
        result = build(body)
        assert "index.rst:4: ERROR: youtube-gallery: " in result.warnings
        assert fragment in result.warnings
        assert result.warnings.count("ERROR") == 1
        assert "sd-card" not in result.html

    def test_same_columns_in_both_spellings_is_not_a_conflict(self, build):
        result = build(
            f".. youtube-gallery::\n   :columns: 1  2 2 3\n   :grid-columns: 1 2 2 3\n"
            f"   :mode: thumbnail\n\n   - {A}\n"
        )
        assert result.warnings == ""
        assert card_titles(result.html) == [A]

    @pytest.mark.parametrize(
        "option",
        [":mode: grid", ":view: playlists", ":limit: -1", ":offset: x",
         ":collection-id: 9bad", ":filter-fields: a b", ":video-width: -3",
         ":searchable: fancy", ":card-link: javascript:alert(1)"],
        ids=["mode", "view", "limit", "offset", "collection-id", "filter-fields",
             "video-width", "search-variant", "card-link-scheme"],
    )
    def test_an_invalid_option_value_is_a_docutils_error(self, build, option):
        result = build(f".. youtube-gallery::\n   {option}\n\n   - {A}\n")
        assert "index.rst:4: ERROR: Error in \"youtube-gallery\" directive" in (
            result.warnings
        )
        assert "sd-card" not in result.html

    def test_conflicting_search_variants_fail_closed(self, build):
        result = build(
            ".. youtube-gallery::\n   :interactive: classic\n"
            f"   :search-variant: pill-overflow\n   :mode: thumbnail\n\n   - {A}\n"
        )
        assert "ERROR: conflicting search variants were supplied" in result.warnings
        assert "sd-card" not in result.html

    def test_an_empty_list_selection_warns_and_says_so(self, build):
        result = build(
            f".. youtube-gallery::\n   :match: zzz\n   :mode: list\n\n   - {A}\n"
        )
        assert (
            "index.rst:4: WARNING: youtube-gallery: no items matched "
            "(1 in catalog, 0 after filtering)." in result.warnings
        )
        assert "ERROR" not in result.warnings
        assert "<p>No items matched.</p>" in result.html

    def test_an_empty_card_selection_warns_once_and_says_so(self, build):
        # run(): "Empty queries emit a paragraph and a build warning."
        result = build(
            f".. youtube-gallery::\n   :match: zzz\n   :mode: thumbnail\n\n   - {A}\n"
        )
        assert "No items matched." in result.html
        assert "no items matched (1 in catalog, 0 after filtering)" in result.warnings
        assert "ERROR" not in result.warnings


@ignore_env_app_deprecation
class TestBuildFiles:
    def test_a_catalog_file_is_read_relative_to_the_document(self, build):
        result = build(
            ".. youtube-gallery:: data/c.yaml\n   :mode: list\n",
            {"data/c.yaml": f"- id: {A}\n  title: From file ç\n"},
        )
        assert result.warnings == ""
        assert "From file ç</a>" in result.html

    def test_the_catalog_option_names_a_file(self, build):
        result = build(
            ".. youtube-gallery::\n   :catalog: data/c.yaml\n   :mode: list\n",
            {"data/c.yaml": f"- id: {A}\n  title: From option\n"},
        )
        assert result.warnings == ""
        assert "From option</a>" in result.html

    @pytest.mark.parametrize(
        "argument, files, fragment",
        [
            ("../../../etc/passwd", None, "outside the documentation source directory"),
            ("/etc/passwd", None, "outside the documentation source directory"),
            ("nope.yaml", None, "catalog file not found"),
            ("c.yaml", {"c.yaml": f"- id: {A}\n  title: caf\xe9\n".encode("latin-1")},
             "Data files must be UTF-8 encoded"),
        ],
        ids=["traversal", "absolute", "missing", "latin-1"],
    )
    def test_a_bad_file_reference_is_a_located_error(
        self, build, argument, files, fragment
    ):
        result = build(f".. youtube-gallery:: {argument}\n", files)
        assert "index.rst:4: ERROR: youtube-gallery: " in result.warnings
        assert fragment in result.warnings
        assert "root:" not in result.html
        assert "sd-card" not in result.html


def test_reading_a_catalog_file_uses_no_deprecated_sphinx_api(build):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build(".. youtube-gallery:: c.yaml\n   :mode: list\n", {"c.yaml": f"- {A}\n"})
    deprecations = [
        str(item.message)
        for item in caught
        if issubclass(item.category, (DeprecationWarning, PendingDeprecationWarning))
        and "BuildEnvironment.app" in str(item.message)
    ]
    assert deprecations == []

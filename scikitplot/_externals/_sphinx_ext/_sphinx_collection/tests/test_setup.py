"""
Tests for ``_sphinx_collection.setup``: asset emission, the rebuild digest,
incremental-build invalidation and the post-build integrity check.

Every function only touches a handful of application attributes, so a small
fake application is an honest stand-in; no Sphinx build is run.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOST_ROOT = ROOT.parents[2]

CSS_NAME = "sk-collection.css"
JS_NAME = "sk-collection.js"
SEARCHABLE = '<div class="sk-collection sk-collection-searchable {extra}">'
STATUS = '<p class="sk-collection-status" data-sk-collection-status-source="document">'
CONTRACT_CLASS = "sk-collection-controls-status-results-v4"


def _module(name: str):
    externals = str(HOST_ROOT / "scikitplot" / "_externals")
    if externals not in sys.path:
        sys.path.insert(0, externals)
    return importlib.import_module("_sphinx_ext._sphinx_collection." + name)


@pytest.fixture(scope="module")
def setup_mod():
    return _module("setup")


@pytest.fixture(scope="module")
def assets():
    return _module("assets")


class FakeApp:
    """The slice of ``sphinx.application.Sphinx`` this module uses."""

    def __init__(self, outdir, builder_format="html"):
        self.outdir = outdir
        self.builder = SimpleNamespace(format=builder_format)
        self.css_files = []
        self.js_files = []
        self.config_values = []

    def add_css_file(self, name, **kwargs):
        self.css_files.append((name, kwargs))

    def add_js_file(self, name, **kwargs):
        self.js_files.append((name, kwargs))

    def add_config_value(self, name, default, rebuild, types=()):
        self.config_values.append((name, default, rebuild, list(types)))


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, message, *args, **kwargs):
        self.warnings.append(message % args if args else message)


@pytest.fixture
def warnings_log(setup_mod, monkeypatch):
    recorder = _Logger()
    monkeypatch.setattr(setup_mod, "logger", recorder)
    return recorder


def _env(*docs):
    return SimpleNamespace(found_docs=set(docs))


def _good_page(count=1):
    block = SEARCHABLE.format(extra=CONTRACT_CLASS) + STATUS + "1 of 1 cards</p></div>"
    return "<html><body>" + block * count + "</body></html>"


# -- collection_asset_revision ------------------------------------------------


def test_revision_is_sha256_of_css_and_js(setup_mod, assets):
    expected = hashlib.sha256(
        (assets.ASSET_CSS + "\0" + assets.ASSET_JS).encode("utf-8")
    ).hexdigest()
    assert setup_mod.collection_asset_revision() == expected
    assert setup_mod.collection_asset_revision() == expected


@pytest.mark.parametrize(
    ("css", "js"),
    [
        pytest.param("a{}", "b()", id="baseline"),
        pytest.param("a{}x", "b()", id="css-changed"),
        pytest.param("a{}", "b()x", id="js-changed"),
        pytest.param("a{}b", "()", id="boundary-moved"),
        pytest.param("", "", id="both-empty"),
        pytest.param("é{}", "日本()", id="non-ascii"),
    ],
)
def test_revision_tracks_asset_content(setup_mod, monkeypatch, css, js):
    monkeypatch.setattr(setup_mod, "ASSET_CSS", css)
    monkeypatch.setattr(setup_mod, "ASSET_JS", js)
    revision = setup_mod.collection_asset_revision()
    assert revision == hashlib.sha256((css + "\0" + js).encode("utf-8")).hexdigest()
    assert len(revision) == 64


def test_revision_distinguishes_moved_boundary(setup_mod, monkeypatch):
    seen = set()
    for css, js in [("ab", "c"), ("a", "bc"), ("abc", ""), ("", "abc")]:
        monkeypatch.setattr(setup_mod, "ASSET_CSS", css)
        monkeypatch.setattr(setup_mod, "ASSET_JS", js)
        seen.add(setup_mod.collection_asset_revision())
    assert len(seen) == 4


# -- register_collection_asset_revision ---------------------------------------


def test_register_adds_html_scoped_config_values(setup_mod, tmp_path):
    app = FakeApp(tmp_path)
    setup_mod.register_collection_asset_revision(app)
    assert app.config_values == [
        (
            "sk_collection_asset_revision",
            setup_mod.collection_asset_revision(),
            "html",
            [str],
        ),
        ("sk_collection_ui_contract", "controls-status-results-v4", "html", [str]),
    ]
    assert not list(tmp_path.iterdir())


# -- ensure_assets ------------------------------------------------------------


def test_ensure_assets_writes_and_registers(setup_mod, assets, tmp_path, warnings_log):
    app = FakeApp(tmp_path / "out")
    setup_mod.ensure_assets(app)
    static = tmp_path / "out" / "_static"
    assert (static / CSS_NAME).read_bytes() == assets.ASSET_CSS.encode("utf-8")
    assert (static / JS_NAME).read_bytes() == assets.ASSET_JS.encode("utf-8")
    assert sorted(path.name for path in static.iterdir()) == [CSS_NAME, JS_NAME]
    assert app.css_files == [(CSS_NAME, {})]
    assert app.js_files == [(JS_NAME, {"defer": "defer"})]
    assert app._sk_collection_assets_changed is True
    assert warnings_log.warnings == []


def test_ensure_assets_accepts_a_string_outdir(setup_mod, tmp_path):
    app = FakeApp(str(tmp_path))
    setup_mod.ensure_assets(app)
    assert (tmp_path / "_static" / CSS_NAME).is_file()


def test_ensure_assets_is_idempotent(setup_mod, tmp_path):
    app = FakeApp(tmp_path)
    for _ in range(3):
        setup_mod.ensure_assets(app)
    assert app.css_files == [(CSS_NAME, {})]
    assert app.js_files == [(JS_NAME, {"defer": "defer"})]


def test_ensure_assets_does_not_rewrite_current_files(setup_mod, tmp_path):
    setup_mod.ensure_assets(FakeApp(tmp_path))
    static = tmp_path / "_static"
    before = {path.name: path.stat().st_mtime_ns for path in static.iterdir()}
    inodes = {path.name: path.stat().st_ino for path in static.iterdir()}
    fresh = FakeApp(tmp_path)
    setup_mod.ensure_assets(fresh)
    assert {path.name: path.stat().st_mtime_ns for path in static.iterdir()} == before
    assert {path.name: path.stat().st_ino for path in static.iterdir()} == inodes
    assert fresh._sk_collection_assets_changed is False
    # A new application still registers the files with itself.
    assert fresh.css_files == [(CSS_NAME, {})]


@pytest.mark.parametrize(
    "stale",
    [
        pytest.param(CSS_NAME, id="stale-css"),
        pytest.param(JS_NAME, id="stale-js"),
    ],
)
def test_ensure_assets_replaces_stale_output(setup_mod, assets, tmp_path, stale):
    static = tmp_path / "_static"
    setup_mod.ensure_assets(FakeApp(tmp_path))
    (static / stale).write_text("/* old build */", encoding="utf-8")
    app = FakeApp(tmp_path)
    setup_mod.ensure_assets(app)
    assert (static / CSS_NAME).read_bytes() == assets.ASSET_CSS.encode("utf-8")
    assert (static / JS_NAME).read_bytes() == assets.ASSET_JS.encode("utf-8")
    assert app._sk_collection_assets_changed is True
    assert sorted(path.name for path in static.iterdir()) == [CSS_NAME, JS_NAME]


def test_ensure_assets_keeps_an_earlier_change_signal(setup_mod, tmp_path):
    setup_mod.ensure_assets(FakeApp(tmp_path))
    app = FakeApp(tmp_path)
    app._sk_collection_assets_changed = True
    setup_mod.ensure_assets(app)
    assert app._sk_collection_assets_changed is True


def test_ensure_assets_reregisters_when_the_digest_changes(
    setup_mod, tmp_path, monkeypatch
):
    app = FakeApp(tmp_path)
    setup_mod.ensure_assets(app)
    monkeypatch.setattr(setup_mod, "ASSET_CSS", "body{color:red}")
    setup_mod.ensure_assets(app)
    assert (tmp_path / "_static" / CSS_NAME).read_text(encoding="utf-8") == (
        "body{color:red}"
    )
    assert app._sk_collection_assets_registered == setup_mod.collection_asset_revision()
    assert len(app.css_files) == 2


@pytest.mark.parametrize(
    "app_factory",
    [
        pytest.param(lambda out: FakeApp(out, "latex"), id="latex-builder"),
        pytest.param(lambda out: FakeApp(out, None), id="builder-without-format"),
        pytest.param(lambda out: SimpleNamespace(outdir=out), id="no-builder-yet"),
        pytest.param(lambda out: SimpleNamespace(outdir=out, builder=None), id="none"),
    ],
)
def test_ensure_assets_ignores_non_html_builders(setup_mod, tmp_path, app_factory):
    app = app_factory(tmp_path)
    setup_mod.ensure_assets(app)
    assert not list(tmp_path.iterdir())
    assert not hasattr(app, "_sk_collection_assets_registered")


def test_ensure_assets_warns_and_allows_retry(setup_mod, tmp_path, warnings_log):
    blocker = tmp_path / "out"
    blocker.write_text("a file where the output directory should be", encoding="utf-8")
    app = FakeApp(blocker)
    setup_mod.ensure_assets(app)  # must not raise
    (warning,) = warnings_log.warnings
    assert warning.startswith("Could not write gallery assets: ")
    assert app.css_files == []
    assert app.js_files == []
    assert not hasattr(app, "_sk_collection_assets_registered")
    # Once the obstacle is gone the same application succeeds.
    blocker.unlink()
    setup_mod.ensure_assets(app)
    assert (blocker / "_static" / JS_NAME).is_file()
    assert app.css_files == [(CSS_NAME, {})]
    assert len(warnings_log.warnings) == 1


def test_failed_replace_leaves_no_temporary_file_and_keeps_old_asset(
    setup_mod, tmp_path, monkeypatch, warnings_log
):
    static = tmp_path / "_static"
    static.mkdir()
    (static / CSS_NAME).write_text("old", encoding="utf-8")

    def refuse(src, dst):
        raise PermissionError("read-only static directory")

    monkeypatch.setattr(setup_mod.os, "replace", refuse)
    app = FakeApp(tmp_path)
    setup_mod.ensure_assets(app)
    assert len(warnings_log.warnings) == 1
    assert "read-only static directory" in warnings_log.warnings[0]
    assert sorted(path.name for path in static.iterdir()) == [CSS_NAME]
    assert (static / CSS_NAME).read_text(encoding="utf-8") == "old"
    assert app.css_files == []


# -- _write_asset_atomic ------------------------------------------------------


def test_write_reports_whether_bytes_changed(setup_mod, tmp_path):
    path = tmp_path / "asset.css"
    assert setup_mod._write_asset_atomic(path, "a{}") is True
    assert setup_mod._write_asset_atomic(path, "a{}") is False
    assert setup_mod._write_asset_atomic(path, "b{}") is True
    assert path.read_text(encoding="utf-8") == "b{}"
    assert [entry.name for entry in tmp_path.iterdir()] == ["asset.css"]


def test_write_encodes_utf8_without_newline_translation(setup_mod, tmp_path):
    path = tmp_path / "asset.js"
    content = "/* é 日本語 */\r\nline\n"
    setup_mod._write_asset_atomic(path, content)
    assert path.read_bytes() == content.encode("utf-8")


def test_write_proceeds_when_existing_file_is_unreadable(
    setup_mod, tmp_path, monkeypatch
):
    path = tmp_path / "asset.css"
    path.write_text("old", encoding="utf-8")

    def unreadable(self):
        raise PermissionError("cannot read")

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    assert setup_mod._write_asset_atomic(path, "new") is True
    monkeypatch.undo()
    assert path.read_text(encoding="utf-8") == "new"


def test_write_failure_propagates_and_cleans_up(setup_mod, tmp_path, monkeypatch):
    path = tmp_path / "asset.css"

    def refuse(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(setup_mod.os, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        setup_mod._write_asset_atomic(path, "data")
    assert list(tmp_path.iterdir()) == []


def test_cleanup_failure_does_not_mask_the_original_error(
    setup_mod, tmp_path, monkeypatch
):
    path = tmp_path / "asset.css"

    def vanish_then_refuse(src, dst):
        os.unlink(src)
        raise OSError("replace failed")

    monkeypatch.setattr(setup_mod.os, "replace", vanish_then_refuse)
    with pytest.raises(OSError, match="replace failed"):
        setup_mod._write_asset_atomic(path, "data")
    assert list(tmp_path.iterdir()) == []


def test_write_into_missing_directory_fails(setup_mod, tmp_path):
    with pytest.raises(OSError):
        setup_mod._write_asset_atomic(tmp_path / "missing" / "asset.css", "data")


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_emitted_assets_are_world_readable_like_other_static_files(setup_mod, tmp_path):
    previous = os.umask(0o022)
    try:
        setup_mod.ensure_assets(FakeApp(tmp_path))
    finally:
        os.umask(previous)
    mode = stat.S_IMODE((tmp_path / "_static" / CSS_NAME).stat().st_mode)
    assert mode & 0o044 == 0o044, oct(mode)


# -- collection_assets_outdated / remember_collection_asset_revision ----------


def test_first_html_build_invalidates_every_document(setup_mod, tmp_path):
    app = FakeApp(tmp_path)
    env = _env("index", "b/page", "a")
    assert setup_mod.collection_assets_outdated(app, env, set(), set(), set()) == [
        "a",
        "b/page",
        "index",
    ]


def test_remembered_revision_makes_next_build_incremental(setup_mod, tmp_path):
    app = FakeApp(tmp_path)
    env = _env("index", "a")
    setup_mod.remember_collection_asset_revision(app, env)
    assert env._sk_collection_asset_revision == setup_mod.collection_asset_revision()
    assert env._sk_collection_ui_contract == "controls-status-results-v4"
    assert setup_mod.collection_assets_outdated(app, env, set(), set(), set()) == []


def test_changed_digest_invalidates_surviving_documents(
    setup_mod, tmp_path, monkeypatch
):
    app = FakeApp(tmp_path)
    env = _env("index", "a", "gone")
    setup_mod.remember_collection_asset_revision(app, env)
    monkeypatch.setattr(setup_mod, "ASSET_JS", "/* new script */")
    result = setup_mod.collection_assets_outdated(app, env, set(), set(), {"gone"})
    assert result == ["a", "index"]


def test_changed_ui_contract_invalidates_documents(setup_mod, tmp_path):
    app = FakeApp(tmp_path)
    env = _env("index")
    setup_mod.remember_collection_asset_revision(app, env)
    env._sk_collection_ui_contract = "controls-status-results-v3"
    assert setup_mod.collection_assets_outdated(app, env, set(), set(), set()) == [
        "index"
    ]


def test_physical_asset_replacement_invalidates_once(setup_mod, tmp_path):
    app = FakeApp(tmp_path)
    env = _env("index", "a")
    setup_mod.remember_collection_asset_revision(app, env)
    setup_mod.ensure_assets(app)
    assert app._sk_collection_assets_changed is True
    assert setup_mod.collection_assets_outdated(app, env, [], [], []) == ["a", "index"]
    # The write signal is consumed, so a long-lived app is incremental again.
    assert app._sk_collection_assets_changed is False
    assert setup_mod.collection_assets_outdated(app, env, [], [], []) == []


@pytest.mark.parametrize(
    "removed",
    [
        pytest.param(None, id="none"),
        pytest.param(set(), id="empty-set"),
        pytest.param([], id="empty-list"),
        pytest.param({"not-a-doc"}, id="unknown-docname"),
    ],
)
def test_outdated_tolerates_removed_argument_shapes(setup_mod, tmp_path, removed):
    result = setup_mod.collection_assets_outdated(
        FakeApp(tmp_path), _env("b", "a"), set(), set(), removed
    )
    assert result == ["a", "b"]


def test_outdated_with_no_documents(setup_mod, tmp_path):
    assert setup_mod.collection_assets_outdated(FakeApp(tmp_path), _env(), 0, 0, 0) == []


def test_outdated_result_is_sorted_and_repeatable(setup_mod, tmp_path):
    docs = [f"doc{n:03d}" for n in range(200)]
    results = [
        setup_mod.collection_assets_outdated(
            FakeApp(tmp_path), _env(*docs), set(), set(), set()
        )
        for _ in range(3)
    ]
    assert results == [docs, docs, docs]


@pytest.mark.parametrize(
    "builder_format",
    [
        pytest.param("latex", id="latex"),
        pytest.param("", id="empty-format"),
        pytest.param(None, id="no-format"),
    ],
)
def test_non_html_builders_neither_invalidate_nor_remember(
    setup_mod, tmp_path, builder_format
):
    app = FakeApp(tmp_path, builder_format)
    app._sk_collection_assets_changed = True
    env = _env("index")
    assert setup_mod.collection_assets_outdated(app, env, set(), set(), set()) == []
    setup_mod.remember_collection_asset_revision(app, env)
    assert not hasattr(env, "_sk_collection_asset_revision")
    assert not hasattr(env, "_sk_collection_ui_contract")
    # The pending signal is left for the HTML builder that owns it.
    assert app._sk_collection_assets_changed is True


def test_outdated_without_a_builder(setup_mod):
    assert setup_mod.collection_assets_outdated(
        SimpleNamespace(), _env("index"), set(), set(), set()
    ) == []


# -- verify_collection_assets -------------------------------------------------


@pytest.fixture
def built(setup_mod, tmp_path):
    """An output directory holding this build's assets and one clean page."""
    app = FakeApp(tmp_path)
    setup_mod.ensure_assets(app)
    (tmp_path / "index.html").write_text(_good_page(), encoding="utf-8")
    return app


def _verify_error(setup_mod, app):
    from sphinx.errors import ExtensionError

    with pytest.raises(ExtensionError) as info:
        setup_mod.verify_collection_assets(app, None)
    message = str(info.value)
    assert "Shared collection UI integrity failure" in message
    assert "Refusing to publish a mixed-version gallery UI." in message
    return message


def test_verify_accepts_a_consistent_build(setup_mod, built):
    assert setup_mod.verify_collection_assets(built, None) is None


def test_verify_accepts_pages_without_collections(setup_mod, built, tmp_path):
    (tmp_path / "plain.html").write_text("<p>sk-collection only</p>", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.html").write_text(_good_page(3), encoding="utf-8")
    assert setup_mod.verify_collection_assets(built, None) is None


def test_verify_never_masks_an_existing_build_failure(setup_mod, tmp_path):
    app = FakeApp(tmp_path)  # nothing was emitted at all
    assert setup_mod.verify_collection_assets(app, RuntimeError("boom")) is None


@pytest.mark.parametrize(
    "builder_format",
    [pytest.param("latex", id="latex"), pytest.param(None, id="no-format")],
)
def test_verify_ignores_non_html_builders(setup_mod, tmp_path, builder_format):
    app = FakeApp(tmp_path, builder_format)
    assert setup_mod.verify_collection_assets(app, None) is None


def test_verify_reports_missing_assets(setup_mod, tmp_path):
    message = _verify_error(setup_mod, FakeApp(tmp_path))
    assert f"{CSS_NAME}: unreadable" in message
    assert f"{JS_NAME}: unreadable" in message


@pytest.mark.parametrize(
    ("name", "content"),
    [
        pytest.param(CSS_NAME, "/* overwritten later */", id="css-overwritten"),
        pytest.param(JS_NAME, "", id="js-truncated"),
        pytest.param(JS_NAME, None, id="js-with-trailing-byte"),
    ],
)
def test_verify_reports_late_overwrites(setup_mod, assets, built, tmp_path, name, content):
    path = tmp_path / "_static" / name
    if content is None:
        path.write_bytes(assets.ASSET_JS.encode("utf-8") + b"\n")
    else:
        path.write_text(content, encoding="utf-8")
    message = _verify_error(setup_mod, built)
    assert f"{name}: emitted bytes do not match extension source" in message
    other = JS_NAME if name == CSS_NAME else CSS_NAME
    assert other + ":" not in message


@pytest.mark.parametrize(
    "page",
    [
        pytest.param(
            SEARCHABLE.format(extra=CONTRACT_CLASS) + "</div>",
            id="status-marker-missing",
        ),
        pytest.param(
            SEARCHABLE.format(extra="") + STATUS + "</p></div>",
            id="contract-class-missing",
        ),
        pytest.param(
            _good_page(1) + SEARCHABLE.format(extra=CONTRACT_CLASS) + "</div>",
            id="second-gallery-lacks-status",
        ),
        pytest.param(SEARCHABLE.format(extra="") + "</div>", id="pre-contract-html"),
    ],
)
def test_verify_reports_stale_gallery_html(setup_mod, built, tmp_path, page):
    (tmp_path / "stale.html").write_text(page, encoding="utf-8")
    message = _verify_error(setup_mod, built)
    assert "missing the document-owned status sibling: stale.html" in message
    assert "index.html" not in message


def test_verify_lists_stale_pages_relative_and_sorted(setup_mod, built, tmp_path):
    stale = SEARCHABLE.format(extra="") + "</div>"
    (tmp_path / "b").mkdir()
    for name in ("b/two.html", "a.html", "c.html"):
        (tmp_path / name).write_text(stale, encoding="utf-8")
    message = _verify_error(setup_mod, built)
    # POSIX form on every platform: the message does not depend on the
    # machine that built the site.
    expected = ", ".join(["a.html", "b/two.html", "c.html"])
    assert "status sibling: " + expected + ". Refusing" in message
    assert str(tmp_path) not in message


def test_verify_caps_the_stale_page_listing_at_twelve(setup_mod, built, tmp_path):
    stale = SEARCHABLE.format(extra="") + "</div>"
    for index in range(20):
        (tmp_path / f"stale{index:02d}.html").write_text(stale, encoding="utf-8")
    message = _verify_error(setup_mod, built)
    assert [f"stale{index:02d}.html" in message for index in range(20)] == (
        [True] * 12 + [False] * 8
    )


def test_verify_combines_asset_and_html_failures(setup_mod, built, tmp_path):
    (tmp_path / "_static" / CSS_NAME).write_text("x", encoding="utf-8")
    (tmp_path / "stale.html").write_text(
        SEARCHABLE.format(extra="") + "</div>", encoding="utf-8"
    )
    message = _verify_error(setup_mod, built)
    assert f"{CSS_NAME}: emitted bytes do not match" in message
    assert "status sibling: stale.html" in message


def test_verify_skips_undecodable_and_non_html_files(setup_mod, built, tmp_path):
    stale = (SEARCHABLE.format(extra="") + "</div>").encode("utf-8")
    (tmp_path / "binary.html").write_bytes(b"\xff\xfe" + stale)
    (tmp_path / "notes.txt").write_bytes(stale)
    (tmp_path / "page.htm").write_bytes(stale)
    assert setup_mod.verify_collection_assets(built, None) is None


def test_verify_does_not_modify_the_output(setup_mod, built, tmp_path):
    before = {p: p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    setup_mod.verify_collection_assets(built, None)
    after = {p: p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    assert after == before


# -- mark_used and exports ----------------------------------------------------


def test_mark_used_is_a_side_effect_free_hook(setup_mod):
    directive = SimpleNamespace()
    assert setup_mod.mark_used(directive) is None
    assert vars(directive) == {}
    assert setup_mod.mark_used(None) is None


def test_public_names(setup_mod):
    assert sorted(setup_mod.__all__) == [
        "collection_asset_revision",
        "collection_assets_outdated",
        "ensure_assets",
        "mark_used",
        "register_collection_asset_revision",
        "remember_collection_asset_revision",
        "verify_collection_assets",
    ]
    assert all(callable(getattr(setup_mod, name)) for name in setup_mod.__all__)

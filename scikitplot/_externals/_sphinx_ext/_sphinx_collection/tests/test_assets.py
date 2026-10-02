from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
# The directory that contains ``scikitplot/``: the docs source in the
# documentation checkout, the repository root in the library checkout.
HOST_ROOT = ROOT.parents[2]
ASSETS_PATH = ROOT / "_sphinx_collection" / "assets.py"


def _docs_source() -> Path:
    """
    Return the documentation source directory, or skip where there is none.

    The stack is deployed in a documentation checkout, beside the site's
    ``conf.py``, and in the library checkout, where there is no site. A test
    of the site's build configuration can only run where the site is; in the
    library checkout it skips with that reason rather than failing on a
    ``conf.py`` that was never meant to exist there.
    """
    if not (HOST_ROOT / "conf.py").is_file():
        pytest.skip(
            "no documentation site owns this extension stack in this checkout; "
            "the site's build configuration is tested in the documentation checkout"
        )
    return HOST_ROOT


def _assets_module():
    externals = str(HOST_ROOT / "scikitplot" / "_externals")
    if externals not in sys.path:
        sys.path.insert(0, externals)
    return importlib.import_module("_sphinx_ext._sphinx_collection.assets")


def test_compact_controls_use_one_inline_shell() -> None:
    assets = _assets_module()
    css = assets.ASSET_CSS
    js = assets.ASSET_JS

    assert ".sk-collection-controls {" in css
    assert ".sk-collection-primary-row {" in css
    assert ".sk-collection-primary-row--pill-overflow" in css
    assert ".sk-collection-search-field {" in css
    assert ".sk-collection-search-field--pill" in css
    assert ".sk-collection-disclosure {" in css
    assert ".sk-collection-disclosure--overflow" in css
    assert ".sk-collection-overflow {" not in css

    assert "var controls = element('div','sk-collection-controls');" in js
    assert "var searchVariant=config.searchVariant==='classic'?'classic':'pill-overflow';" in js
    assert "sk-collection-primary-row--'+searchVariant" in js
    assert "sk-collection-search-field--pill" in js
    assert "sk-collection-disclosure--overflow" in js
    assert "sk-collection-overflow-icon" in js
    assert "controls.append(primary,panel);" in js
    assert "controls.append(status,primary,panel);" not in js
    assert "if(status.parentElement!==root){root.insertBefore(status,root.firstChild);}root.insertBefore(controls,status);status.after(chips);chips.after(suggestions);" in js
    assert "sk-collection-overflow='" not in js


def test_match_status_is_document_owned_sibling_after_controls() -> None:
    assets = _assets_module()
    css = assets.ASSET_CSS
    js = assets.ASSET_JS
    browser = (ROOT / "_sphinx_collection" / "_browser.py").read_text(encoding="utf-8")
    gallery = (ROOT / "_sphinx_gallery_grid" / "directive.py").read_text(encoding="utf-8")

    assert 'def status_node(count):' in browser
    assert '_STATUS_NODE_SOURCE_KEY = "sk_collection_status_source"' in browser
    assert 'def is_document_status_node(node):' in browser
    assert 'nodes.raw(markup, markup, format="html")' in browser
    assert 'markup = node.astext()' in browser
    assert "f'<p hidden class=\"{STATUS_CLASS}\" role=\"status\" '" in browser
    assert "f'{STATUS_SOURCE_ATTRIBUTE}=\"{STATUS_SOURCE_DOCUMENT}\">'" in browser
    assert 'f"{count} of {count} cards</p>"' in browser
    assert 'status_node,' in gallery
    assert 'wrapper += status_node(len(self._browser_records))' in gallery
    assert gallery.index('wrapper += metadata_node(self._browser_records, browser_options)') < gallery.index('wrapper += status_node(len(self._browser_records))') < gallery.index('wrapper += rendered')

    assert "var status=element('p','sk-collection-status');status.setAttribute('role','status')" not in js
    assert "var status=prepareStatus(root);" in js
    assert "function directStatus(root)" in js
    assert "function prepareStatus(root)" in js
    assert "status.setAttribute('data-sk-collection-status-source','runtime-fallback');" in js
    assert "status.hidden=false;" in js
    assert "controls.append(primary,panel);" in js
    assert "controls.append(status,primary,panel);" not in js
    assert "if(status.parentElement!==root){root.insertBefore(status,root.firstChild);}root.insertBefore(controls,status);status.after(chips);chips.after(suggestions);" in js
    assert "var UI_CONTRACT='controls-status-results-v4';" in js
    assert "data-sk-collection-status-placement','sibling'" in js
    assert "if (embedded) {" in js
    assert "if (status && status !== embedded) embedded.remove();" in js
    assert "else { status = embedded; }" in js
    assert "if (status && status.previousElementSibling !== controls) controls.after(status);" in js
    assert "if (root.hasAttribute('data-sk-enhanced')) { prepareStatus(root); stampUiContract(root); return; }" in js
    assert "root.setAttribute('data-sk-enhanced','true');stampUiContract(root);" in js
    assert ".sk-collection-status {\n  margin:0 0 .65rem;" in css

def test_search_is_live_and_ime_safe() -> None:
    js = _assets_module().ASSET_JS

    assert "input.addEventListener('input',function(event){if(!event.isComposing)apply();});" in js
    assert "input.addEventListener('compositionend',apply);" in js
    assert "searchWrap.addEventListener('submit',function(event){event.preventDefault();apply();input.focus();});" in js
    assert "if(event.key==='Enter' && event.isComposing)event.preventDefault();" in js


def test_disclosure_is_inline_not_popup_autoclose() -> None:
    js = _assets_module().ASSET_JS

    assert "toggle.setAttribute('aria-haspopup','true')" in js
    assert "toggle.setAttribute('aria-expanded','false')" in js
    assert "polyline.setAttribute('points','6 9 12 15 18 9')" in js
    assert "if(polyline)polyline.setAttribute('points'" in js
    assert "panel.hidden=!open" in js
    assert "open?'6 15 12 9 18 15':'6 9 12 15 18 9'" in js
    assert "document.addEventListener('pointerdown'" not in js
    assert "More options" in js


def test_youtube_layers_keep_collection_as_browser_ui_owner() -> None:
    collection = (ROOT / "_sphinx_collection" / "README.md").read_text(encoding="utf-8")
    core = (ROOT / "_sphinx_youtube_core" / "__init__.py").read_text(encoding="utf-8")
    gallery = (ROOT / "_sphinx_youtube_gallery" / "README.md").read_text(encoding="utf-8")
    contrib = (ROOT / "_sphinxcontrib_youtube" / "__init__.py").read_text(encoding="utf-8")
    directive = (ROOT / "_sphinx_youtube_gallery" / "directive.py").read_text(encoding="utf-8")

    assert "owns the progressive browser UI" in collection
    assert "owns no browser/search controls" in core
    assert "same compact search/disclosure interaction as" in gallery
    assert "centralized" in contrib and "_sphinx_collection" in contrib
    assert "one implementation" in directive
    assert "for key in (\"searchable\", \"interactive\")" in directive
    assert "primary search row comes first" in gallery
    assert "live result count is a separate row" in gallery
    assert "left-aligned result count, then one primary row" not in gallery


def test_expanded_panel_has_grouped_information_architecture() -> None:
    assets = _assets_module()
    css = assets.ASSET_CSS
    js = assets.ASSET_JS

    assert ".sk-collection-panel-section {" in css
    assert ".sk-collection-view-grid {" in css
    assert ".sk-collection-tools-grid {" in css
    assert ".sk-collection-tool[open] { grid-column:1 / -1;" in css
    assert ".sk-collection-tool-body {" in css
    assert "sk-collection-close" not in css

    assert "sk-collection-panel-title','View'" in js
    assert "sk-collection-panel-title','Gallery tools'" in js
    assert "sk-collection-tool sk-collection-add-details" in js
    assert "sk-collection-tool sk-collection-preferences" in js
    assert "sk-collection-tool sk-collection-export" in js
    assert "Restore original gallery" in js
    assert "tool.addEventListener('toggle'" in js
    assert "other!==tool&&other.parentElement===toolsGrid" in js
    assert "Close options" not in js
    assert "Reset view clears search, filters and sorting" not in js


def test_expanded_panel_keeps_escape_as_single_close_path() -> None:
    js = _assets_module().ASSET_JS

    assert "panel.addEventListener('keydown',function(event){if(event.key==='Escape')" in js
    assert "toggle.focus()" in js
    assert "close.addEventListener" not in js


def test_search_variant_is_one_shared_presentation_contract() -> None:
    assets = _assets_module()
    browser = (ROOT / "_sphinx_collection" / "_browser.py").read_text(encoding="utf-8")
    gallery = (ROOT / "_sphinx_gallery_grid" / "directive.py").read_text(encoding="utf-8")
    youtube = (ROOT / "_sphinx_youtube_gallery" / "directive.py").read_text(encoding="utf-8")
    conf = (_docs_source() / "conf.py").read_text(encoding="utf-8")

    assert assets.SEARCH_VARIANTS == ("pill-overflow", "classic")
    assert '"searchVariant": options.get("search-variant", "pill-overflow")' in browser
    assert '"search-variant": search_variant_option' in gallery
    assert '"search_variant": search_variant_option' in gallery
    assert '"interactive": search_variant_option' in gallery
    assert '"searchable": search_variant_option' in gallery
    assert 'app.add_config_value(' in gallery
    assert '"collection_search_variant", "pill-overflow", "env", types=[str]' in gallery
    assert "collection_search_variant must be 'pill-overflow' or 'classic'" in gallery
    assert 'self.config.collection_search_variant' in gallery
    assert '"search-variant": search_variant_option' in youtube
    assert '"search_variant": search_variant_option' in youtube
    assert '"interactive": search_variant_option' in youtube
    assert '"searchable": search_variant_option' in youtube
    assert 'options["search-variant"] = resolve_search_variant(' in youtube
    assert 'collection_search_variant = "pill-overflow"  # alternative: "classic"' in conf


def test_per_directive_search_variant_supports_flags_shorthand_aliases_and_conflict_rejection() -> None:
    import sys

    source_root = ROOT.parent
    sys.path.insert(0, str(source_root))
    try:
        from _sphinx_ext._search_variant import resolve_search_variant, search_variant_option
    finally:
        sys.path.pop(0)

    assert search_variant_option(None) is None
    assert search_variant_option(" CLASSIC ") == "classic"
    assert resolve_search_variant({"interactive": None}, "pill-overflow") == "pill-overflow"
    assert resolve_search_variant({"interactive": "classic"}, "pill-overflow") == "classic"
    assert resolve_search_variant({"searchable": "pill-overflow"}, "classic") == "pill-overflow"
    assert resolve_search_variant({"search_variant": "classic"}, "pill-overflow") == "classic"
    assert resolve_search_variant({"search-variant": "classic", "interactive": "classic"}, "pill-overflow") == "classic"
    assert resolve_search_variant({"interactive": None, "search_variant": "classic"}, "pill-overflow") == "classic"
    assert resolve_search_variant({"searchable": "pill-overflow", "search_variant": "pill-overflow"}, "classic") == "pill-overflow"
    try:
        resolve_search_variant({"interactive": "classic", "search-variant": "pill-overflow"}, "pill-overflow")
    except ValueError as exc:
        assert "conflicting search variants" in str(exc)
    else:
        raise AssertionError("conflicting per-directive search variants must fail closed")


def test_leaf_youtube_layers_remain_search_ui_free() -> None:
    core = (ROOT / "_sphinx_youtube_core" / "__init__.py").read_text(encoding="utf-8")
    contrib = (ROOT / "_sphinxcontrib_youtube" / "__init__.py").read_text(encoding="utf-8")
    assert "owns no browser/search controls" in core
    assert "centralized" in contrib and "_sphinx_collection" in contrib
    assert "search-variant" not in core
    assert "search-variant" not in contrib


def test_collection_assets_are_html_only_and_atomically_written() -> None:
    setup = (ROOT / "_sphinx_collection" / "setup.py").read_text(encoding="utf-8")
    assert 'getattr(getattr(app, "builder", None), "format", None) != "html"' in setup
    assert 'def _write_asset_atomic(path: Path, content: str) -> bool' in setup
    assert 'os.replace(temporary, path)' in setup
    assert 'path.read_bytes() == data' in setup
    assert '.write_text(ASSET_CSS' not in setup
    assert '.write_text(ASSET_JS' not in setup


def test_collection_asset_registration_runtime_is_html_only_and_idempotent(tmp_path) -> None:
    import ast
    import os
    import tempfile
    from types import SimpleNamespace

    source_path = ROOT / "_sphinx_collection" / "setup.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"_write_asset_atomic", "ensure_assets"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]

    class Logger:
        def warning(self, *args, **kwargs):
            raise AssertionError(f"unexpected asset warning: {args!r}")

    namespace = {
        "Path": Path,
        "os": os,
        "tempfile": tempfile,
        "_FLAG": "_registered",
        "_CHANGED_FLAG": "_changed",
        "_CSS_NAME": "sk-collection.css",
        "_JS_NAME": "sk-collection.js",
        "CONTRACT_CLASS": "sk-collection-controls-status-results-v4",
        "STATUS_SOURCE_ATTRIBUTE": "data-sk-collection-status-source",
        "STATUS_SOURCE_DOCUMENT": "document",
        "ASSET_CSS": "body{display:block}",
        "ASSET_JS": "console.log('ok');",
        "collection_asset_revision": lambda: "rev-1",
        "logger": Logger(),
        "Any": object,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), namespace)
    ensure = namespace["ensure_assets"]

    calls = []
    app = SimpleNamespace(
        builder=SimpleNamespace(format="latex"),
        outdir=str(tmp_path / "latex"),
        add_css_file=lambda *args, **kwargs: calls.append(("css", args, kwargs)),
        add_js_file=lambda *args, **kwargs: calls.append(("js", args, kwargs)),
    )
    ensure(app)
    assert calls == []
    assert not (tmp_path / "latex").exists()

    app.builder.format = "html"
    app.outdir = str(tmp_path / "html")
    ensure(app)
    assert [row[0] for row in calls] == ["css", "js"]
    assert (tmp_path / "html" / "_static" / "sk-collection.css").read_text() == "body{display:block}"
    assert (tmp_path / "html" / "_static" / "sk-collection.js").read_text() == "console.log('ok');"
    before = (tmp_path / "html" / "_static" / "sk-collection.js").stat().st_mtime_ns
    ensure(app)
    after = (tmp_path / "html" / "_static" / "sk-collection.js").stat().st_mtime_ns
    assert before == after
    assert [row[0] for row in calls] == ["css", "js"]

    # A reloaded asset module in a long-lived builder can expose a new digest
    # on the same Sphinx app.  Registration must follow the digest, not a bool.
    namespace["collection_asset_revision"] = lambda: "rev-2"
    namespace["ASSET_JS"] = "console.log('new');"
    ensure(app)
    assert (tmp_path / "html" / "_static" / "sk-collection.js").read_text() == "console.log('new');"
    assert [row[0] for row in calls] == ["css", "js", "css", "js"]


def test_collection_asset_revision_invalidates_incremental_html_pages() -> None:
    import ast
    import hashlib
    from types import SimpleNamespace

    source_path = ROOT / "_sphinx_collection" / "setup.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {
        "collection_asset_revision",
        "collection_assets_outdated",
        "remember_collection_asset_revision",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {
        "Any": object,
        "hashlib": hashlib,
        "ASSET_CSS": "old-css",
        "ASSET_JS": "old-js",
        "_ENV_REVISION": "_asset_revision",
        "_ENV_UI_CONTRACT": "_ui_contract",
        "COLLECTION_UI_CONTRACT": "controls-status-results-v4",
        "_CHANGED_FLAG": "_assets_changed",
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    revision = namespace["collection_asset_revision"]
    outdated = namespace["collection_assets_outdated"]
    remember = namespace["remember_collection_asset_revision"]

    app = SimpleNamespace(builder=SimpleNamespace(format="html"))
    env = SimpleNamespace(found_docs={"examples/index", "examples/youtube-gallery", "removed"})

    assert outdated(app, env, set(), set(), {"removed"}) == [
        "examples/index",
        "examples/youtube-gallery",
    ]
    assert not hasattr(env, "_asset_revision")

    remember(app, env)
    assert env._asset_revision == revision()
    assert env._ui_contract == "controls-status-results-v4"
    assert outdated(app, env, set(), set(), set()) == []

    # Structural contract changes invalidate HTML even if CSS/JS bytes happen
    # to be unchanged. This is what forces old embedded-status doctrees/pages
    # to be reread and rewritten.
    env._ui_contract = "status-sibling-v3-document"
    assert outdated(app, env, set(), set(), set()) == [
        "examples/index",
        "examples/youtube-gallery",
        "removed",
    ]
    remember(app, env)

    namespace["ASSET_JS"] = "new-js"
    assert outdated(app, env, set(), set(), set()) == [
        "examples/index",
        "examples/youtube-gallery",
        "removed",
    ]

    # A physically replaced output asset is independently sufficient to force
    # rewrite even if an environment revision happens to match.
    remember(app, env)
    app._assets_changed = True
    assert outdated(app, env, set(), set(), set()) == [
        "examples/index",
        "examples/youtube-gallery",
        "removed",
    ]
    assert app._assets_changed is False
    assert outdated(app, env, set(), set(), set()) == []

    app.builder.format = "latex"
    assert outdated(app, env, set(), set(), set()) == []


def test_gallery_grid_registers_collection_asset_revision_lifecycle() -> None:
    gallery = (ROOT / "_sphinx_gallery_grid" / "directive.py").read_text(encoding="utf-8")
    assert "register_collection_asset_revision(app)" in gallery
    assert 'app.connect("env-get-outdated", collection_assets_outdated)' in gallery
    assert 'app.connect("env-updated", remember_collection_asset_revision)' in gallery
    assert 'app.connect("build-finished", verify_collection_assets)' in gallery


def test_collection_asset_revision_uses_sphinx_native_html_rebuild_contract() -> None:
    import ast
    from types import SimpleNamespace

    source_path = ROOT / "_sphinx_collection" / "setup.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "register_collection_asset_revision"
    )
    calls = []
    namespace = {
        "Any": object,
        "_CONFIG_REVISION": "sk_collection_asset_revision",
        "_CONFIG_UI_CONTRACT": "sk_collection_ui_contract",
        "COLLECTION_UI_CONTRACT": "controls-status-results-v4",
        "collection_asset_revision": lambda: "digest-123",
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    app = SimpleNamespace(
        add_config_value=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    namespace["register_collection_asset_revision"](app)
    assert calls == [
        (("sk_collection_asset_revision", "digest-123", "html"), {"types": [str]}),
        (("sk_collection_ui_contract", "controls-status-results-v4", "html"), {"types": [str]}),
    ]


def test_final_collection_asset_integrity_fails_closed_on_stale_output(tmp_path) -> None:
    import ast
    from types import SimpleNamespace

    source_path = ROOT / "_sphinx_collection" / "setup.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_collection_assets"
    )

    class AssetIntegrityError(RuntimeError):
        pass

    namespace = {
        "Any": object,
        "Path": Path,
        "ExtensionError": AssetIntegrityError,
        "ASSET_CSS": "current-css",
        "ASSET_JS": "current-js",
        "_CSS_NAME": "sk-collection.css",
        "_JS_NAME": "sk-collection.js",
        "CONTRACT_CLASS": "sk-collection-controls-status-results-v4",
        "STATUS_SOURCE_ATTRIBUTE": "data-sk-collection-status-source",
        "STATUS_SOURCE_DOCUMENT": "document",
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    verify = namespace["verify_collection_assets"]
    static = tmp_path / "html" / "_static"
    static.mkdir(parents=True)
    (static / "sk-collection.css").write_text("current-css", encoding="utf-8")
    (static / "sk-collection.js").write_text("current-js", encoding="utf-8")
    app = SimpleNamespace(
        builder=SimpleNamespace(format="html"), outdir=str(tmp_path / "html")
    )
    verify(app, None)

    # Old searchable HTML without the document-owned status sibling is also
    # rejected even when the CSS/JS bytes themselves are current.
    stale_page = tmp_path / "html" / "examples" / "index.html"
    stale_page.parent.mkdir(parents=True)
    stale_page.write_text(
        '<div class="sk-collection sk-collection-searchable docutils container"></div>',
        encoding="utf-8",
    )
    try:
        verify(app, None)
    except AssetIntegrityError as exc:
        assert "document-owned status sibling" in str(exc)
        assert "examples/index.html" in str(exc)
    else:
        raise AssertionError("stale searchable gallery HTML must fail closed")
    stale_page.write_text(
        '<div class="sk-collection sk-collection-searchable sk-collection-controls-status-results-v4 docutils container">'
        '<p hidden class="sk-collection-status" data-sk-collection-status-source="document"></p>'
        '</div>',
        encoding="utf-8",
    )
    verify(app, None)

    (static / "sk-collection.js").write_text("stale-js", encoding="utf-8")
    try:
        verify(app, None)
    except AssetIntegrityError as exc:
        assert "Shared collection UI integrity failure" in str(exc)
        assert "sk-collection.js" in str(exc)
        assert "mixed-version gallery UI" in str(exc)
    else:
        raise AssertionError("stale final collection assets must fail closed")

    # Do not replace an existing build exception with an asset-integrity error.
    verify(app, RuntimeError("earlier build failure"))
    app.builder.format = "latex"
    verify(app, None)


def test_docs_build_has_explicit_local_and_installed_extension_authorities() -> None:
    conf = (_docs_source() / "conf.py").read_text(encoding="utf-8")
    makefile = (_docs_source().parent / "Makefile").read_text(encoding="utf-8")
    make_bat = (_docs_source().parent / "make.bat").read_text(encoding="utf-8")
    namespace_init = (ROOT / "__init__.py").read_text(encoding="utf-8")

    assert '_MODE_ENV = "SCIKITPLOT_SPHINX_EXT_MODE"' in conf
    assert '_ALLOWED_SPHINX_EXT_MODES = frozenset({"auto", "local", "installed"})' in conf
    assert '_REQUIRED_PRIVATE_MODULES = (' in conf
    assert '"_sphinx_youtube_core"' in conf
    assert '"_sphinx_collection"' in conf
    assert 'installed_namespace = "scikitplot._externals._sphinx_ext"' in conf
    assert 'namespace = "_sphinx_ext"' in conf
    assert '_EXPECTED_SPHINX_EXT_STACK_API = 1' in conf
    assert '_COLLECTION_UI_CONTRACT_EXPECTED = "controls-status-results-v4"' in conf
    assert 'SPHINX_EXT_STACK_API = 1' in namespace_init
    assert 'SPHINX_EXT_MODE ?= local' in makefile
    assert 'SCIKITPLOT_SPHINX_EXT_MODE="$(SPHINX_EXT_MODE)"' in makefile
    assert 'make SPHINX_EXT_MODE=installed html' in makefile
    assert 'set SCIKITPLOT_SPHINX_EXT_MODE=local' in make_bat
    assert 'SCIKITPLOT_SPHINX_EXT_MODE=installed' in make_bat
    assert 'Sphinx extension provenance mismatch' not in conf
    assert 'expected a module under' not in conf


def test_local_authority_isolated_from_preimported_scikitplot(tmp_path) -> None:
    """A loaded foreign scikitplot package must not capture local docs extensions."""
    import subprocess

    fake = tmp_path / "foreign"
    package = fake / "scikitplot"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "ORIGIN = 'foreign-scikitplot'\n", encoding="utf-8"
    )
    externals = _docs_source() / "scikitplot" / "_externals"
    conf_path = _docs_source() / "conf.py"
    code = f"""
import os
import runpy
import sys
from pathlib import Path
sys.path.insert(0, {str(fake)!r})
import scikitplot
assert Path(scikitplot.__file__).resolve().is_relative_to(Path({str(fake)!r}).resolve())
os.environ['SCIKITPLOT_SPHINX_EXT_MODE'] = 'local'
ns = runpy.run_path({str(conf_path)!r})
assert ns['_SPHINX_EXT_MODE'] == 'local'
assert ns['_SPHINX_EXT_NAMESPACE'] == '_sphinx_ext'
private = [name for name in ns['extensions'] if name.startswith('_sphinx_ext.')]
assert '_sphinx_ext._sphinx_gallery_grid' in private
assert '_sphinx_ext._sphinx_youtube_gallery' in private
assert '_sphinx_ext._sphinx_ai_learn' in private
import _sphinx_ext
assert Path(_sphinx_ext.__file__).resolve().is_relative_to(Path({str(externals)!r}).resolve())
assert _sphinx_ext.SPHINX_EXT_STACK_API == 1
from _sphinx_ext._sphinx_collection.contract import COLLECTION_UI_CONTRACT
assert COLLECTION_UI_CONTRACT == 'controls-status-results-v4'
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_installed_authority_uses_scikitplot_namespace_without_local_path_injection(tmp_path) -> None:
    """Stable mode must stay inside the installed scikitplot package namespace."""
    import subprocess

    fake = tmp_path / "site"
    root = fake / "scikitplot" / "_externals" / "_sphinx_ext"
    root.mkdir(parents=True)
    for package in (fake / "scikitplot", fake / "scikitplot" / "_externals", root):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (root / "__init__.py").write_text("SPHINX_EXT_STACK_API = 1\n", encoding="utf-8")
    collection = root / "_sphinx_collection"
    collection.mkdir()
    (collection / "__init__.py").write_text("", encoding="utf-8")
    (collection / "contract.py").write_text(
        "COLLECTION_UI_CONTRACT = 'controls-status-results-v4'\n", encoding="utf-8"
    )
    for module_name in ("_extension_setup.py", "_search_variant.py"):
        (root / module_name).write_text("", encoding="utf-8")
    required = (
        "_sphinx_youtube_core",
        "_pydata_component_list",
        "_sphinx_gallery_grid",
        "_sphinxcontrib_youtube",
        "_sphinx_youtube_gallery",
        "_sphinx_ai_assistant",
        "_sphinx_feedback",
        "_sphinx_ai_learn",
    )
    for name in required:
        package = root / name
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")

    conf_path = _docs_source() / "conf.py"
    code = f"""
import os
import runpy
import sys
from pathlib import Path
sys.path.insert(0, {str(fake)!r})
os.environ['SCIKITPLOT_SPHINX_EXT_MODE'] = 'installed'
ns = runpy.run_path({str(conf_path)!r})
assert ns['_SPHINX_EXT_MODE'] == 'installed'
assert ns['_SPHINX_EXT_NAMESPACE'] == 'scikitplot._externals._sphinx_ext'
assert ns['scikitplot_sphinx_ext_mode'] == 'installed'
assert ns['scikitplot_sphinx_ext_namespace'] == 'scikitplot._externals._sphinx_ext'
private = [name for name in ns['extensions'] if name.startswith('scikitplot._externals._sphinx_ext.')]
assert len(private) == 7
import scikitplot._externals._sphinx_ext as stack
assert Path(stack.__file__).resolve().is_relative_to(Path({str(fake)!r}).resolve())
local_externals = str(Path({str(_docs_source())!r}) / 'scikitplot' / '_externals')
assert local_externals not in sys.path
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_auto_authority_fails_closed_when_local_and_installed_are_both_available(tmp_path) -> None:
    """Auto mode must never guess between two complete extension authorities."""
    import subprocess

    fake = tmp_path / "site"
    root = fake / "scikitplot" / "_externals" / "_sphinx_ext"
    root.mkdir(parents=True)
    for package in (fake / "scikitplot", fake / "scikitplot" / "_externals", root):
        (package / "__init__.py").write_text("", encoding="utf-8")
    conf_path = _docs_source() / "conf.py"
    code = f"""
import os
import runpy
import sys
sys.path.insert(0, {str(fake)!r})
os.environ['SCIKITPLOT_SPHINX_EXT_MODE'] = 'auto'
try:
    runpy.run_path({str(conf_path)!r})
except RuntimeError as exc:
    message = str(exc)
    assert 'Both local and installed' in message
    assert 'SCIKITPLOT_SPHINX_EXT_MODE=local' in message
    assert 'SCIKITPLOT_SPHINX_EXT_MODE=installed' in message
else:
    raise AssertionError('ambiguous auto mode must fail closed')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_installed_authority_rejects_old_stack_api(tmp_path) -> None:
    """Stable mode must fail clearly when the installed library is too old."""
    import subprocess

    fake = tmp_path / "site"
    root = fake / "scikitplot" / "_externals" / "_sphinx_ext"
    root.mkdir(parents=True)
    for package in (fake / "scikitplot", fake / "scikitplot" / "_externals", root):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (root / "__init__.py").write_text("SPHINX_EXT_STACK_API = 0\n", encoding="utf-8")
    conf_path = _docs_source() / "conf.py"
    code = f"""
import os
import runpy
import sys
sys.path.insert(0, {str(fake)!r})
os.environ['SCIKITPLOT_SPHINX_EXT_MODE'] = 'installed'
try:
    runpy.run_path({str(conf_path)!r})
except RuntimeError as exc:
    message = str(exc)
    assert 'stack API' in message
    assert 'expected 1' in message
    assert 'got 0' in message
else:
    raise AssertionError('old installed extension stack must fail closed')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_private_extension_stack_uses_relative_cross_extension_imports() -> None:
    """Sibling extension code must not pin itself to either outer namespace."""
    import ast

    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "_sphinx_ext" or alias.name.startswith(
                        ("_sphinx_ext.", "scikitplot._externals._sphinx_ext")
                    ):
                        offenders.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module == "_sphinx_ext" or node.module.startswith(
                    ("_sphinx_ext.", "scikitplot._externals._sphinx_ext")
                ):
                    offenders.append((path, node.lineno, node.module))
    assert offenders == []


def test_collection_status_has_one_structural_owner_across_all_four_layers() -> None:
    collection_browser = (ROOT / "_sphinx_collection" / "_browser.py").read_text(encoding="utf-8")
    gallery_grid = (ROOT / "_sphinx_gallery_grid" / "directive.py").read_text(encoding="utf-8")
    youtube_core = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "_sphinx_youtube_core").glob("*.py"))
    )
    youtube_gallery = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "_sphinx_youtube_gallery").glob("*.py"))
    )

    # _sphinx_collection defines the one status primitive. gallery-grid places
    # it as a root child. YouTube layers delegate and never emit competing UI.
    assert 'def status_node(count):' in collection_browser
    assert 'STATUS_CLASS' in collection_browser
    assert 'wrapper += status_node(len(self._browser_records))' in gallery_grid
    assert 'classes.extend((SEARCHABLE_CLASS, CONTRACT_CLASS))' in gallery_grid
    assert 'class="sk-collection-status"' not in gallery_grid
    assert "sk-collection-status" not in youtube_core
    assert "sk-collection-status" not in youtube_gallery
    assert "sk-collection-controls" not in youtube_core
    assert "sk-collection-controls" not in youtube_gallery
    assert "controls -> status -> cards" in youtube_core
    assert "controls -> status -> cards" in youtube_gallery
    assert "expected exactly one document-owned" in youtube_gallery
    assert "if is_document_status_node(child)" in youtube_gallery
    assert ".rawsource" not in youtube_gallery


def test_four_collection_layers_use_package_relative_sibling_imports() -> None:
    """Shared collection dependencies must follow the selected package root."""
    package_sources = {}
    for name in (
        "_sphinx_collection",
        "_sphinx_gallery_grid",
        "_sphinx_youtube_core",
        "_sphinx_youtube_gallery",
    ):
        package = ROOT / name
        package_sources[name] = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(package.glob("*.py"))
        )

    for source in package_sources.values():
        assert "from scikitplot._externals._sphinx_ext" not in source
        assert "import scikitplot._externals._sphinx_ext" not in source

    assert "from .assets import ASSET_CSS, ASSET_JS" in package_sources["_sphinx_collection"]
    assert "from .._sphinx_collection import (" in package_sources["_sphinx_gallery_grid"]
    assert "from .._sphinx_collection import (" in package_sources["_sphinx_youtube_gallery"]
    assert "from .._sphinx_collection._browser import (" in package_sources["_sphinx_youtube_gallery"]
    assert "is_document_status_node" in package_sources["_sphinx_youtube_gallery"]

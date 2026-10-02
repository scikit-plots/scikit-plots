"""Import boundaries and lazy namespace discovery."""
from pathlib import Path
import subprocess
import sys


def test_core_import_has_no_optional_dependencies():
    root = Path(__file__).resolve().parents[3]
    code = (
        "import sys; sys.path.insert(0, " + repr(str(root)) + "); "
        "import _sphinx_ext; from _sphinx_ext import _sphinx_ai_learn; "
        "assert '_sphinx_ai_learn' in dir(_sphinx_ext); "
        "assert callable(_sphinx_ai_learn.load_content_tree); assert callable(_sphinx_ai_learn.materialize); "
        "assert not any(k.split('.')[0] in ('sphinx','docutils','bs4','fastapi') for k in sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_extension_version_has_one_owner():
    from _sphinx_ext._sphinx_ai_learn import __version__

    assert __version__ == "0.49.0"
    source = (Path(__file__).resolve().parents[1] / "_sphinx.py").read_text(encoding="utf-8")
    assert 'from . import __version__' in source
    assert source.count('"version": __version__') == 2
    assert '"version": "0.49.0"' not in source
    assert source.count('"env_version": _ENV_VERSION') == 2
    assert 'app.connect("env-merge-info", _merge_feedback_consumers)' in source


def test_parallel_feedback_consumer_merge_is_filtered_and_idempotent():
    import ast
    from types import SimpleNamespace

    source_path = Path(__file__).resolve().parents[1] / "_sphinx.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"_merge_feedback_consumers", "_purge_feedback_consumer"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), namespace)

    env = SimpleNamespace(_ai_learn_feedback_consumers={
        "topic-a": {"existing/doc", "worker/one"},
        "stale-record": {"worker/two"},
    })
    other = SimpleNamespace(_ai_learn_feedback_consumers={
        "topic-a": {"worker/one", "outside/doc"},
        "topic-b": {"worker/two", "outside/doc"},
    })
    merge = namespace["_merge_feedback_consumers"]
    merge(None, env, {"worker/one", "worker/two"}, other)
    assert env._ai_learn_feedback_consumers == {
        "topic-a": {"existing/doc", "worker/one"},
        "topic-b": {"worker/two"},
    }
    # Re-merging the same worker data is idempotent and does not resurrect the
    # stale record that used to be consumed by worker/two.
    merge(None, env, {"worker/one", "worker/two"}, other)
    assert env._ai_learn_feedback_consumers == {
        "topic-a": {"existing/doc", "worker/one"},
        "topic-b": {"worker/two"},
    }
    # A re-read with no feedback dependencies removes the old master mapping.
    empty = SimpleNamespace(_ai_learn_feedback_consumers={})
    merge(None, env, {"worker/two"}, empty)
    assert env._ai_learn_feedback_consumers == {
        "topic-a": {"existing/doc", "worker/one"},
    }

    purge = namespace["_purge_feedback_consumer"]
    purge(None, env, "worker/one")
    purge(None, env, "worker/two")
    assert env._ai_learn_feedback_consumers == {"topic-a": {"existing/doc"}}
    purge(None, env, "existing/doc")
    assert env._ai_learn_feedback_consumers == {}



def test_optional_extension_switches_are_validated_before_setup_side_effects():
    source = (Path(__file__).resolve().parents[1] / "_sphinx.py").read_text(encoding="utf-8")
    namespace_guard = source.index('check_namespace(app, root)')
    design_setup = source.index('app.setup_extension("sphinx_design")')
    config_registration = source.index('app.add_config_value("ai_learn_content_root"')
    assert namespace_guard < design_setup < config_registration
    media_check = source.index('if not isinstance(media, bool):')
    media_setup = source.index('app.setup_extension(root + "._sphinx_gallery_grid")')
    runtime_check = source.index('if runtime not in ("none", "assistant"):')
    runtime_setup = source.index('app.setup_extension(root + "._sphinx_ai_assistant")')
    assert media_check < media_setup
    assert runtime_check < runtime_setup
    assert source.count('ai_learn_media must be true or false') == 2
    # The success guard is committed only after all registration side effects.
    # A dependency/config failure cannot turn a half-registered app into a false success.
    assert source.index('app._ai_learn_registered = True') > source.index('app.connect("doctree-resolved", _resolve_routes)')


def test_ai_learn_config_paths_are_idempotent():
    import ast
    from types import SimpleNamespace

    source_path = Path(__file__).resolve().parents[1] / "_sphinx.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_append_unique_config_path")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source_path), "exec"), namespace)
    config = SimpleNamespace(html_static_path=["custom"], templates_path=None)
    helper = namespace["_append_unique_config_path"]
    helper(config, "html_static_path", "/tmp/assets")
    helper(config, "html_static_path", "/tmp/assets")
    helper(config, "templates_path", "/tmp/templates")
    helper(config, "templates_path", "/tmp/templates")
    assert config.html_static_path == ["custom", "/tmp/assets"]
    assert config.templates_path == ["/tmp/templates"]


def test_optional_youtube_environment_state_is_versioned_purged_and_parallel_safe():
    root = Path(__file__).resolve().parents[2] / "_sphinxcontrib_youtube"
    setup = (root / "__init__.py").read_text(encoding="utf-8")
    utils = (root / "utils.py").read_text(encoding="utf-8")
    assert '"env_version": 2' in setup
    assert '"video_download_thumbnails", False' in setup
    assert '"video_download_max_total_bytes"' in setup
    assert 'app.connect("config-inited", utils.validate_download_config)' in setup
    assert 'app.connect("env-purge-doc", utils.purge_download_images)' in setup
    assert 'app.connect("env-merge-info", utils.merge_download_images)' in setup
    assert 'def purge_download_images(app, env, docname):' in utils
    assert 'video_remote_images_by_doc' in utils
    assert 'env.images.add_file(env.docname, remote_images[url])' in utils
    assert 'for docname in set(docnames or ()):' in utils
    assert 'purge_download_images(app, env, docname)' in utils
    assert 'if not isinstance(getattr(app.env, "video_remote_images", None), dict):' in utils
    assert 'if output_path not in static_paths:' not in utils
    assert 'html_static_path' not in utils
    assert 'if not getattr(app.config, "video_download_thumbnails", False):' in utils
    assert 'DEFAULT_DOWNLOAD_MAX_TOTAL_BYTES' in utils


def test_optional_youtube_thumbnail_state_purge_and_parallel_merge_are_scoped():
    import ast
    from types import SimpleNamespace

    source_path = Path(__file__).resolve().parents[2] / "_sphinxcontrib_youtube" / "utils.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"purge_download_images", "merge_download_images"}
    fns = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(source_path), "exec"), namespace)
    purge = namespace["purge_download_images"]
    merge = namespace["merge_download_images"]

    env = SimpleNamespace(
        video_remote_images={"u1": "p1", "u2": "p2", "old": "po"},
        video_remote_images_by_doc={"doc/a": {"u1", "old"}, "doc/b": {"u1", "u2"}},
    )
    purge(None, env, "doc/a")
    assert env.video_remote_images == {"u1": "p1", "u2": "p2"}
    assert env.video_remote_images_by_doc == {"doc/b": {"u1", "u2"}}

    other = SimpleNamespace(
        video_remote_images={"u3": "p3", "outside": "px"},
        video_remote_images_by_doc={"doc/a": {"u3"}, "outside/doc": {"outside"}},
    )
    merge(None, env, {"doc/a"}, other)
    assert env.video_remote_images == {"u1": "p1", "u2": "p2", "u3": "p3"}
    assert env.video_remote_images_by_doc == {"doc/b": {"u1", "u2"}, "doc/a": {"u3"}}

    purge(None, env, "doc/b")
    assert env.video_remote_images == {"u3": "p3"}
    purge(None, env, "doc/a")
    assert env.video_remote_images == {}
    assert env.video_remote_images_by_doc == {}


def test_optional_youtube_download_config_is_bounded_and_fail_closed():
    import ast
    from types import SimpleNamespace

    source_path = Path(__file__).resolve().parents[2] / "_sphinxcontrib_youtube" / "utils.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"_bounded_config_int", "validate_download_config"}
    fns = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]

    class ConfigError(Exception):
        pass

    namespace = {
        "ConfigError": ConfigError,
        "_MAX_DOWNLOAD_LIMIT": 1000,
        "_MAX_DOWNLOAD_MAX_BYTES": 32 * 1024 * 1024,
        "_MAX_DOWNLOAD_MAX_TOTAL_BYTES": 512 * 1024 * 1024,
    }
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(source_path), "exec"), namespace)
    validate = namespace["validate_download_config"]

    good = SimpleNamespace(
        video_download_thumbnails=False,
        video_download_limit=200,
        video_download_max_bytes=8 * 1024 * 1024,
        video_download_max_total_bytes=64 * 1024 * 1024,
    )
    validate(None, good)
    good.video_download_thumbnails = True
    validate(None, good)

    bad_values = [
        ("video_download_thumbnails", 1),
        ("video_download_limit", True),
        ("video_download_limit", -1),
        ("video_download_limit", 1001),
        ("video_download_max_bytes", 0),
        ("video_download_max_bytes", 33 * 1024 * 1024),
        ("video_download_max_total_bytes", 0),
        ("video_download_max_total_bytes", 513 * 1024 * 1024),
    ]
    for field, value in bad_values:
        cfg = SimpleNamespace(
            video_download_thumbnails=False,
            video_download_limit=200,
            video_download_max_bytes=8 * 1024 * 1024,
            video_download_max_total_bytes=64 * 1024 * 1024,
        )
        setattr(cfg, field, value)
        try:
            validate(None, cfg)
        except ConfigError:
            pass
        else:
            raise AssertionError(f"{field}={value!r} should fail")

    cfg = SimpleNamespace(
        video_download_thumbnails=True,
        video_download_limit=0,
        video_download_max_bytes=8 * 1024 * 1024,
        video_download_max_total_bytes=64 * 1024 * 1024,
    )
    try:
        validate(None, cfg)
    except ConfigError:
        pass
    else:
        raise AssertionError("enabled thumbnail downloading requires a positive limit")

    cfg.video_download_limit = 1
    cfg.video_download_max_total_bytes = 1024
    try:
        validate(None, cfg)
    except ConfigError:
        pass
    else:
        raise AssertionError("aggregate budget below per-file budget should fail")


def test_optional_youtube_thumbnail_io_is_opt_in_and_https_only():
    source = (
        Path(__file__).resolve().parents[2] / "_sphinxcontrib_youtube" / "utils.py"
    ).read_text(encoding="utf-8")
    assert 'if getattr(env.config, "video_download_thumbnails", False):' in source
    assert 'url = _safe_thumbnail_url(self._thumbnail_url.format(video_id))' in source
    assert 'parsed.scheme != "https"' in source
    assert 'port not in (None, 443)' in source
    assert 'if not getattr(app.config, "video_download_thumbnails", False):' in source
    assert 'if "latex" not in app.builder.name:' in source
    assert '_received_total[0] + chunk_size > _max_total_bytes' in source


def test_optional_youtube_thumbnail_cache_path_and_url_are_revalidated():
    import ast
    from hashlib import sha256
    from pathlib import Path as _Path

    source_path = Path(__file__).resolve().parents[2] / "_sphinxcontrib_youtube" / "utils.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"_safe_thumbnail_url", "_thumbnail_path"}
    fns = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"sha256": sha256, "Path": _Path, "THUMBNAIL_DIR": "_video_thumbnail", "_THUMBNAIL_HOSTS": frozenset({"i3.ytimg.com", "vumbnail.com"})}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(source_path), "exec"), namespace)
    safe = namespace["_safe_thumbnail_url"]
    path_for = namespace["_thumbnail_path"]

    good = "https://i3.ytimg.com/vi/abcdefghijk/maxresdefault.jpg"
    assert safe(good) == good
    assert safe("https://i3.ytimg.com:443/vi/abcdefghijk/maxresdefault.jpg") is not None
    assert safe("http://i3.ytimg.com/vi/abcdefghijk/maxresdefault.jpg") is None
    assert safe("https://evil.example/x.jpg") is None
    assert safe("https://user@vumbnail.com/x.jpg") is None
    assert safe("https://example.com:444/x.jpg") is None
    assert safe("https://example.com/x.jpg#fragment") is None
    assert safe("https://[broken/x.jpg") is None

    cache_path = path_for("https://vumbnail.com/../../escape.jpg")
    assert cache_path.parent == _Path("_video_thumbnail")
    assert cache_path.name.endswith(".jpg")
    assert len(cache_path.stem) == 64
    int(cache_path.stem, 16)

    source = source_path.read_text(encoding="utf-8")
    assert "dst = Path(app.outdir) / _thumbnail_path(src)" in source
    assert "env.video_remote_images[src]" not in source
    assert "allow_redirects=False" in source
    assert 'if not content_type.lower().startswith("image/")' in source


def test_youtube_subscribe_config_rejects_malformed_and_nonstandard_authority():
    source = (Path(__file__).resolve().parents[1] / "_sphinx.py").read_text(encoding="utf-8")
    assert "_MAX_SUBSCRIBE_URL = 2048" in source
    assert "ord(ch) < 32 or ord(ch) == 127" in source
    assert "for ch in youtube_subscribe_url" in source
    assert "except ValueError as exc:" in source
    assert "port = parsed.port" in source
    assert "or port not in (None, 443)" in source


def test_ai_learn_buttons_ratings_config_is_strict_balanced_and_html_only():
    import ast

    source_path = Path(__file__).resolve().parents[1] / "_sphinx.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted_assignments = {
        "_AI_LEARN_BUTTON_RATING_POSITIONS",
        "_AI_LEARN_BUTTONS_RATINGS_DEFAULT",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted_assignments
            for target in node.targets
        ):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_normalize_ai_learn_buttons_ratings":
            selected.append(node)
    class ConfigError(Exception):
        pass
    namespace = {"ConfigError": ConfigError}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    normalize = namespace["_normalize_ai_learn_buttons_ratings"]

    assert normalize(None) == {
        "left_button_rating": "left",
        "right_button_rating": "right",
    }
    assert normalize({"left_button_rating": " RIGHT "}) == {
        "left_button_rating": "right",
        "right_button_rating": "right",
    }
    assert normalize({"right_button_rating": "LEFT"}) == {
        "left_button_rating": "left",
        "right_button_rating": "left",
    }
    for bad in ([], "left", 1, True):
        try:
            normalize(bad)
        except ConfigError as exc:
            assert "dictionary" in str(exc)
        else:
            raise AssertionError(f"expected ConfigError for {bad!r}")
    for bad in (
        {"left_button_rating": "center"},
        {"right_button_rating": 1},
        {"unknown_button_rating": "left"},
        {"unknown_button_rating": "left", 1: "right"},
    ):
        try:
            normalize(bad)
        except ConfigError:
            pass
        else:
            raise AssertionError(f"expected ConfigError for {bad!r}")

    assert '"ai_learn_buttons_ratings"' in source
    registration = source.index('app.add_config_value(\n        "ai_learn_buttons_ratings",')
    assert '"html",' in source[registration:registration + 260]
    assert 'app._ai_learn_buttons_ratings = _normalize_ai_learn_buttons_ratings(' in source

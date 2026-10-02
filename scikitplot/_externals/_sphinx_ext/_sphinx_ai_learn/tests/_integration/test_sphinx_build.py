"""Real Sphinx builds for the JSON materializer lifecycle."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from _sphinx_ext._sphinx_ai_learn._materialize import canonical_record_json_files

EXT = Path(__file__).resolve().parents[4]


def _write_record_tree(root, subjects, prompts=()):
    root.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        for relative, raw in canonical_record_json_files(subject, prompts).items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)


def _topic(title="Custom topic"):
    return {
        "id": "topic-a",
        "kind": "topic",
        "title": title,
        "summary": "</script><script>window.INJECTED=true</script>",
        "created_at": "2026-09-19T12:10:00Z",
        "domains": [],
        "related": [],
        "sections": [
            {
                "id": "summary",
                "title": "Summary",
                "body": ".. raw:: html\n\n   <img src=x onerror=alert(1)>",
                "citations": [],
                "links": [],
            }
        ],
    }


def build(tmp_path, theme="pydata_sphinx_theme", namespace="_sphinx_ext", extra=""):
    source = tmp_path / "source"
    source.mkdir()
    _write_record_tree(source / "learn-ai", [_topic()])
    root = str(EXT)
    config = "import sys\nsys.path.insert(0, " + repr(root) + ")\n"
    if namespace.startswith("scikitplot"):
        config += (
            "import types\n"
            "for name, path in [('scikitplot', " + repr(str(EXT.parent)) + "), "
            "('scikitplot._externals', " + repr(root) + ")]:\n"
            "    module = types.ModuleType(name); module.__path__ = [path]; sys.modules[name] = module\n"
        )
    config += (
        "project = 'Learn test'\n"
        "extensions = [" + repr(namespace + "._sphinx_ai_learn") + "]\n"
        "html_theme = " + repr(theme) + "\n"
        "ai_learn_content_root = 'learn-ai'\n"
    ) + extra
    (source / "conf.py").write_text(config)
    (source / "index.rst").write_text("Home\n====\n\n.. toctree::\n\n   hub\n")
    (source / "hub.rst").write_text(
        "Hub\n===\n\n.. ai-learn::\n\n.. ai-learn::\n   :subject: topic-a\n"
    )
    out = tmp_path / "html"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "-j",
            "2",
            "-W",
            str(source),
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    return result, source, out


@pytest.mark.parametrize("theme", ["pydata_sphinx_theme", "furo", "alabaster"])
def test_theme_build_static_fallback_and_page_scoped_assets(tmp_path, theme):
    result, source, out = build(tmp_path, theme)
    assert result.returncode == 0, result.stdout + result.stderr
    hub = (out / "hub.html").read_text()
    assert hub.count("data-skplt-learn-ai-mount") == 2
    assert "ai-learn.js" in hub
    assert "ai-learn.js" not in (out / "index.html").read_text()
    assert "<script>window.INJECTED" not in hub
    assert "<img src=x" not in hub
    assert "Custom topic" in hub and "raw:: html" in hub
    assert (out / "_static" / "ai-learn.css").is_file()
    assert list((source / "learn-ai").rglob("*.rst"))


def test_vendored_namespace_fixture(tmp_path):
    result, _, _ = build(tmp_path, namespace="scikitplot._externals._sphinx_ext")
    assert result.returncode == 0, result.stdout + result.stderr


def test_invalid_runtime_stops_build(tmp_path):
    result, _, _ = build(tmp_path, extra="ai_learn_runtime = 'arbitrary'\n")
    assert result.returncode != 0
    assert "ai_learn_runtime" in result.stderr


def test_json_change_invalidates_existing_doctree(tmp_path):
    result, source, out = build(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    index_json = next((source / "learn-ai/topics").glob("*/index.json"))
    data = json.loads(index_json.read_text())
    data["record"]["title"] = "Updated custom title"
    index_json.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "-W", str(source), str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Updated custom title" in (out / "hub.html").read_text()


@pytest.mark.parametrize("builder", ["html", "dirhtml"])
def test_materialized_record_routes_and_generic_resources(tmp_path, builder):
    source = tmp_path / "source"
    source.mkdir()
    subjects = []
    for kind in ("topic", "source", "problem", "audio", "document", "whiteboard", "video"):
        record = {
            "id": "record-" + kind,
            "kind": kind,
            "title": kind.capitalize(),
            "created_at": "2026-09-16T00:00:00Z",
            "domains": ["astronomy"],
            "related": [],
            "sections": [],
        }
        if kind in ("source", "whiteboard", "video"):
            record["url"] = "https://example.org/" + kind
        if kind == "audio":
            record["media"] = {
                "type": "audio",
                "src": "/_static/integration-audio.mp3",
                "mime_type": "audio/mpeg",
            }
        elif kind == "document":
            record["media"] = {
                "type": "document",
                "src": "/_static/integration-document.txt",
                "mime_type": "text/plain",
            }
        elif kind == "whiteboard":
            record["media"] = {
                "image": "/_static/integration-whiteboard.svg",
                "alt": "Integration whiteboard diagram",
            }
        subjects.append(record)
    static = source / "_static"
    static.mkdir()
    (static / "integration-whiteboard.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" '
        'viewBox="0 0 320 180"><rect width="320" height="180" fill="white"/>'
        '<text x="20" y="90">integration whiteboard</text></svg>'
    )
    (static / "integration-audio.mp3").write_bytes(b"ID3")
    (static / "integration-document.txt").write_text("integration document\n")
    _write_record_tree(source / "learn-ai", subjects)
    # The explorer under test is a page of its own, defined in canonical JSON
    # like every index page. Records alone produce detail pages and no
    # explorer, so a records-only tree built cleanly and never reached the
    # media-card branch this test exists to cover.
    whiteboards = source / "learn-ai" / "whiteboards"
    (whiteboards / "index.json").write_text(
        json.dumps(
            {
                "contract": "learn.page.v1",
                "view": "media-gallery",
                "kind": "whiteboard",
                "title": "Whiteboards",
                "hide_secondary_sidebar": True,
            },
            sort_keys=True,
        )
    )
    (whiteboards / "new.json").write_text(
        json.dumps(
            {
                "contract": "learn.page.v1",
                "view": "media-create",
                "kind": "whiteboard",
                "title": "Create a Whiteboard",
                "hide_secondary_sidebar": True,
            },
            sort_keys=True,
        )
    )
    # Routes are deterministic and can be referenced before Sphinx materializes RST.
    from _sphinx_ext._sphinx_ai_learn._materialize import record_docpath

    topic_route = record_docpath(subjects[0])
    (source / "index.rst").write_text(
        "Home\n====\n\n.. toctree::\n\n   learn-ai/"
        + topic_route
        + "\n   learn-ai/whiteboards/index\n"
    )
    (source / "conf.py").write_text(
        "import sys\nsys.path.insert(0, "
        + repr(str(EXT))
        + ")\nextensions = ['_sphinx_ext._sphinx_ai_learn']\n"
        + "ai_learn_content_root = 'learn-ai'\n"
        + "ai_learn_domains = ['astronomy']\n"
    )
    out = tmp_path / builder
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", builder, "-j", "2", "-W", str(source), str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    route = Path("learn-ai") / topic_route
    # A record's docname ends in ``/index``. The dirhtml builder writes such a
    # document to ``<dir>/index.html`` exactly as the html builder does; only a
    # docname with another last component becomes ``<name>/index.html``.
    if builder == "html" or route.name == "index":
        page = out / route.with_suffix(".html")
    else:
        page = out / route / "index.html"
    rendered = page.read_text()
    assert "Topic" in rendered
    assert "astronomy" in rendered
    assert "<iframe" not in rendered

    # Regression: the Whiteboard explorer must execute its media-card branch in
    # a real Sphinx parse/build. V41 crashed here by calling an undefined
    # _rst_text helper while synthesizing an ``.. image::`` directive.
    whiteboard_pages = list((out / "learn-ai" / "whiteboards").rglob("*.html"))
    assert whiteboard_pages
    whiteboard_html = "\n".join(path.read_text() for path in whiteboard_pages)
    assert "learn-whiteboard-card" in whiteboard_html
    assert "learn-whiteboard-index-image" in whiteboard_html
    assert "Integration whiteboard diagram" in whiteboard_html


def test_invalid_canonical_tree_fails_before_page_directives(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    root = source / "learn-ai"
    _write_record_tree(root, [_topic()])
    # An unowned section is an invalid canonical tree and must fail at config-inited.
    (root / "orphan.json").write_text(
        json.dumps(
            {
                "contract": "learn.section.v1",
                "record_id": "topic-a",
                "section": {
                    "id": "orphan",
                    "title": "Orphan",
                    "body": "",
                    "citations": [],
                    "links": [],
                },
            }
        )
    )
    (source / "index.rst").write_text("Home\n====\n\n.. ai-learn::\n")
    (source / "conf.py").write_text(
        "import sys\nsys.path.insert(0, "
        + repr(str(EXT))
        + ")\nextensions = ['_sphinx_ext._sphinx_ai_learn']\n"
        + "ai_learn_content_root = 'learn-ai'\n"
    )
    out = tmp_path / "html"
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", str(source), str(out)],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Unable to materialize AI Learn canonical JSON" in output
    assert "Unknown topic/catalog identifier" not in output


def test_environment_signature_invalidates_only_owned_learn_tree():
    from types import SimpleNamespace

    from _sphinx_ext._sphinx_ai_learn._sphinx import (
        _outdated_learn_documents,
        _purge_feedback_consumer,
        _remember_environment_signature,
    )

    app = SimpleNamespace(
        config=SimpleNamespace(ai_learn_content_root="learn-ai"),
        _ai_learn_content_digest="abc123",
        _ai_learn_feedback_digests={"topic-example": "feedback-a"},
        _ai_learn_routes={"topic-example": "topics/example/index"},
    )
    env = SimpleNamespace(
        found_docs={
            "index",
            "examples/index",
            "learn-ai/index",
            "learn-ai/topics/example/index",
            "learn-ai/topics/example/summary",
        }
    )
    assert _outdated_learn_documents(app, env, set(), set(), set()) == [
        "learn-ai/index",
        "learn-ai/topics/example/index",
        "learn-ai/topics/example/summary",
    ]
    env._ai_learn_feedback_consumers = {"topic-example": {"examples/index"}}
    _remember_environment_signature(app, env)
    assert _outdated_learn_documents(app, env, set(), set(), set()) == []

    app._ai_learn_feedback_digests = {"topic-example": "feedback-b"}
    assert _outdated_learn_documents(app, env, set(), set(), set()) == [
        "examples/index",
        "learn-ai/topics/example/index",
        "learn-ai/topics/example/summary",
    ]

    _purge_feedback_consumer(app, env, "examples/index")
    assert env._ai_learn_feedback_consumers == {}
